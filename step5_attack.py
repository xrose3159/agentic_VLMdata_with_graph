"""第五步：攻击式过滤（Attacker-based filtering）。

灵感：MTA-Agent / VSearcher / WebSTAR 的"生成很多 → 攻击 → 只留通不过攻击的"。

对每道 L3 题跑 4 个攻击器：
  - no_tool         : 只看图直接答（测试模型先验/视觉 shortcut）
  - no_image_search : 给 web/visit/code 但禁 image_search
  - no_visit        : 给 web/image/code 但禁 visit
  - no_code         : 给 web/image/visit 但禁 code_interpreter

任何一个攻击成功（答案命中 ground truth）→ reject 这道题。

分层 attacker：
  Tier A: Qwen3-VL-30B-A3B（便宜广筛，跑全量）
  Tier B: Gemini 3 Flash（更强复判，只跑 hard bucket，且不对 image_heavy/all_tools 做最终判决，因为 reverse API 还没真接）
  Tier C: Claude Sonnet 4.5（小规模校准，500 题以内）

用法：
    python step5_attack.py output/questions/sku_*.json --tier A --workers 4
    python step5_attack.py output/questions/sku_8846.json --tier B
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from openai import OpenAI

# 复用 eval_agent 里的工具实现
from eval_agent import (
    extract_json,
    tool_code_interpreter,
    tool_image_search,
    tool_visit,
    tool_web_search,
)

# ============================================================
# Tier 配置（攻击器分层）
# ============================================================

TIER_CONFIG = {
    "A": {
        "name": "Qwen3-VL-30B-A3B",
        "model": "qwen3-vl-30b-a3b-instruct",
        "base_url": "http://35.220.164.252:3888/v1",
        "api_key": "sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH",
        "scope": "all_l3",          # A 层跑全量 L3
        "block_image_heavy": False, # A 层可以对 image_heavy 做攻击（用于记录，但下游可以 ignore）
        "max_steps": 8,
    },
    "B": {
        "name": "Gemini 3 Flash",
        "model": "gemini-3-flash-preview-nothinking",
        "base_url": "http://35.220.164.252:3888/v1",
        "api_key": "sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH",
        "scope": "hard_passed_a",   # B 层只跑 A 通过的 hard bucket
        "block_image_heavy": True,  # ★ B 层不对 image_heavy/all_tools 做最终 reject（reverse 未接入）
        "max_steps": 10,
    },
    "C": {
        "name": "Claude Sonnet 4.5",
        "model": "claude-sonnet-4-5",
        "base_url": "http://35.220.164.252:3888/v1",
        "api_key": "sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH",
        "scope": "calibration_only", # C 层只用于小规模校准
        "block_image_heavy": True,
        "max_steps": 12,
    },
}

# ============================================================
# 4 个攻击器：工具白名单
# ============================================================

ATTACKS = {
    "no_tool":         {"allowed": set(), "label": "纯视觉/先验"},
    "no_image_search": {"allowed": {"web_search", "visit", "code_interpreter"}, "label": "禁图搜"},
    "no_visit":        {"allowed": {"web_search", "image_search", "code_interpreter"}, "label": "禁深读"},
    "no_code":         {"allowed": {"web_search", "image_search", "visit"}, "label": "禁代码"},
}

# ============================================================
# Bucket-aware 攻击调度（对齐 hard bucket 的语义）
# ============================================================

# Phase A0: 所有 L3 题都过 no_tool 这一关（便宜广筛）
# Phase A1: 根据工具签名 + bucket + is_ultra_long 动态决定攻击
# 不再只看 primary bucket 查表 → 改为函数 _compute_a1_attacks

# 旧表保留兼容（fallback 用，bucket 到攻击的默认映射）
BUCKET_PHASE_A1_ATTACKS: dict[str, list[str]] = {
    "image_heavy":  ["no_image_search"],
    "visit_heavy":  ["no_visit"],
    "code_heavy":   ["no_code"],
    "all_tools":    ["no_image_search", "no_visit", "no_code"],
    "ultra_long":   ["no_image_search", "no_visit", "no_code"],
    "chain_heavy":  [],
    "standard":     [],
    "":             [],
}


def _compute_a1_attacks(q: dict) -> list[str]:
    """从题目的 tool_sequence + bucket + is_ultra_long 动态决定 A1 攻击列表。

    原则：攻击调度跟"实际依赖的工具"对齐，不跟单一主 bucket 绑死。
    - image_heavy 先跑 no_image_search
    - 如果 is_ultra_long=True 且工具里还有 visit/code，继续补跑对应 ablation
    - visit_heavy 跑 no_visit
    - code_heavy 跑 no_code
    - chain_heavy / standard → 止步（A0 够了）
    """
    bucket = q.get("hard_bucket", "") or "standard"
    is_ul = q.get("is_ultra_long", False)
    tool_seq = q.get("tool_sequence", [])
    tools_in_plan = {s.get("tool") for s in tool_seq if isinstance(s, dict)}

    attacks: list[str] = []

    # 主 bucket 决定主攻击
    if bucket == "all_tools":
        return ["no_image_search", "no_visit", "no_code"]
    if bucket == "image_heavy":
        attacks.append("no_image_search")
    elif bucket == "visit_heavy":
        attacks.append("no_visit")
    elif bucket == "code_heavy":
        attacks.append("no_code")
    elif bucket == "ultra_long":
        # pure ultra_long（没被工具类 bucket 抢走的长题）
        return ["no_image_search", "no_visit", "no_code"]
    elif bucket in ("chain_heavy", "standard", ""):
        return []

    # 补充：如果 is_ultra_long + 工具里还有别的依赖，加对应 ablation
    if is_ul:
        if "visit" in tools_in_plan and "no_visit" not in attacks:
            attacks.append("no_visit")
        if "code_interpreter" in tools_in_plan and "no_code" not in attacks:
            attacks.append("no_code")
        if "image_search" in tools_in_plan and "no_image_search" not in attacks:
            attacks.append("no_image_search")

    return attacks

# max_steps 分层：no_tool 固定 1 call；单一 ablation 4；all_tools 三连 ablation 6
MAX_STEPS_BY_PHASE: dict[str, int] = {
    "A0_no_tool":         1,
    "A1_single_ablation": 4,
    "A1_cascade":         6,
}


def _build_attack_system_prompt(allowed: set[str]) -> str:
    """根据允许的工具集构建 system prompt。"""
    if not allowed:
        return """你是一个多模态智能体。你只能根据图片内容和你的先验知识回答问题，不能调用任何外部工具。

请直接给出答案，格式：
{"tool": "final_answer", "answer": "你的答案"}

如果不知道答案，回答：
{"tool": "final_answer", "answer": "unknown"}

不要输出 JSON 之外的任何内容。"""

    tools_desc = []
    if "code_interpreter" in allowed:
        tools_desc.append("""1. code_interpreter — 执行 Python（裁剪/OCR/计算）
   {"tool": "code_interpreter", "code": "..."}""")
    if "web_search" in allowed:
        tools_desc.append("""2. web_search — 网络搜索
   {"tool": "web_search", "query": "..."}""")
    if "image_search" in allowed:
        tools_desc.append("""3. image_search — 图片搜索
   {"tool": "image_search", "search_type": "text", "query": "..."}""")
    if "visit" in allowed:
        tools_desc.append("""4. visit — 访问网页
   {"tool": "visit", "url": "https://..."}""")

    tools_text = "\n\n".join(tools_desc)
    return f"""你是一个多模态智能体，需要回答关于图片的问题。你只能使用以下工具：

{tools_text}

每次只输出一个 JSON 工具调用。最后用 {{"tool": "final_answer", "answer": "..."}} 给出答案。

注意：除上面列出的工具外，禁止调用其他任何工具。"""


# ============================================================
# 受限攻击 agent loop
# ============================================================

def run_attack_agent(
    question: str,
    image_path: str,
    client: OpenAI,
    model: str,
    allowed_tools: set[str],
    max_steps: int = 8,
) -> dict:
    """带工具白名单的 agent loop。"""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    work_dir = tempfile.mkdtemp(prefix="attacker_")

    system = _build_attack_system_prompt(allowed_tools)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": f"问题：{question}\n\n请回答这个问题。"},
        ]},
    ]

    trace = []
    final_answer = None

    # no_tool 攻击：只允许 1 步
    actual_max_steps = 1 if not allowed_tools else max_steps

    for step in range(actual_max_steps):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=600,
                temperature=0.2,
                timeout=120,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            trace.append({"step": step + 1, "error": str(e)[:200]})
            break

        call = extract_json(raw)
        if not call:
            trace.append({"step": step + 1, "raw": raw[:200], "error": "json_parse_failed"})
            break

        tool = call.get("tool", "")
        entry = {"step": step + 1, "tool": tool}

        if tool == "final_answer":
            final_answer = call.get("answer", "")
            entry["answer"] = final_answer
            trace.append(entry)
            break

        # 工具白名单检查
        if tool not in allowed_tools:
            entry["blocked"] = True
            entry["reason"] = f"tool_not_allowed:{tool}"
            trace.append(entry)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"工具 {tool} 在本轮被禁用。请使用允许的工具或直接给出 final_answer。"})
            continue

        # 执行
        result_text = ""
        try:
            if tool == "code_interpreter":
                result_text = tool_code_interpreter(
                    call.get("code", ""), os.path.abspath(image_path), work_dir
                )
            elif tool == "web_search":
                result_text = tool_web_search(call.get("query", ""))
            elif tool == "image_search":
                result_text = tool_image_search(
                    call.get("search_type", "text"),
                    query=call.get("query", ""),
                )
            elif tool == "visit":
                result_text = tool_visit(call.get("url", ""), call.get("goal", ""))
            else:
                result_text = f"未知工具: {tool}"
        except Exception as e:
            result_text = f"工具执行异常: {e}"

        entry["output"] = result_text[:300]
        trace.append(entry)

        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"工具结果:\n{result_text[:1500]}\n\n请继续。"})

    return {"answer": final_answer, "trace": trace, "n_steps": len(trace)}


# ============================================================
# 答案命中检查
# ============================================================

def _normalize_for_match(s: str) -> str:
    s = (s or "").strip().lower()
    # 去掉常见标点
    for ch in ",.!?；。，！？\"'《》()[]【】（）":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def is_answer_correct(model_ans: str | None, ground_truth: str) -> bool:
    """模型答案是否包含 ground truth。"""
    if not model_ans:
        return False
    if not ground_truth or ground_truth.lower() in ("unknown", "rank_winner", ""):
        # 占位答案不可校验
        return False
    m = _normalize_for_match(model_ans)
    g = _normalize_for_match(ground_truth)
    if not m or not g:
        return False
    # 直接子串
    if g in m:
        return True
    # 关键词命中：所有 ≥3 字符的词都在 m 里
    g_words = [w for w in g.split() if len(w) >= 3]
    if g_words and all(w in m for w in g_words):
        return True
    return False


# ============================================================
# 攻击单道题
# ============================================================

def attack_one_question(
    q: dict,
    image_path: str,
    client: OpenAI,
    tier: dict,
    only_attacks: tuple[str, ...] | None = None,
    max_steps_override: int | None = None,
) -> dict:
    """对一道题跑攻击器，返回每个攻击的结果。

    only_attacks：可选，只跑指定的攻击（如 ('no_tool',)）。默认跑全部 4 个。
    max_steps_override：覆盖 tier.max_steps（用于 bucket-aware 调度的分层 max_steps）。
    """
    question_text = q.get("question", "")
    ground_truth = q.get("answer", "")

    attack_results = {}
    overall_attack_success = False

    attack_iter = [(name, ATTACKS[name]) for name in (only_attacks or list(ATTACKS.keys()))]
    effective_max_steps = max_steps_override if max_steps_override is not None else tier["max_steps"]

    for attack_name, attack_def in attack_iter:
        t0 = time.time()
        result = run_attack_agent(
            question_text,
            image_path,
            client,
            tier["model"],
            attack_def["allowed"],
            max_steps=effective_max_steps,
        )
        elapsed = time.time() - t0
        ans = result["answer"]
        correct = is_answer_correct(ans, ground_truth)

        attack_results[attack_name] = {
            "label": attack_def["label"],
            "model_answer": ans,
            "correct": correct,
            "n_steps": result["n_steps"],
            "elapsed_sec": round(elapsed, 1),
        }
        if correct:
            overall_attack_success = True
            break  # 早停

    return {
        "attack_results": attack_results,
        "attack_breached": overall_attack_success,
        "tier_used": tier["name"],
    }


# ============================================================
# Bucket-aware 两阶段攻击调度
# ============================================================

def run_two_phase_attack(
    question_files: list[str],
    tier: dict,
    workers_a0: int = 4,
    workers_a1: int = 2,
    only_levels: tuple[int, ...] = (3,),
) -> dict:
    """两阶段 bucket-aware 攻击：
      Phase A0: 所有 L3 题只跑 no_tool（workers_a0 并发，max_steps=1 固定）
      Phase A1: A0 通过的题按 bucket 跑 bucket-specific 攻击（workers_a1 并发，max_steps 分层）

    返回聚合的 run stats。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ---------- 统一做图像路径定位 ----------
    def _resolve_img(qfile: str, data: dict) -> str:
        img_id = data.get("image_id") or os.path.splitext(os.path.basename(qfile))[0]
        img_path = data.get("image_path", "")
        if not os.path.exists(img_path):
            for base in ("output/images", "images", "try"):
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    cand = os.path.join(base, f"{img_id}{ext}")
                    if os.path.exists(cand):
                        return cand
        return img_path

    client = OpenAI(
        api_key=tier["api_key"],
        base_url=tier["base_url"],
        http_client=httpx.Client(trust_env=False),
    )

    # =====================================================
    # Phase A0: no_tool on all L3, workers=workers_a0
    # =====================================================
    print(f"\n{'='*60}")
    print(f"Phase A0: no_tool 全量广筛 (workers={workers_a0})")
    print(f"{'='*60}")
    t_a0 = time.time()

    a0_targets = []  # [(qfile, data, q), ...]
    for qfile in question_files:
        try:
            with open(qfile, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[SKIP] {qfile}: read failed {e}")
            continue
        img_path = _resolve_img(qfile, data)
        if not os.path.exists(img_path):
            print(f"[SKIP] {qfile}: image not found")
            continue
        for lvl in only_levels:
            for q in data.get(f"level_{lvl}", []):
                # 跳过占位答案
                if (q.get("answer") or "").lower() in ("rank_winner", "unknown", ""):
                    continue
                a0_targets.append((qfile, data, q, img_path))

    def _run_a0(target):
        qfile, data, q, img_path = target
        return qfile, data, q, attack_one_question(
            q, img_path, client, tier,
            only_attacks=("no_tool",),
            max_steps_override=MAX_STEPS_BY_PHASE["A0_no_tool"],
        )

    a0_results: dict[str, list] = {}  # {qfile: [(q, res), ...]}
    with ThreadPoolExecutor(max_workers=workers_a0) as pool:
        futs = {pool.submit(_run_a0, t): t for t in a0_targets}
        for fut in as_completed(futs):
            try:
                qfile, data, q, res = fut.result()
            except Exception as e:
                print(f"  [A0 ERROR] {e}")
                continue
            a0_results.setdefault(qfile, []).append((data, q, res))

    # 写回 A0 结果
    a0_breached = 0
    a0_total = 0
    files_to_save: set[str] = set()
    for qfile, items in a0_results.items():
        for data, q, res in items:
            _apply_attack_result(q, res, tier)
            a0_total += 1
            if res.get("attack_breached"):
                a0_breached += 1
            files_to_save.add(qfile)
        # 找到这个 file 的 data 并保存
    # 保存每个文件（同一 file 的 data 共享）
    for qfile in files_to_save:
        items = a0_results.get(qfile) or []
        if items:
            data = items[0][0]  # 所有 items 共享同一 data ref
            with open(qfile, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Phase A0: {a0_total} 题攻击，{a0_breached} 攻破，{a0_total - a0_breached} 通过（{time.time()-t_a0:.0f}s）")

    # =====================================================
    # Phase A1: bucket-specific on A0-passed, workers=workers_a1
    # =====================================================
    print(f"\n{'='*60}")
    print(f"Phase A1: bucket-specific 攻击 (workers={workers_a1})")
    print(f"{'='*60}")
    t_a1 = time.time()

    a1_targets = []  # [(qfile, data, q, img_path, bucket_attacks, max_steps), ...]
    for qfile, items in a0_results.items():
        for data, q, res in items:
            if res.get("attack_breached"):
                continue  # A0 已攻破，跳过
            # 用 tool signature 动态决定攻击，不再纯查 bucket 表
            bucket_attacks = _compute_a1_attacks(q)
            if not bucket_attacks:
                continue  # chain_heavy / standard → 止步
            img_path = _resolve_img(qfile, data)
            # max_steps: cascade（多种 ablation）用 6，单一用 4
            max_steps = (
                MAX_STEPS_BY_PHASE["A1_cascade"] if len(bucket_attacks) >= 3
                else MAX_STEPS_BY_PHASE["A1_single_ablation"]
            )
            a1_targets.append((qfile, data, q, img_path, tuple(bucket_attacks), max_steps))

    def _run_a1(target):
        qfile, data, q, img_path, attacks, max_steps = target
        return qfile, data, q, attack_one_question(
            q, img_path, client, tier,
            only_attacks=attacks,
            max_steps_override=max_steps,
        )

    a1_breached = 0
    a1_total = 0
    a1_file_dirty: set[str] = set()
    if a1_targets:
        with ThreadPoolExecutor(max_workers=workers_a1) as pool:
            futs = {pool.submit(_run_a1, t): t for t in a1_targets}
            for fut in as_completed(futs):
                try:
                    qfile, data, q, res = fut.result()
                except Exception as e:
                    print(f"  [A1 ERROR] {e}")
                    continue
                # Merge：A1 的 attack_results 合并到 A0 已写的字段
                _merge_attack_result(q, res, tier)
                a1_total += 1
                if res.get("attack_breached"):
                    a1_breached += 1
                a1_file_dirty.add(qfile)

    # 保存 A1 修改过的文件
    for qfile in a1_file_dirty:
        items = a0_results.get(qfile) or []
        if items:
            data = items[0][0]
            with open(qfile, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Phase A1: {a1_total} 题攻击，{a1_breached} 攻破（{time.time()-t_a1:.0f}s）")

    # 汇总
    total_ns = (time.time() - t_a0)
    total_breached = a0_breached + a1_breached
    total_attacked = a0_total
    return {
        "phase_a0": {"total": a0_total, "breached": a0_breached, "elapsed_sec": round(time.time()-t_a0, 0)},
        "phase_a1": {"total": a1_total, "breached": a1_breached, "elapsed_sec": round(time.time()-t_a1, 0)},
        "total_attacked": total_attacked,
        "total_breached": total_breached,
        "total_passed": total_attacked - total_breached,
        "total_elapsed_sec": round(total_ns, 0),
    }


def _merge_attack_result(q: dict, res: dict, tier: dict) -> None:
    """把 Phase A1 的 attack_results 合并到之前 A0 写入的字段里。"""
    key = f"attack_{tier['name'].split()[0].lower()}"
    existing = q.get(key)
    if not isinstance(existing, dict):
        # 没有 A0 结果，直接 apply
        _apply_attack_result(q, res, tier)
        return
    # 合并 attack_results 字典
    prev_ar = existing.get("attack_results") or {}
    new_ar = res.get("attack_results") or {}
    merged = {**prev_ar, **new_ar}
    existing["attack_results"] = merged
    # 如果 A1 有任何 breach，更新整体 breached
    if res.get("attack_breached"):
        existing["attack_breached"] = True
        bucket = q.get("hard_bucket", "")
        # image_heavy / all_tools 在 B/C 层不参与 filter reject
        block = tier.get("block_image_heavy") and bucket in ("image_heavy", "all_tools")
        if not block:
            existing["breached_for_filter"] = True
            q["hard_attacker_passed"] = False


# ============================================================
# Batch 处理一个 questions JSON 文件
# ============================================================

def attack_question_file(
    qfile: str,
    tier: dict,
    workers: int = 2,
    only_levels: tuple[int, ...] = (3,),
    only_attacks: tuple[str, ...] | None = None,
    max_questions: int | None = None,
) -> dict:
    """对一个 questions JSON 跑攻击，写回 attack_results 字段。"""
    with open(qfile, encoding="utf-8") as f:
        data = json.load(f)
    img_id = data.get("image_id") or os.path.splitext(os.path.basename(qfile))[0]
    img_path = data.get("image_path", "")
    if not os.path.exists(img_path):
        # 尝试候选目录
        for base in ("output/images", "images", "try"):
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                cand = os.path.join(base, f"{img_id}{ext}")
                if os.path.exists(cand):
                    img_path = cand
                    break
            if os.path.exists(img_path):
                break
    if not os.path.exists(img_path):
        return {"img_id": img_id, "error": f"image not found", "n_attacked": 0}

    client = OpenAI(
        api_key=tier["api_key"],
        base_url=tier["base_url"],
        http_client=httpx.Client(trust_env=False),
    )

    # 收集要攻击的题
    targets = []
    for lvl in only_levels:
        for q in data.get(f"level_{lvl}", []):
            # 跳过占位答案
            if (q.get("answer") or "").lower() in ("rank_winner", "unknown", ""):
                continue
            # 按 tier scope 过滤
            bucket = q.get("hard_bucket", "")
            if tier["scope"] == "hard_passed_a":
                # B 层只跑 A 已通过的 hard bucket
                prev = q.get("attack_results", {})
                if not prev or prev.get("attack_breached", True):
                    continue
                if bucket not in ("ultra_long", "visit_heavy", "code_heavy", "chain_heavy", "all_tools", "image_heavy"):
                    continue
            targets.append(q)

    if max_questions:
        targets = targets[:max_questions]

    n_attacked = 0
    n_breached = 0

    def _do(q):
        return attack_one_question(q, img_path, client, tier, only_attacks=only_attacks)

    if workers > 1 and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_do, q): q for q in targets}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"error": str(e)[:200], "attack_breached": False, "attack_results": {}}
                _apply_attack_result(q, res, tier)
                n_attacked += 1
                if res.get("attack_breached"):
                    n_breached += 1
    else:
        for q in targets:
            try:
                res = _do(q)
            except Exception as e:
                res = {"error": str(e)[:200], "attack_breached": False, "attack_results": {}}
            _apply_attack_result(q, res, tier)
            n_attacked += 1
            if res.get("attack_breached"):
                n_breached += 1

    # 写回
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "img_id": img_id,
        "n_attacked": n_attacked,
        "n_breached": n_breached,
        "n_passed": n_attacked - n_breached,
    }


def _apply_attack_result(q: dict, res: dict, tier: dict) -> None:
    """把攻击结果写回 question dict。"""
    bucket = q.get("hard_bucket", "")
    breached = res.get("attack_breached", False)

    # 关键约束：B/C 不对 image_heavy / all_tools 做最终 reject
    block = tier.get("block_image_heavy") and bucket in ("image_heavy", "all_tools")
    if block:
        # 仍然记录攻击结果，但标记为不计入 final reject
        breached_for_filter = False
        note = "tier_blocked_for_image_heavy"
    else:
        breached_for_filter = breached
        note = ""

    q[f"attack_{tier['name'].split()[0].lower()}"] = {
        "tier": tier["name"],
        "attack_results": res.get("attack_results", {}),
        "attack_breached": breached,
        "breached_for_filter": breached_for_filter,
        "note": note,
    }
    # 全局 hard_attacker_passed 字段：只看实际生效的攻击结果
    prev = q.get("hard_attacker_passed", True)  # 没攻击过默认 True
    q["hard_attacker_passed"] = prev and (not breached_for_filter)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="L3 题目攻击式过滤")
    parser.add_argument("question_files", nargs="+",
                        help="output/questions/*.json 路径或 glob")
    parser.add_argument("--tier", choices=["A", "B", "C"], default="A",
                        help="攻击器分层（A=便宜广筛，B=Hard 复判，C=校准）")
    parser.add_argument("--mode", choices=["legacy", "two_phase"], default="two_phase",
                        help="legacy=每题跑全部 4 攻击；two_phase=先 no_tool 广筛再 bucket 专属（推荐）")
    parser.add_argument("--workers", type=int, default=2,
                        help="legacy 模式的并发（two_phase 模式请用 --workers-a0 / --workers-a1）")
    parser.add_argument("--workers-a0", type=int, default=4,
                        help="two_phase: Phase A0 (no_tool) 的并发，默认 4")
    parser.add_argument("--workers-a1", type=int, default=2,
                        help="two_phase: Phase A1 (bucket 专属 ablation) 的并发，默认 2")
    parser.add_argument("--levels", type=str, default="3",
                        help="要攻击的 level，逗号分隔（默认只攻击 L3）")
    parser.add_argument("--attacks", type=str, default="",
                        help="[legacy] 只跑指定的攻击，逗号分隔")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="[legacy] 每张图最多攻击 N 道题")
    args = parser.parse_args()

    tier = TIER_CONFIG[args.tier]
    levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    only_attacks = tuple(a.strip() for a in args.attacks.split(",") if a.strip()) or None

    files = []
    for pat in args.question_files:
        if any(c in pat for c in "*?["):
            files.extend(sorted(glob.glob(pat)))
        elif os.path.isfile(pat):
            files.append(pat)
    if not files:
        print(f"找不到文件: {args.question_files}")
        sys.exit(1)

    print(f"攻击器: {tier['name']}（{args.tier} 层）  模式: {args.mode}")
    print(f"目标: {len(files)} 个 question 文件，levels={levels}")
    print("=" * 60)

    if args.mode == "two_phase":
        summary = run_two_phase_attack(
            files, tier,
            workers_a0=args.workers_a0,
            workers_a1=args.workers_a1,
            only_levels=levels,
        )
        print()
        print("=" * 60)
        print(f"总耗时: {summary['total_elapsed_sec']}s")
        print(f"Phase A0: {summary['phase_a0']['total']} 题 / 攻破 {summary['phase_a0']['breached']} / {summary['phase_a0']['elapsed_sec']}s")
        print(f"Phase A1: {summary['phase_a1']['total']} 题 / 攻破 {summary['phase_a1']['breached']} / {summary['phase_a1']['elapsed_sec']}s")
        print(f"总攻击: {summary['total_attacked']}  攻破: {summary['total_breached']}  通过: {summary['total_passed']}")
        return

    # ---------- legacy mode（保留兼容）----------
    t0 = time.time()
    total_attacked = 0
    total_breached = 0
    for fp in files:
        ts = time.time()
        result = attack_question_file(
            fp, tier,
            workers=args.workers,
            only_levels=levels,
            only_attacks=only_attacks,
            max_questions=args.max_questions,
        )
        elapsed = time.time() - ts
        if "error" in result:
            print(f"[FAIL] {result['img_id']} {elapsed:.0f}s  {result['error']}")
            continue
        total_attacked += result["n_attacked"]
        total_breached += result["n_breached"]
        print(f"[DONE] {result['img_id']}  {elapsed:.0f}s  攻击={result['n_attacked']} 攻破={result['n_breached']} 通过={result['n_passed']}")

    print("=" * 60)
    print(f"总耗时: {time.time()-t0:.0f}s")
    print(f"总攻击: {total_attacked}  攻破: {total_breached}  通过: {total_attacked - total_breached}")


if __name__ == "__main__":
    main()
