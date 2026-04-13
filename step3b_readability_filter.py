"""Step 3b: 轻量链可读性判别器（post-filter）。

三层过滤架构：
  Layer 1 — 规则预筛（0 LLM call，确定性拦截明确不自然的题）
  Layer 2 — LLM 复核（只对灰区题，flow/motivation/naturalness 三维 1-5 分）
  Layer 3 — hash 缓存（按 question+reasoning_path hash，避免重复调 LLM）

任一项 <= 2 → `readability_flag="rejected"` / `"rejected_by_rule"`。

**只写入新字段，不删题**。下游 `aggregate_hard_split.py --apply-readability-filter` 可选过滤。

用法：
    python3 step3b_readability_filter.py 'output/questions/*.json' --workers 4
    python3 step3b_readability_filter.py 'output/questions/*.json' --force
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from core.vlm import call_vlm_json

# ============================================================
# Layer 3: hash 缓存
# ============================================================
CACHE_DIR = os.path.join("output", ".cache", "readability")


def _cache_key(q: dict) -> str:
    """按 question + reasoning_path 做 hash（确定性，题内容不变则 key 不变）。"""
    payload = json.dumps({
        "question": q.get("question", ""),
        "reasoning_path": q.get("reasoning_path", {}),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_get(key: str) -> dict | None:
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_set(key: str, result: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


# ============================================================
# Layer 1: 规则预筛
# ============================================================

# "的" 字堆砌检测用的正则
# 间隔 < 5 字符（比如"所属的品牌的创始人的"），避免误杀"他的亲生孩子的出生日期"这种合理结构
_DE_PATTERN = re.compile(r"的.{1,4}的.{1,4}的")

# 关系词堆砌检测
_RELATION_WORDS = {"所属", "对应", "母公司", "所有者", "创始人", "旗下", "隶属", "生产商", "制造商"}


def _rule_prefilter(q: dict) -> tuple[bool, str]:
    """Layer 1 规则预筛。返回 (reject, reason)。

    命中任一规则 → reject=True。
    """
    question = q.get("question", "")
    rp = q.get("reasoning_path") or {}
    family = q.get("family", "")

    # 1. 定语堆叠：连续 3 个"的"间隔很短
    if _DE_PATTERN.search(question):
        return True, "modifier_stacking_3de"

    # 2. 关系词堆砌：≥3 个不同关系词出现在同一句里
    matched_rel_words = sum(1 for w in _RELATION_WORDS if w in question)
    if matched_rel_words >= 3:
        return True, "relation_word_stacking"

    # 3. 跨域 compare：两个 branch head entity type 完全不同
    if family in ("compare", "compare_then_follow"):
        branches = rp.get("branches") or []
        if len(branches) >= 2:
            types = set()
            for b in branches:
                # 从 edge 的 entity 或 region info 尝试推 type
                edge = b.get("edge") or {}
                # 用 branch 的 entity key 里的前缀推断（如 "entity:Dr Pepper" 没有 type 信息）
                # 退而求其次：如果两个 branch 的 edge.relation 语义域完全不同（一个是人物属性，一个是建筑属性）
                # 这里简化为：不做跨域检测（太难做准，容易误杀）
                pass
            # 跨域检测暂不启用（false positive 太高，先只用规则 1 和 2）

    # 4. 末跳 lex 过低（multi_hop only）
    hops = rp.get("hop_chain") or []
    if hops and isinstance(hops[-1], dict):
        edge = hops[-1].get("edge") or {}
        lex = edge.get("lexicalizability", 1.0)
        if isinstance(lex, (int, float)) and lex < 0.3:
            return True, "last_hop_low_lex"

    return False, ""


READABILITY_PROMPT = """你是题目自然度审查员。评估下面这道多跳推理题的"可读性"。

你需要给出三个 1-5 分数：

1. **flow**（推理流畅度）—— 从第一步到最后一步的语义衔接是否自然
   - 5 = 每一跳都顺理成章，像人会顺着想的
   - 3 = 链条结构上对但中间某一跳有点跳跃
   - 1 = 最后一跳强行拐弯（例如前两跳都在讲球员生涯，突然问他家乡城市的人口）

2. **motivation**（提问动机）—— 人为什么会问这道题
   - 5 = 完全能想象真人在哪种场景下会好奇这个
   - 3 = 问法正确但不是人会真问的（像考试凑题）
   - 1 = 完全没有提问动机（例如"这家公司的员工编号前缀是什么"）

3. **naturalness**（问句自然度）—— 中文问句是否像真人写的
   - 5 = 流畅中文，没有翻译腔
   - 3 = 能懂但有点生硬，类似 LLM 翻译
   - 1 = 机翻腔、语法怪、或者概念堆砌

规则：
- 不要做事实核查，只看"这题问得合理吗"
- 答案本身是对是错不影响评分
- 简单题（一跳、直接事实）通常分数高，这是正常的

输出严格 JSON（不加 markdown 代码块）：
{
  "flow": <1-5>,
  "motivation": <1-5>,
  "naturalness": <1-5>,
  "note": "一句话解释最低分那项的理由（或 'ok' 如果都 >=4）"
}"""


def _summarize_hop_chain(hop_chain: list[dict]) -> str:
    """把 hop_chain 串成一行可读摘要给 LLM 看。"""
    if not hop_chain or not isinstance(hop_chain, list):
        return "（无 hop_chain）"
    parts = []
    for i, h in enumerate(hop_chain):
        if not isinstance(h, dict):
            continue
        rel = h.get("relation", "?")
        val = h.get("value", "?")
        parts.append(f"hop{i+1}[{rel}] → {val}")
    return " → ".join(parts)


def _summarize_branches(branches: list[dict]) -> str:
    """compare / set_merge 家族用：概括两个 branch 的结构。"""
    if not branches or not isinstance(branches, list):
        return "（无 branches）"
    parts = []
    for i, b in enumerate(branches):
        if not isinstance(b, dict):
            continue
        ent = b.get("entity", "?")
        edge = b.get("edge") or {}
        rel = edge.get("relation", "?") if isinstance(edge, dict) else "?"
        val = b.get("value", b.get("fact", "?"))
        parts.append(f"branch{i+1}[{ent} -{rel}→ {val}]")
    return " | ".join(parts)


def _build_user_content(q: dict) -> str:
    """给 LLM 的输入：问题、答案、推理路径摘要。"""
    question = q.get("question", "")
    answer = q.get("answer", "")
    family = q.get("family", "")
    bucket = q.get("hard_bucket", "")
    rp = q.get("reasoning_path") or {}

    chain_summary = ""
    if rp.get("hop_chain"):
        chain_summary = f"\nhop_chain: {_summarize_hop_chain(rp['hop_chain'])}"
    if rp.get("branches"):
        chain_summary += f"\nbranches: {_summarize_branches(rp['branches'])}"

    return (
        f"题目：{question}\n"
        f"答案：{answer}\n"
        f"family: {family}  bucket: {bucket}"
        f"{chain_summary}"
    )


def score_one(q: dict, max_retries: int = 2, use_cache: bool = True) -> dict:
    """三层过滤：rule → cache → LLM。返回分数 dict 或 error dict。"""
    # Layer 1: 规则预筛
    reject, reason = _rule_prefilter(q)
    if reject:
        return {
            "flow": 0, "motivation": 0, "naturalness": 0,
            "min_score": 0,
            "readability_flag": "rejected_by_rule",
            "note": reason,
        }

    # Layer 3: 缓存查找
    cache_key = _cache_key(q)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached and "readability_flag" in cached:
            return cached

    # Layer 2: LLM 复核
    user = _build_user_content(q)
    try:
        resp = call_vlm_json(
            READABILITY_PROMPT,
            user,
            max_tokens=250,
            temperature=0.2,
            max_attempts=max_retries + 1,
        )
    except Exception as e:
        return {"error": str(e)[:200]}

    if not isinstance(resp, dict):
        return {"error": "non_dict_response"}

    # 提取分数，容忍字段类型
    def _to_int_1_5(v: Any) -> int:
        try:
            iv = int(v)
            return max(1, min(5, iv))
        except (TypeError, ValueError):
            return 0

    flow = _to_int_1_5(resp.get("flow"))
    motivation = _to_int_1_5(resp.get("motivation"))
    naturalness = _to_int_1_5(resp.get("naturalness"))
    note = str(resp.get("note", ""))[:200]

    if flow == 0 or motivation == 0 or naturalness == 0:
        return {"error": "invalid_score", "raw": str(resp)[:200]}

    min_score = min(flow, motivation, naturalness)
    flag = "rejected" if min_score <= 2 else "ok"

    result = {
        "flow": flow,
        "motivation": motivation,
        "naturalness": naturalness,
        "min_score": min_score,
        "readability_flag": flag,
        "note": note,
    }
    # 写缓存（只缓存成功打分的，不缓存 error）
    if use_cache:
        _cache_set(cache_key, result)
    return result


def process_file(
    fp: str,
    force: bool = False,
    max_retries: int = 2,
) -> dict:
    """处理单个 question 文件，为每道 L3 打分并写回。

    Returns stats: {total, scored, skipped, rejected_rule, rejected_llm, ok, errors, cached}
    """
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total": 0, "scored": 0, "skipped": 0,
        "rejected_rule": 0, "rejected_llm": 0, "ok": 0, "errors": 0, "cached": 0,
    }
    dirty = False

    for q in data.get("level_3", []):
        stats["total"] += 1

        if not force and "readability_flag" in q:
            stats["skipped"] += 1
            continue

        result = score_one(q, max_retries=max_retries)
        if "error" in result:
            stats["errors"] += 1
            q["readability_flag"] = "error"
            q["readability_error"] = result["error"]
            dirty = True
            continue

        flag = result["readability_flag"]
        q["readability_flag"] = flag
        q["readability_scores"] = {
            "flow": result.get("flow", 0),
            "motivation": result.get("motivation", 0),
            "naturalness": result.get("naturalness", 0),
            "min_score": result.get("min_score", 0),
            "note": result.get("note", ""),
        }
        stats["scored"] += 1
        if flag == "rejected_by_rule":
            stats["rejected_rule"] += 1
        elif flag == "rejected":
            stats["rejected_llm"] += 1
        else:
            stats["ok"] += 1
        dirty = True

    if dirty:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Step 3b: 轻量链可读性过滤")
    ap.add_argument("question_files", nargs="+",
                    help="question JSON 文件路径或 glob")
    ap.add_argument("--workers", type=int, default=4,
                    help="并发 worker 数（默认 4）")
    ap.add_argument("--force", action="store_true",
                    help="强制重打分（默认只打没有 readability_flag 字段的题）")
    ap.add_argument("--max-retries", type=int, default=1,
                    help="单题打分失败重试次数")
    args = ap.parse_args()

    # 展开 glob
    files: list[str] = []
    for pat in args.question_files:
        if any(c in pat for c in "*?["):
            files.extend(sorted(glob.glob(pat)))
        elif os.path.isfile(pat):
            files.append(pat)
    files = sorted(set(files))
    if not files:
        print(f"No files match: {args.question_files}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} files (workers={args.workers}, force={args.force})...")
    t0 = time.time()

    total_stats = {
        "total": 0, "scored": 0, "skipped": 0,
        "rejected_rule": 0, "rejected_llm": 0, "ok": 0, "errors": 0, "cached": 0,
    }
    rejected_log: list[dict] = []

    def _worker(fp: str) -> tuple[str, dict]:
        return fp, process_file(fp, force=args.force, max_retries=args.max_retries)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, fp): fp for fp in files}
        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                fp2, stats = fut.result()
            except Exception as e:
                print(f"[ERR] {fp}: {e}", file=sys.stderr)
                continue

            for k, v in stats.items():
                if k in total_stats:
                    total_stats[k] += v
            rej_total = stats.get('rejected_rule', 0) + stats.get('rejected_llm', 0)
            print(f"[{time.time()-t0:5.0f}s] {os.path.basename(fp2):<45} "
                  f"scored={stats['scored']:>2} rule_rej={stats.get('rejected_rule',0):>2} "
                  f"llm_rej={stats.get('rejected_llm',0):>2} ok={stats.get('ok',0):>2} "
                  f"err={stats['errors']:>2} skip={stats['skipped']:>2}",
                  flush=True)

            # 收集 rejected 题目用于最终 summary
            with open(fp2, encoding="utf-8") as f:
                data = json.load(f)
            for q in data.get("level_3", []):
                if q.get("readability_flag") in ("rejected", "rejected_by_rule"):
                    rejected_log.append({
                        "img_id": data.get("image_id") or os.path.splitext(os.path.basename(fp2))[0],
                        "question_id": q.get("question_id"),
                        "bucket": q.get("hard_bucket"),
                        "family": q.get("family"),
                        "scores": q.get("readability_scores"),
                        "question": q.get("question", "")[:150],
                        "answer": q.get("answer", "")[:100],
                    })

    elapsed = time.time() - t0
    print(f"\n=== DONE in {elapsed:.0f}s ===")
    print(f"total:         {total_stats['total']}")
    print(f"scored:        {total_stats['scored']}")
    print(f"skipped:       {total_stats['skipped']} (already had flag)")
    print(f"ok:            {total_stats['ok']}")
    print(f"rejected_rule: {total_stats['rejected_rule']} (Layer 1 heuristic)")
    print(f"rejected_llm:  {total_stats['rejected_llm']} (Layer 2 LLM)")
    print(f"errors:        {total_stats['errors']}")

    if rejected_log:
        # 去重（同一题可能在多轮中被多次 collect）
        seen = set()
        unique_rejected = []
        for r in rejected_log:
            key = (r["img_id"], r["question_id"])
            if key in seen:
                continue
            seen.add(key)
            unique_rejected.append(r)

        print(f"\n=== Rejected examples ({len(unique_rejected)}) ===")
        for r in unique_rejected[:20]:
            s = r["scores"] or {}
            print(f"  [{r['img_id']}:{r['question_id']}] {r['bucket']}/{r['family']} "
                  f"flow={s.get('flow')} mot={s.get('motivation')} nat={s.get('naturalness')}")
            print(f"    Q: {r['question']}")
            print(f"    A: {r['answer']}")
            note = s.get("note", "")
            if note and note != "ok":
                print(f"    note: {note}")


if __name__ == "__main__":
    main()
