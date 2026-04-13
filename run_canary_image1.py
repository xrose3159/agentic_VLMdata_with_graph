"""定向 canary 攻击：image 1（新版 lens 数据）

攻击配置：
  全部 6 道 L3：no_tool
  L3_05 / L3_06 (image_heavy):  + no_image_search
  L3_02 (visit_heavy):           + no_visit
  L3_04 (code_heavy):            + no_code

通过标准（严格模式）：
  L3_05/06 no_image_search 必须 FAIL（否则 image_search 可被替代 → image_heavy 是假的）
  L3_02    no_visit        必须 FAIL
  L3_04    no_code          必须 FAIL
  L3_01/03 no_tool          应该 FAIL（chain_heavy 不该被纯模型先验秒答）
"""
from __future__ import annotations
import json
import os
import sys
import time

import httpx
from openai import OpenAI

from step5_attack import (
    ATTACKS,
    TIER_CONFIG,
    attack_one_question,
    is_answer_correct,
    run_attack_agent,
)

# 每个 question_id → 要跑的攻击列表
CANARY_PLAN: dict[str, list[str]] = {
    "L3_01": ["no_tool"],                          # chain_heavy，验证不被先验秒答
    "L3_02": ["no_tool", "no_visit"],              # visit_heavy
    "L3_03": ["no_tool"],                          # chain_heavy
    "L3_04": ["no_tool", "no_code"],               # code_heavy (rank_winner 占位已知 bug)
    "L3_05": ["no_tool", "no_image_search"],       # image_heavy
    "L3_06": ["no_tool", "no_image_search"],       # image_heavy
}

# 预期（True = 应该 FAIL 攻击，即题太硬模型答不出）
EXPECTED_FAIL: dict[str, dict[str, bool]] = {
    "L3_01": {"no_tool": True},
    "L3_02": {"no_tool": True, "no_visit": True},
    "L3_03": {"no_tool": True},
    "L3_04": {"no_tool": True, "no_code": True},
    "L3_05": {"no_tool": True, "no_image_search": True},
    "L3_06": {"no_tool": True, "no_image_search": True},
}


def main():
    # 读 image 1 的 questions
    qfile = "output/questions/1.json"
    with open(qfile) as f:
        data = json.load(f)
    img_path = data.get("image_path", "")
    if not os.path.exists(img_path):
        for cand in ("output/images/1.jpg", "try/1.jpg"):
            if os.path.exists(cand):
                img_path = cand
                break
    print(f"image: {img_path}")
    print()

    l3_questions = {q["question_id"]: q for q in data.get("level_3", [])}
    print(f"L3 questions in file: {list(l3_questions.keys())}")
    print()

    tier = TIER_CONFIG["A"]  # Qwen3-VL-30B-A3B
    client = OpenAI(
        api_key=tier["api_key"],
        base_url=tier["base_url"],
        http_client=httpx.Client(trust_env=False),
    )

    results = {}
    total_calls = 0
    total_pass = 0   # 符合预期（攻击失败 = 题硬）
    total_miss = 0   # 与预期不符（攻击成功 = 题有 shortcut）

    t0 = time.time()
    for qid, attacks in CANARY_PLAN.items():
        q = l3_questions.get(qid)
        if not q:
            print(f"[SKIP] {qid}: not in file")
            continue
        print(f"\n{'='*60}")
        print(f"{qid} [{q.get('family','')} / {q.get('hard_bucket','')}]")
        print(f"  Q: {q['question'][:100]}")
        print(f"  GT: {q['answer']}")
        print(f"  attacks: {attacks}")
        results[qid] = {"attacks": {}}

        for attack_name in attacks:
            attack_def = ATTACKS[attack_name]
            print(f"\n  --- {attack_name} ({attack_def['label']}) ---")
            ts = time.time()
            try:
                result = run_attack_agent(
                    q["question"],
                    img_path,
                    client,
                    tier["model"],
                    attack_def["allowed"],
                    max_steps=tier["max_steps"],
                )
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
            elapsed = time.time() - ts
            total_calls += 1
            ans = result.get("answer") or ""
            correct = is_answer_correct(ans, q["answer"])
            expected_fail = EXPECTED_FAIL.get(qid, {}).get(attack_name, True)
            # 预期：expected_fail=True 意味着"期望攻击失败"（题硬）
            # correct=True 意味着"攻击成功"（题软）
            match_expectation = (not correct) if expected_fail else correct

            status = "✓ 符合预期" if match_expectation else "✗ 不符预期"
            if match_expectation:
                total_pass += 1
            else:
                total_miss += 1

            print(f"    [{status}] {elapsed:.0f}s, {result['n_steps']} 步")
            print(f"    model_ans: {str(ans)[:100]}")
            breach = "BREACHED (攻击成功 → 题软)" if correct else "PASSED (攻击失败 → 题硬)"
            print(f"    verdict: {breach}")

            results[qid]["attacks"][attack_name] = {
                "model_answer": ans,
                "attack_success": correct,
                "expected_fail": expected_fail,
                "match_expectation": match_expectation,
                "elapsed_sec": round(elapsed, 1),
                "n_steps": result["n_steps"],
            }

    print(f"\n{'='*60}")
    print(f"总耗时: {time.time()-t0:.0f}s, 总调用: {total_calls}")
    print(f"符合预期: {total_pass}/{total_calls}")
    print(f"不符预期: {total_miss}/{total_calls}")
    print()
    print("=== 逐题判定 ===")
    for qid, r in results.items():
        q = l3_questions[qid]
        print(f"\n{qid} [{q.get('hard_bucket','')}]")
        for an, ar in r["attacks"].items():
            mark = "✓" if ar["match_expectation"] else "✗"
            breach_lbl = "攻破" if ar["attack_success"] else "未攻破"
            exp_lbl = "期望攻破" if not ar["expected_fail"] else "期望未攻破"
            print(f"  {mark} {an}: {breach_lbl} ({exp_lbl})  ans='{ar['model_answer'][:50]}'")

    # 写结果
    out_path = "output/canary_image1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "image_id": "1",
            "image_path": img_path,
            "tier": tier["name"],
            "model": tier["model"],
            "canary_plan": CANARY_PLAN,
            "expected_fail": EXPECTED_FAIL,
            "results": results,
            "total_calls": total_calls,
            "total_pass_expectation": total_pass,
            "total_miss_expectation": total_miss,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果写入: {out_path}")


if __name__ == "__main__":
    main()
