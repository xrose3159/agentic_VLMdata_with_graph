"""Stress suite analysis: per-category breach breakdown.

不动 step6 主框架，只用它的 _get_attack_results helper。读
output/stats/stress_category_map.json 做 category 维度分解。

输出：
  - per-category bucket × attack breach rate
  - Tier A vs Tier B 对比
  - image_heavy × no_image_search breach rate（阈值决策点）
  - 每个 breach 的 shortcut 样本（用于手工标注 shortcut_path_type）

用法：
    python analyze_stress_suite.py \
        --questions 'output/questions/*.json' \
        --category-map output/stats/stress_category_map.json \
        --out output/stats/stress_breach_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any

from step6_attack_stats import _get_attack_results, _which_attack_breached

ATTACK_NAMES = ["no_tool", "no_image_search", "no_visit", "no_code"]
BUCKETS = ["standard", "chain_heavy", "visit_heavy", "code_heavy",
           "ultra_long", "image_heavy", "all_tools"]
TIER_FILTERS = {"A": "qwen3", "B": "gemini", "C": "claude"}


def _empty_cell() -> dict:
    return {
        "total": 0,
        "attacked": 0,
        "breached": 0,
        "by_bucket": {b: {"total": 0, "breached": 0} for b in BUCKETS},
        "by_attack": {a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES},
        "bucket_attack_matrix": {
            b: {a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES}
            for b in BUCKETS
        },
    }


def _safe_rate(num: int, den: int) -> float:
    return round(num / max(1, den), 4)


def analyze(question_files: list[str], category_map: dict[str, str], tier: str) -> dict:
    tier_filter = TIER_FILTERS.get(tier)

    per_cat: dict[str, dict] = defaultdict(_empty_cell)
    global_cell = _empty_cell()
    breached_examples: list[dict] = []

    for qf in question_files:
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARN: failed to read {qf}: {e}", file=sys.stderr)
            continue
        img_id = data.get("image_id") or os.path.splitext(os.path.basename(qf))[0]
        cat = category_map.get(img_id, "unknown")
        cell_cat = per_cat[cat]

        for q in data.get("level_3", []):
            bucket = q.get("hard_bucket", "") or "standard"
            family = q.get("family", "") or "unknown"

            cell_cat["total"] += 1
            global_cell["total"] += 1
            if bucket not in cell_cat["by_bucket"]:
                cell_cat["by_bucket"][bucket] = {"total": 0, "breached": 0}
                cell_cat["bucket_attack_matrix"][bucket] = {
                    a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES
                }
            if bucket not in global_cell["by_bucket"]:
                global_cell["by_bucket"][bucket] = {"total": 0, "breached": 0}
                global_cell["bucket_attack_matrix"][bucket] = {
                    a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES
                }
            cell_cat["by_bucket"][bucket]["total"] += 1
            global_cell["by_bucket"][bucket]["total"] += 1

            attack_rec = _get_attack_results(q, tier_filter=tier_filter)
            if attack_rec is None:
                continue

            cell_cat["attacked"] += 1
            global_cell["attacked"] += 1
            ar = attack_rec.get("attack_results") or {}
            was_breached = attack_rec.get("attack_breached", False)

            for an in ATTACK_NAMES:
                r = ar.get(an)
                if not isinstance(r, dict):
                    continue
                cell_cat["by_attack"][an]["tried"] += 1
                cell_cat["bucket_attack_matrix"][bucket][an]["tried"] += 1
                global_cell["by_attack"][an]["tried"] += 1
                global_cell["bucket_attack_matrix"][bucket][an]["tried"] += 1
                if r.get("correct"):
                    cell_cat["by_attack"][an]["breached"] += 1
                    cell_cat["bucket_attack_matrix"][bucket][an]["breached"] += 1
                    global_cell["by_attack"][an]["breached"] += 1
                    global_cell["bucket_attack_matrix"][bucket][an]["breached"] += 1

            if was_breached:
                cell_cat["breached"] += 1
                cell_cat["by_bucket"][bucket]["breached"] += 1
                global_cell["breached"] += 1
                global_cell["by_bucket"][bucket]["breached"] += 1

                which = _which_attack_breached(attack_rec)
                breached_examples.append({
                    "category": cat,
                    "img_id": img_id,
                    "question_id": q.get("question_id", ""),
                    "bucket": bucket,
                    "family": family,
                    "breached_by_attack": which,
                    "tier": tier,
                    "model_answer": (ar.get(which, {}).get("model_answer") if which else "")[:200],
                    "ground_truth": q.get("answer", ""),
                    "question": q.get("question", ""),
                    "shortcut_path_type": None,
                })

    def _attach_rates(cell: dict):
        cell["breach_rate"] = _safe_rate(cell["breached"], cell["attacked"])
        for b, bv in cell["by_bucket"].items():
            bv["rate"] = _safe_rate(bv["breached"], bv["total"])
        for a, av in cell["by_attack"].items():
            av["rate"] = _safe_rate(av["breached"], av["tried"])
        for b, bm in cell["bucket_attack_matrix"].items():
            for a, av in bm.items():
                av["rate"] = _safe_rate(av["breached"], av["tried"])

    _attach_rates(global_cell)
    for cell in per_cat.values():
        _attach_rates(cell)

    # Threshold verdict: image_heavy × no_image_search（全 stress suite 维度）
    ih_cell = global_cell["bucket_attack_matrix"].get("image_heavy", {})
    nis = ih_cell.get("no_image_search", {"tried": 0, "breached": 0, "rate": 0.0})
    ih_nis_rate = nis["rate"]
    if nis["tried"] == 0:
        verdict = "no_image_heavy_data"
        verdict_note = (
            "image_heavy × no_image_search tried=0, cannot evaluate threshold"
        )
    elif ih_nis_rate <= 0.15:
        verdict = "keep_B_path"
        verdict_note = "breach rate within safe zone"
    elif ih_nis_rate <= 0.30:
        verdict = "inspect_motif_concentration"
        verdict_note = "breach rate in mid zone, inspect shortcut concentration"
    else:
        verdict = "implement_A_prime_coarse_anchor_substitution"
        verdict_note = "breach rate exceeds 30%, trigger A' fix"

    return {
        "tier": tier,
        "total_stress_images": len(set(category_map.keys())),
        "global": global_cell,
        "per_category": dict(per_cat),
        "breached_examples": breached_examples,
        "threshold_verdict": {
            "image_heavy_no_image_search_rate": ih_nis_rate,
            "image_heavy_no_image_search_tried": nis["tried"],
            "image_heavy_no_image_search_breached": nis["breached"],
            "recommended_action": verdict,
            "note": verdict_note,
        },
    }


def print_summary(result: dict) -> None:
    t = result["tier"]
    print(f"=== Stress Suite Breach Analysis (Tier {t}) ===")
    g = result["global"]
    print(f"总 L3: {g['total']}  已攻击: {g['attacked']}  攻破: {g['breached']}  "
          f"breach_rate: {g['breach_rate']*100:.1f}%")
    print()
    print("--- 全局 by_bucket ---")
    for b in BUCKETS:
        bv = g["by_bucket"].get(b)
        if not bv or bv["total"] == 0:
            continue
        print(f"  {b:<14} total={bv['total']:>3}  breached={bv['breached']:>2} ({bv['rate']*100:>5.1f}%)")

    print()
    print("--- 全局 bucket × attack 矩阵 ---")
    for b in BUCKETS:
        bm = g["bucket_attack_matrix"].get(b, {})
        bucket_total = g["by_bucket"].get(b, {}).get("total", 0)
        if bucket_total == 0:
            continue
        print(f"  {b}:")
        for a in ATTACK_NAMES:
            av = bm.get(a, {})
            if av.get("tried", 0) == 0:
                continue
            mark = "✗" if av["breached"] > 0 else " "
            print(f"    {mark} {a:<18} tried={av['tried']:>3}  breached={av['breached']:>2} ({av['rate']*100:>5.1f}%)")

    print()
    print("--- per-category breach ---")
    print(f"  {'category':<10} {'total':>6} {'attack':>7} {'breach':>7} {'rate':>7}")
    for cat in sorted(result["per_category"].keys()):
        c = result["per_category"][cat]
        print(f"  {cat:<10} {c['total']:>6} {c['attacked']:>7} {c['breached']:>7} "
              f"{c['breach_rate']*100:>6.1f}%")

    print()
    print("--- per-category bucket × attack (only non-empty) ---")
    for cat in sorted(result["per_category"].keys()):
        c = result["per_category"][cat]
        print(f"  [{cat}]")
        for b in BUCKETS:
            bm = c["bucket_attack_matrix"].get(b, {})
            total_b = c["by_bucket"].get(b, {}).get("total", 0)
            if total_b == 0:
                continue
            breached_b = c["by_bucket"].get(b, {}).get("breached", 0)
            print(f"    {b:<14} total={total_b:>2} breached={breached_b:>2}")
            for a in ATTACK_NAMES:
                av = bm.get(a, {})
                if av.get("tried", 0) == 0:
                    continue
                mark = "✗" if av["breached"] > 0 else " "
                print(f"      {mark} {a:<18} tried={av['tried']:>2}  "
                      f"breached={av['breached']:>2} ({av['rate']*100:>5.1f}%)")

    print()
    print("--- threshold verdict ---")
    tv = result["threshold_verdict"]
    print(f"  image_heavy × no_image_search: "
          f"{tv['image_heavy_no_image_search_breached']}/{tv['image_heavy_no_image_search_tried']} "
          f"= {tv['image_heavy_no_image_search_rate']*100:.1f}%")
    print(f"  → {tv['recommended_action']}")
    print(f"    {tv['note']}")

    if result["breached_examples"]:
        print()
        print(f"--- breached examples ({len(result['breached_examples'])}) ---")
        for ex in result["breached_examples"][:20]:
            print(f"  [{ex['category']}/{ex['img_id']}:{ex['question_id']}] "
                  f"{ex['bucket']}/{ex['family']} × {ex['breached_by_attack']}")
            print(f"    GT:    {ex['ground_truth'][:80]}")
            print(f"    MODEL: {ex['model_answer'][:80]}")
            print(f"    Q:     {ex['question'][:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True,
                    help="glob for question files, e.g. 'output/questions/*.json'")
    ap.add_argument("--category-map", required=True,
                    help="stress category map JSON")
    ap.add_argument("--tier", choices=["A", "B", "C"], default="A")
    ap.add_argument("--out", default="output/stats/stress_breach_analysis.json")
    args = ap.parse_args()

    files = sorted(glob.glob(args.questions))
    if not files:
        print(f"No files match: {args.questions}", file=sys.stderr)
        sys.exit(1)

    with open(args.category_map, encoding="utf-8") as f:
        cmap = json.load(f)

    # Filter question files to those in category map
    filtered = []
    for fp in files:
        img_id = os.path.splitext(os.path.basename(fp))[0]
        if img_id in cmap:
            filtered.append(fp)
    if not filtered:
        print(f"No question files match category map", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(filtered)} question files (tier={args.tier})...")
    result = analyze(filtered, cmap, args.tier)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.out}")
    print()
    print_summary(result)


if __name__ == "__main__":
    main()
