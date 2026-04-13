"""Step6: 从 attack 结果聚合 breach taxonomy 统计。

输入：output/questions/*.json（含 step5_attack 写入的 attack_results 字段）
输出：output/stats/attack_breach_stats.json

统计维度：
  - by_bucket: 每个 hard bucket 的 breach rate + 各攻击细分
  - by_family: 每个 closure family 的 breach rate
  - by_attack: 每种攻击的 breach 数
  - breached_examples: 被攻破题目的完整 context，含 shortcut_path_type=null 待手工标注

阈值决策：
  image_heavy × no_image_search rate
    ≤ 0.15  → keep_B_path
    0.15-0.30 → inspect_motif_concentration
    > 0.30  → implement_A_prime_coarse_anchor_substitution

用法：
    python step6_attack_stats.py output/questions/sku_*.json
    python step6_attack_stats.py output/questions/sku_*.json -o custom_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any


ATTACK_NAMES = ["no_tool", "no_image_search", "no_visit", "no_code"]


def _extract_anchor_types(question: dict) -> list[str]:
    """从 reasoning_path 或 entities_involved 抽出锚点 entity_type 列表。"""
    rp = question.get("reasoning_path", {}) or {}
    types: list[str] = []
    # multi_hop: 从 hop_chain 第一个 hop 取 entity
    hop_chain = rp.get("hop_chain") or []
    if hop_chain and isinstance(hop_chain, list):
        # 不直接能拿到 entity_type，需要回查 graph，但 question json 里没有 graph
        # 退而求其次：用 obfuscated_entities 的描述反推
        pass
    # fallback: 用 frame.visible_refs 里的描述关键词
    frame = question.get("frame", {}) or {}
    refs = frame.get("visible_refs") or []
    for ref in refs:
        if not isinstance(ref, str):
            continue
        if any(k in ref for k in ("球员", "人", "球星")):
            types.append("person")
        elif any(k in ref for k in ("广告牌", "品牌", "标志", "logo", "广告")):
            types.append("brand")
        elif any(k in ref for k in ("海报",)):
            types.append("product")
        else:
            types.append("other")
    return types or ["unknown"]


def _extract_answer_entity_type(question: dict) -> str:
    """从 frame.answer_type 或 answer 字面推断类型。"""
    frame = question.get("frame", {}) or {}
    tt = (frame.get("answer_type") or "").upper()
    if tt in ("TIME", "QUANTITY", "LOCATION", "PERSON", "ORG"):
        return tt
    # 从 answer 字面推断
    ans = (question.get("answer") or "").strip()
    if not ans:
        return "UNKNOWN"
    import re
    if re.match(r"^\d{4}", ans):  # 四位年份开头
        return "TIME"
    if re.match(r"^[\d,\.]+$", ans):  # 纯数字
        return "QUANTITY"
    return "OTHER"


def _extract_hop_chain_summary(question: dict) -> str:
    """multi_hop 用，串成 'hop1[plays_for] → hop2[won_championship_in]'。"""
    rp = question.get("reasoning_path", {}) or {}
    hop_chain = rp.get("hop_chain") or []
    if not hop_chain:
        return ""
    parts = []
    for i, hop in enumerate(hop_chain):
        rel = hop.get("relation", "") if isinstance(hop, dict) else ""
        parts.append(f"hop{i+1}[{rel}]")
    return " → ".join(parts)


def _get_attack_results(question: dict, tier_filter: str | None = None) -> dict[str, dict] | None:
    """从 question 字段里找 step5_attack 写入的 attack_results。

    step5_attack 写入的字段名是 `attack_qwen3-vl-30b-a3b` 等（按 model 名 tier），
    或者是 `attack_qwen3-vl-30b-a3b-instruct` 视 tier name 格式而定。

    tier_filter: "qwen3" / "gemini" / "claude" / None。
      - None: 返回第一个找到的（legacy 行为，两 tier 都跑时只读第一个）
      - 否则：只匹配 key 里含该子串的字段
    """
    # step5 写入的 key 是 attack_{tier_name_first_word_lower}
    # Tier A: "Qwen3-VL-30B-A3B" → "qwen3-vl-30b-a3b"
    # Tier B: "Gemini 3 Flash" → "gemini"
    for k in question:
        if not k.startswith("attack_"):
            continue
        if not isinstance(question[k], dict):
            continue
        if tier_filter and tier_filter.lower() not in k.lower():
            continue
        ar = question[k].get("attack_results")
        if isinstance(ar, dict):
            return question[k]  # 返回整个 attack 记录（含 breached, etc）
    return None


def _which_attack_breached(attack_record: dict) -> str | None:
    """返回第一个命中 GT 的攻击名。"""
    ar = attack_record.get("attack_results") or {}
    for name, result in ar.items():
        if isinstance(result, dict) and result.get("correct"):
            return name
    return None


def compute_attack_stats(question_files: list[str], tier_filter: str | None = None) -> dict:
    """聚合所有 questions/*.json 的攻击结果为 breach taxonomy。

    tier_filter: "qwen3" / "gemini" / "claude" / None。
      只聚合该 tier 的 attack 字段，None 则走第一个（legacy）。
    """
    total_l3 = 0
    attacked = 0
    breached = 0

    by_bucket: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "breached": 0,
        "attack_breakdown": {
            a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES
        },
    })
    by_family: dict[str, dict] = defaultdict(lambda: {"total": 0, "breached": 0})
    by_attack: dict[str, dict] = {a: {"tried": 0, "breached": 0} for a in ATTACK_NAMES}

    breached_examples: list[dict] = []

    for qf in question_files:
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARN: failed to read {qf}: {e}", file=sys.stderr)
            continue
        img_id = data.get("image_id") or os.path.splitext(os.path.basename(qf))[0]

        for q in data.get("level_3", []):
            total_l3 += 1
            bucket = q.get("hard_bucket", "") or "standard"
            family = q.get("family", "") or "unknown"

            by_bucket[bucket]["total"] += 1
            by_family[family]["total"] += 1

            attack_rec = _get_attack_results(q, tier_filter=tier_filter)
            if attack_rec is None:
                continue  # 该题没被攻击

            attacked += 1
            ar = attack_rec.get("attack_results") or {}
            was_breached = attack_rec.get("attack_breached", False)

            for an in ATTACK_NAMES:
                r = ar.get(an)
                if not isinstance(r, dict):
                    continue  # 该攻击未跑（早停）
                by_bucket[bucket]["attack_breakdown"][an]["tried"] += 1
                by_attack[an]["tried"] += 1
                if r.get("correct"):
                    by_bucket[bucket]["attack_breakdown"][an]["breached"] += 1
                    by_attack[an]["breached"] += 1

            if was_breached:
                breached += 1
                by_bucket[bucket]["breached"] += 1
                by_family[family]["breached"] += 1

                which = _which_attack_breached(attack_rec)
                breached_examples.append({
                    "img_id": img_id,
                    "question_id": q.get("question_id", ""),
                    "bucket": bucket,
                    "family": family,
                    "anchor_types": _extract_anchor_types(q),
                    "answer_entity_type": _extract_answer_entity_type(q),
                    "breached_by_attack": which,
                    "model_answer": (ar.get(which, {}).get("model_answer") if which else "")[:200],
                    "ground_truth": q.get("answer", ""),
                    "question": q.get("question", ""),
                    "hop_chain_summary": _extract_hop_chain_summary(q),
                    "tool_sequence_len": len(q.get("tool_sequence", [])),
                    "shortcut_path_type": None,  # 待手工标注
                })

    # 计算比例
    def _add_rate(d: dict, num_key: str = "breached", denom_key: str = "total"):
        d["rate"] = round(d[num_key] / max(1, d[denom_key]), 4)

    for bk, bv in by_bucket.items():
        _add_rate(bv)
        for an, av in bv["attack_breakdown"].items():
            _add_rate(av, "breached", "tried")
    for fv in by_family.values():
        _add_rate(fv)
    for av in by_attack.values():
        _add_rate(av, "breached", "tried")

    # Threshold verdict
    ih = by_bucket.get("image_heavy", {})
    ih_nis_rate = (ih.get("attack_breakdown") or {}).get("no_image_search", {}).get("rate", 0.0)
    if ih_nis_rate <= 0.15:
        verdict = "keep_B_path"
    elif ih_nis_rate <= 0.30:
        verdict = "inspect_motif_concentration"
    else:
        verdict = "implement_A_prime_coarse_anchor_substitution"

    stats = {
        "total_l3": total_l3,
        "attacked": attacked,
        "breached": breached,
        "overall_breach_rate": round(breached / max(1, attacked), 4),
        "by_bucket": dict(by_bucket),
        "by_family": dict(by_family),
        "by_attack": by_attack,
        "breached_examples": breached_examples,
        "threshold_verdict": {
            "image_heavy_no_image_search_rate": round(ih_nis_rate, 4),
            "recommended_action": verdict,
        },
    }
    return stats


def print_summary(stats: dict) -> None:
    """Print a human-readable summary."""
    print(f"总 L3: {stats['total_l3']}  已攻击: {stats['attacked']}  攻破: {stats['breached']}"
          f"  总 breach rate: {stats['overall_breach_rate']*100:.1f}%")
    print()
    print("=== by_bucket ===")
    for bk, bv in sorted(stats["by_bucket"].items()):
        print(f"  {bk:15s} total={bv['total']:3d} breached={bv['breached']:2d} ({bv['rate']*100:.0f}%)")
        for an, av in bv["attack_breakdown"].items():
            if av["tried"] == 0:
                continue
            mark = "✗" if av["breached"] else " "
            print(f"      {mark} {an:20s} tried={av['tried']:2d} breached={av['breached']:2d} ({av['rate']*100:.0f}%)")
    print()
    print("=== by_family ===")
    for fk, fv in sorted(stats["by_family"].items()):
        print(f"  {fk:20s} total={fv['total']:3d} breached={fv['breached']:2d} ({fv['rate']*100:.0f}%)")
    print()
    print("=== by_attack ===")
    for an in ATTACK_NAMES:
        av = stats["by_attack"][an]
        print(f"  {an:20s} tried={av['tried']:3d} breached={av['breached']:2d} ({av['rate']*100:.0f}%)")
    print()
    print(f"=== threshold verdict ===")
    tv = stats["threshold_verdict"]
    print(f"  image_heavy × no_image_search rate: {tv['image_heavy_no_image_search_rate']*100:.1f}%")
    print(f"  → {tv['recommended_action']}")

    if stats["breached_examples"]:
        print()
        print(f"=== breached_examples ({len(stats['breached_examples'])}) ===")
        for ex in stats["breached_examples"][:10]:
            print(f"  [{ex['img_id']}:{ex['question_id']}] {ex['bucket']} × {ex['breached_by_attack']}")
            print(f"      GT: {ex['ground_truth'][:60]}")
            print(f"      MODEL: {ex['model_answer'][:60]}")
            print(f"      Q: {ex['question'][:100]}")


def main():
    parser = argparse.ArgumentParser(description="Step6: attack breach taxonomy stats")
    parser.add_argument("question_files", nargs="+",
                        help="output/questions/*.json paths or globs")
    parser.add_argument("-o", "--output", default="output/stats/attack_breach_stats.json",
                        help="output JSON path")
    parser.add_argument("--tier", default=None,
                        help="filter tier: 'qwen3' / 'gemini' / 'claude' (omit = first found)")
    args = parser.parse_args()

    files: list[str] = []
    for pat in args.question_files:
        if any(c in pat for c in "*?["):
            files.extend(sorted(glob.glob(pat)))
        elif os.path.isfile(pat):
            files.append(pat)
    if not files:
        print(f"No files found: {args.question_files}")
        sys.exit(1)

    print(f"Aggregating attack stats from {len(files)} files (tier={args.tier or 'any'})...")
    stats = compute_attack_stats(files, tier_filter=args.tier)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {args.output}")
    print()
    print_summary(stats)


if __name__ == "__main__":
    main()
