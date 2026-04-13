"""v0 hard split aggregator.

聚合所有通过 Tier A + Tier B 攻击过滤的 L3 题为 provisional 训练数据集。

输入：
  - output/questions/*.json      （含 attack_qwen3-vl-30b-a3b / attack_gemini / hard_attacker_passed 字段）
  - output/stats/stress_category_map.json  （39 张 stress suite 图的 category 标签）

输出：
  - output/final/level_3_hard_split.jsonl        （每行一条 L3 record）
  - output/stats/hard_split_stats.json           （breakdown + audit）

设计原则：
  - 只读磁盘现有数据，不改 Step3 管道
  - 过滤：只保留 hard_attacker_passed != False 的 L3 题
  - 每条 record 含 trajectory distillation 所需的全部字段 + 压缩的 attack_provenance 审计

用法：
    python3 aggregate_hard_split.py
    python3 aggregate_hard_split.py --out output/final/level_3_hard_split.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any

from core.config import FINAL_DIR, STATS_DIR, FILTERED_IMAGE_DIR

QUESTIONS_GLOB = "output/questions/*.json"
STRESS_CATEGORY_MAP = "output/stats/stress_category_map.json"
DEFAULT_OUT_JSONL = os.path.join(FINAL_DIR, "level_3_hard_split.jsonl")
DEFAULT_STATS = os.path.join(STATS_DIR, "hard_split_stats.json")

IMAGE_SEARCH_DIRS = (FILTERED_IMAGE_DIR, "images", "try")


def _find_image(img_id: str) -> str | None:
    """按 img_id 找源图，依次查 FILTERED_IMAGE_DIR / images / try。

    复用 step3_generate._find_image 的查找策略，避免跨模块导入带来的副作用。
    """
    for base in IMAGE_SEARCH_DIRS:
        if not base:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = os.path.join(base, f"{img_id}{ext}")
            if os.path.exists(cand):
                return cand
    return None


def _no_tool_breached(q: dict) -> bool:
    """任意一层攻击的 no_tool 攻击命中 GT → 该题有纯先验 shortcut。

    step5 的 block_image_heavy 逻辑会把 image_heavy/all_tools bucket 的所有 Tier B 攻击
    标成 breached_for_filter=False（因为反图搜索工具没真接），但 no_tool 攻击不涉及任何
    工具——它测的就是"模型看图 + 读题能否直接答"——bucket 豁免对它不适用。所以这里
    对 no_tool 单独加一道闸，不依赖 hard_attacker_passed 判断。
    """
    for tier_field in ("attack_qwen3-vl-30b-a3b", "attack_gemini"):
        rec = q.get(tier_field)
        if not isinstance(rec, dict):
            continue
        ar = rec.get("attack_results") or {}
        no_tool_r = ar.get("no_tool")
        if isinstance(no_tool_r, dict) and no_tool_r.get("correct"):
            return True
    return False


def _is_hard(q: dict, *, apply_readability_filter: bool = False) -> bool:
    """Hard split 的成员资格判定：
      1. hard_attacker_passed != False（常规 attack filter 结果）
      2. 没有任何 tier 的 no_tool 攻击命中 GT（hard-split-specific 补丁）
      3. （可选）readability_flag != 'rejected'（step3b 可读性过滤，默认关）
    """
    if q.get("hard_attacker_passed", True) is False:
        return False
    if _no_tool_breached(q):
        return False
    if apply_readability_filter and q.get("readability_flag") == "rejected":
        return False
    return True


def _lookup_category(img_id: str, stress_map: dict[str, str]) -> str:
    """stress 图走 category_map，sku 图统一归为 'sku'。"""
    cat = stress_map.get(img_id)
    if cat:
        return cat
    if img_id.startswith("sku_"):
        return "sku"
    return "unknown"


def _summarize_attack(q: dict, tier_field: str) -> tuple[bool, str | None]:
    """从 q[attack_xxx] 抽 (breached, first_breached_attack_name)。

    tier_field 如 'attack_qwen3-vl-30b-a3b' 或 'attack_gemini'.
    """
    rec = q.get(tier_field)
    if not isinstance(rec, dict):
        return False, None
    breached = bool(rec.get("attack_breached"))
    if not breached:
        return False, None
    ar = rec.get("attack_results") or {}
    # 返回第一个命中 GT 的攻击名
    for attack_name, result in ar.items():
        if isinstance(result, dict) and result.get("correct"):
            return True, attack_name
    return True, None  # breached 但找不到具体 attack（边界情况）


def _build_record(
    img_id: str,
    img_path: str,
    category: str,
    q: dict,
) -> dict:
    qid = q.get("question_id", "")
    tier_a_breached, _ = _summarize_attack(q, "attack_qwen3-vl-30b-a3b")
    tier_b_breached, tier_b_attack = _summarize_attack(q, "attack_gemini")

    return {
        "id": f"{img_id}_{qid}",
        "image_id": img_id,
        "image_path": img_path,
        "category": category,
        "level": 3,
        "question_id": qid,
        "question": q.get("question", ""),
        "answer": q.get("answer", ""),
        "family": q.get("family"),
        "hard_bucket": q.get("hard_bucket"),
        "code_skill_tags": q.get("code_skill_tags", []) or [],
        "reasoning_path": q.get("reasoning_path", {}) or {},
        "tool_sequence": q.get("tool_sequence", []) or [],
        "entities_involved": q.get("entities_involved", []) or [],
        "obfuscation_applied": q.get("obfuscation_applied"),
        "obfuscated_entities": q.get("obfuscated_entities", []) or [],
        "rationale": q.get("rationale"),
        "attack_provenance": {
            "tier_a_breached": tier_a_breached,
            "tier_b_breached": tier_b_breached,
            "tier_b_breached_by_attack": tier_b_attack,
        },
    }


def aggregate_hard_split(
    questions_glob: str = QUESTIONS_GLOB,
    category_map_path: str = STRESS_CATEGORY_MAP,
    apply_readability_filter: bool = False,
) -> tuple[list[dict], dict]:
    """读取所有 question 文件，过滤出 hard L3 题。返回 (records, stats)。

    **只处理**符合以下条件的 question 文件：
      - img_id 在 stress_category_map 里（39 张 stress suite 图），或
      - img_id 以 "sku_" 开头（4 张 sku 图）

    其他文件（比如 images/ 下预存的 img_* 等旧 pipeline 产物）一律跳过。
    这保证 hard split 只包含本验证周期真正攻击过的题。
    """
    try:
        with open(category_map_path, encoding="utf-8") as f:
            stress_map = json.load(f)
    except FileNotFoundError:
        print(f"WARN: {category_map_path} not found, stress images will be 'unknown'",
              file=sys.stderr)
        stress_map = {}

    files = sorted(glob.glob(questions_glob))
    # 排除 randomwalk shadow / 其他非 question 文件
    files = [f for f in files if "_randomwalk_shadow" not in f]

    records: list[dict] = []
    audit = {
        "total_considered": 0,
        "files_skipped_unknown_source": 0,
        "tier_a_breached_count": 0,
        "tier_b_breached_count": 0,
        "dropped_hard_attacker_passed_false": 0,
        "dropped_no_tool_breach_override": 0,
        "dropped_readability_rejected": 0,
        "readability_filter_applied": apply_readability_filter,
        "kept": 0,
        "dropped": 0,
        "missing_image": 0,
    }

    for fp in files:
        # 先从文件名判定来源，只处理本周期的 sku + stress suite
        fname_iid = os.path.splitext(os.path.basename(fp))[0]
        if fname_iid not in stress_map and not fname_iid.startswith("sku_"):
            audit["files_skipped_unknown_source"] += 1
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"WARN: failed to read {fp}: {e}", file=sys.stderr)
            continue

        img_id = data.get("image_id") or os.path.splitext(os.path.basename(fp))[0]
        img_path = data.get("image_path") or ""
        if not img_path or not os.path.exists(img_path):
            resolved = _find_image(img_id)
            if resolved:
                img_path = resolved

        category = _lookup_category(img_id, stress_map)

        for q in data.get("level_3", []):
            audit["total_considered"] += 1

            # 审计两 tier 的 breach 情况（不影响保留决定；保留决定只看 hard_attacker_passed）
            tier_a_rec = q.get("attack_qwen3-vl-30b-a3b") or {}
            tier_b_rec = q.get("attack_gemini") or {}
            if tier_a_rec.get("attack_breached"):
                audit["tier_a_breached_count"] += 1
            if tier_b_rec.get("attack_breached"):
                audit["tier_b_breached_count"] += 1

            # 分别记录三种过滤原因，便于审计
            hap_false = q.get("hard_attacker_passed", True) is False
            nt_breach = _no_tool_breached(q)
            rd_rejected = apply_readability_filter and q.get("readability_flag") == "rejected"

            if hap_false:
                audit["dropped_hard_attacker_passed_false"] += 1
            if nt_breach and not hap_false:
                audit["dropped_no_tool_breach_override"] += 1
            if rd_rejected and not hap_false and not nt_breach:
                audit["dropped_readability_rejected"] += 1
            if hap_false or nt_breach or rd_rejected:
                audit["dropped"] += 1
                continue

            if not img_path:
                audit["missing_image"] += 1
                # 仍然保留，但 image_path 为空字符串（下游 asserter 会捕获）

            rec = _build_record(img_id, img_path, category, q)
            records.append(rec)

    # 排序
    records.sort(key=lambda r: (r["category"], r["image_id"], r["question_id"]))
    audit["kept"] = len(records)

    # 统计
    by_category = Counter(r["category"] for r in records)
    by_bucket = Counter(r.get("hard_bucket") or "unknown" for r in records)
    by_family = Counter(r.get("family") or "unknown" for r in records)
    per_cat_bucket: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        per_cat_bucket[r["category"]][r.get("hard_bucket") or "unknown"] += 1

    stats = {
        "total": len(records),
        "by_category": dict(sorted(by_category.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_family": dict(sorted(by_family.items())),
        "per_category_bucket_matrix": {
            cat: dict(sorted(cnt.items())) for cat, cnt in sorted(per_cat_bucket.items())
        },
        "attack_audit": audit,
    }
    return records, stats


def write_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_summary(stats: dict) -> None:
    print(f"\n=== hard split summary ===")
    print(f"Total kept: {stats['total']}")
    print()
    print("By category:")
    for cat, n in stats["by_category"].items():
        print(f"  {cat:<10} {n:>4}")
    print()
    print("By bucket:")
    for b, n in stats["by_bucket"].items():
        print(f"  {b:<14} {n:>4}")
    print()
    print("By family:")
    for f, n in stats["by_family"].items():
        print(f"  {f:<22} {n:>4}")
    print()
    print("Per-category × bucket matrix:")
    buckets = sorted({b for m in stats["per_category_bucket_matrix"].values() for b in m})
    header = f"  {'cat':<10} " + "".join(f"{b[:11]:>13}" for b in buckets)
    print(header)
    for cat, mat in stats["per_category_bucket_matrix"].items():
        row = f"  {cat:<10} " + "".join(f"{mat.get(b, 0):>13}" for b in buckets)
        print(row)
    print()
    print("Attack audit:")
    aa = stats["attack_audit"]
    print(f"  total_considered:       {aa['total_considered']}")
    print(f"  tier_a_breached_count:  {aa['tier_a_breached_count']}")
    print(f"  tier_b_breached_count:  {aa['tier_b_breached_count']}")
    print(f"  dropped:                {aa['dropped']}")
    print(f"  kept:                   {aa['kept']}")
    print(f"  missing_image:          {aa['missing_image']}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate v0 hard split (L3 only) from attack-filtered question files")
    ap.add_argument("--questions-glob", default=QUESTIONS_GLOB,
                    help=f"glob for question files (default: {QUESTIONS_GLOB})")
    ap.add_argument("--category-map", default=STRESS_CATEGORY_MAP,
                    help=f"stress suite category map JSON (default: {STRESS_CATEGORY_MAP})")
    ap.add_argument("--out", default=DEFAULT_OUT_JSONL,
                    help=f"output JSONL path (default: {DEFAULT_OUT_JSONL})")
    ap.add_argument("--stats-out", default=DEFAULT_STATS,
                    help=f"output stats JSON (default: {DEFAULT_STATS})")
    ap.add_argument("--apply-readability-filter", action="store_true",
                    help="also drop records with readability_flag='rejected' (from step3b)")
    args = ap.parse_args()

    print(f"Aggregating from: {args.questions_glob}")
    print(f"readability_filter: {args.apply_readability_filter}")
    records, stats = aggregate_hard_split(
        questions_glob=args.questions_glob,
        category_map_path=args.category_map,
        apply_readability_filter=args.apply_readability_filter,
    )

    write_jsonl(args.out, records)
    print(f"Wrote {len(records)} records to {args.out}")

    os.makedirs(os.path.dirname(args.stats_out), exist_ok=True)
    with open(args.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Wrote stats to {args.stats_out}")

    print_summary(stats)


if __name__ == "__main__":
    main()
