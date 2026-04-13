"""诊断：sports 图的 image_heavy 闭合可用性。

n_walks=48 跑 8 张 sports 图，在选题前截取所有 L3 closure 的 bucket 分布，
记录 4 个关键指标：

1. 原始 image_heavy closures 数量（bucket 分类后、check_tool_irreducibility 之后）
2. 1-anchor + image_search_needed + follow≥2 的 closures（潜在 image_resolve_follow 子型）
3. 被 ultra_long 抢走的 image_search_needed closures
4. person/Lens 证据是否主要被 walker 吃成了 multi_hop / ultra_long

不改主代码，只读 + 临时 monkey-patch n_walks。
"""
import os, sys, json, time
from collections import Counter, defaultdict
import core.vlm
core.vlm.MODEL_NAME = 'gemini-3-flash-preview-nothinking'
from core.checkpoint import _ckpt_path

# Monkey-patch n_walks
import step3_trajectory
original_generate = step3_trajectory.generate_questions

def patched_generate(entity_json_path, image_path=None, *, seed=42, tau=0.7, n_walks=48, **kw):
    return original_generate(entity_json_path, image_path=image_path, seed=seed, tau=tau, n_walks=n_walks, **kw)

step3_trajectory.generate_questions = patched_generate

# Also need to intercept the internal closure data BEFORE selection.
# We'll read the checkpoint after generation and analyze.
from step3_generate import generate_questions
from step3_trajectory import _classify_hard_bucket, compile_tool_plan, check_tool_irreducibility

sports = sorted([os.path.splitext(f)[0] for f in os.listdir('images')
                 if f.startswith('sports_') and f.endswith('.png')])
print(f"Diagnostic: {len(sports)} sports images, n_walks=48")

# Clear step3 checkpoints
for iid in sports:
    cp = _ckpt_path(3, iid)
    if os.path.exists(cp): os.remove(cp)

# Run Step3 and analyze each
grand = {
    "total_l3_closures": 0,
    "by_bucket_raw": Counter(),        # bucket BEFORE irr check
    "by_bucket_after_irr": Counter(),  # bucket AFTER irr check (降级后)
    "image_heavy_raw": 0,
    "image_heavy_after_irr": 0,
    "single_anchor_image_needed_follow2": 0,  # 潜在 image_resolve_follow
    "stolen_by_ultra_long": 0,         # image_search_needed 但归入 ultra_long
    "person_entity_in_closures": 0,    # 闭合里包含 person 实体的
    "per_image": {},
}

for iid in sports:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {iid}", flush=True)

    r = generate_questions(f'output/entities/{iid}.json')
    dt = time.time() - t0

    if r is None:
        print(f"  SKIPPED (None)")
        continue

    # 从 checkpoint 读完整结果（包含 metadata）
    cp_data = json.load(open(_ckpt_path(3, iid)))
    meta = cp_data.get("metadata", {})

    # 获取 L3 题
    l3 = r.get("level_3", [])
    l3_buckets = Counter(q.get("hard_bucket", "standard") for q in l3)

    print(f"  L3 produced: {len(l3)} — {dict(l3_buckets)}")
    print(f"  time: {dt:.0f}s")

    # 尝试从 metadata 获取 tool_irr_rejects
    irr_rejects = meta.get("tool_irreducibility_rejects", [])
    if isinstance(irr_rejects, list):
        for rej in irr_rejects:
            if isinstance(rej, dict):
                grand["by_bucket_after_irr"][rej.get("reason", "?")] += 1

    # 分析每道 L3 的结构
    img_data = {
        "l3_count": len(l3),
        "buckets": dict(l3_buckets),
        "image_heavy_count": l3_buckets.get("image_heavy", 0),
        "single_anchor_image_candidates": 0,
        "ultra_long_with_image_search": 0,
    }

    for q in l3:
        bucket = q.get("hard_bucket", "standard")
        family = q.get("family", "")
        rp = q.get("reasoning_path", {})
        anchors = rp.get("anchors", [])
        hops = rp.get("hop_chain", [])
        branches = rp.get("branches", [])
        tool_seq = q.get("tool_sequence", [])

        grand["total_l3_closures"] += 1
        grand["by_bucket_raw"][bucket] += 1

        has_image_search = any(s.get("tool") == "image_search" for s in tool_seq)
        has_image_needed_anchor = len(anchors) >= 1  # proxy: all anchors in sports are persons now
        n_anchors = len(anchors)
        n_follow = len(hops) + len(branches)

        # 指标 2: 1 锚点 + image_search + follow≥2
        if n_anchors == 1 and has_image_search and n_follow >= 2:
            img_data["single_anchor_image_candidates"] += 1
            grand["single_anchor_image_needed_follow2"] += 1

        # 指标 3: ultra_long 抢走 image_search
        if bucket == "ultra_long" and has_image_search:
            img_data["ultra_long_with_image_search"] += 1
            grand["stolen_by_ultra_long"] += 1

        # 指标 4: person entity 在闭合里
        for hop in hops:
            if isinstance(hop, dict):
                ent = hop.get("entity", "")
                # person proposal entities 通常有描述性名字
                if "person" in ent.lower() or "jersey" in ent.lower() or "player" in ent.lower():
                    grand["person_entity_in_closures"] += 1
                    break

    grand["image_heavy_raw"] += img_data["image_heavy_count"]
    grand["per_image"][iid] = img_data
    print(f"  image_heavy: {img_data['image_heavy_count']}")
    print(f"  1-anchor + image_search + follow≥2: {img_data['single_anchor_image_candidates']}")
    print(f"  ultra_long stealing image_search: {img_data['ultra_long_with_image_search']}")

print(f"\n{'='*60}")
print(f"=== GRAND DIAGNOSTIC SUMMARY (n_walks=48, 8 sports) ===")
print(f"{'='*60}")
print(f"Total L3 closures: {grand['total_l3_closures']}")
print(f"By bucket: {dict(grand['by_bucket_raw'])}")
print()
print(f"1. image_heavy closures (current rule): {grand['image_heavy_raw']}")
print(f"2. 1-anchor + image_search + follow≥2 (潜在 image_resolve_follow): {grand['single_anchor_image_needed_follow2']}")
print(f"3. ultra_long that have image_search (被抢走): {grand['stolen_by_ultra_long']}")
print(f"4. person entity referenced in closures: {grand['person_entity_in_closures']}")
print()

# 关键判断
if grand["image_heavy_raw"] == 0 and grand["single_anchor_image_needed_follow2"] > 0:
    print("★ 结论：image_heavy 定义太窄。")
    print(f"  有 {grand['single_anchor_image_needed_follow2']} 个闭合满足 '1-anchor + image_search + follow≥2'")
    print(f"  但因 ≥2 锚点门槛被排除在 image_heavy 外。")
    print(f"  → 建议拆 image_heavy 为 image_resolve_follow + image_compare 两个子型")
elif grand["image_heavy_raw"] == 0 and grand["stolen_by_ultra_long"] > 0:
    print("★ 结论：ultra_long 优先级太高，抢走了 image_heavy 候选。")
    print(f"  {grand['stolen_by_ultra_long']} 个含 image_search 的闭合被 ultra_long 吃掉。")
    print(f"  → 建议调整 ultra_long vs image_heavy 的优先级")
elif grand["image_heavy_raw"] > 0:
    print(f"★ 结论：n_walks=48 确实能产出 image_heavy（{grand['image_heavy_raw']} 个）。")
    print("  → n_walks=24 可能只是采样不够")
else:
    print("★ 结论：即使 n_walks=48 也没有 image_heavy 或潜在候选。")
    print("  → 需要更深层的 bucket 定义修改或 Step2 图谱结构调整")
