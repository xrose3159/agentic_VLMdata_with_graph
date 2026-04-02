"""
端到端测试 step2 完整流程：
  2a. VLM + SAM3 实体提取
  2b-0. 跨实体关联发现
  2b-1. 搜索计划生成
  2b-2. 并行搜索 + Jina 深读
  2b-3. 三元组提取
  2b-3.5. 跨实体三元组合并
  2b-4. 多轮扩展搜索

用法:
  srun -p belt_road --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    /mnt/petrelfs/shangxiaoran/anaconda3/envs/math/bin/python test_step2_full.py

  指定图片:
    ... python test_step2_full.py --image images/img_0010.jpg
"""

import json
import os
import sys
import time
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


SEP = "=" * 72
SUB = "-" * 60


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def print_entities(entities):
    for e in entities:
        bbox = e.get("bbox")
        bbox_str = (f"[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                    if bbox else "无")
        nearby = e.get("nearby_entities", [])
        nearby_str = ", ".join(
            n.get("description", n.get("id", "?")) if isinstance(n, dict) else str(n)
            for n in nearby[:3]
        )
        print(f"  {e.get('id', '?'):5s} | {e.get('name', '?')[:28]:28s} | "
              f"type={e.get('type', '?'):10s} | "
              f"bbox={bbox_str:26s} | "
              f"conf={e.get('confidence', 0):.3f} | "
              f"loc={e.get('location_in_image', '?')}")
        if nearby_str:
            print(f"        └─ 附近: {nearby_str}")


def print_triples(triples, label="三元组"):
    if not triples:
        print(f"  (无{label})")
        return
    for i, t in enumerate(triples, 1):
        src_tag = f" [{t['source']}]" if t.get("source") else ""
        print(f"  {i:3d}. ({t.get('head', '?')}, {t.get('relation', '?')}, {t.get('tail', '?')}){src_tag}")
        fact = t.get("fact", "")
        if fact:
            print(f"       事实: {fact[:100]}")


def main():
    parser = argparse.ArgumentParser(description="端到端测试 step2 完整流程")
    parser.add_argument("--image", type=str, default=None, help="指定测试图片路径")
    args = parser.parse_args()

    if args.image:
        img_path = args.image
    else:
        import glob as g
        from core.config import FILTERED_IMAGE_DIR
        candidates = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            candidates.extend(g.glob(os.path.join(FILTERED_IMAGE_DIR, ext)))
        candidates.sort()
        if not candidates:
            imgs_dir = os.path.join(os.path.dirname(__file__), "images")
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                candidates.extend(g.glob(os.path.join(imgs_dir, ext)))
            candidates.sort()
        if not candidates:
            print("没有找到可用图片")
            sys.exit(1)
        img_path = candidates[0]

    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # 清除该图的 checkpoint，确保完整运行
    from core.checkpoint import CHECKPOINT_DIR
    ckpt_file = os.path.join(CHECKPOINT_DIR, "step2", f"{img_id}.json")
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)
        print(f"已清除旧 checkpoint: {ckpt_file}")

    print(SEP)
    print(f"  step2 完整流程测试")
    print(f"  图片: {img_path}  (id={img_id})")
    print(SEP)
    print()

    from step2_enrich import enrich_image

    t_start = time.time()
    result = enrich_image(img_path)
    t_total = time.time() - t_start

    if result is None:
        print(f"\n enrich_image 返回 None（实体太少或提取失败）")
        print(f"耗时: {fmt_time(t_total)}")
        sys.exit(1)

    # ========== 输出详细结果 ==========

    print(f"\n{SEP}")
    print(f"  完整结果（耗时 {fmt_time(t_total)}）")
    print(SEP)

    # 1) 图片描述 & 领域
    print(f"\n{'[图片描述]':>12}")
    desc = result.get("image_description", "")
    print(f"  {desc[:500]}")
    print(f"\n{'[领域]':>12}  {result.get('domain', '?')}")

    # 2) 实体
    entities = result.get("entities", [])
    print(f"\n{SUB}")
    print(f"  2a. 实体提取结果 — 共 {len(entities)} 个实体")
    print(SUB)
    print_entities(entities)

    # 3) 搜索计划
    plans = result.get("search_plans", [])
    searchable_plans = [p for p in plans if not p.get("skip", False)]
    skipped_plans = [p for p in plans if p.get("skip", False)]
    print(f"\n{SUB}")
    print(f"  2b-1. 搜索计划 — {len(searchable_plans)} 个需搜索, {len(skipped_plans)} 个跳过")
    print(SUB)
    for p in searchable_plans:
        queries = p.get("queries", [])
        print(f"  {p.get('entity_id', '?'):5s} {p.get('entity_name', '?')[:30]:30s}")
        for q in queries:
            print(f"        → \"{q.get('query', '')}\"  ({q.get('purpose', '')})")
    if skipped_plans:
        print(f"  跳过: {', '.join(p.get('entity_name', '?') for p in skipped_plans)}")

    # 4) 搜索结果
    search_results = result.get("search_results", [])
    total_searches = sum(len(sr.get("searches", [])) for sr in search_results)
    total_deep = sum(
        sum(len(s.get("deep_reads", [])) for s in sr.get("searches", []))
        for sr in search_results
    )
    print(f"\n{SUB}")
    print(f"  2b-2. 搜索结果 — {len(search_results)} 个实体, {total_searches} 次搜索, {total_deep} 篇深读")
    print(SUB)
    for sr in search_results:
        rnd = sr.get("round", 1)
        rnd_tag = f" [R{rnd}]" if rnd > 1 else ""
        n_res = sum(len(s.get("results", [])) for s in sr.get("searches", []))
        n_deep = sum(len(s.get("deep_reads", [])) for s in sr.get("searches", []))
        print(f"  {sr.get('entity_id', '?'):8s} {sr.get('entity_name', '?')[:30]:30s}"
              f"  {n_res} 条结果, {n_deep} 篇深读{rnd_tag}")
        for s in sr.get("searches", []):
            q = s.get("query", "")
            n = len(s.get("results", []))
            print(f"          \"{q[:60]}\" → {n} 条")

    # 5) 三元组
    triples = result.get("triples", [])
    cross = [t for t in triples if t.get("source") == "cross_entity"]
    regular = [t for t in triples if t.get("source") != "cross_entity"]
    print(f"\n{SUB}")
    print(f"  2b-3 + 2b-3.5 + 2b-4. 三元组 — 共 {len(triples)} 个 "
          f"(常规={len(regular)}, 跨实体桥接={len(cross)})")
    print(SUB)
    print_triples(triples)

    # 6) 保存完整 JSON
    out_json = f"test_full_{img_id}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n{SUB}")
    print(f"  完整 JSON 已保存: {out_json}")
    print(f"  总耗时: {fmt_time(t_total)}")
    print(SUB)


if __name__ == "__main__":
    main()
