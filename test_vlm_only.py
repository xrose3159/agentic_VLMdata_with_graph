"""
单张图片的分步测试脚本：纯 VLM 实体提取 → 搜索计划 → 并行搜索 → 三元组提取

用法：
    python test_vlm_only.py                        # 默认使用 output/images/img_0007.jpg
    python test_vlm_only.py output/images/img_0010.jpg
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 确保在项目根目录下运行
sys.path.insert(0, os.path.dirname(__file__))

from step2_enrich import (
    VLM_ENTITY_PROMPT,
    SEARCH_PLAN_PROMPT,
    TRIPLE_EXTRACTION_PROMPT,
    extract_entities_vlm,
    web_search,
    _deep_read_top_urls,
    _normalize_triple_entities,
    _find_cross_entity_relations,
    _add_spatial_fallback,
)
from core.vlm import call_vlm_json
from core.config import ENTITY_DIR


# ============================================================
# 打印工具
# ============================================================

def banner(title: str):
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step_header(step: str, desc: str):
    print(f"\n{'─' * 64}")
    print(f"  [{step}] {desc}")
    print(f"{'─' * 64}")


def print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ============================================================
# 主测试流程
# ============================================================

def run_test(img_path: str):
    if not os.path.isfile(img_path):
        print(f"[错误] 图片不存在: {img_path}")
        sys.exit(1)

    img_id = os.path.splitext(os.path.basename(img_path))[0]

    banner(f"VLM-Only 实体提取测试 | {img_id}")
    print(f"  图片路径 : {img_path}")
    print(f"  图片大小 : {os.path.getsize(img_path) / 1024:.1f} KB")

    # ──────────────────────────────────────────────────────────
    # STEP 1 · VLM 实体识别（bbox + location）
    # ──────────────────────────────────────────────────────────
    step_header("STEP 1", "VLM 实体识别 — 输出 BBox + 位置描述")
    t0 = time.time()
    entity_data = extract_entities_vlm(img_path, img_id=img_id)
    elapsed = time.time() - t0

    entities       = entity_data.get("entities", [])
    image_desc     = entity_data.get("image_description", "")
    domain         = entity_data.get("domain", "other")

    print(f"\n[图片描述]")
    print(f"  {image_desc}")
    print(f"\n[领域] {domain}")
    print(f"\n[识别到 {len(entities)} 个实体]  耗时 {elapsed:.1f}s")

    if not entities:
        print("\n[警告] 未识别到任何实体，终止测试")
        return

    for e in entities:
        crop_exists = "✓" if e.get("crop_path") and os.path.isfile(e["crop_path"]) else "✗"
        print(
            f"  {e['id']:>3}  {e['name']:<30}"
            f"  type={e.get('type','?'):<12}"
            f"  bbox={str(e.get('bbox')):<26}"
            f"  位置={e.get('location_in_image',''):<14}"
            f"  crop={crop_exists}"
        )

    print(f"\n[裁剪图片目录] {os.path.join(ENTITY_DIR, 'crops', img_id)}/")
    for e in entities:
        cp = e.get("crop_path")
        if cp and os.path.isfile(cp):
            size_kb = os.path.getsize(cp) / 1024
            print(f"    {os.path.basename(cp)}  ({size_kb:.1f} KB)")

    # 选出高置信度实体（最多 5 个），后续所有步骤基于此子集
    high_conf = [e for e in entities if e.get("confidence_level") in ("high", "medium")][:5]
    if not high_conf:
        high_conf = entities[:5]
    print(f"\n  [高置信度实体] {len(high_conf)} 个: {[e['name'] for e in high_conf]}")

    # ──────────────────────────────────────────────────────────
    # STEP 2 · 跨实体关联发现（只在 high_conf 实体之间查找）
    # ──────────────────────────────────────────────────────────
    step_header("STEP 2", "跨实体关联发现（仅限被搜索的实体）")
    t0 = time.time()
    cross_triples = _find_cross_entity_relations(img_path, high_conf, img_id)
    elapsed = time.time() - t0
    print(f"  发现跨实体三元组 {len(cross_triples)} 个  耗时 {elapsed:.1f}s")
    for t in cross_triples:
        print(f"    ({t['head']})  ──[{t['relation']}]──▶  ({t['tail']})")

    # ──────────────────────────────────────────────────────────
    # STEP 3 · 搜索计划生成
    # ──────────────────────────────────────────────────────────
    step_header("STEP 3", "LLM 生成搜索计划")

    entities_json_str = json.dumps(high_conf, ensure_ascii=False, indent=2)
    t0 = time.time()
    search_plan_data = call_vlm_json(
        SEARCH_PLAN_PROMPT.format(image_description=image_desc, entities_json=entities_json_str),
        "请为每个实体生成搜索计划。",
        max_tokens=2048,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed:.1f}s")

    if search_plan_data is None:
        print("  [错误] 搜索计划生成失败")
        return

    plans = search_plan_data.get("search_plans", [])
    searchable = [p for p in plans if not p.get("skip", False)]
    skipped    = [p for p in plans if p.get("skip", False)]
    print(f"\n  需搜索: {len(searchable)} 个实体   跳过: {len(skipped)} 个实体")

    for p in skipped:
        print(f"    ✗ {p['entity_name']}  原因: {p.get('skip_reason','')}")

    for p in searchable:
        print(f"\n  ✓ {p['entity_id']} {p['entity_name']}")
        for q in p.get("queries", []):
            print(f"      查询: \"{q['query']}\"")
            print(f"      目的: {q['purpose']}")

    # ──────────────────────────────────────────────────────────
    # STEP 4 · 并行执行搜索
    # ──────────────────────────────────────────────────────────
    step_header("STEP 4", "并行执行网络搜索（SerpAPI）")
    search_tasks = []
    for plan in searchable:
        eid   = plan.get("entity_id", "")
        ename = plan.get("entity_name", "")
        for q_item in plan.get("queries", []):
            query = q_item.get("query", "")
            if query:
                search_tasks.append((eid, ename, query, q_item.get("purpose", "")))

    print(f"  共 {len(search_tasks)} 条搜索任务，并行执行...")

    search_results_map = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(web_search, q, 5): (eid, ename, q, purpose)
                   for eid, ename, q, purpose in search_tasks}
        for fut in as_completed(futures):
            eid, ename, query, purpose = futures[fut]
            sr = fut.result()
            sr["purpose"] = purpose
            search_results_map[(eid, query)] = sr
            n = len(sr.get("results", []))
            err = f"  ⚠ {sr['error']}" if sr.get("error") else ""
            print(f"    [{eid}] \"{query}\" → {n} 条结果{err}")
    elapsed = time.time() - t0
    print(f"  搜索完成，耗时 {elapsed:.1f}s")

    # 整理成 entity 维度的结构
    all_search_results = []
    for plan in searchable:
        eid   = plan.get("entity_id", "")
        ename = plan.get("entity_name", "")
        entity_results = {"entity_id": eid, "entity_name": ename, "searches": []}
        for q_item in plan.get("queries", []):
            query = q_item.get("query", "")
            if query and (eid, query) in search_results_map:
                entity_results["searches"].append(search_results_map[(eid, query)])
        all_search_results.append(entity_results)

    # 展示每条搜索的 top snippet
    print(f"\n[搜索摘要]")
    for er in all_search_results:
        print(f"\n  ▶ {er['entity_name']} ({er['entity_id']})")
        for s in er.get("searches", []):
            print(f"    查询: \"{s.get('query','')}\"")
            for r in s.get("results", [])[:3]:
                if r.get("type") == "knowledge_graph":
                    print(f"      [知识面板] {r['content'][:120]}")
                elif r.get("type") == "answer":
                    print(f"      [精选摘要] {r['content'][:120]}")
                else:
                    print(f"      [{r.get('title','')[:40]}] {r.get('snippet','')[:100]}")

    # ──────────────────────────────────────────────────────────
    # STEP 5 · Jina 深度读取（每条搜索最多 1 篇）
    # ──────────────────────────────────────────────────────────
    step_header("STEP 5", "Jina Reader 深度读取（每条搜索取 1 篇正文）")

    jina_tasks = []
    for i, er in enumerate(all_search_results):
        for j, s in enumerate(er.get("searches", [])):
            jina_tasks.append((i, j, s))

    print(f"  search_results_map keys (前3): {list(search_results_map.keys())[:3]}")
    print(f"  all_search_results searches counts: {[len(er.get('searches',[])) for er in all_search_results]}")
    print(f"  jina_tasks 数量: {len(jina_tasks)}")

    def _jina_task(task):
        i, j, s = task
        return i, j, _deep_read_top_urls(s, max_urls=5, max_chars=2000)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, j, deep in pool.map(_jina_task, jina_tasks):
            all_search_results[i]["searches"][j]["deep_reads"] = deep
    elapsed = time.time() - t0
    print(f"  Jina 读取完成，耗时 {elapsed:.1f}s")

    total_deep = sum(
        len(s.get("deep_reads", []))
        for er in all_search_results
        for s in er.get("searches", [])
    )
    print(f"  成功深读 {total_deep} 篇网页")

    for er in all_search_results:
        for s in er.get("searches", []):
            for d in s.get("deep_reads", []):
                print(f"\n    [{er['entity_name']}] {d['url']}")
                preview = d["content"][:200].replace("\n", " ")
                print(f"    {preview}...")

    # ──────────────────────────────────────────────────────────
    # STEP 6 · 三元组提取
    # ──────────────────────────────────────────────────────────
    step_header("STEP 6", "LLM 从搜索结果提取事实三元组")

    entities_summary = "\n".join(
        f"- {e['id']}: {e['name']} (类型={e.get('type','?')}, 位置={e.get('location_in_image','?')})"
        for e in high_conf
    )

    search_results_text = ""
    for sr in all_search_results:
        search_results_text += f"\n### {sr['entity_name']} ({sr['entity_id']})\n"
        for s in sr.get("searches", []):
            search_results_text += f"\n搜索词: \"{s.get('query','')}\" (目的: {s.get('purpose','')})\n"
            for r in s.get("results", []):
                if r.get("type") == "knowledge_graph":
                    search_results_text += f"  [知识面板] {r['content']}\n"
                elif r.get("type") == "answer":
                    search_results_text += f"  [精选摘要] {r['content']}\n"
                else:
                    search_results_text += f"  [{r.get('title','')}] {r.get('snippet','')[:300]}\n"
            for d in s.get("deep_reads", []):
                search_results_text += f"\n  [深度读取: {d.get('title','')}]\n  {d['content'][:1500]}\n"

    t0 = time.time()
    triple_data = call_vlm_json(
        TRIPLE_EXTRACTION_PROMPT.format(
            entities_summary=entities_summary,
            search_results=search_results_text,
        ),
        "请严格基于搜索结果提取事实三元组。不要编造搜索结果中没有的信息。",
        max_tokens=4096,
        temperature=0.3,
    )
    elapsed = time.time() - t0
    print(f"  耗时 {elapsed:.1f}s")

    if triple_data is None:
        print("  [错误] 三元组提取失败")
        return

    triples = triple_data.get("triples", [])
    triples = triples + cross_triples
    before = len(triples)
    triples = _normalize_triple_entities(triples, entities)
    print(f"\n  提取到 {before} 个三元组，规范化后 {len(triples)} 个")

    # 空间关系兜底
    spatial_added = _add_spatial_fallback(triples, high_conf)
    if spatial_added:
        print(f"  空间关系兜底: 补充 {len(spatial_added)} 条"
              f" → {[(t['head'], t['tail']) for t in spatial_added]}")
    print()

    for i, t in enumerate(triples, 1):
        print(f"  {i:>3}. ({t.get('head','?')})  ──[{t.get('relation','?')}]──▶  ({t.get('tail','?')})")
        print(f"        事实: {t.get('fact','')}")
        snippet = t.get("source_snippet", "")
        if snippet:
            print(f"        来源: {snippet[:100]}")

    # ──────────────────────────────────────────────────────────
    # 汇总
    # ──────────────────────────────────────────────────────────
    banner("测试完成 · 汇总")
    print(f"  图片      : {img_path}")
    print(f"  识别实体  : {len(entities)} 个")
    print(f"  搜索查询  : {len(search_tasks)} 条")
    print(f"  深度读取  : {total_deep} 篇")
    print(f"  三元组    : {len(triples)} 个")

    # 写出完整 JSON 结果（同时写到 output/entities/ 供 test_pipeline.py --from-step 3 使用）
    import os as _os
    _os.makedirs(ENTITY_DIR, exist_ok=True)
    out_path = _os.path.join(ENTITY_DIR, f"{img_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "img_id": img_id,
            "image_description": image_desc,
            "domain": domain,
            "entities": entities,
            "search_plans": plans,
            "search_results": all_search_results,
            "triples": triples,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  完整结果已写入: {out_path}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM-Only 单张图片分步测试")
    parser.add_argument(
        "img_path",
        nargs="?",
        default="output/images/img_0007.jpg",
        help="要测试的图片路径（默认 output/images/img_0007.jpg）",
    )
    args = parser.parse_args()
    run_test(args.img_path)
