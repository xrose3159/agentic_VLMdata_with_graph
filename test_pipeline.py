"""
单张图片全流程测试脚本：Step1 → Step2 → Step3 → Step4

每一步都打印详细输出，方便逐步观察。

用法：
    python test_pipeline.py                           # 默认用 images/ 第一张图
    python test_pipeline.py images/img_0007.jpg
    python test_pipeline.py images/img_0007.jpg --skip-step1   # 跳过 step1（图已在 output/images/）
    python test_pipeline.py images/img_0007.jpg --from-step 3  # 从 step3 开始（需已有 entity json）
"""

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# 打印工具
# ============================================================

def banner(title: str):
    print("\n" + "═" * 68)
    print(f"  {title}")
    print("═" * 68)

def step_header(n: int, title: str):
    print(f"\n{'─' * 68}")
    print(f"  STEP {n} · {title}")
    print(f"{'─' * 68}")

def section(title: str):
    print(f"\n  ▌ {title}")

def ok(msg):   print(f"  ✓ {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def info(msg): print(f"    {msg}")


# ============================================================
# STEP 1 · 图片筛选
# ============================================================

def run_step1(img_path: str) -> str | None:
    """将单张图片送入 step1 评分，通过则复制到 output/images/，返回目标路径。"""
    from step1_filter import evaluate_image as score_image
    from core.config import FILTERED_IMAGE_DIR

    step_header(1, "图片筛选（VLM 五维打分）")

    t0 = time.time()
    result = score_image(img_path)
    elapsed = time.time() - t0

    if result is None:
        warn("VLM 打分失败")
        return None

    scores = result.get("scores", {})
    total  = result.get("total_score", 0)
    passed = result.get("pass", False)
    category = result.get("category", "unknown")

    section("评分详情")
    dims = ["entity_richness", "detail_depth", "external_linkage", "entity_relations", "naturalness"]
    for d in dims:
        bar = "█" * scores.get(d, 0) + "░" * (5 - scores.get(d, 0))
        print(f"    {d:<22} [{bar}] {scores.get(d, 0)}/5")
    print(f"\n    总分: {total}/25   类别: {category}   耗时: {elapsed:.1f}s")

    section("图片描述")
    info(result.get("image_description", "（无）")[:200])

    if not passed:
        warn(f"图片未通过筛选（总分 {total} < 18 或某维度 < 3），跳过后续步骤")
        return None

    os.makedirs(FILTERED_IMAGE_DIR, exist_ok=True)
    img_id   = os.path.splitext(os.path.basename(img_path))[0]
    dst_ext  = os.path.splitext(img_path)[1]
    dst_path = os.path.join(FILTERED_IMAGE_DIR, f"{img_id}{dst_ext}")
    if os.path.abspath(img_path) != os.path.abspath(dst_path):
        shutil.copy2(img_path, dst_path)
    ok(f"通过筛选 → {dst_path}")
    return dst_path


# ============================================================
# STEP 2 · 实体提取与知识图谱扩展
# ============================================================

def run_step2(img_path: str) -> dict | None:
    from step2_enrich import enrich_image

    step_header(2, "实体提取 + 知识图谱扩展")
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    t0 = time.time()
    result = enrich_image(img_path)
    elapsed = time.time() - t0

    if result is None:
        warn("实体提取失败或实体数不足")
        return None

    entities = result.get("entities", [])
    triples  = result.get("triples", [])

    # ── 实体 ──
    section(f"识别到 {len(entities)} 个实体")
    for e in entities:
        crop_ok = "✓" if e.get("crop_path") and os.path.isfile(e["crop_path"]) else "✗"
        print(
            f"    {e['id']:>3}  {e['name']:<28}"
            f"  type={e.get('type','?'):<12}"
            f"  位置={e.get('location_in_image',''):<16}"
            f"  crop={crop_ok}"
        )

    # ── 搜索计划摘要 ──
    plans = result.get("search_plans", [])
    searchable = [p for p in plans if not p.get("skip")]
    skipped    = [p for p in plans if p.get("skip")]
    section(f"搜索计划：{len(searchable)} 个实体搜索，{len(skipped)} 个跳过")
    for p in skipped:
        info(f"✗ {p['entity_name']}  ({p.get('skip_reason','')})")
    for p in searchable:
        queries = [q["query"] for q in p.get("queries", [])]
        info(f"✓ {p['entity_name']}  →  {' | '.join(queries)}")

    # ── 搜索结果摘要 ──
    all_sr = result.get("search_results", [])
    total_results = sum(len(s.get("results", [])) for er in all_sr for s in er.get("searches", []))
    total_deep    = sum(len(s.get("deep_reads", [])) for er in all_sr for s in er.get("searches", []))
    section(f"搜索结果：{total_results} 条摘要  {total_deep} 篇深读")

    # ── 三元组 ──
    cross = [t for t in triples if t.get("source") == "cross_entity"]
    section(f"三元组：{len(triples)} 个（其中跨实体 {len(cross)} 个）")
    for t in triples[:15]:
        marker = " 🔗" if t.get("source") == "cross_entity" else ""
        print(f"    ({t['head']})  ──[{t['relation']}]──▶  ({t['tail']}){marker}")
    if len(triples) > 15:
        info(f"... 省略 {len(triples)-15} 条")

    print(f"\n  耗时 {elapsed:.1f}s")
    return result


# ============================================================
# STEP 3 · 分层问题生成
# ============================================================

def run_step3(entity_file: str) -> dict | None:
    from step3_generate import generate_questions, find_motifs, motif_to_skeleton

    step_header(3, "分层问题生成（L1 / L2 / L3）")

    with open(entity_file, encoding="utf-8") as f:
        entity_data = json.load(f)
    entities = entity_data.get("entities", [])
    triples  = entity_data.get("triples", [])

    section("知识图谱建图与 Motif 探测")
    motifs, nodes = find_motifs(triples, entities)
    in_image_nodes = [n for n in nodes.values() if n["in_image"]]
    info(f"节点总数: {len(nodes)}  图中实体节点: {len(in_image_nodes)}")
    info(f"Motif 数量: Bridge={len(motifs['bridge'])}  MultiHop={len(motifs['multihop'])}  Comparative={len(motifs['comparative'])}")

    for mtype, label in [("bridge", "Bridge L2"), ("multihop", "MultiHop L3"), ("comparative", "Comparative L2")]:
        mlist = motifs.get(mtype, [])
        if not mlist:
            continue
        print(f"\n  [{label} Motif]")
        for m in mlist[:5]:
            skeleton = motif_to_skeleton(m, nodes)
            anchors = skeleton.get("visual_anchors", {})
            target  = skeleton.get("target_answer", "?")
            anchor_str = "  +  ".join(f"{k}={v}" for k, v in anchors.items())
            print(f"    {anchor_str}  →  答案: {target}")

    section("LLM 润色生成自然语言问题")
    t0 = time.time()
    result = generate_questions(entity_file)
    elapsed = time.time() - t0

    if result is None:
        warn("问题生成失败")
        return None

    meta = result.get("metadata", {})
    print(f"\n  生成完成  L1={meta.get('level_1_count',0)}  "
          f"L2={meta.get('level_2_count',0)}  L3={meta.get('level_3_count',0)}  "
          f"耗时 {elapsed:.1f}s")

    for level_key, label in [("level_1", "L1"), ("level_2", "L2"), ("level_3", "L3")]:
        qs = result.get(level_key, [])
        if not qs:
            continue
        section(f"{label} 题目（共 {len(qs)} 道）")
        for q in qs:
            print(f"\n  [{q.get('question_id','')}]")
            print(f"    问题: {q.get('question','')}")
            print(f"    答案: {q.get('answer','')}")
            tools = " → ".join(s.get("tool","?") for s in q.get("tool_sequence", []))
            print(f"    工具: {tools}")
            obf = q.get("obfuscated_entities", [])
            if obf:
                print(f"    模糊化: {obf}")

    return result


# ============================================================
# STEP 4 · 模糊化验证与修正
# ============================================================

def run_step4(q_file: str) -> dict | None:
    from step4_verify import verify_image_questions, structural_check

    step_header(4, "模糊化验证与修正")

    with open(q_file, encoding="utf-8") as f:
        question_data = json.load(f)

    # 结构性检查（不调用 VLM，立即显示）
    section("结构性检查")
    issues, auto_fixes, drop_ids = structural_check(question_data)
    if not issues and not drop_ids:
        ok("无结构性问题")
    for iss in issues:
        warn(iss)
    if auto_fixes:
        ok(f"自动修正 {auto_fixes} 个无效工具名")
    if drop_ids:
        for qid, reason in drop_ids.items():
            warn(f"丢弃 {qid}: {reason}")

    # VLM 模糊化验证
    section("VLM 模糊化验证（L2 / L3）")
    t0 = time.time()
    result = verify_image_questions(q_file)
    elapsed = time.time() - t0

    if result is None:
        warn("验证失败")
        return None

    struct_n = len(result.get("structural_issues", []))
    fix_n    = result.get("obfuscation_fixes", 0)
    rej_n    = result.get("rejected_count", 0)

    ok(f"验证完成  结构问题={struct_n}  模糊化修正={fix_n}  丢弃题目={rej_n}  耗时 {elapsed:.1f}s")

    # 展示修正详情
    vlm_ver = result.get("vlm_verification") or {}
    for v in vlm_ver.get("verifications", []):
        qid     = v.get("question_id", "?")
        correct = v.get("obfuscation_correct", True)
        reason  = v.get("reason", "")
        fixed   = v.get("fixed_question", "")
        if not correct:
            print(f"\n  [{qid}] 模糊化不合格")
            info(f"原因: {reason}")
            if fixed:
                info(f"修正: {fixed}")
        else:
            print(f"  [{qid}] ✓ 模糊化合格")

    # 展示最终题目
    with open(q_file, encoding="utf-8") as f:
        final_data = json.load(f)

    section("最终题目汇总")
    for level_key, label in [("level_1", "L1"), ("level_2", "L2"), ("level_3", "L3")]:
        qs = final_data.get(level_key, [])
        for q in qs:
            prefix = "  [已修正]" if q.get("question_original") else "         "
            print(f"\n{prefix} [{q.get('question_id','')}]")
            print(f"    问题: {q.get('question','')}")
            print(f"    答案: {q.get('answer','')}")
            tools = " → ".join(s.get("tool","?") for s in q.get("tool_sequence", []))
            print(f"    工具: {tools}")

    return result


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="单张图片全流程测试")
    parser.add_argument("img_path", nargs="?", default=None,
                        help="图片路径（默认自动取 images/ 第一张）")
    parser.add_argument("--skip-step1", action="store_true",
                        help="跳过 step1，图片已在 output/images/")
    parser.add_argument("--from-step", type=int, default=1, choices=[1, 2, 3, 4],
                        help="从哪一步开始（默认 1）")
    args = parser.parse_args()

    # 自动找图片
    if args.img_path is None:
        import glob as _glob
        imgs = sorted(_glob.glob("images/*.jpg") + _glob.glob("images/*.jpeg") +
                      _glob.glob("images/*.png") + _glob.glob("images/*.webp"))
        if not imgs:
            print("images/ 目录下没有图片")
            sys.exit(1)
        args.img_path = imgs[0]

    img_path = args.img_path
    img_id   = os.path.splitext(os.path.basename(img_path))[0]

    banner(f"全流程测试 · {img_id}")
    print(f"  图片: {img_path}  ({os.path.getsize(img_path)/1024:.1f} KB)")

    from core.config import FILTERED_IMAGE_DIR, ENTITY_DIR, QUESTION_DIR

    filtered_img  = os.path.join(FILTERED_IMAGE_DIR, os.path.basename(img_path))
    entity_file   = os.path.join(ENTITY_DIR, f"{img_id}.json")
    question_file = os.path.join(QUESTION_DIR, f"{img_id}.json")

    start = args.from_step

    # ── Step 1 ──
    if start <= 1 and not args.skip_step1:
        filtered_img = run_step1(img_path) or filtered_img
        if not os.path.exists(filtered_img):
            sys.exit(0)
    else:
        if not os.path.exists(filtered_img):
            print(f"\n[错误] 跳过 step1 但图片不存在: {filtered_img}")
            sys.exit(1)
        step_header(1, "图片筛选（已跳过）")
        ok(f"使用已有图片: {filtered_img}")

    # ── Step 2 ──
    if start <= 2:
        result2 = run_step2(filtered_img)
        if result2 is None:
            sys.exit(0)
    else:
        if not os.path.exists(entity_file):
            print(f"\n[错误] 跳过 step2 但实体文件不存在: {entity_file}")
            sys.exit(1)
        step_header(2, "实体提取（已跳过）")
        d = json.load(open(entity_file))
        ok(f"使用已有实体文件: {entity_file}")
        info(f"实体数: {len(d.get('entities',[]))}  三元组数: {len(d.get('triples',[]))}")

    # ── Step 3 ──
    if start <= 3:
        result3 = run_step3(entity_file)
        if result3 is None:
            sys.exit(0)
    else:
        if not os.path.exists(question_file):
            print(f"\n[错误] 跳过 step3 但问题文件不存在: {question_file}")
            sys.exit(1)
        step_header(3, "问题生成（已跳过）")
        d = json.load(open(question_file))
        meta = d.get("metadata", {})
        ok(f"使用已有问题文件  L1={meta.get('level_1_count',0)} "
           f"L2={meta.get('level_2_count',0)} L3={meta.get('level_3_count',0)}")

    # ── Step 4 ──
    run_step4(question_file)

    # ── 汇总 ──
    banner("测试完成 · 汇总")
    if os.path.exists(entity_file):
        d = json.load(open(entity_file))
        info(f"实体数   : {len(d.get('entities',[]))}")
        info(f"三元组数 : {len(d.get('triples',[]))}")
    if os.path.exists(question_file):
        d = json.load(open(question_file))
        meta = d.get("metadata", {})
        info(f"题目总数 : {meta.get('total_questions',0)}"
             f"  L1={meta.get('level_1_count',0)}"
             f"  L2={meta.get('level_2_count',0)}"
             f"  L3={meta.get('level_3_count',0)}")
    print()


if __name__ == "__main__":
    main()
