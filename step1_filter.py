"""
第一步：筛选高信息密度图片。

从 images/ 目录的候选图片中，用 VLM 评估打分，筛选出约50张高信息密度图片。
每张图片按5个维度打分（1-5），总分>=18且每项>=3为通过。

用法：
    python step1_filter.py
    python step1_filter.py --source images --workers 4
"""

import argparse
import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import (
    SOURCE_IMAGE_DIR, FILTERED_IMAGE_DIR, STATS_DIR,
    FILTER_MIN_TOTAL, FILTER_MIN_PER_DIM, FILTER_DIMENSIONS, MAX_WORKERS,
)
from core.vlm import call_vlm_json
from core.image_utils import file_to_b64, copy_image
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step1", "step1_filter.log")


# ============================================================
# Prompt
# ============================================================
IMAGE_FILTER_PROMPT = """\
请评估这张图片是否适合用于多模态agentic训练数据生成。

评估维度（每项1-5分）：
1. 实体丰富度(entity_richness)：图中有多少个可独立识别的实体（文字、数字、品牌、型号等）？
2. 信息层次性(detail_depth)：是否有需要放大/裁剪才能看清的细节？
3. 外部知识关联(external_linkage)：图中实体是否能关联到可搜索的外部知识（产品规格、百科信息等）？
4. 多实体关系(entity_relations)：实体之间是否存在对比、包含、空间等可推理关系？
5. 自然真实性(naturalness)：图片是否来自真实场景（非AI生成、非纯文本截图）？

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{
    "scores": {"entity_richness": int, "detail_depth": int, "external_linkage": int, "entity_relations": int, "naturalness": int},
    "total_score": int,
    "category": "产品货架|技术图纸|地图路线|菜单价目|体育数据|图表报表|其他",
    "brief_description": "图片的一句话描述",
    "estimated_entities": int,
    "pass": bool
}

注意：total_score 是五项分数之和，pass 为 true 当且仅当 total_score>=18 且每项>=3。"""


# ============================================================
# 单张图片处理
# ============================================================
def evaluate_image(img_path: str) -> dict | None:
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    # 检查点
    if is_done(1, img_id):
        logger.info(f"  [{img_id}] 已有检查点，跳过 VLM 调用")
        return load_checkpoint(1, img_id)

    logger.info(f"  [{img_id}] 开始评估...")
    try:
        image_b64 = file_to_b64(img_path)
    except Exception as e:
        logger.error(f"  [{img_id}] 图片读取失败: {e}")
        return None

    data = call_vlm_json(
        IMAGE_FILTER_PROMPT,
        "请根据这张图片完成评估。",
        image_b64=image_b64,
        max_tokens=1024,
        temperature=0.3,
    )

    if data is None:
        logger.warning(f"  [{img_id}] VLM 返回解析失败")
        return None

    # 校正 pass 字段
    scores = data.get("scores", {})
    total = sum(scores.get(d, 0) for d in FILTER_DIMENSIONS)
    all_above_min = all(scores.get(d, 0) >= FILTER_MIN_PER_DIM for d in FILTER_DIMENSIONS)
    data["total_score"] = total
    data["pass"] = total >= FILTER_MIN_TOTAL and all_above_min

    result = {"img_id": img_id, "img_path": img_path, **data}
    save_checkpoint(1, img_id, result)
    logger.info(f"  [{img_id}] 总分={total} pass={data['pass']} cat={data.get('category', '?')}")
    return result


# ============================================================
# 主流程
# ============================================================
def main(source_dir: str = SOURCE_IMAGE_DIR, workers: int = MAX_WORKERS, limit: int = 0):
    logger.info("=" * 60)
    logger.info(f"第一步：筛选高信息密度图片  来源={source_dir}  limit={limit or '无限制'}")
    logger.info("=" * 60)

    # 收集所有图片
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(source_dir, p)))
    img_paths.sort()
    if limit > 0:
        img_paths = img_paths[:limit]
    logger.info(f"找到 {len(img_paths)} 张候选图片")

    # 并发评估
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_image, p): p for p in img_paths}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    # 分拣
    passed = [r for r in results if r.get("pass")]
    failed = [r for r in results if not r.get("pass")]
    logger.info(f"评估完成：通过 {len(passed)} / {len(results)}  未通过 {len(failed)}")

    # 复制通过的图片到 output/images/
    os.makedirs(FILTERED_IMAGE_DIR, exist_ok=True)
    for r in passed:
        copy_image(r["img_path"], FILTERED_IMAGE_DIR)

    # 按类别统计
    cat_counts = {}
    for r in passed:
        cat = r.get("category", "其他")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # 写入统计
    os.makedirs(STATS_DIR, exist_ok=True)
    stats = {
        "total_candidates": len(img_paths),
        "total_evaluated": len(results),
        "total_passed": len(passed),
        "total_failed": len(failed),
        "category_distribution": cat_counts,
        "all_scores": sorted(results, key=lambda x: x.get("total_score", 0), reverse=True),
    }
    stats_path = os.path.join(STATS_DIR, "filter_scores.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"统计写入: {stats_path}")

    # 输出分类分布
    logger.info("类别分布:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {cnt}")

    logger.info("=" * 60)
    logger.info(f"第一步完成！{len(passed)} 张图片已复制到 {FILTERED_IMAGE_DIR}")
    logger.info("=" * 60)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第一步：筛选高信息密度图片")
    parser.add_argument("--source", default=SOURCE_IMAGE_DIR, help="候选图片目录")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N张图片（0=全部）")
    args = parser.parse_args()
    main(source_dir=args.source, workers=args.workers, limit=args.limit)
