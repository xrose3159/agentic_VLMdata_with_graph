"""
Agentic MLLM 训练数据生成 Pipeline 编排器。

四步流程：
  1. 筛选高信息密度图片（~50张）
  2. 实体提取 + 知识图谱扩展
  3. 分层生成问题（L1:8 + L2:6 + L3:3 = 17题/张）
  4. 模糊化验证与修正

用法：
    python pipeline.py                        # 运行全部4步
    python pipeline.py --start-from 2         # 从第2步开始
    python pipeline.py --only 1               # 只运行第1步
    python pipeline.py --workers 8            # 设置并发数
    python pipeline.py --limit 20             # 第一步只处理前20张图片
"""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logging_setup import get_logger

logger = get_logger("pipeline", "pipeline.log")


def _run_step2_step3_pipeline(workers: int):
    """step2 和 step3 流水线执行：step2 完成一张图就立即提交 step3。"""
    from step2_enrich import enrich_image
    from step3_generate import generate_questions
    from core.config import FILTERED_IMAGE_DIR, ENTITY_DIR, TAVILY_API_KEY

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(FILTERED_IMAGE_DIR, p)))
    img_paths.sort()
    logger.info(f"找到 {len(img_paths)} 张筛选后图片")

    if not img_paths:
        logger.error("没有找到筛选后的图片，请先运行第一步")
        return

    # step2 并发数受 Tavily API 限制，step3 无限制
    step2_workers = min(workers, 3) if TAVILY_API_KEY else workers
    step3_workers = workers

    step3_pool = ThreadPoolExecutor(max_workers=step3_workers)
    step3_futures = []

    results_step2 = 0
    results_step3 = 0

    with ThreadPoolExecutor(max_workers=step2_workers) as step2_pool:
        step2_futures = {step2_pool.submit(enrich_image, p): p for p in img_paths}

        for fut in as_completed(step2_futures):
            img_path = step2_futures[fut]
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            try:
                result = fut.result()
            except Exception as e:
                logger.error(f"  [{img_id}] step2 异常: {e}")
                continue

            if result is None:
                continue
            results_step2 += 1

            # step2 完成后立即提交 step3
            entity_file = os.path.join(ENTITY_DIR, f"{img_id}.json")
            if os.path.exists(entity_file):
                step3_futures.append(step3_pool.submit(generate_questions, entity_file))

    # 等待所有 step3 任务完成
    for fut in as_completed(step3_futures):
        try:
            r = fut.result()
            if r is not None:
                results_step3 += 1
        except Exception as e:
            logger.error(f"  step3 异常: {e}")

    step3_pool.shutdown(wait=True)

    logger.info(f"step2 完成: {results_step2}/{len(img_paths)} 张图片")
    logger.info(f"step3 完成: {results_step3}/{results_step2} 张图片")

    # 聚合最终结果
    from step3_generate import aggregate_final
    stats = aggregate_final()
    logger.info(f"总计 {stats['total_questions']} 道题")


def run_pipeline(start_from: int = 1, only: int | None = None, workers: int = 4, limit: int = 0):
    steps = {
        1: ("筛选高信息密度图片", "step1_filter"),
        4: ("模糊化验证", "step4_verify"),
    }

    to_run = [only] if only else list(range(start_from, 5))

    logger.info("=" * 60)
    logger.info("Agentic MLLM 训练数据生成 Pipeline")
    logger.info(f"将执行步骤: {to_run}  并发数: {workers}  limit: {limit or '无限制'}")
    logger.info("=" * 60)

    t0 = time.time()

    for step_num in to_run:
        if step_num == 1:
            logger.info(f"\n{'='*60}")
            logger.info(f">>> 第 1 步：筛选高信息密度图片")
            logger.info(f"{'='*60}")
            ts = time.time()
            try:
                module = __import__("step1_filter")
                module.main(workers=workers, limit=limit)
            except Exception as e:
                logger.error(f"第 1 步执行失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
            logger.info(f"第 1 步耗时: {time.time() - ts:.1f}s")

        elif step_num in (2, 3):
            # step2 和 step3 流水线执行，只处理一次
            if step_num == 2 or (step_num == 3 and 2 not in to_run):
                logger.info(f"\n{'='*60}")
                logger.info(f">>> 第 2+3 步：实体提取 + 问题生成（流水线）")
                logger.info(f"{'='*60}")
                ts = time.time()
                try:
                    if step_num == 3 and 2 not in to_run:
                        # 只跑 step3
                        module = __import__("step3_generate")
                        module.main(workers=workers)
                    else:
                        _run_step2_step3_pipeline(workers=workers)
                except Exception as e:
                    logger.error(f"第 2+3 步执行失败: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return
                logger.info(f"第 2+3 步耗时: {time.time() - ts:.1f}s")
            # step_num == 3 but 2 is in to_run → already handled above, skip

        elif step_num == 4:
            logger.info(f"\n{'='*60}")
            logger.info(f">>> 第 4 步：模糊化验证")
            logger.info(f"{'='*60}")
            ts = time.time()
            try:
                module = __import__("step4_verify")
                module.main(workers=workers)
            except Exception as e:
                logger.error(f"第 4 步执行失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
            logger.info(f"第 4 步耗时: {time.time() - ts:.1f}s")

    total = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline 完成！总耗时: {total:.1f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic MLLM 训练数据生成 Pipeline")
    parser.add_argument("--start-from", type=int, default=1, help="从第几步开始（1-4）")
    parser.add_argument("--only", type=int, default=None, help="只运行某一步")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--limit", type=int, default=0, help="第一步只处理前N张图片（0=全部）")
    args = parser.parse_args()
    run_pipeline(start_from=args.start_from, only=args.only, workers=args.workers, limit=args.limit)
