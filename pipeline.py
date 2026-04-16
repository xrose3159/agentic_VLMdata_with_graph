"""
Agentic MLLM 训练数据生成 Pipeline 编排器。

每张图独立走完 Step2→Step3，互不影响。
worker 跑完一张图立即取下一张，不等其他 worker。

用法：
    python pipeline.py                        # 运行全部图片
    python pipeline.py --workers 10           # 10 并发
    python pipeline.py --source selected_images/retail/  # 指定图片目录
    python pipeline.py --limit 30             # 只跑前 30 张
"""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.logging_setup import get_logger

logger = get_logger("pipeline", "pipeline.log")


def _process_one_image(img_path: str) -> dict:
    """单张图的完整 pipeline：Step2(enrich) → Step3(generate)。

    每张图独立执行，互不依赖。返回统计信息。
    """
    from step1_graph import enrich_image
    from step2_generate import generate_questions
    from core.config import ENTITY_DIR

    img_id = os.path.splitext(os.path.basename(img_path))[0]
    t0 = time.time()

    # Step2: 实体提取 + 知识图谱
    try:
        result = enrich_image(img_path)
    except Exception as e:
        logger.error(f"[{img_id}] Step2 异常: {e}")
        return {"img_id": img_id, "status": "step2_error", "error": str(e)}

    if result is None:
        return {"img_id": img_id, "status": "step2_skip"}

    t_step2 = time.time() - t0

    # Step3: 问题生成
    entity_file = os.path.join(ENTITY_DIR, f"{img_id}.json")
    if not os.path.exists(entity_file):
        return {"img_id": img_id, "status": "no_entity_file", "step2_time": t_step2}

    t1 = time.time()
    try:
        q_result = generate_questions(entity_file)
    except Exception as e:
        logger.error(f"[{img_id}] Step2 异常: {e}")
        return {"img_id": img_id, "status": "step3_error", "step2_time": t_step2, "error": str(e)}

    t_step3 = time.time() - t1
    total = time.time() - t0

    n_q = sum(len(q_result.get(c, [])) for c in ("retrieval", "code", "hybrid")) if q_result else 0
    logger.info(f"[{img_id}] 完成 Step2={t_step2:.0f}s Step3={t_step3:.0f}s 总={total:.0f}s 题目={n_q}")

    return {
        "img_id": img_id,
        "status": "ok",
        "step2_time": t_step2,
        "step3_time": t_step3,
        "total_time": total,
        "n_entities": len(result.get("entities", [])),
        "n_triples": len(result.get("triples", [])),
        "n_questions": n_q,
    }


def run_pipeline(workers: int = 10, source_dir: str = "", limit: int = 0):
    from core.config import IMAGE_DIR

    search_dirs = [source_dir] if source_dir else [IMAGE_DIR, "images"]
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    img_paths = []
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(d, p)))
            img_paths.extend(glob.glob(os.path.join(d, "**", p), recursive=True))
        img_paths = list(dict.fromkeys(img_paths))  # 去重保序
        if img_paths:
            logger.info(f"图片来源: {d} ({len(img_paths)} 张)")
            break
    img_paths.sort()

    if limit > 0:
        img_paths = img_paths[:limit]

    if not img_paths:
        logger.error("没有找到图片。请把图片放到 images/ 目录，或用 --source 指定目录")
        return

    logger.info("=" * 60)
    logger.info(f"Pipeline 启动: {len(img_paths)} 张图片, {workers} 并发")
    logger.info("每张图独立走 Step2→Step3，互不等待")
    logger.info("=" * 60)

    t0 = time.time()
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one_image, p): p for p in img_paths}
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                results.append(r)
                status = r.get("status", "?")
                img_id = r.get("img_id", "?")
                if status == "ok":
                    logger.info(f"[{done}/{len(img_paths)}] {img_id} ✓ "
                                f"L3={r['n_questions']} ({r['total_time']:.0f}s)")
                else:
                    logger.warning(f"[{done}/{len(img_paths)}] {img_id} → {status}")
            except Exception as e:
                logger.error(f"[{done}/{len(img_paths)}] 异常: {e}")

    total_time = time.time() - t0
    ok = [r for r in results if r.get("status") == "ok"]

    logger.info("=" * 60)
    logger.info(f"Pipeline 完成！")
    logger.info(f"  总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info(f"  成功: {len(ok)}/{len(img_paths)} 张图片")
    logger.info(f"  L3 题目: {sum(r['n_questions'] for r in ok)} 道")
    logger.info(f"  平均每张: Step2={sum(r['step2_time'] for r in ok)/max(len(ok),1):.0f}s "
                f"Step3={sum(r['step3_time'] for r in ok)/max(len(ok),1):.0f}s")
    logger.info("=" * 60)

    # 聚合最终输出
    from step2_generate import aggregate_final
    stats = aggregate_final()
    logger.info(f"聚合完成: {stats['total_questions']} 道题")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic MLLM 训练数据生成 Pipeline")
    parser.add_argument("--workers", type=int, default=10, help="并发数（默认10）")
    parser.add_argument("--source", type=str, default="", help="图片来源目录")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张图片")
    args = parser.parse_args()
    run_pipeline(workers=args.workers, source_dir=args.source, limit=args.limit)
