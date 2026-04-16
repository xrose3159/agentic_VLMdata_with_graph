"""高并发批量生成题目。

用法：
    python batch_generate.py images/                    # 处理 images/ 下所有图片
    python batch_generate.py /path/to/photos/ -w 20     # 20 并发
    python batch_generate.py images/ --no-cache          # 禁用 API 缓存
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import ENTITY_DIR, QUESTION_DIR
from core.logging_setup import get_logger

logger = get_logger("batch", "batch_generate.log")


def process_one(img_path: str) -> dict:
    """单张图：Step1 构建图谱 → Step2 出题。"""
    from step1_graph import enrich_image
    from step2_generate import generate_questions

    img_id = os.path.splitext(os.path.basename(img_path))[0]
    r = {"img_id": img_id, "img_path": img_path}
    t0 = time.time()

    # Step1: 构建图谱
    try:
        s1 = enrich_image(img_path)
    except Exception as e:
        r["error"] = f"step1: {e}"
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    if s1 is None:
        r["error"] = "step1: skipped (too few entities)"
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    r["step1_sec"] = round(time.time() - t0, 1)
    r["n_entities"] = len(s1.get("entities", []))
    r["n_triples"] = len(s1.get("triples", []))

    # Step2: 出题
    entity_file = os.path.join(ENTITY_DIR, f"{img_id}.json")
    t1 = time.time()
    try:
        s2 = generate_questions(entity_file)
    except Exception as e:
        r["error"] = f"step2: {e}"
        r["step2_sec"] = round(time.time() - t1, 1)
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    r["step2_sec"] = round(time.time() - t1, 1)
    r["total_sec"] = round(time.time() - t0, 1)

    if s2 is None:
        r["error"] = "step2: no questions generated"
        return r

    # 统计
    questions = []
    for cat in ("retrieval", "code", "hybrid"):
        for q in s2.get(cat, []):
            questions.append(q)
    r["n_questions"] = len(questions)
    r["by_category"] = {
        cat: len(s2.get(cat, [])) for cat in ("retrieval", "code", "hybrid")
    }
    r["by_difficulty"] = {
        "easy": sum(1 for q in questions if q.get("difficulty") == "easy"),
        "hard": sum(1 for q in questions if q.get("difficulty") == "hard"),
    }
    return r


def main():
    parser = argparse.ArgumentParser(description="高并发批量生成题目")
    parser.add_argument("image_dir", help="图片目录路径")
    parser.add_argument("-w", "--workers", type=int, default=10, help="并发数（默认 10）")
    parser.add_argument("--no-cache", action="store_true", help="禁用 API 缓存")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["DISABLE_API_CACHE"] = "1"

    # 收集图片
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        img_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))
    img_paths = sorted(set(img_paths))
    if args.limit > 0:
        img_paths = img_paths[:args.limit]

    if not img_paths:
        print(f"在 {args.image_dir} 下没有找到图片")
        return

    print(f"{'='*60}")
    print(f"  批量生成: {len(img_paths)} 张图片, {args.workers} 并发")
    print(f"{'='*60}")

    t_start = time.time()
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, p): p for p in img_paths}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            img_id = r["img_id"]
            if "error" in r:
                print(f"  [{done}/{len(img_paths)}] {img_id} ✗ {r['error']}")
            else:
                print(f"  [{done}/{len(img_paths)}] {img_id} ✓ "
                      f"Step1={r['step1_sec']}s Step2={r['step2_sec']}s "
                      f"实体={r['n_entities']} 三元组={r['n_triples']} "
                      f"题目={r['n_questions']} ({r['by_difficulty']})")

    wall = round(time.time() - t_start, 1)
    ok = [r for r in results if "error" not in r]
    total_q = sum(r.get("n_questions", 0) for r in ok)

    print(f"\n{'='*60}")
    print(f"  完成！")
    print(f"  总耗时: {wall}s ({wall/60:.1f}min)")
    print(f"  成功: {len(ok)}/{len(img_paths)} 张图片")
    print(f"  总题目: {total_q} 道")
    if ok:
        print(f"  平均: Step1={sum(r['step1_sec'] for r in ok)/len(ok):.0f}s "
              f"Step2={sum(r['step2_sec'] for r in ok)/len(ok):.0f}s")
    print(f"{'='*60}")

    # 聚合
    from step2_generate import aggregate_final
    stats = aggregate_final()
    print(f"聚合完成: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
