"""批量跑 Step3 生题：并发处理多张图的 entities JSON → questions JSON."""
import os
import sys
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 支持指定生题用的 VLM 模型
import core.vlm
core.vlm.MODEL_NAME = os.environ.get("STEP3_MODEL", "gemini-3-flash-preview-nothinking")

from step3_trajectory import generate_questions  # noqa: E402

OUTPUT_DIR = "output/questions"


def _find_image(img_id: str) -> str | None:
    """根据 img_id 找源图（images/<img_id>.{jpg,png,webp}）。"""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = os.path.join("images", f"{img_id}{ext}")
        if os.path.exists(p):
            return p
        p = os.path.join("try", f"{img_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def _run_one(entity_json: str, n_walks: int, seed: int, suffix: str) -> tuple[str, dict, float, str]:
    img_id = os.path.splitext(os.path.basename(entity_json))[0]
    img_path = _find_image(img_id)
    start = time.time()
    try:
        result = generate_questions(
            entity_json,
            image_path=img_path,
            seed=seed,
            n_walks=n_walks,
        )
        # 写出
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_name = f"{img_id}{suffix}.json" if suffix else f"{img_id}.json"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return img_id, result, time.time() - start, out_path
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return img_id, {"error": str(exc)}, time.time() - start, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_files", nargs="+",
                        help="Entity JSON 文件路径（或 glob，例如 'output/entities/sku_*.json'）")
    parser.add_argument("--workers", type=int, default=4, help="并发图数")
    parser.add_argument("--n-walks", type=int, default=24, help="每图走步数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suffix", type=str, default="",
                        help="输出文件名后缀（例如 _multihop 会生成 sku_8845_multihop.json）")
    args = parser.parse_args()

    # 展开 glob
    import glob
    files: list[str] = []
    for pat in args.entity_files:
        if any(ch in pat for ch in "*?["):
            files.extend(sorted(glob.glob(pat)))
        elif os.path.isfile(pat):
            files.append(pat)
        else:
            print(f"[WARN] {pat} 不存在，跳过")
    if not files:
        print("没有找到任何 entity JSON 文件")
        sys.exit(1)

    print(f"准备处理 {len(files)} 个 entity JSON（并发 {args.workers}，n_walks={args.n_walks}）")
    print("=" * 70)
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, f, args.n_walks, args.seed, args.suffix): f
            for f in files
        }
        all_results = []
        for fut in as_completed(futures):
            img_id, result, elapsed, out_path = fut.result()
            if "error" in result:
                print(f"[FAIL] {img_id}  {elapsed:.0f}s  {result['error']}")
                continue
            meta = result.get("metadata", {})
            selected = meta.get("selected", {})
            bucket = meta.get("hard_bucket_dist", {})
            n_l1 = len(result.get("level_1", []))
            n_l2 = len(result.get("level_2", []))
            n_l3 = len(result.get("level_3", []))
            max_tool_len = 0
            for lk in ("level_1", "level_2", "level_3"):
                for q in result.get(lk, []):
                    ts = q.get("tool_sequence", [])
                    if len(ts) > max_tool_len:
                        max_tool_len = len(ts)
            print(f"[DONE] {img_id}  {elapsed:.0f}s  L1={n_l1} L2={n_l2} L3={n_l3}  "
                  f"bucket={bucket}  max_tools={max_tool_len}  → {out_path}")
            all_results.append((img_id, result, elapsed))

    print("=" * 70)
    total_time = time.time() - t0
    total_l3 = sum(len(r[1].get("level_3", [])) for r in all_results)
    total_multi_hop = sum(
        1 for r in all_results for q in r[1].get("level_3", [])
        if q.get("family") == "multi_hop"
    )
    print(f"全部完成：{len(all_results)}/{len(files)} 张，总耗时 {total_time:.0f}s")
    print(f"L3 题目总数：{total_l3}，其中 multi_hop：{total_multi_hop}")


if __name__ == "__main__":
    main()
