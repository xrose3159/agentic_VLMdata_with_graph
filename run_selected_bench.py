"""从 selected_images 每个子文件夹选 3 张图，并发生成题目，汇总到一个 JSON。"""
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 禁用缓存
os.environ["DISABLE_API_CACHE"] = "1"

from step1_graph import enrich_image
from step2_generate import generate_questions
from core.config import ENTITY_DIR

SELECTED_DIR = "selected_images"
PER_FOLDER = 3
WORKERS = 20


def collect_images():
    images = []
    for folder in sorted(os.listdir(SELECTED_DIR)):
        folder_path = os.path.join(SELECTED_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted(glob.glob(os.path.join(folder_path, "*.png")) +
                       glob.glob(os.path.join(folder_path, "*.jpg")) +
                       glob.glob(os.path.join(folder_path, "*.jpeg")) +
                       glob.glob(os.path.join(folder_path, "*.webp")))
        for f in files[:PER_FOLDER]:
            images.append((folder, f))
    return images


def process_one(folder, img_path):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    # 清 checkpoint
    for d in ("step1", "step2"):
        p = f"output/.checkpoints/{d}/{img_id}.json"
        if os.path.exists(p): os.remove(p)
    for p in (f"output/entities/{img_id}.json", f"output/questions/{img_id}.json"):
        if os.path.exists(p): os.remove(p)

    r = {"img_id": img_id, "folder": folder, "img_path": img_path}
    t0 = time.time()

    # Step1
    try:
        s1 = enrich_image(img_path)
    except Exception as e:
        r["error"] = f"step1: {e}"
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    if s1 is None:
        r["error"] = "step1: skipped"
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    r["step1_sec"] = round(time.time() - t0, 1)
    r["n_entities"] = len(s1.get("entities", []))
    r["n_triples"] = len(s1.get("triples", []))

    # Step2
    ef = os.path.join(ENTITY_DIR, f"{img_id}.json")
    # 清 step2 checkpoint
    ck = f"output/.checkpoints/step2/{img_id}.json"
    if os.path.exists(ck): os.remove(ck)

    t1 = time.time()
    try:
        s2 = generate_questions(ef)
    except Exception as e:
        r["error"] = f"step2: {e}"
        r["step2_sec"] = round(time.time() - t1, 1)
        r["total_sec"] = round(time.time() - t0, 1)
        return r
    r["step2_sec"] = round(time.time() - t1, 1)
    r["total_sec"] = round(time.time() - t0, 1)

    if not s2:
        r["error"] = "step2: no output"
        return r

    questions = []
    for cat in ("retrieval", "code", "hybrid"):
        for q in s2.get(cat, []):
            questions.append(q)
    r["n_questions"] = len(questions)
    r["questions"] = questions
    return r


def main():
    images = collect_images()
    print(f"{'='*60}")
    print(f"  {len(images)} 张图 ({PER_FOLDER}/文件夹 × {len(set(f for f,_ in images))} 文件夹), {WORKERS} 并发")
    print(f"{'='*60}")

    t_start = time.time()
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_one, folder, path): (folder, path) for folder, path in images}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            results.append(r)
            if "error" in r:
                print(f"  [{done}/{len(images)}] {r['folder']}/{r['img_id']} ✗ {r['error']}")
            else:
                print(f"  [{done}/{len(images)}] {r['folder']}/{r['img_id']} ✓ "
                      f"实体={r['n_entities']} 三元组={r['n_triples']} 题目={r['n_questions']} "
                      f"({r['total_sec']}s)")

    wall = round(time.time() - t_start, 1)
    results.sort(key=lambda x: (x["folder"], x["img_id"]))

    ok = [r for r in results if "error" not in r]
    total_q = sum(r.get("n_questions", 0) for r in ok)

    # 汇总 JSON
    summary = {
        "wall_time_sec": wall,
        "total_images": len(images),
        "success": len(ok),
        "failed": len(images) - len(ok),
        "total_questions": total_q,
        "results": []
    }
    for r in results:
        entry = {
            "img_id": r["img_id"],
            "folder": r["folder"],
            "img_path": r["img_path"],
        }
        if "error" in r:
            entry["error"] = r["error"]
        else:
            entry["n_entities"] = r["n_entities"]
            entry["n_triples"] = r["n_triples"]
            entry["step1_sec"] = r["step1_sec"]
            entry["step2_sec"] = r["step2_sec"]
            entry["total_sec"] = r["total_sec"]
            entry["questions"] = r.get("questions", [])
        summary["results"].append(entry)

    out_path = "selected_images_questions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  完成！总耗时 {wall}s ({wall/60:.1f}min)")
    print(f"  成功 {len(ok)}/{len(images)}, 总题目 {total_q}")
    print(f"  输出: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
