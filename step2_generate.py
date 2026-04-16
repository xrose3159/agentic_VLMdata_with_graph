"""
第二步：基于 step3_trajectory 的 heterogeneous solve graph + random walk 生题。

这个文件是主 pipeline 的 Step2 入口，底层生题逻辑在 `step3_trajectory.py`。

本模块提供：
  1. API：`generate_questions(entity_file)`、`aggregate_final()`、`main()`
     供 pipeline.py / step3_verify.py 继续使用
  2. 三元组清洗 / 图构建工具（_sanitize_triples、_build_nx_graph、_match_entity）
     供旧脚本复用
  3. 调用 step3_trajectory.generate_questions 做实际的 walk + 生题

用法：
    python step3_generate.py                 # 处理全部 entity 文件
    python step3_generate.py --workers 4
"""

import argparse
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import (
    ENTITY_DIR, QUESTION_DIR, FINAL_DIR, STATS_DIR,
    IMAGE_DIR, MAX_WORKERS, MODEL_NAME,
)
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step2", "step2_generate.log")


# ============================================================
# Utility: tail_type 归一化 / 三元组清洗
# 被 trajectory_runtime、experimental/stochastic_step3 等复用
# ============================================================

_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}


def _normalize_tail_type(value) -> str:
    if not isinstance(value, str):
        return "OTHER"
    normalized = value.strip().upper()
    return normalized if normalized in _TAIL_TYPE_ENUM else "OTHER"


def _normalize_unit(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_scalar_value(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            if re.fullmatch(r"-?\d+", text):
                return int(text)
            if re.fullmatch(r"-?\d+\.\d+", text):
                return float(text)
        except ValueError:
            pass
        return text
    return str(value)


def _sanitize_triples(triples: list) -> list:
    """兼容旧数据：缺失 tail_type 时默认 OTHER。"""
    normalized = []
    for t in triples or []:
        if not isinstance(t, dict):
            continue
        item = dict(t)
        item["tail_type"] = _normalize_tail_type(item.get("tail_type"))
        item["unit"] = _normalize_unit(item.get("unit"))
        item["normalized_value"] = _normalize_scalar_value(item.get("normalized_value"))
        item.pop("relation_family", None)
        normalized.append(item)
    return normalized


# ============================================================

# ============================================================
# 主入口：generate_questions(entity_file) — 包装 trajectory_runtime
# ============================================================

N_WALKS = 24
SEED = 42
IMAGE_SEARCH_DIRS = ("images", "try")  # 备选图片目录


def _find_image(img_id: str) -> str | None:
    """按 img_id 找源图，依次查 IMAGE_DIR / images / try。"""
    for base in (IMAGE_DIR, *IMAGE_SEARCH_DIRS):
        if not base:
            continue
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            cand = os.path.join(base, f"{img_id}{ext}")
            if os.path.exists(cand):
                return cand
    return None


def generate_questions(entity_file: str) -> dict | None:
    """根据 Step2 entity_file 生成分层问题 JSON。

    - 内部调用 `step3_trajectory.generate_questions`
    - 处理 checkpoint 和文件写出
    - 返回与旧 API 兼容的 result dict
    """
    try:
        with open(entity_file, encoding="utf-8") as f:
            entity_data = json.load(f)
    except Exception as e:
        logger.error(f"读取 entity_file 失败 {entity_file}: {e}")
        return None

    img_id = entity_data.get("img_id") or os.path.splitext(os.path.basename(entity_file))[0]

    if is_done(2, img_id):
        logger.info(f"  [{img_id}] 已有检查点，跳过")
        return load_checkpoint(2, img_id)

    img_path = _find_image(img_id)
    if img_path is None:
        logger.warning(f"  [{img_id}] 找不到图片（查过 IMAGE_DIR / images / try）")
        return None

    logger.info(f"  [{img_id}] 开始生成问题（step3_trajectory, n_walks={N_WALKS}）...")
    t0 = time.time()
    try:
        # 延迟导入避免循环依赖（step3_trajectory 会从本模块 import _sanitize_triples）
        from step2_question import generate_questions as _tr_generate
        result = _tr_generate(
            entity_file,
            image_path=img_path,
            seed=SEED,
            n_walks=N_WALKS,
        )
    except Exception as e:
        logger.error(f"  [{img_id}] step3_trajectory 异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

    elapsed = time.time() - t0

    # 补充 image_id / image_path / 耗时字段
    result["image_id"] = img_id
    result["image_path"] = img_path
    result["generation_time_sec"] = round(elapsed, 2)

    # 写出 questions 文件
    os.makedirs(QUESTION_DIR, exist_ok=True)
    out_path = os.path.join(QUESTION_DIR, f"{img_id}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"  [{img_id}] 写入问题文件失败: {e}")

    n_ret = len(result.get("retrieval", []))
    n_code = len(result.get("code", []))
    n_hyb = len(result.get("hybrid", []))
    logger.info(
        f"  [{img_id}] 完成 {elapsed:.0f}s  retrieval={n_ret} code={n_code} hybrid={n_hyb}"
    )

    save_checkpoint(2, img_id, result)
    return result


# ============================================================
# 聚合: 把所有 questions/*.json 合并成最终 JSONL
# ============================================================

def aggregate_final():
    """将 output/questions/*.json 聚合为 output/final/*.jsonl 和 stats。"""
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    q_files = sorted(
        p for p in glob.glob(os.path.join(QUESTION_DIR, "*.json"))
        if not p.endswith("_randomwalk_shadow.json")
    )
    logger.info(f"聚合 {len(q_files)} 个问题文件...")

    all_by_cat = {"retrieval": [], "code": [], "hybrid": []}

    for qf in q_files:
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"读取 {qf} 失败: {e}")
            continue

        img_id = data.get("image_id", os.path.splitext(os.path.basename(qf))[0])
        img_path = data.get("image_path", f"output/images/{img_id}.jpg")

        for cat in ("retrieval", "code", "hybrid"):
            for q in data.get(cat, []):
                record = {
                    "id": f"{img_id}_{q.get('question_id', '')}",
                    "image_id": img_id,
                    "image_path": img_path,
                    "category": cat,
                    "question_id": q.get("question_id", ""),
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "tools": q.get("tools", []),
                    "reasoning": q.get("reasoning", ""),
                }
                all_by_cat[cat].append(record)

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for cat in ("retrieval", "code", "hybrid"):
        write_jsonl(os.path.join(FINAL_DIR, f"{cat}.jsonl"), all_by_cat[cat])

    all_questions = all_by_cat["retrieval"] + all_by_cat["code"] + all_by_cat["hybrid"]
    write_jsonl(os.path.join(FINAL_DIR, "all_questions.jsonl"), all_questions)

    stats = {
        "total_questions": len(all_questions),
        "retrieval": len(all_by_cat["retrieval"]),
        "code": len(all_by_cat["code"]),
        "hybrid": len(all_by_cat["hybrid"]),
        "images_with_questions": len(q_files),
    }
    with open(os.path.join(STATS_DIR, "question_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"��合完成: retrieval={stats['retrieval']} code={stats['code']} hybrid={stats['hybrid']} "
        f"总计={stats['total_questions']}"
    )
    return stats


# ============================================================
# 主流程
# ============================================================

def main(workers: int = MAX_WORKERS):
    logger.info("=" * 60)
    logger.info("Step3：基于图谱生成问题")
    logger.info("=" * 60)

    entity_files = sorted(glob.glob(os.path.join(ENTITY_DIR, "*.json")))
    logger.info(f"找到 {len(entity_files)} 个实体文件")

    if not entity_files:
        logger.error("没有找到实体文件，请先运行第一步")
        return

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_questions, ef): ef for ef in entity_files}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                logger.error(f"生成问题异常: {e}")

    logger.info(f"问题生成完成：{len(results)}/{len(entity_files)} 张图片")

    stats = aggregate_final()

    logger.info("=" * 60)
    logger.info(f"第二步完成！总计 {stats['total_questions']} 道题")
    logger.info(f"retrieval={stats['retrieval']} code={stats['code']} hybrid={stats['hybrid']}")
    logger.info(f"L3 bucket 分布: {stats['l3_bucket_dist']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第二步：trajectory-based 分层问题生成")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    args = parser.parse_args()
    main(workers=args.workers)
