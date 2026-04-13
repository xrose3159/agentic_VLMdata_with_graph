"""
第三步：基于 step3_trajectory 的 heterogeneous solve graph + random walk 生题。

这个文件是主 pipeline 的 Step3 入口，底层生题逻辑在 `step3_trajectory.py`。

本模块提供：
  1. 旧 API：`generate_questions(entity_file)`、`aggregate_final()`、`main()`
     供 pipeline.py / step4_verify.py 继续使用
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

import networkx as nx

from core.config import (
    ENTITY_DIR, QUESTION_DIR, FINAL_DIR, STATS_DIR,
    FILTERED_IMAGE_DIR, MAX_WORKERS, MODEL_NAME,
)
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step3", "step3_generate.log")


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
        item["provenance"] = str(item.get("provenance") or item.get("source") or "").strip().lower()
        item.pop("relation_family", None)
        normalized.append(item)
    return normalized


# ============================================================
# Utility: 实体匹配 + NetworkX 图构建
# 旧 pipeline 保留的工具，trajectory_runtime 不直接用但
# experimental/ 下的其他 runtime 和 stochastic_step3 还在依赖。
# ============================================================

_FUZZY_TYPES = {"brand", "person", "location", "product"}
_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "for", "and", "or", "is",
    "logo", "text", "sign", "label", "brand", "body", "piece", "fabric",
    "标志", "品牌", "文字", "演奏者",
}


def _build_entity_index(entities: list):
    exact_map = {}
    keyword_map = {}
    for e in entities:
        key = e["name"].strip().lower()
        exact_map[key] = e
        if e.get("type", "other") in _FUZZY_TYPES:
            for w in key.split():
                if w not in _STOP_WORDS and len(w) >= 3:
                    keyword_map[w] = e
    return exact_map, keyword_map


def _match_entity(key: str, exact_map: dict, keyword_map: dict, entities: list):
    """返回 (is_in_image, location, entity_type)。"""
    if key in exact_map:
        e = exact_map[key]
        return True, e.get("location_in_image", ""), e.get("type", "")
    for kw, e in keyword_map.items():
        if e.get("type") == "brand":
            if key.startswith(kw):
                return True, e.get("location_in_image", ""), e.get("type", "")
        else:
            if kw in set(key.split()) or kw in key:
                return True, e.get("location_in_image", ""), e.get("type", "")
    for e in entities:
        e_name = e["name"].strip().lower()
        if len(e_name) < 3:
            continue
        if e_name in key or key in e_name:
            return True, e.get("location_in_image", ""), e.get("type", "")
    return False, "", ""


def _build_nx_graph(triples: list, entities: list):
    """从三元组建 NetworkX DiGraph；节点带 in_image / location / name 属性。"""
    exact_map, keyword_map = _build_entity_index(entities)
    nodes: dict = {}
    G = nx.DiGraph()

    def _ensure_node(name: str, tail_type: str = "OTHER"):
        key = name.strip().lower()
        normalized_tail_type = _normalize_tail_type(tail_type)
        if key not in nodes:
            is_img, loc, etype = _match_entity(key, exact_map, keyword_map, entities)
            nodes[key] = {
                "name": name,
                "in_image": is_img,
                "location": loc,
                "tail_type": normalized_tail_type,
                "entity_type": etype,
            }
            G.add_node(
                key,
                in_image=is_img,
                location=loc,
                name=name,
                tail_type=normalized_tail_type,
                entity_type=etype,
            )
        elif normalized_tail_type != "OTHER" and nodes[key].get("tail_type", "OTHER") == "OTHER":
            nodes[key]["tail_type"] = normalized_tail_type
            G.nodes[key]["tail_type"] = normalized_tail_type
        return key

    for t in triples:
        head = t.get("head", "").strip()
        tail = t.get("tail", "").strip()
        if not head or not tail:
            continue
        tail_type = _normalize_tail_type(t.get("tail_type"))
        h_key = _ensure_node(head)
        t_key = _ensure_node(tail, tail_type=tail_type)
        if h_key == t_key:
            continue
        if not G.has_edge(h_key, t_key):
            G.add_edge(
                h_key, t_key,
                relation=t.get("relation", ""),
                fact=t.get("fact", ""),
                source_snippet=t.get("source_snippet", ""),
                tail_type=tail_type,
                normalized_value=_normalize_scalar_value(t.get("normalized_value")),
                unit=_normalize_unit(t.get("unit")),
                source=t.get("source", ""),
                provenance=t.get("provenance", ""),
            )
        elif (
            G[h_key][t_key].get("tail_type", "OTHER") == "OTHER"
            and tail_type != "OTHER"
        ):
            G[h_key][t_key]["tail_type"] = tail_type

    logger.info(f"    图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G, nodes


# ============================================================
# 主入口：generate_questions(entity_file) — 包装 trajectory_runtime
# ============================================================

N_WALKS = 24
SEED = 42
IMAGE_SEARCH_DIRS = ("images", "try")  # 备选图片目录


def _find_image(img_id: str) -> str | None:
    """按 img_id 找源图，依次查 FILTERED_IMAGE_DIR / images / try。"""
    for base in (FILTERED_IMAGE_DIR, *IMAGE_SEARCH_DIRS):
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

    if is_done(3, img_id):
        logger.info(f"  [{img_id}] 已有检查点，跳过")
        return load_checkpoint(3, img_id)

    img_path = _find_image(img_id)
    if img_path is None:
        logger.warning(f"  [{img_id}] 找不到图片（查过 FILTERED_IMAGE_DIR / images / try）")
        return None

    logger.info(f"  [{img_id}] 开始生成问题（step3_trajectory, n_walks={N_WALKS}）...")
    t0 = time.time()
    try:
        # 延迟导入避免循环依赖（step3_trajectory 会从本模块 import _sanitize_triples）
        from step3_trajectory import generate_questions as _tr_generate
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

    n_l1 = len(result.get("level_1", []))
    n_l2 = len(result.get("level_2", []))
    n_l3 = len(result.get("level_3", []))
    meta = result.get("metadata", {})
    logger.info(
        f"  [{img_id}] 完成 {elapsed:.0f}s  L1={n_l1} L2={n_l2} L3={n_l3}  "
        f"bucket={meta.get('hard_bucket_dist', {})}"
    )

    save_checkpoint(3, img_id, result)
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

    all_l1, all_l2, all_l3 = [], [], []

    for qf in q_files:
        try:
            with open(qf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"读取 {qf} 失败: {e}")
            continue

        img_id = data.get("image_id", os.path.splitext(os.path.basename(qf))[0])
        img_path = data.get("image_path", f"output/images/{img_id}.jpg")

        for level_key, level_num, container in [
            ("level_1", 1, all_l1),
            ("level_2", 2, all_l2),
            ("level_3", 3, all_l3),
        ]:
            for q in data.get(level_key, []):
                record = {
                    "id": f"{img_id}_{q.get('question_id', '')}",
                    "image_id": img_id,
                    "image_path": img_path,
                    "level": level_num,
                    "question_id": q.get("question_id", ""),
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "tool_sequence": q.get("tool_sequence", []),
                    "family": q.get("family", ""),
                    "hard_bucket": q.get("hard_bucket", ""),
                    "code_skill_tags": q.get("code_skill_tags", []),
                    "reasoning_path": q.get("reasoning_path", {}),
                    "entities_involved": q.get("entities_involved", []),
                    "obfuscation_applied": q.get("obfuscation_applied", False),
                    "obfuscated_entities": q.get("obfuscated_entities", []),
                    "rationale": q.get("rationale", ""),
                    "metadata": {
                        "generation_model": MODEL_NAME,
                        "family": q.get("family", ""),
                    },
                }
                container.append(record)

    def write_jsonl(path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(FINAL_DIR, "level_1_all.jsonl"), all_l1)
    write_jsonl(os.path.join(FINAL_DIR, "level_2_all.jsonl"), all_l2)
    write_jsonl(os.path.join(FINAL_DIR, "level_3_all.jsonl"), all_l3)

    all_questions = all_l1 + all_l2 + all_l3
    write_jsonl(os.path.join(FINAL_DIR, "all_questions.jsonl"), all_questions)

    # L3 bucket 和 family 分布
    from collections import Counter
    bucket_counter = Counter()
    family_counter = Counter()
    for r in all_l3:
        bucket_counter[r.get("hard_bucket", "") or "standard"] += 1
        family_counter[r.get("family", "")] += 1

    stats = {
        "total_questions": len(all_questions),
        "level_1_count": len(all_l1),
        "level_2_count": len(all_l2),
        "level_3_count": len(all_l3),
        "images_with_questions": len(q_files),
        "l3_bucket_dist": dict(bucket_counter),
        "l3_family_dist": dict(family_counter),
    }
    with open(os.path.join(STATS_DIR, "question_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        f"聚合完成: L1={len(all_l1)} L2={len(all_l2)} L3={len(all_l3)} "
        f"总计={len(all_questions)}  L3 bucket={dict(bucket_counter)}"
    )
    return stats


# ============================================================
# 主流程
# ============================================================

def main(workers: int = MAX_WORKERS):
    logger.info("=" * 60)
    logger.info("第三步：Heterogeneous solve graph + random walk 生题")
    logger.info("=" * 60)

    entity_files = sorted(glob.glob(os.path.join(ENTITY_DIR, "*.json")))
    logger.info(f"找到 {len(entity_files)} 个实体文件")

    if not entity_files:
        logger.error("没有找到实体文件，请先运行第二步")
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
    logger.info(f"第三步完成！总计 {stats['total_questions']} 道题")
    logger.info(f"L1={stats['level_1_count']} L2={stats['level_2_count']} L3={stats['level_3_count']}")
    logger.info(f"L3 bucket 分布: {stats['l3_bucket_dist']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第三步：trajectory-based 分层问题生成")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    args = parser.parse_args()
    main(workers=args.workers)
