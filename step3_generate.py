"""
第三步：基于三元组的分层问题生成。

流程：
  3a. 从三元组建 NetworkX 有向图
  3b. GraphEnv + typed action + Beam Search 提议 QuestionProgram
  3c. verifier 做硬约束筛选，按全局分数选择 L2/L3
  3d. constrained realization 生成自然语言问题 + 工具序列

用法：
    python step3_generate.py
    python step3_generate.py --workers 4
"""

import argparse
import glob
import json
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from core.config import ENTITY_DIR, QUESTION_DIR, FINAL_DIR, STATS_DIR, FILTERED_IMAGE_DIR, MAX_WORKERS, MODEL_NAME
from core.vlm import call_vlm_json
from core.image_utils import file_to_b64
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step3", "step3_generate.log")


# ============================================================
# tail_type 规范化
# ============================================================

_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}
_COMPARATIVE_TAIL_TYPES = {"TIME", "QUANTITY"}


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
# 3a. 实体匹配工具
# ============================================================

_FUZZY_TYPES = {"brand", "person", "location", "product"}
_STOP_WORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "and", "or", "is",
               "logo", "text", "sign", "label", "brand", "body", "piece", "fabric",
               "标志", "品牌", "文字", "演奏者"}


def _build_entity_index(entities: list):
    exact_map = {}   # lower_name → entity
    keyword_map = {} # keyword → entity
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


# ============================================================
# 3b. 建 NetworkX 有向图
# ============================================================

def _build_nx_graph(triples: list, entities: list):
    """从三元组建 NetworkX DiGraph，节点带 in_image / location / name 属性。"""
    exact_map, keyword_map = _build_entity_index(entities)
    nodes = {}  # lower_key → {name, in_image, location, tail_type}

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
            G.add_edge(h_key, t_key,
                       relation=t.get("relation", ""),
                       fact=t.get("fact", ""),
                source_snippet=t.get("source_snippet", ""),
                tail_type=tail_type,
                normalized_value=_normalize_scalar_value(t.get("normalized_value")),
                unit=_normalize_unit(t.get("unit")),
                source=t.get("source", ""),
                provenance=t.get("provenance", ""))
        elif (
            G[h_key][t_key].get("tail_type", "OTHER") == "OTHER"
            and tail_type != "OTHER"
        ):
            G[h_key][t_key]["tail_type"] = tail_type
        if G[h_key][t_key].get("normalized_value", "") in ("", None) and t.get("normalized_value") not in ("", None):
            G[h_key][t_key]["normalized_value"] = _normalize_scalar_value(t.get("normalized_value"))
        if not G[h_key][t_key].get("unit") and t.get("unit"):
            G[h_key][t_key]["unit"] = _normalize_unit(t.get("unit"))
        if not G[h_key][t_key].get("source") and t.get("source"):
            G[h_key][t_key]["source"] = t.get("source", "")
        if not G[h_key][t_key].get("provenance") and t.get("provenance"):
            G[h_key][t_key]["provenance"] = t.get("provenance", "")

    logger.info(f"    图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G, nodes


# ============================================================
# 3c. 旧 Motif 主路径（已废弃，仅保留历史实现供参考）
# ============================================================

_BAD_ANSWERS = {"true", "false", "yes", "no", "none", "n/a", "unknown", "various", "multiple"}
_LOW_VALUE_RELATIONS = {
    "related_to", "associated_with", "connected_to",
    "located_left_of", "located_right_of", "located_above", "located_below",
}
_LOOKUP_TAIL_TYPE_BONUS = {
    "PERSON": 2.0,
    "ORG": 1.8,
    "LOCATION": 1.6,
    "TIME": 1.4,
    "QUANTITY": 1.4,
    "OTHER": 0.0,
}
_EXTREMUM_TAIL_TYPES = {"TIME", "QUANTITY"}
_EXTREMUM_COMPARE_TEXT = {
    "TIME": "时间先后（如更早/更晚）",
    "QUANTITY": "数值大小（必要时统一单位）",
}
_EXTREMUM_TARGET_PREFIX = {
    "TIME": "时间极值",
    "QUANTITY": "数值极值",
}
_SAME_TARGET_TYPES = {"PERSON", "ORG", "LOCATION"}
_SAME_GROUP_TYPES = {"LOCATION", "ORG"}
_QUANTITY_SCALE = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}
_PATH_FOLLOW_TARGET_TYPES = {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"}


def _answer_ok(name: str) -> bool:
    s = name.strip()
    if not s or len(s) < 3:
        return False
    if s.lower() in _BAD_ANSWERS:
        return False
    return True


def _relation_slug(rel: str) -> str:
    return rel.strip().lower().replace(" ", "_")


def _edge_relation_key(data: dict) -> str:
    return _relation_slug(data.get("relation", ""))


def _edge_tail_type(nodes: dict, node_key: str, data: dict | None = None) -> str:
    if data is not None and data.get("tail_type"):
        return _normalize_tail_type(data.get("tail_type"))
    return _normalize_tail_type(nodes.get(node_key, {}).get("tail_type", "OTHER"))


def _edge_evidence_score(data: dict) -> float:
    score = 0.0
    if data.get("fact"):
        score += 1.0
    if data.get("source_snippet"):
        score += 0.6
    if data.get("source") == "cross_entity":
        score += 0.3
    return score


def _relation_semantic_bucket(relation_key: str, target_type: str) -> str:
    rel = relation_key or ""
    target_type = _normalize_tail_type(target_type)

    if target_type == "PERSON":
        if any(k in rel for k in ("founder", "founded_by", "ceo", "president", "chairman", "leader", "owner")):
            return "person_role"
        if any(k in rel for k in ("author", "writer", "lyrics", "composer", "music_by", "director", "designed_by", "created_by")):
            return "person_creator"
        if any(k in rel for k in ("born_in", "born_on", "died_in", "died_on", "spouse", "parent")):
            return "person_biographical"
        return "person_other"

    if target_type == "ORG":
        if any(k in rel for k in ("parent", "owned_by", "subsidiary", "division", "group", "brand_of")):
            return "org_ownership"
        if any(k in rel for k in ("sponsored_by", "regulated_by", "partner", "publisher", "distributor", "label")):
            return "org_affiliation"
        if any(k in rel for k in ("listed_on", "member_of", "league", "association")):
            return "org_membership"
        return "org_other"

    if target_type == "LOCATION":
        if any(k in rel for k in ("headquartered", "located_in", "based_in", "born_in", "from", "resides_in", "origin", "country", "city", "state")):
            return "location_membership"
        if any(k in rel for k in ("performed_at", "premiered_at", "staged_at", "held_at", "filmed_in", "shot_in", "opened_at")):
            return "location_occurrence"
        if any(k in rel for k in ("formed_by_intersection", "intersection", "crosses", "runs_through", "borders", "connects")):
            return "location_structure"
        return "location_other"

    return "other"


def _value_relation_bucket(relation_key: str, target_type: str) -> str:
    rel = relation_key or ""
    target_type = _normalize_tail_type(target_type)

    if target_type == "TIME":
        if any(k in rel for k in ("founded", "established", "formed", "opened", "started")):
            return "time_origin"
        if any(k in rel for k in ("released", "premiered", "debut", "published", "aired", "launched")):
            return "time_release"
        if any(k in rel for k in ("closed", "ended", "ceased", "cancelled", "retired")):
            return "time_end"
        if any(k in rel for k in ("born", "birth")):
            return "time_birth"
        if any(k in rel for k in ("died", "death")):
            return "time_death"
        return "time_other"

    if target_type == "QUANTITY":
        if any(k in rel for k in ("revenue", "sales", "gross", "income", "earnings", "budget")):
            return "quantity_money"
        if any(k in rel for k in ("population", "employees", "staff", "people", "attendance")):
            return "quantity_people"
        if any(k in rel for k in ("area", "size", "length", "distance", "height", "weight", "mass")):
            return "quantity_measure"
        if any(k in rel for k in ("price", "cost", "fee", "valuation")):
            return "quantity_price"
        return "quantity_other"

    return "other"


def _relation_tokens(relation_key: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", relation_key or "")
        if token and token not in {"in", "on", "at", "by", "of", "the", "a", "an"}
    }


def _value_relations_compatible(rel_a: str, rel_b: str, target_type: str) -> tuple[bool, float, str]:
    rel_a = rel_a or ""
    rel_b = rel_b or ""
    target_type = _normalize_tail_type(target_type)

    if not rel_a or not rel_b:
        generic = "time_attribute" if target_type == "TIME" else "quantity_attribute"
        return False, 0.0, generic

    if rel_a == rel_b:
        return True, 1.0, rel_a

    bucket_a = _value_relation_bucket(rel_a, target_type)
    bucket_b = _value_relation_bucket(rel_b, target_type)
    if bucket_a == bucket_b and not bucket_a.endswith("other"):
        generic = "time_attribute" if target_type == "TIME" else "quantity_attribute"
        return True, 0.6, generic

    token_overlap = _relation_tokens(rel_a) & _relation_tokens(rel_b)
    if token_overlap:
        generic = "time_attribute" if target_type == "TIME" else "quantity_attribute"
        return True, 0.3, generic

    return False, 0.0, ""


def _relation_natural_text(relation_key: str, target_type: str = "OTHER") -> str:
    rel = (relation_key or "").strip().lower()
    target_type = _normalize_tail_type(target_type)
    if not rel:
        return "相关属性"

    tokens = _relation_tokens(rel)

    def _has_token(*keys: str) -> bool:
        return any(key in tokens for key in keys)

    def _has_phrase(*keys: str) -> bool:
        return any(key in rel for key in keys)

    if _has_phrase("performed_at", "staged_at", "played_at"):
        return "演出地点"
    if _has_phrase("premiered_at", "opened_at"):
        return "首演地点"
    if _has_phrase("headquartered_in", "headquartered_at") or _has_token("headquartered"):
        return "总部所在地"
    if _has_phrase("located_in", "located_at"):
        return "所在地点"
    if _has_token("founder") or _has_phrase("founded_by"):
        return "创始人"
    if _has_token("ceo"):
        return "首席执行官"
    if _has_token("owner") or _has_phrase("owned_by"):
        return "所有者"
    if _has_token("parent", "subsidiary") or _has_phrase("brand_of"):
        return "上级组织"
    if _has_token("author", "writer"):
        return "作者"
    if _has_token("lyrics"):
        return "作词者"
    if _has_token("composer") or _has_phrase("music_by"):
        return "作曲者"
    if _has_token("director") or _has_phrase("directed_by"):
        return "导演"
    if _has_token("revenue", "sales", "gross", "income", "earnings"):
        return "营收"
    if _has_token("population", "employees", "staff", "attendance"):
        return "人数"
    if _has_token("area", "size", "length", "distance", "height", "weight", "mass"):
        return "数值规模"
    if _has_token("price", "cost", "fee", "valuation"):
        return "价格"
    if _has_token("incorporated", "renamed") or _has_phrase("originally_named"):
        return "后续名称"
    if _has_token("country", "city", "state"):
        return "所属地点"
    if _has_token("born", "birth"):
        return "出生时间" if target_type == "TIME" else "出生地"
    if _has_token("died", "death"):
        return "去世时间" if target_type == "TIME" else "去世地"
    if _has_token("closed", "ended", "ceased", "cancelled", "retired"):
        return "结束时间"
    if _has_token("premiered", "premiere", "debut"):
        return "首演时间"
    if _has_token("released", "published", "aired", "launched"):
        return "发布时间"
    if _has_token("founded", "established", "formed", "opened", "started"):
        return "成立时间"

    generic_map = {
        "PERSON": "相关人物",
        "ORG": "相关组织",
        "LOCATION": "相关地点",
        "TIME": "相关时间",
        "QUANTITY": "相关数值",
    }
    return generic_map.get(target_type, "相关属性")


def _path_answer_prompt(target_type: str) -> str:
    target_type = _normalize_tail_type(target_type)
    return {
        "PERSON": "是谁",
        "ORG": "是什么组织或机构",
        "LOCATION": "是哪里",
        "TIME": "是什么时间",
        "QUANTITY": "是多少",
    }.get(target_type, "是什么")


def _join_shared_target_phrase(rel_a: str, rel_b: str, shared_type: str) -> str:
    rel_a = _relation_slug(rel_a)
    rel_b = _relation_slug(rel_b)
    shared_type = _normalize_tail_type(shared_type)

    if shared_type == "LOCATION":
        if all(any(k in rel for k in ("performed_at", "staged_at", "played_at", "premiered_at", "opened_at")) for rel in (rel_a, rel_b)):
            return "共同关联的演出地点"
        if all(any(k in rel for k in ("headquartered", "based_in")) for rel in (rel_a, rel_b)):
            return "共同关联的总部所在地"
        if all(any(k in rel for k in ("born_in", "origin", "from")) for rel in (rel_a, rel_b)):
            return "共同关联的出生地"
        return "共同关联的地点"

    if shared_type == "PERSON":
        if all(any(k in rel for k in ("founder", "founded_by")) for rel in (rel_a, rel_b)):
            return "共同关联的创始人"
        if all("ceo" in rel for rel in (rel_a, rel_b)):
            return "共同关联的负责人"
        if all(any(k in rel for k in ("author", "writer")) for rel in (rel_a, rel_b)):
            return "共同关联的作者"
        if all("lyrics" in rel for rel in (rel_a, rel_b)):
            return "共同关联的作词者"
        if all(any(k in rel for k in ("composer", "music_by")) for rel in (rel_a, rel_b)):
            return "共同关联的作曲者"
        if all(any(k in rel for k in ("director", "directed_by")) for rel in (rel_a, rel_b)):
            return "共同关联的导演"
        return "共同关联到的人物"

    if shared_type == "ORG":
        if all(any(k in rel for k in ("headquartered", "based_in")) for rel in (rel_a, rel_b)):
            return "共同关联的机构"
        if all(any(k in rel for k in ("parent", "owned_by", "brand_of")) for rel in (rel_a, rel_b)):
            return "共同关联的上级组织"
        return "共同关联的组织"

    return {
        "PERSON": "共同关联到的人物",
        "ORG": "共同关联的组织",
        "LOCATION": "共同关联的地点",
    }.get(shared_type, "共同关联的目标")


def _path_follow_clause(relation_key: str, target_type: str) -> str:
    rel = _relation_slug(relation_key)
    target_type = _normalize_tail_type(target_type)

    if any(k in rel for k in ("incorporated", "renamed", "originally_named")):
        return "后来合并注册为什么公司名称"
    if any(k in rel for k in ("performed_at", "staged_at", "played_at")):
        return "演出地点是哪里"
    if any(k in rel for k in ("premiered_at", "opened_at")):
        return "首演地点是哪里"
    if any(k in rel for k in ("headquartered", "based_in")):
        return "总部设在哪里"
    if any(k in rel for k in ("located_in", "located_at")):
        return "位于哪里"
    if any(k in rel for k in ("founder", "founded_by")):
        return "创始人是谁"
    if "ceo" in rel:
        return "首席执行官是谁"
    if any(k in rel for k in ("author", "writer")):
        return "作者是谁"
    if "lyrics" in rel:
        return "作词者是谁"
    if any(k in rel for k in ("composer", "music_by")):
        return "作曲者是谁"
    if any(k in rel for k in ("director", "directed_by")):
        return "导演是谁"
    if any(k in rel for k in ("founded", "established", "formed", "opened", "started")):
        return "成立于哪一年"
    if any(k in rel for k in ("premiered", "premiere", "debut")):
        return "首演时间是什么时候"
    if any(k in rel for k in ("released", "published", "aired", "launched")):
        return "发布时间是什么时候"
    if any(k in rel for k in ("born", "birth")):
        return "出生地是哪里" if target_type == "LOCATION" else "出生时间是什么时候"
    if any(k in rel for k in ("died", "death")):
        return "去世地是哪里" if target_type == "LOCATION" else "去世时间是什么时候"
    if any(k in rel for k in ("revenue", "sales", "gross", "income", "earnings")):
        return "营收是多少"
    if any(k in rel for k in ("population", "employees", "staff", "attendance")):
        return "人数是多少"
    if any(k in rel for k in ("area", "size", "length", "distance", "height", "weight", "mass")):
        return "具体数值是多少"
    if any(k in rel for k in ("price", "cost", "fee", "valuation")):
        return "价格是多少"

    fallback_phrase = _relation_natural_text(relation_key, target_type)
    generic_clause = {
        "PERSON": "对应的人物是谁",
        "ORG": "对应的组织或机构是什么",
        "LOCATION": "对应的地点是哪里",
        "TIME": "对应的时间是什么时候",
        "QUANTITY": "对应的数值是多少",
    }.get(target_type, "对应的答案是什么")
    if fallback_phrase.startswith("相关"):
        return generic_clause
    return f"{fallback_phrase}{_path_answer_prompt(target_type)}"


def _join_location_question_tail(answer_text: str) -> str:
    answer = (answer_text or "").strip().lower()
    if not answer:
        return "位于哪里"
    if any(k in answer for k in (" city", "new york", "london", "paris", "tokyo", "shanghai", "beijing")):
        return "位于哪座城市"
    if any(k in answer for k in ("state", "province", "州", "省")):
        return "位于哪个州或地区"
    if any(k in answer for k in ("country", "nation", "united states", "china", "japan", "france", "britain", "uk")):
        return "位于哪个国家"
    return "位于哪里"


def _join_follow_clause(rel_a: str, rel_b: str, shared_type: str, follow_rel: str, target_type: str, answer_text: str) -> str:
    follow_rel = _relation_slug(follow_rel)
    if any(k in follow_rel for k in ("located_in", "located_at", "based_in")) and target_type == "LOCATION":
        return _join_location_question_tail(answer_text)
    if any(k in follow_rel for k in ("performed_at", "staged_at", "played_at")):
        return "演出地点是哪里"
    if any(k in follow_rel for k in ("premiered_at", "opened_at")):
        return "首演地点是哪里"
    if any(k in follow_rel for k in ("headquartered", "headquartered_in", "headquartered_at")):
        return "总部设在哪里"
    return _path_follow_clause(follow_rel, target_type)


def _selection_phrase(selection_type: str, compare_rel: str) -> str:
    selection_type = _normalize_tail_type(selection_type)
    compare_phrase = _relation_natural_text(compare_rel, selection_type)
    if selection_type == "TIME":
        if compare_phrase == "相关时间":
            return "时间更早"
        return f"{compare_phrase}更早"
    if selection_type == "QUANTITY":
        if compare_phrase == "相关数值":
            return "数值更高"
        return f"{compare_phrase}更高"
    return f"{compare_phrase}更符合条件"


def _same_target_relation_compatible(rel_a: str, rel_b: str, target_type: str) -> tuple[bool, float]:
    """过滤共享目标但语义上不自然的 relation 组合。"""
    if not rel_a or not rel_b:
        return False, 0.0
    if rel_a == rel_b:
        return True, 1.0

    bucket_a = _relation_semantic_bucket(rel_a, target_type)
    bucket_b = _relation_semantic_bucket(rel_b, target_type)
    if bucket_a != bucket_b:
        return False, 0.0

    blocked_buckets = {"location_structure", "location_other", "org_other", "person_other", "other"}
    if bucket_a in blocked_buckets:
        return False, 0.0

    bucket_bonus = {
        "person_role": 0.8,
        "person_creator": 0.8,
        "person_biographical": 0.6,
        "org_ownership": 0.8,
        "org_affiliation": 0.7,
        "org_membership": 0.6,
        "location_membership": 0.7,
        "location_occurrence": 0.5,
    }.get(bucket_a, 0.0)
    return True, bucket_bonus


def _select_top_candidates(
    candidates: list[dict],
    max_n: int,
    signature_fields: tuple[str, ...],
) -> list[dict]:
    """按 score 排序后去重，选出 top-N 候选。"""
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    selected = []
    seen = set()
    for cand in ranked:
        signature = tuple(cand.get(field) for field in signature_fields)
        if signature in seen:
            continue
        seen.add(signature)
        selected.append(cand)
        if len(selected) >= max_n:
            break
    return selected


def _lookup_operation_info(relation: str, tail_type: str) -> dict:
    rel = relation.strip() or "相关信息"
    tail_type = _normalize_tail_type(tail_type)
    default = {
        "operation": "lookup_fact",
        "question_goal": f"查找该实体的[{rel}]",
        "comparison_hint": f"定位该实体的[{rel}]答案",
        "search_reason": f"搜索该实体的 [{rel}]",
        "target_prefix": "单实体查找",
    }
    mapping = {
        "PERSON": {
            "operation": "lookup_person",
            "question_goal": f"查找与该实体相关的人物（[{rel}]）",
            "comparison_hint": f"该实体 →[{rel}]→ [人物答案]",
            "search_reason": f"搜索该实体对应的相关人物信息（[{rel}]）",
            "target_prefix": "人物查找",
        },
        "ORG": {
            "operation": "lookup_org",
            "question_goal": f"查找与该实体相关的组织/机构（[{rel}]）",
            "comparison_hint": f"该实体 →[{rel}]→ [组织答案]",
            "search_reason": f"搜索该实体对应的组织/机构信息（[{rel}]）",
            "target_prefix": "组织查找",
        },
        "LOCATION": {
            "operation": "lookup_location",
            "question_goal": f"查找该实体相关的地点（[{rel}]）",
            "comparison_hint": f"该实体 →[{rel}]→ [地点答案]",
            "search_reason": f"搜索该实体对应的地点信息（[{rel}]）",
            "target_prefix": "地点查找",
        },
        "TIME": {
            "operation": "lookup_time",
            "question_goal": f"查找该实体相关的时间信息（[{rel}]）",
            "comparison_hint": f"该实体 →[{rel}]→ [时间答案]",
            "search_reason": f"搜索该实体对应的时间信息（[{rel}]）",
            "target_prefix": "时间查找",
        },
        "QUANTITY": {
            "operation": "lookup_quantity",
            "question_goal": f"查找该实体相关的数值信息（[{rel}]）",
            "comparison_hint": f"该实体 →[{rel}]→ [数值答案]",
            "search_reason": f"搜索该实体对应的数值信息（[{rel}]）",
            "target_prefix": "数值查找",
        },
    }
    return mapping.get(tail_type, default)


def _extract_year(value: str) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", value)
    if not match:
        return None
    year = int(match.group(1))
    if 1000 <= year <= 2100:
        return year
    return None


def _extract_year_from_edge(nodes: dict, node_key: str, data: dict) -> int | None:
    normalized = data.get("normalized_value")
    if isinstance(normalized, int):
        return normalized if 1000 <= normalized <= 2100 else None
    if isinstance(normalized, float) and normalized.is_integer():
        year = int(normalized)
        return year if 1000 <= year <= 2100 else None
    if isinstance(normalized, str) and normalized.strip():
        year = _extract_year(normalized)
        if year is not None:
            return year
    return _extract_year(nodes.get(node_key, {}).get("name", node_key))


def _extract_quantity(value: str) -> tuple[float | None, str]:
    """从字符串中提取可比较数值和简单单位签名。

    返回 (normalized_numeric_value, unit_signature)。
    仅做轻量解析，适用于同关系、同单位场景。
    """
    if not isinstance(value, str):
        return None, ""

    text = value.strip()
    if not text:
        return None, ""

    match = re.search(
        r"(?P<prefix>[$€£¥]?)\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>[KMBTkmbt]?)\s*(?P<unit>[A-Za-z%$€£¥/.-]*)",
        text,
    )
    if not match:
        return None, ""

    raw_number = match.group("number").replace(",", "")
    try:
        number = float(raw_number)
    except ValueError:
        return None, ""

    scale = match.group("scale").lower()
    if scale:
        number *= _QUANTITY_SCALE.get(scale, 1.0)

    prefix = match.group("prefix").strip().lower()
    unit = match.group("unit").strip().lower()
    unit_signature = f"{prefix}{unit}"
    return number, unit_signature


def _extract_quantity_from_edge(nodes: dict, node_key: str, data: dict) -> tuple[float | None, str]:
    normalized = data.get("normalized_value")
    unit = _normalize_unit(data.get("unit"))
    if isinstance(normalized, (int, float)):
        return float(normalized), unit.lower()
    if isinstance(normalized, str) and normalized.strip():
        try:
            return float(normalized.strip()), unit.lower()
        except ValueError:
            pass
    return _extract_quantity(nodes.get(node_key, {}).get("name", node_key))


def _format_numeric_value(value: float | int | None) -> str:
    if value is None:
        return "?"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_compact_quantity(value: float | int | None, unit_signature: str = "") -> str:
    if value is None:
        return "?"

    abs_value = abs(float(value))
    for suffix, scale in (("T", 1_000_000_000_000.0), ("B", 1_000_000_000.0), ("M", 1_000_000.0), ("K", 1_000.0)):
        if abs_value >= scale:
            compact = _format_numeric_value(float(value) / scale)
            if unit_signature in {"$", "€", "£", "¥"}:
                return f"{unit_signature}{compact}{suffix}"
            return f"{compact}{suffix}{unit_signature}"

    base = _format_numeric_value(value)
    if unit_signature in {"$", "€", "£", "¥"}:
        return f"{unit_signature}{base}"
    return f"{base}{unit_signature}"


def _find_bridge_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """交集桥接：两个图中实体共同指向同一个图外节点。
    形态: [A in_image] →rel_1→ [T] ←rel_2← [B in_image]
    """
    candidates = []
    for target in G.nodes():
        if G.nodes[target].get("in_image"):
            continue
        if not _answer_ok(nodes.get(target, {}).get("name", "")):
            continue
        preds = list(G.predecessors(target))
        img_preds = [p for p in preds if p in in_image_nodes]
        if len(img_preds) < 2:
            continue

        target_type = _edge_tail_type(nodes, target)
        if target_type in _SAME_TARGET_TYPES:
            continue
        for i in range(len(img_preds)):
            for j in range(i + 1, len(img_preds)):
                a = img_preds[i]
                b = img_preds[j]
                edge_a = G[a][target]
                edge_b = G[b][target]
                rel_a = _edge_relation_key(edge_a)
                rel_b = _edge_relation_key(edge_b)
                same_relation_bonus = 0.6 if rel_a == rel_b and rel_a else 0.0
                target_type_bonus = {
                    "PERSON": 1.0,
                    "ORG": 0.9,
                    "LOCATION": 0.8,
                    "TIME": 0.5,
                    "QUANTITY": 0.5,
                    "OTHER": 0.0,
                }.get(target_type, 0.0)
                score = (
                    1.0
                    + _edge_evidence_score(edge_a)
                    + _edge_evidence_score(edge_b)
                    + same_relation_bonus
                    + target_type_bonus
                )
                candidates.append({
                    "motif_type": "bridge",
                    "difficulty": "L2",
                    "source_a": a,
                    "source_b": b,
                    "target": target,
                    "rel_a": edge_a.get("relation", ""),
                    "rel_b": edge_b.get("relation", ""),
                    "fact_a": edge_a.get("fact", ""),
                    "fact_b": edge_b.get("fact", ""),
                    "score": score,
                })
    return _select_top_candidates(
        candidates, max_n, ("source_a", "source_b", "target")
    )


def _find_same_target_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """共享目标：两个图中实体通过人物/组织/地点类目标产生同一性/同组关系。"""
    candidates = []
    for target in G.nodes():
        if G.nodes[target].get("in_image"):
            continue
        if not _answer_ok(nodes.get(target, {}).get("name", "")):
            continue

        target_type = _edge_tail_type(nodes, target)
        if target_type not in _SAME_TARGET_TYPES:
            continue

        preds = [p for p in G.predecessors(target) if p in in_image_nodes]
        if len(preds) < 2:
            continue

        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                a = preds[i]
                b = preds[j]
                edge_a = G[a][target]
                edge_b = G[b][target]
                rel_a = _edge_relation_key(edge_a)
                rel_b = _edge_relation_key(edge_b)
                compatible, semantic_bonus = _same_target_relation_compatible(rel_a, rel_b, target_type)
                if not compatible:
                    continue
                score = (
                    1.2
                    + _edge_evidence_score(edge_a)
                    + _edge_evidence_score(edge_b)
                    + (0.6 if rel_a == rel_b and rel_a else 0.0)
                    + semantic_bonus
                )
                candidates.append({
                    "motif_type": "same_target",
                    "difficulty": "L2",
                    "entity_a": a,
                    "entity_b": b,
                    "target": target,
                    "target_type": target_type,
                    "rel_a": edge_a.get("relation", ""),
                    "rel_b": edge_b.get("relation", ""),
                    "fact_a": edge_a.get("fact", ""),
                    "fact_b": edge_b.get("fact", ""),
                    "score": score,
                })
    return _select_top_candidates(
        candidates, max_n, ("entity_a", "entity_b", "target")
    )


def _find_same_group_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """同组归属：两个图中实体经由不同中间节点，归属于同一个地点/组织上级节点。"""
    parent_buckets: dict[tuple[str, str], list] = defaultdict(list)

    for start in in_image_nodes:
        for _, child, edge_1 in G.out_edges(start, data=True):
            if child in in_image_nodes:
                continue
            child_type = _edge_tail_type(nodes, child, edge_1)
            if child_type not in _SAME_GROUP_TYPES:
                continue
            if not _answer_ok(nodes.get(child, {}).get("name", "")):
                continue

            for _, parent, edge_2 in G.out_edges(child, data=True):
                if parent in in_image_nodes or parent == start:
                    continue
                parent_type = _edge_tail_type(nodes, parent, edge_2)
                if parent_type not in _SAME_GROUP_TYPES:
                    continue
                if not _answer_ok(nodes.get(parent, {}).get("name", "")):
                    continue

                parent_buckets[(parent, parent_type)].append({
                    "entity": start,
                    "child": child,
                    "parent": parent,
                    "child_type": child_type,
                    "parent_type": parent_type,
                    "rel_1": edge_1.get("relation", ""),
                    "rel_2": edge_2.get("relation", ""),
                    "fact_1": edge_1.get("fact", ""),
                    "fact_2": edge_2.get("fact", ""),
                    "score": _edge_evidence_score(edge_1) + _edge_evidence_score(edge_2),
                })

    candidates = []
    for (parent, parent_type), entries in parent_buckets.items():
        if len(entries) < 2:
            continue

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                left = entries[i]
                right = entries[j]
                if left["entity"] == right["entity"]:
                    continue
                if left["child"] == right["child"]:
                    continue

                score = (
                    1.6
                    + left["score"]
                    + right["score"]
                    + (0.6 if left["parent_type"] == right["parent_type"] else 0.0)
                    + (0.4 if left["child_type"] == right["child_type"] else 0.0)
                )
                candidates.append({
                    "motif_type": "same_group",
                    "difficulty": "L2",
                    "entity_a": left["entity"],
                    "entity_b": right["entity"],
                    "child_a": left["child"],
                    "child_b": right["child"],
                    "parent": parent,
                    "parent_type": parent_type,
                    "child_type_a": left["child_type"],
                    "child_type_b": right["child_type"],
                    "rel_a_1": left["rel_1"],
                    "rel_a_2": left["rel_2"],
                    "rel_b_1": right["rel_1"],
                    "rel_b_2": right["rel_2"],
                    "fact_a_1": left["fact_1"],
                    "fact_a_2": left["fact_2"],
                    "fact_b_1": right["fact_1"],
                    "fact_b_2": right["fact_2"],
                    "score": score,
                })

    return _select_top_candidates(
        candidates,
        max_n,
        ("entity_a", "entity_b", "child_a", "child_b", "parent"),
    )


def _find_multihop_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """深度跳跃：A → X → T，要求 X 有其他出边（用于生成限定语）。
    形态: [A in_image] →rel_1→ [X] →rel_2→ [T]，X 还有额外属性 →rel_3→ [Y]
    """
    candidates = []
    for start in in_image_nodes:
        for mid in G.successors(start):
            if mid in in_image_nodes:
                continue
            out_edges = list(G.out_edges(mid, data=True))
            if len(out_edges) < 1:
                continue
            for _, end, end_data in out_edges:
                if end == start or end in in_image_nodes:
                    continue
                if not _answer_ok(nodes.get(end, {}).get("name", "")):
                    continue
                # 选择 X 上最有信息量、且不指向 end 的出边作为限定属性
                disc_edges = [(u, v, d) for u, v, d in out_edges if v != end]
                disc = None
                if disc_edges:
                    best_disc = None
                    best_score = -1.0
                    for _, dv, dd in disc_edges:
                        if not _answer_ok(nodes.get(dv, {}).get("name", "")):
                            continue
                        rel_slug = _edge_relation_key(dd)
                        if rel_slug in _LOW_VALUE_RELATIONS:
                            continue
                        disc_score = (
                            _edge_evidence_score(dd)
                            + (0.8 if _edge_tail_type(nodes, dv, dd) != "OTHER" else 0.0)
                        )
                        if disc_score > best_score:
                            best_score = disc_score
                            best_disc = (dv, dd)
                    if best_disc is not None:
                        dv, dd = best_disc
                        disc = {
                            "node": dv,
                            "node_name": nodes.get(dv, {}).get("name", dv),
                            "relation": dd.get("relation", ""),
                            "fact": dd.get("fact", ""),
                        }
                score = (
                    1.5
                    + _edge_evidence_score(G[start][mid])
                    + _edge_evidence_score(end_data)
                    + (1.0 if disc else 0.0)
                )
                candidates.append({
                    "motif_type": "multihop",
                    "difficulty": "L3",
                    "source": start,
                    "mid": mid,
                    "target": end,
                    "rel_1": G[start][mid]["relation"],
                    "rel_2": G[mid][end]["relation"],
                    "fact_1": G[start][mid].get("fact", ""),
                    "fact_2": G[mid][end].get("fact", ""),
                    "discriminator": disc,
                    "score": score,
                })
    return _select_top_candidates(
        candidates, max_n, ("source", "mid", "target")
    )


def _find_lookup_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 4) -> list:
    """单实体查找：从图中实体出发的一跳高价值事实。
    形态: [A in_image] →rel→ [答案]
    """
    candidates = []
    for start in in_image_nodes:
        for _, target, data in G.out_edges(start, data=True):
            if target in in_image_nodes:
                continue
            if not _answer_ok(nodes.get(target, {}).get("name", "")):
                continue

            rel_slug = _edge_relation_key(data)
            if not rel_slug or rel_slug in _LOW_VALUE_RELATIONS:
                continue

            tail_type = _edge_tail_type(nodes, target, data)
            op_info = _lookup_operation_info(data.get("relation", ""), tail_type)
            score = (
                1.0
                + _LOOKUP_TAIL_TYPE_BONUS.get(tail_type, 0.0)
                + _edge_evidence_score(data)
            )
            candidates.append({
                "motif_type": "lookup",
                "difficulty": "L2",
                "source": start,
                "target": target,
                "relation": data.get("relation", ""),
                "tail_type": tail_type,
                "operation": op_info["operation"],
                "fact": data.get("fact", ""),
                "score": score,
            })
    return _select_top_candidates(candidates, max_n, ("source", "relation", "target"))


def _find_extremum_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 2) -> list:
    """集合极值：图中多个实体具有同类时间/数值属性，问谁最早/最高。"""
    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            tail_type = _edge_tail_type(nodes, succ, data)
            rel_slug = _edge_relation_key(data)
            if tail_type not in _EXTREMUM_TAIL_TYPES:
                continue
            rel_buckets[tail_type].append((node, succ, data, rel_slug))

    candidates = []
    for bucket_type, entries in rel_buckets.items():
        category_groups: dict[str, list] = defaultdict(list)
        for node, succ, data, rel_slug in entries:
            category = _value_relation_bucket(rel_slug, bucket_type)
            category_groups[category].append((node, succ, data, rel_slug))

        for _, group_entries in category_groups.items():
            distinct_entities = []
            seen_nodes = set()
            for node, succ, data, rel_slug in group_entries:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                distinct_entities.append((node, succ, data, rel_slug))

            if len(distinct_entities) < 3:
                continue

            selected = distinct_entities[:3]
            anchor_nodes = [node for node, _, _, _ in selected]
            value_nodes = [succ for _, succ, _, _ in selected]
            relation_label = selected[0][3] if all(rel == selected[0][3] for _, _, _, rel in selected) else (
                "time_attribute" if bucket_type == "TIME" else "quantity_attribute"
            )
            score = sum(_edge_evidence_score(data) for _, _, data, _ in selected) + len(selected)
            candidates.append({
                "motif_type": "extremum",
                "difficulty": "L2",
                "comparison_type": bucket_type,
                "relation": relation_label,
                "entities": anchor_nodes,
                "values": value_nodes,
                "facts": [data.get("fact", "") for _, _, data, _ in selected],
                "score": score,
            })

    return _select_top_candidates(candidates, max_n, ("comparison_type", "relation"))


def _find_delta_time_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 2) -> list:
    """时间差：两个图中实体的时间属性可解析为年份时，问相差多少年。"""
    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            if _edge_tail_type(nodes, succ, data) != "TIME":
                continue
            year = _extract_year_from_edge(nodes, succ, data)
            if year is None:
                continue
            rel_buckets["TIME"].append((node, succ, data, year, _edge_relation_key(data)))

    candidates = []
    for _, entries in rel_buckets.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_node, a_val, a_data, year_a, rel_a = entries[i]
                b_node, b_val, b_data, year_b, rel_b = entries[j]
                if a_node == b_node or year_a == year_b:
                    continue
                compatible, relation_bonus, relation_label = _value_relations_compatible(rel_a, rel_b, "TIME")
                if not compatible:
                    continue
                delta_years = abs(year_a - year_b)
                score = (
                    1.5
                    + _edge_evidence_score(a_data)
                    + _edge_evidence_score(b_data)
                    + relation_bonus
                    + (0.5 if delta_years >= 2 else 0.0)
                )
                candidates.append({
                    "motif_type": "delta_time",
                    "difficulty": "L2",
                    "entity_a": a_node,
                    "entity_b": b_node,
                    "relation": relation_label,
                    "value_a": a_val,
                    "value_b": b_val,
                    "year_a": year_a,
                    "year_b": year_b,
                    "delta_years": delta_years,
                    "fact_a": a_data.get("fact", ""),
                    "fact_b": b_data.get("fact", ""),
                    "score": score,
                })
    return _select_top_candidates(candidates, max_n, ("entity_a", "entity_b", "relation"))


def _find_delta_quantity_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 2) -> list:
    """数值差：两个图中实体的数值属性可解析且单位签名一致时，问相差多少。"""
    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            if _edge_tail_type(nodes, succ, data) != "QUANTITY":
                continue
            value_num, unit_sig = _extract_quantity_from_edge(nodes, succ, data)
            if value_num is None:
                continue
            rel_buckets["QUANTITY"].append((node, succ, data, value_num, unit_sig, _edge_relation_key(data)))

    candidates = []
    for _, entries in rel_buckets.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_node, a_val, a_data, num_a, unit_a, rel_a = entries[i]
                b_node, b_val, b_data, num_b, unit_b, rel_b = entries[j]
                if a_node == b_node or num_a == num_b:
                    continue
                if unit_a != unit_b:
                    continue
                compatible, relation_bonus, relation_label = _value_relations_compatible(rel_a, rel_b, "QUANTITY")
                if not compatible:
                    continue

                delta_value = abs(num_a - num_b)
                score = (
                    1.5
                    + _edge_evidence_score(a_data)
                    + _edge_evidence_score(b_data)
                    + relation_bonus
                    + (0.5 if delta_value > 0 else 0.0)
                )
                candidates.append({
                    "motif_type": "delta_quantity",
                    "difficulty": "L2",
                    "entity_a": a_node,
                    "entity_b": b_node,
                    "relation": relation_label,
                    "value_a": a_val,
                    "value_b": b_val,
                    "numeric_a": num_a,
                    "numeric_b": num_b,
                    "unit_signature": unit_a,
                    "delta_value": delta_value,
                    "fact_a": a_data.get("fact", ""),
                    "fact_b": b_data.get("fact", ""),
                    "score": score,
                })
    return _select_top_candidates(candidates, max_n, ("entity_a", "entity_b", "relation"))


def _find_sum_threshold_quantity_motifs(
    G: nx.DiGraph,
    in_image_nodes: set,
    nodes: dict,
    max_n: int = 2,
) -> list:
    """数量求和阈值：A+B 与 C 做比较，要求同关系、可解析、单位一致。"""
    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            if _edge_tail_type(nodes, succ, data) != "QUANTITY":
                continue
            value_num, unit_sig = _extract_quantity_from_edge(nodes, succ, data)
            if value_num is None:
                continue
            rel_buckets["QUANTITY"].append((node, succ, data, value_num, unit_sig, _edge_relation_key(data)))

    candidates = []
    for _, entries in rel_buckets.items():
        deduped = []
        seen_entities = set()
        for node, succ, data, value_num, unit_sig, rel_slug in entries:
            if node in seen_entities:
                continue
            seen_entities.add(node)
            deduped.append((node, succ, data, value_num, unit_sig, rel_slug))

        if len(deduped) < 3:
            continue

        for i in range(len(deduped)):
            for j in range(i + 1, len(deduped)):
                for k in range(len(deduped)):
                    if k in {i, j}:
                        continue

                    a_node, a_val, a_data, num_a, unit_a, rel_a = deduped[i]
                    b_node, b_val, b_data, num_b, unit_b, rel_b = deduped[j]
                    c_node, c_val, c_data, num_c, unit_c, rel_c = deduped[k]
                    if len({a_node, b_node, c_node}) < 3:
                        continue
                    if unit_a != unit_b or unit_a != unit_c:
                        continue
                    ok_ab, bonus_ab, relation_label_ab = _value_relations_compatible(rel_a, rel_b, "QUANTITY")
                    ok_ac, bonus_ac, relation_label_ac = _value_relations_compatible(rel_a, rel_c, "QUANTITY")
                    if not (ok_ab and ok_ac):
                        continue

                    sum_value = num_a + num_b
                    if sum_value == num_c:
                        continue

                    margin = abs(sum_value - num_c)
                    exceeds = sum_value > num_c
                    score = (
                        2.0
                        + _edge_evidence_score(a_data)
                        + _edge_evidence_score(b_data)
                        + _edge_evidence_score(c_data)
                        + bonus_ab
                        + bonus_ac
                        + (0.6 if margin > 0 else 0.0)
                    )
                    candidates.append({
                        "motif_type": "sum_threshold_quantity",
                        "difficulty": "L2",
                        "entity_a": a_node,
                        "entity_b": b_node,
                        "entity_c": c_node,
                        "relation": relation_label_ab if relation_label_ab == relation_label_ac else "quantity_attribute",
                        "value_a": a_val,
                        "value_b": b_val,
                        "value_c": c_val,
                        "numeric_a": num_a,
                        "numeric_b": num_b,
                        "numeric_c": num_c,
                        "sum_value": sum_value,
                        "unit_signature": unit_a,
                        "exceeds": exceeds,
                        "margin": margin,
                        "fact_a": a_data.get("fact", ""),
                        "fact_b": b_data.get("fact", ""),
                        "fact_c": c_data.get("fact", ""),
                        "score": score,
                    })

    return _select_top_candidates(
        candidates,
        max_n,
        ("entity_a", "entity_b", "entity_c", "relation"),
    )


def _find_comparative_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """平行对比：按目标节点 tail_type 分桶（仅 TIME / QUANTITY）。
    形态: [A in_image] →rel_A→ [val_A]  vs  [B in_image] →rel_B→ [val_B]
    """
    # TIME/QUANTITY → [(in_image_node, value_node, edge_data)]
    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            tail_type = _normalize_tail_type(
                nodes.get(succ, {}).get("tail_type", data.get("tail_type", "OTHER"))
            )
            if tail_type in _COMPARATIVE_TAIL_TYPES:
                rel_buckets[tail_type].append((node, succ, data))

    comparatives = []
    seen_pairs: set = set()
    for bucket_type, entries in rel_buckets.items():
        if len(entries) < 2:
            continue

        candidate_pairs = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a_node, a_val, a_data = entries[i]
                b_node, b_val, b_data = entries[j]
                if a_node == b_node:
                    continue

                a_rel_norm = _edge_relation_key(a_data)
                b_rel_norm = _edge_relation_key(b_data)
                compatible, relation_bonus, relation_label = _value_relations_compatible(
                    a_rel_norm, b_rel_norm, bucket_type
                )
                if not compatible:
                    continue
                same_relation = int(bool(a_rel_norm) and a_rel_norm == b_rel_norm)
                distinct_values = int(a_val != b_val)
                score = (
                    same_relation
                    + distinct_values
                    + relation_bonus
                    + _edge_evidence_score(a_data)
                    + _edge_evidence_score(b_data)
                )

                candidate_pairs.append({
                    "a_node": a_node,
                    "a_val": a_val,
                    "a_data": a_data,
                    "a_rel_norm": a_rel_norm,
                    "b_node": b_node,
                    "b_val": b_val,
                    "b_data": b_data,
                    "b_rel_norm": b_rel_norm,
                    "relation_label": relation_label,
                    "same_relation": same_relation,
                    "distinct_values": distinct_values,
                    "score": score,
                })

        candidate_pairs.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        for cand in candidate_pairs:
            a_node = cand["a_node"]
            b_node = cand["b_node"]
            a_val = cand["a_val"]
            b_val = cand["b_val"]
            a_data = cand["a_data"]
            b_data = cand["b_data"]
            a_rel_norm = cand["a_rel_norm"]
            b_rel_norm = cand["b_rel_norm"]

            pair_key = (
                bucket_type,
                tuple(sorted([a_node, b_node])),
                tuple(sorted([a_val, b_val])),
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            relation = a_rel_norm if a_rel_norm and a_rel_norm == b_rel_norm else (
                cand.get("relation_label") or ("time_value" if bucket_type == "TIME" else "quantity_value")
            )

            comparatives.append({
                "motif_type": "comparative",
                "difficulty": "L2",
                "entity_a": a_node,
                "entity_b": b_node,
                "comparison_type": bucket_type,
                "relation": relation,
                "relation_a": a_rel_norm,
                "relation_b": b_rel_norm,
                "value_a": a_val,
                "value_b": b_val,
                "fact_a": a_data.get("fact", ""),
                "fact_b": b_data.get("fact", ""),
                "score": cand.get("score", 0.0),
            })
            if len(comparatives) >= max_n:
                return comparatives
    return comparatives


def _find_select_then_follow_motifs(
    G: nx.DiGraph,
    in_image_nodes: set,
    nodes: dict,
    max_n: int = 3,
) -> list:
    """路径依赖题：先在 TIME/QUANTITY 上选出胜者，再从胜者继续追一跳找到答案。"""
    candidates = []

    rel_buckets: dict[str, list] = defaultdict(list)
    for node in in_image_nodes:
        for _, succ, data in G.out_edges(node, data=True):
            tail_type = _edge_tail_type(nodes, succ, data)
            if tail_type == "TIME":
                year = _extract_year_from_edge(nodes, succ, data)
                if year is None:
                    continue
                rel_buckets["TIME"].append((node, succ, data, year, _edge_relation_key(data)))
            elif tail_type == "QUANTITY":
                value_num, unit_sig = _extract_quantity_from_edge(nodes, succ, data)
                if value_num is None:
                    continue
                rel_buckets["QUANTITY"].append((node, succ, data, value_num, unit_sig, _edge_relation_key(data)))

    def _pick_follow_edge(entity_key: str, excluded_targets: set[str], excluded_relation: str) -> tuple[str, dict] | None:
        best = None
        best_score = -1.0
        for _, target, follow_data in G.out_edges(entity_key, data=True):
            if target in in_image_nodes or target in excluded_targets:
                continue
            if not _answer_ok(nodes.get(target, {}).get("name", "")):
                continue
            follow_tail_type = _edge_tail_type(nodes, target, follow_data)
            if follow_tail_type not in _PATH_FOLLOW_TARGET_TYPES:
                continue
            follow_rel = _edge_relation_key(follow_data)
            if not follow_rel or follow_rel in _LOW_VALUE_RELATIONS:
                continue
            if follow_rel == excluded_relation:
                continue
            score = (
                _edge_evidence_score(follow_data)
                + _LOOKUP_TAIL_TYPE_BONUS.get(follow_tail_type, 0.0)
                + (0.5 if follow_tail_type in {"PERSON", "ORG", "LOCATION"} else 0.0)
            )
            if score > best_score:
                best = (target, follow_data)
                best_score = score
        return best

    # TIME path reasoning
    time_entries = rel_buckets.get("TIME", [])
    for i in range(len(time_entries)):
        for j in range(i + 1, len(time_entries)):
            a_node, a_val, a_data, year_a, rel_a = time_entries[i]
            b_node, b_val, b_data, year_b, rel_b = time_entries[j]
            if a_node == b_node or year_a == year_b:
                continue
            compatible, relation_bonus, relation_label = _value_relations_compatible(rel_a, rel_b, "TIME")
            if not compatible:
                continue

            winner = a_node if year_a < year_b else b_node
            loser = b_node if winner == a_node else a_node
            winner_value = year_a if winner == a_node else year_b
            loser_value = year_b if winner == a_node else year_a
            winner_val_node = a_val if winner == a_node else b_val
            loser_val_node = b_val if winner == a_node else a_val
            winner_data = a_data if winner == a_node else b_data
            loser_data = b_data if winner == a_node else a_data

            follow = _pick_follow_edge(winner, {winner_val_node, loser_val_node}, relation_label)
            if not follow:
                continue
            follow_target, follow_data = follow
            score = (
                2.2
                + relation_bonus
                + _edge_evidence_score(winner_data)
                + _edge_evidence_score(loser_data)
                + _edge_evidence_score(follow_data)
                + (0.5 if abs(year_a - year_b) >= 2 else 0.0)
            )
            candidates.append({
                "motif_type": "select_then_follow",
                "difficulty": "L3",
                "selection_type": "TIME",
                "entity_a": a_node,
                "entity_b": b_node,
                "winner": winner,
                "loser": loser,
                "comparison_relation": relation_label,
                "value_a": a_val,
                "value_b": b_val,
                "winner_value": winner_value,
                "loser_value": loser_value,
                "follow_target": follow_target,
                "follow_relation": follow_data.get("relation", ""),
                "follow_tail_type": _edge_tail_type(nodes, follow_target, follow_data),
                "fact_a": a_data.get("fact", ""),
                "fact_b": b_data.get("fact", ""),
                "follow_fact": follow_data.get("fact", ""),
                "score": score,
            })

    # QUANTITY path reasoning
    quantity_entries = rel_buckets.get("QUANTITY", [])
    for i in range(len(quantity_entries)):
        for j in range(i + 1, len(quantity_entries)):
            a_node, a_val, a_data, num_a, unit_a, rel_a = quantity_entries[i]
            b_node, b_val, b_data, num_b, unit_b, rel_b = quantity_entries[j]
            if a_node == b_node or num_a == num_b or unit_a != unit_b:
                continue
            compatible, relation_bonus, relation_label = _value_relations_compatible(rel_a, rel_b, "QUANTITY")
            if not compatible:
                continue

            winner = a_node if num_a > num_b else b_node
            loser = b_node if winner == a_node else a_node
            winner_value = num_a if winner == a_node else num_b
            loser_value = num_b if winner == a_node else num_a
            winner_val_node = a_val if winner == a_node else b_val
            loser_val_node = b_val if winner == a_node else a_val
            winner_data = a_data if winner == a_node else b_data
            loser_data = b_data if winner == a_node else a_data

            follow = _pick_follow_edge(winner, {winner_val_node, loser_val_node}, relation_label)
            if not follow:
                continue
            follow_target, follow_data = follow
            score = (
                2.2
                + relation_bonus
                + _edge_evidence_score(winner_data)
                + _edge_evidence_score(loser_data)
                + _edge_evidence_score(follow_data)
                + (0.5 if abs(num_a - num_b) > 0 else 0.0)
            )
            candidates.append({
                "motif_type": "select_then_follow",
                "difficulty": "L3",
                "selection_type": "QUANTITY",
                "entity_a": a_node,
                "entity_b": b_node,
                "winner": winner,
                "loser": loser,
                "comparison_relation": relation_label,
                "value_a": a_val,
                "value_b": b_val,
                "winner_value": winner_value,
                "loser_value": loser_value,
                "unit_signature": unit_a,
                "follow_target": follow_target,
                "follow_relation": follow_data.get("relation", ""),
                "follow_tail_type": _edge_tail_type(nodes, follow_target, follow_data),
                "fact_a": a_data.get("fact", ""),
                "fact_b": b_data.get("fact", ""),
                "follow_fact": follow_data.get("fact", ""),
                "score": score,
            })

    return _select_top_candidates(
        candidates,
        max_n,
        ("selection_type", "entity_a", "entity_b", "winner", "follow_target"),
    )


def _find_join_then_follow_motifs(
    G: nx.DiGraph,
    in_image_nodes: set,
    nodes: dict,
    max_n: int = 3,
) -> list:
    """路径依赖题：两个图中实体先汇合到同一隐藏节点，再从共享节点继续追一跳。"""
    candidates = []
    for target in G.nodes():
        if G.nodes[target].get("in_image"):
            continue
        if not _answer_ok(nodes.get(target, {}).get("name", "")):
            continue

        target_type = _edge_tail_type(nodes, target)
        if target_type not in _SAME_TARGET_TYPES:
            continue

        preds = [p for p in G.predecessors(target) if p in in_image_nodes]
        if len(preds) < 2:
            continue

        best_follow = None
        best_follow_score = -1.0
        for _, follow_target, follow_data in G.out_edges(target, data=True):
            if follow_target in in_image_nodes:
                continue
            if not _answer_ok(nodes.get(follow_target, {}).get("name", "")):
                continue
            follow_tail_type = _edge_tail_type(nodes, follow_target, follow_data)
            if follow_tail_type not in _PATH_FOLLOW_TARGET_TYPES:
                continue
            follow_rel = _edge_relation_key(follow_data)
            if not follow_rel or follow_rel in _LOW_VALUE_RELATIONS:
                continue
            info_gain_bonus = 0.0
            if follow_tail_type in {"PERSON", "ORG"}:
                info_gain_bonus += 0.8
            elif follow_tail_type == "LOCATION":
                info_gain_bonus += 0.2
                if target_type == "LOCATION" and any(k in follow_rel for k in ("located_in", "located_at", "based_in")):
                    info_gain_bonus -= 0.5
            follow_score = (
                _edge_evidence_score(follow_data)
                + _LOOKUP_TAIL_TYPE_BONUS.get(follow_tail_type, 0.0)
                + info_gain_bonus
            )
            if follow_score > best_follow_score:
                best_follow = (follow_target, follow_data)
                best_follow_score = follow_score

        if not best_follow:
            continue

        follow_target, follow_data = best_follow
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                a = preds[i]
                b = preds[j]
                edge_a = G[a][target]
                edge_b = G[b][target]
                rel_a = _edge_relation_key(edge_a)
                rel_b = _edge_relation_key(edge_b)
                compatible, semantic_bonus = _same_target_relation_compatible(rel_a, rel_b, target_type)
                if not compatible:
                    continue

                score = (
                    2.1
                    + _edge_evidence_score(edge_a)
                    + _edge_evidence_score(edge_b)
                    + _edge_evidence_score(follow_data)
                    + semantic_bonus
                    + (0.4 if rel_a == rel_b and rel_a else 0.0)
                )
                candidates.append({
                    "motif_type": "join_then_follow",
                    "difficulty": "L3",
                    "entity_a": a,
                    "entity_b": b,
                    "shared_target": target,
                    "shared_target_type": target_type,
                    "rel_a": edge_a.get("relation", ""),
                    "rel_b": edge_b.get("relation", ""),
                    "fact_a": edge_a.get("fact", ""),
                    "fact_b": edge_b.get("fact", ""),
                    "follow_target": follow_target,
                    "follow_relation": follow_data.get("relation", ""),
                    "follow_tail_type": _edge_tail_type(nodes, follow_target, follow_data),
                    "follow_fact": follow_data.get("fact", ""),
                    "score": score,
                })

    return _select_top_candidates(
        candidates,
        max_n,
        ("entity_a", "entity_b", "shared_target", "follow_target"),
    )


# ============================================================
# 3d. Motif 探测入口
# ============================================================

def find_motifs(triples: list, entities: list,
                max_bridge: int = 3, max_multihop: int = 3,
                max_comparative: int = 3, max_lookup: int = 4,
                max_extremum: int = 2, max_same_target: int = 3,
                max_same_group: int = 3,
                max_delta_time: int = 2, max_delta_quantity: int = 2,
                max_sum_threshold_quantity: int = 2,
                max_path_reasoning: int = 3):
    """建图并运行多类探测器，返回 (motifs_dict, nodes_dict)。"""
    G, nodes = _build_nx_graph(triples, entities)
    in_image_nodes = {k for k, v in nodes.items() if v["in_image"]}

    bridges     = _find_bridge_motifs(G, in_image_nodes, nodes, max_bridge)
    same_targets = _find_same_target_motifs(G, in_image_nodes, nodes, max_same_target)
    same_groups = _find_same_group_motifs(G, in_image_nodes, nodes, max_same_group)
    multihops   = _find_multihop_motifs(G, in_image_nodes, nodes, max_multihop)
    comparatives = _find_comparative_motifs(G, in_image_nodes, nodes, max_comparative)
    lookups = _find_lookup_motifs(G, in_image_nodes, nodes, max_lookup)
    extremums = _find_extremum_motifs(G, in_image_nodes, nodes, max_extremum)
    delta_times = _find_delta_time_motifs(G, in_image_nodes, nodes, max_delta_time)
    delta_quantities = _find_delta_quantity_motifs(G, in_image_nodes, nodes, max_delta_quantity)
    sum_threshold_quantities = _find_sum_threshold_quantity_motifs(
        G, in_image_nodes, nodes, max_sum_threshold_quantity
    )
    select_then_follow = _find_select_then_follow_motifs(
        G, in_image_nodes, nodes, max_path_reasoning
    )
    join_then_follow = _find_join_then_follow_motifs(
        G, in_image_nodes, nodes, max_path_reasoning
    )
    path_reasonings = _select_top_candidates(
        select_then_follow + join_then_follow,
        max_path_reasoning,
        ("motif_type", "entity_a", "entity_b", "follow_target"),
    )

    logger.info(
        f"    Motif 探测: Lookup={len(lookups)}  Bridge={len(bridges)}"
        f"  SameTarget={len(same_targets)}"
        f"  SameGroup={len(same_groups)}"
        f"  MultiHop={len(multihops)}  Comparative={len(comparatives)}"
        f"  Extremum={len(extremums)}  DeltaTime={len(delta_times)}"
        f"  DeltaQuantity={len(delta_quantities)}"
        f"  SumThresholdQuantity={len(sum_threshold_quantities)}"
        f"  PathReasoning={len(path_reasonings)}"
        f" (SelectThenFollow={len(select_then_follow)}, JoinThenFollow={len(join_then_follow)})"
    )
    return {
        "lookup": lookups,
        "bridge": bridges,
        "same_target": same_targets,
        "same_group": same_groups,
        "multihop": multihops,
        "comparative": comparatives,
        "extremum": extremums,
        "delta_time": delta_times,
        "delta_quantity": delta_quantities,
        "sum_threshold_quantity": sum_threshold_quantities,
        "select_then_follow": select_then_follow,
        "join_then_follow": join_then_follow,
        "path_reasoning": path_reasonings,
    }, nodes


# ============================================================
# 3e. Motif → 结构化骨架 JSON
# ============================================================

def motif_to_skeleton(motif: dict, nodes: dict) -> dict:
    """将一个 Motif 转为传给 LLM 的结构化骨架。"""
    mtype = motif["motif_type"]

    def _obf(key: str) -> str:
        n = nodes.get(key, {})
        if n.get("in_image") and n.get("location"):
            return f"[图中: {n['location']}]"
        return "[隐藏]"

    def _name(key: str) -> str:
        return nodes.get(key, {}).get("name", key)

    def _program() -> dict:
        if mtype == "lookup":
            return {
                "program_type": "lookup",
                "operation": motif.get("operation", "lookup_fact"),
                "target_type": motif.get("tail_type", "OTHER"),
                "anchors": [motif["source"]],
                "hidden_nodes": [motif["target"]],
                "relation_keys": [_relation_slug(motif.get("relation", ""))],
            }
        if mtype == "extremum":
            return {
                "program_type": "extremum",
                "operation": "argmax_or_argmin",
                "target_type": motif.get("comparison_type", "OTHER"),
                "anchors": motif.get("entities", []),
                "hidden_nodes": motif.get("values", []),
                "relation_keys": [motif.get("relation", "")],
            }
        if mtype == "same_target":
            return {
                "program_type": "same_entity",
                "operation": "same_target_check",
                "target_type": motif.get("target_type", "OTHER"),
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["target"]],
                "relation_keys": [
                    _relation_slug(motif.get("rel_a", "")),
                    _relation_slug(motif.get("rel_b", "")),
                ],
            }
        if mtype == "same_group":
            return {
                "program_type": "same_group",
                "operation": "shared_parent_check",
                "target_type": motif.get("parent_type", "OTHER"),
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["child_a"], motif["child_b"], motif["parent"]],
                "relation_keys": [
                    _relation_slug(motif.get("rel_a_1", "")),
                    _relation_slug(motif.get("rel_a_2", "")),
                    _relation_slug(motif.get("rel_b_1", "")),
                    _relation_slug(motif.get("rel_b_2", "")),
                ],
            }
        if mtype == "bridge":
            return {
                "program_type": "bridge",
                "operation": "intersection",
                "target_type": _normalize_tail_type(nodes.get(motif["target"], {}).get("tail_type", "OTHER")),
                "anchors": [motif["source_a"], motif["source_b"]],
                "hidden_nodes": [motif["target"]],
                "relation_keys": [
                    _relation_slug(motif.get("rel_a", "")),
                    _relation_slug(motif.get("rel_b", "")),
                ],
            }
        if mtype == "multihop":
            return {
                "program_type": "constrained_multihop",
                "operation": "two_hop_lookup",
                "target_type": _normalize_tail_type(nodes.get(motif["target"], {}).get("tail_type", "OTHER")),
                "anchors": [motif["source"]],
                "hidden_nodes": [motif["mid"], motif["target"]],
                "relation_keys": [
                    _relation_slug(motif.get("rel_1", "")),
                    _relation_slug(motif.get("rel_2", "")),
                ],
            }
        if mtype == "comparative":
            return {
                "program_type": "compare",
                "operation": "compare_values",
                "target_type": motif.get("comparison_type", "OTHER"),
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["value_a"], motif["value_b"]],
                "relation_keys": [motif.get("relation", "")],
            }
        if mtype == "delta_time":
            return {
                "program_type": "delta",
                "operation": "time_difference",
                "target_type": "TIME",
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["value_a"], motif["value_b"]],
                "relation_keys": [motif.get("relation", "")],
            }
        if mtype == "delta_quantity":
            return {
                "program_type": "delta",
                "operation": "quantity_difference",
                "target_type": "QUANTITY",
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["value_a"], motif["value_b"]],
                "relation_keys": [motif.get("relation", "")],
            }
        if mtype == "sum_threshold_quantity":
            return {
                "program_type": "threshold",
                "operation": "sum_then_compare",
                "target_type": "QUANTITY",
                "anchors": [motif["entity_a"], motif["entity_b"], motif["entity_c"]],
                "hidden_nodes": [motif["value_a"], motif["value_b"], motif["value_c"]],
                "relation_keys": [motif.get("relation", "")],
            }
        if mtype == "select_then_follow":
            return {
                "program_type": "path_reasoning",
                "operation": "select_then_follow",
                "target_type": motif.get("follow_tail_type", "OTHER"),
                "selection_type": motif.get("selection_type", "OTHER"),
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [
                    motif["winner"],
                    motif.get("follow_target"),
                    motif.get("value_a"),
                    motif.get("value_b"),
                ],
                "relation_keys": [
                    motif.get("comparison_relation", ""),
                    _relation_slug(motif.get("follow_relation", "")),
                ],
            }
        if mtype == "join_then_follow":
            return {
                "program_type": "path_reasoning",
                "operation": "join_then_follow",
                "target_type": motif.get("follow_tail_type", "OTHER"),
                "shared_target_type": motif.get("shared_target_type", "OTHER"),
                "anchors": [motif["entity_a"], motif["entity_b"]],
                "hidden_nodes": [motif["shared_target"], motif["follow_target"]],
                "relation_keys": [
                    _relation_slug(motif.get("rel_a", "")),
                    _relation_slug(motif.get("rel_b", "")),
                    _relation_slug(motif.get("follow_relation", "")),
                ],
            }
        return {
            "program_type": mtype,
            "operation": "unknown",
            "target_type": "OTHER",
            "anchors": [],
            "hidden_nodes": [],
            "relation_keys": [],
        }

    program = _program()

    if mtype == "lookup":
        s, t = motif["source"], motif["target"]
        rel = motif.get("relation", "相关信息")
        op_info = _lookup_operation_info(rel, motif.get("tail_type", "OTHER"))
        return {
            "motif_type": "single_entity_lookup",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(s),
            },
            "reasoning_graph": {
                "operation": op_info["question_goal"],
                "step_1": f"{_obf(s)} →[{rel}]→ [答案]",
                "step_hint": op_info["comparison_hint"],
                "fact": motif.get("fact", ""),
            },
            "target_answer": f"{op_info['target_prefix']}: {_name(t)}",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(s)}"},
                {"tool": "web_search",
                 "reason": op_info["search_reason"]},
            ],
        }

    if mtype == "extremum":
        entities = motif.get("entities", [])
        values = motif.get("values", [])
        facts = motif.get("facts", [])
        comparison_type = motif.get("comparison_type", "OTHER")
        relation = motif.get("relation", "相关属性")

        visual_anchors = {
            f"Entity_{chr(ord('A') + idx)}": _obf(key)
            for idx, key in enumerate(entities)
        }
        reasoning_graph = {}
        for idx, (entity_key, value_key) in enumerate(zip(entities, values)):
            label = chr(ord("A") + idx)
            reasoning_graph[f"query_{label}"] = (
                f"{_obf(entity_key)} →[{relation}]→ {_name(value_key)}"
            )
            if idx < len(facts) and facts[idx]:
                reasoning_graph[f"fact_{label.lower()}"] = facts[idx]
        reasoning_graph["comparison"] = (
            f"在这些图中实体里，比较它们的{_EXTREMUM_COMPARE_TEXT.get(comparison_type, '相关属性')}"
        )

        return {
            "motif_type": "set_extremum",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": visual_anchors,
            "reasoning_graph": reasoning_graph,
            "target_answer": (
                f"{_EXTREMUM_TARGET_PREFIX.get(comparison_type, '集合极值')}: "
                + " vs ".join(f"{_name(e)}({_name(v)})" for e, v in zip(entities, values))
            ),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": "裁剪识别这些图中实体"},
                {"tool": "web_search",
                 "reason": f"分别搜索这些实体的 [{relation}]"},
                {"tool": "code_interpreter",
                 "reason": "编写 Python 代码比较这些值并找出极值"},
            ],
        }

    if mtype == "same_target":
        a, b, t = motif["entity_a"], motif["entity_b"], motif["target"]
        target_type = motif.get("target_type", "OTHER")
        if target_type == "PERSON":
            op_text = "判断这两个相关人物是否是同一个人"
            target_prefix = "同一人物"
        elif target_type == "ORG":
            op_text = "判断这两个关联组织是否是同一个组织"
            target_prefix = "同一组织"
        elif target_type == "LOCATION":
            op_text = "判断这两个关联地点是否属于同一地点/同一地区"
            target_prefix = "同一地点"
        else:
            op_text = "判断两者是否共享同一目标"
            target_prefix = "共享目标"

        return {
            "motif_type": "same_target_relation",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "operation": op_text,
                "query_A": f"{_obf(a)} →[{motif['rel_a']}]→ [隐藏目标]",
                "query_B": f"{_obf(b)} →[{motif['rel_b']}]→ [隐藏目标]",
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
            },
            "target_answer": f"{target_prefix}: 是，都是 {_name(t)}",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索 [{motif['rel_a']}] 和 [{motif['rel_b']}]，判断是否指向同一{target_prefix}"},
            ],
        }

    if mtype == "same_group":
        a, b, p = motif["entity_a"], motif["entity_b"], motif["parent"]
        parent_type = motif.get("parent_type", "OTHER")
        if parent_type == "LOCATION":
            op_text = "判断两者是否归属于同一地点层级（如同一国家/州/地区）"
            target_prefix = "同一地点归属"
        elif parent_type == "ORG":
            op_text = "判断两者是否归属于同一上级组织/集团"
            target_prefix = "同一组织归属"
        else:
            op_text = "判断两者是否归属于同一上级目标"
            target_prefix = "同一归属"

        return {
            "motif_type": "same_group_relation",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "operation": op_text,
                "path_A": (
                    f"{_obf(a)} →[{motif['rel_a_1']}]→ [隐藏中间节点A] "
                    f"→[{motif['rel_a_2']}]→ [共享上级]"
                ),
                "path_B": (
                    f"{_obf(b)} →[{motif['rel_b_1']}]→ [隐藏中间节点B] "
                    f"→[{motif['rel_b_2']}]→ [共享上级]"
                ),
                "fact_a_1": motif.get("fact_a_1", ""),
                "fact_a_2": motif.get("fact_a_2", ""),
                "fact_b_1": motif.get("fact_b_1", ""),
                "fact_b_2": motif.get("fact_b_2", ""),
            },
            "target_answer": f"{target_prefix}: 是，都属于 {_name(p)}",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": "分别搜索两个实体的相关地点/组织信息，并继续追溯其上级归属"},
                {"tool": "code_interpreter",
                 "reason": "整理两条归属链并判断是否归于同一上级目标"},
            ],
        }

    if mtype == "bridge":
        a, b, t = motif["source_a"], motif["source_b"], motif["target"]
        return {
            "motif_type": "bridge_intersection",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "condition_1": f"{_obf(a)} →[{motif['rel_a']}]→ [隐藏目标]",
                "condition_2": f"{_obf(b)} →[{motif['rel_b']}]→ [隐藏目标]",
                "fact_1": motif.get("fact_a", ""),
                "fact_2": motif.get("fact_b", ""),
            },
            "target_answer": _name(t),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索 [{motif['rel_a']}] 和 [{motif['rel_b']}]，寻找共同交集"},
            ],
        }

    elif mtype == "multihop":
        s, m, t = motif["source"], motif["mid"], motif["target"]
        disc = motif.get("discriminator")
        disc_text = ""
        if disc:
            disc_text = f"{_name(m)} →[{disc['relation']}]→ {disc['node_name']}"
        return {
            "motif_type": "multihop_path",
            "difficulty": "L3",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(s),
            },
            "reasoning_graph": {
                "step_1": f"{_obf(s)} →[{motif['rel_1']}]→ [隐藏中间节点]",
                "step_2": f"[隐藏中间节点] →[{motif['rel_2']}]→ [答案]",
                "fact_1": motif.get("fact_1", ""),
                "fact_2": motif.get("fact_2", ""),
                "discriminator": disc_text,
            },
            "target_answer": _name(t),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(s)}"},
                {"tool": "web_search",
                 "reason": f"搜索 [{motif['rel_1']}] 找到中间节点"},
                {"tool": "visit",
                 "reason": f"访问详情页查询 [{motif['rel_2']}]"},
            ],
        }

    elif mtype == "comparative":
        a, b = motif["entity_a"], motif["entity_b"]
        comparison_type = motif.get("comparison_type", "OTHER")
        rel_a = motif.get("relation_a") or motif.get("relation", "")
        rel_b = motif.get("relation_b") or motif.get("relation", "")
        rel_a_text = rel_a or "相关属性"
        rel_b_text = rel_b or "相关属性"

        if comparison_type == "TIME":
            comparison_text = "比较两者时间先后（更早/更晚）"
            target_prefix = "时间对比"
            compare_reason = "编写 Python 代码解析年份/日期并比较先后，得出结论"
        elif comparison_type == "QUANTITY":
            comparison_text = "比较两者数值大小（必要时统一单位）"
            target_prefix = "数值对比"
            compare_reason = "编写 Python 代码做单位换算与数值比较，得出结论"
        else:
            comparison_text = f"比较两者的 [{motif.get('relation', '相关属性')}]"
            target_prefix = "属性对比"
            compare_reason = "编写 Python 代码比较两个数值，得出结论"

        if rel_a_text == rel_b_text:
            search_reason = f"分别搜索两个实体的 [{rel_a_text}]"
        else:
            search_reason = f"分别搜索两个实体的属性（A: [{rel_a_text}], B: [{rel_b_text}]）"

        return {
            "motif_type": "comparative",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "query_A": f"{_obf(a)} →[{rel_a_text}]→ {_name(motif['value_a'])}",
                "query_B": f"{_obf(b)} →[{rel_b_text}]→ {_name(motif['value_b'])}",
                "comparison": comparison_text,
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
            },
            "target_answer": (
                f"{target_prefix}: "
                f"{_name(a)}({_name(motif['value_a'])}) vs {_name(b)}({_name(motif['value_b'])})"
            ),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": search_reason},
                {"tool": "code_interpreter",
                 "reason": compare_reason},
            ],
        }

    elif mtype == "delta_time":
        a, b = motif["entity_a"], motif["entity_b"]
        rel = motif.get("relation", "时间属性")
        year_a = motif.get("year_a")
        year_b = motif.get("year_b")
        delta_years = motif.get("delta_years")
        return {
            "motif_type": "delta_time",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "query_A": f"{_obf(a)} →[{rel}]→ {year_a}",
                "query_B": f"{_obf(b)} →[{rel}]→ {year_b}",
                "comparison": "计算两个时间值之间相差多少年",
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
            },
            "target_answer": f"时间差: {delta_years} 年（{_name(a)}={year_a}, {_name(b)}={year_b}）",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索两个实体的 [{rel}]"},
                {"tool": "code_interpreter",
                 "reason": "编写 Python 代码计算两个年份之间的差值"},
            ],
        }

    elif mtype == "delta_quantity":
        a, b = motif["entity_a"], motif["entity_b"]
        rel = motif.get("relation", "数值属性")
        delta_value = motif.get("delta_value")
        unit_signature = motif.get("unit_signature", "")
        unit_text = unit_signature or "同单位数值"
        delta_text = _format_compact_quantity(delta_value, unit_signature)
        return {
            "motif_type": "delta_quantity",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "query_A": f"{_obf(a)} →[{rel}]→ {_name(motif['value_a'])}",
                "query_B": f"{_obf(b)} →[{rel}]→ {_name(motif['value_b'])}",
                "comparison": f"计算两个{unit_text}数值之间相差多少",
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
            },
            "target_answer": (
                f"数值差: {delta_text}"
                + f"（{_name(a)}={_name(motif['value_a'])}, {_name(b)}={_name(motif['value_b'])}）"
            ),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索两个实体的 [{rel}]"},
                {"tool": "code_interpreter",
                 "reason": "编写 Python 代码解析两个数值并计算差值"},
            ],
        }

    elif mtype == "sum_threshold_quantity":
        a, b, c = motif["entity_a"], motif["entity_b"], motif["entity_c"]
        rel = motif.get("relation", "数值属性")
        unit_signature = motif.get("unit_signature", "")
        unit_text = unit_signature or "同单位数值"
        sum_text = _format_compact_quantity(motif.get("sum_value"), unit_signature)
        threshold_text = _format_compact_quantity(motif.get("numeric_c"), unit_signature)
        verdict = "是" if motif.get("exceeds") else "否"
        return {
            "motif_type": "sum_threshold_quantity",
            "difficulty": "L2",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
                "Entity_C": _obf(c),
            },
            "reasoning_graph": {
                "query_A": f"{_obf(a)} →[{rel}]→ {_name(motif['value_a'])}",
                "query_B": f"{_obf(b)} →[{rel}]→ {_name(motif['value_b'])}",
                "query_C": f"{_obf(c)} →[{rel}]→ {_name(motif['value_c'])}",
                "comparison": f"计算前两个{unit_text}数值之和，并判断是否超过第三个值",
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
                "fact_c": motif.get("fact_c", ""),
            },
            "target_answer": (
                f"求和阈值判断: {verdict}，"
                f"{sum_text} {'>' if motif.get('exceeds') else '<'} {threshold_text}"
            ),
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)}、{_obf(b)} 和 {_obf(c)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索三个实体的 [{rel}]"},
                {"tool": "code_interpreter",
                 "reason": "编写 Python 代码解析三个数值，计算前两个之和并与第三个比较"},
            ],
        }

    elif mtype == "select_then_follow":
        a, b = motif["entity_a"], motif["entity_b"]
        winner = motif["winner"]
        selection_type = motif.get("selection_type", "TIME")
        compare_rel = motif.get("comparison_relation", "相关属性")
        follow_rel = motif.get("follow_relation", "相关信息")
        winner_value = motif.get("winner_value")
        loser_value = motif.get("loser_value")
        if selection_type == "TIME":
            step_1 = "先比较两个图中实体的时间先后，选出更早的那个实体"
            compare_reason = "编写 Python 代码比较两个年份/日期，确定哪一个实体满足条件"
        else:
            step_1 = "先比较两个图中实体的数值大小，选出数值更高的那个实体"
            compare_reason = "编写 Python 代码比较两个数值，确定哪一个实体满足条件"

        return {
            "motif_type": "path_reasoning_select_then_follow",
            "difficulty": "L3",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "step_1": (
                    f"{_obf(a)} 与 {_obf(b)} 比较 [{compare_rel}]，"
                    f"选出满足条件的那个实体"
                ),
                "step_2": (
                    f"对上一步选中的实体继续查询 [{follow_rel}]，"
                    f"得到最终答案"
                ),
                "comparison_detail": (
                    f"{_name(a)}={winner_value if winner == a else loser_value}, "
                    f"{_name(b)}={winner_value if winner == b else loser_value}"
                ),
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
                "follow_fact": motif.get("follow_fact", ""),
            },
            "target_answer": f"路径推理答案: {_name(motif['follow_target'])}",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索两个实体的 [{compare_rel}]"},
                {"tool": "code_interpreter",
                 "reason": compare_reason},
                {"tool": "web_search",
                 "reason": f"继续搜索被选中实体的 [{follow_rel}]"},
            ],
        }

    elif mtype == "join_then_follow":
        a, b = motif["entity_a"], motif["entity_b"]
        shared = motif["shared_target"]
        follow_rel = motif.get("follow_relation", "相关信息")
        rel_a = motif.get("rel_a", "相关关系")
        rel_b = motif.get("rel_b", "相关关系")
        shared_phrase = _join_shared_target_phrase(rel_a, rel_b, motif.get("shared_target_type", "OTHER"))
        return {
            "motif_type": "path_reasoning_join_then_follow",
            "difficulty": "L3",
            "program": program,
            "visual_anchors": {
                "Entity_A": _obf(a),
                "Entity_B": _obf(b),
            },
            "reasoning_graph": {
                "step_1": (
                    f"{_obf(a)} 和 {_obf(b)} 分别追查后，汇合到同一个[{shared_phrase}]"
                ),
                "step_2": (
                    f"再从这个共享目标继续查询 [{follow_rel}]，"
                    f"得到最终答案"
                ),
                "shared_hint": shared_phrase,
                "fact_a": motif.get("fact_a", ""),
                "fact_b": motif.get("fact_b", ""),
                "follow_fact": motif.get("follow_fact", ""),
            },
            "target_answer": f"路径推理答案: {_name(motif['follow_target'])}",
            "tool_plan": [
                {"tool": "code_interpreter",
                 "reason": f"裁剪识别 {_obf(a)} 和 {_obf(b)}"},
                {"tool": "web_search",
                 "reason": f"分别搜索两个实体的 [{rel_a}] 和 [{rel_b}]，找出共享目标"},
                {"tool": "web_search",
                 "reason": f"继续搜索共享目标的 [{follow_rel}]"},
            ],
        }

    raise ValueError(f"未知 motif_type: {mtype}")


# ============================================================
# 3f. LLM 润色
# ============================================================

TOOLS_BLOCK = """\
## 可用工具（只能使用以下4种，禁止使用其他名称）
1. web_search(query) — 搜索网络获取文本信息
2. image_search(search_type, query/image_url) — 文字搜图或以图搜图
3. visit(url, goal) — 访问网页提取内容
4. code_interpreter(code) — 执行Python代码（OCR、裁剪、计算等所有图像处理操作）

⚠️ tool 字段只能填写上述4个名称之一，严禁使用 crop/ocr/reverse_image_search/count 等名称。"""

OUTPUT_FORMAT = """\
## 输出格式
请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "questions": [
        {{
            "skeleton_index": 0,
            "question": "润色后的问题",
            "answer": "答案",
            "tool_sequence": [
                {{"step": 1, "tool": "工具名", "action": "操作描述", "input": "输入内容", "expected_output": "预期输出"}}
            ],
            "obfuscated_entities": ["被模糊化的图中实体的位置描述"],
            "rationale": "简述推理链和工具必要性"
        }}
    ]
}}"""

# ── Motif 类型专属指令 ──
MOTIF_RULES = {
    "single_entity_lookup": """\
**推理拓扑：单实体查找（L2）**
图中一个实体通过一条高价值事实边直接连接到答案。
- 问题结构要根据答案类型自适应：
  人物类可问“相关人物是谁”，地点类可问“位于哪里/相关地点是哪里”，时间类可问“发生于何时/哪一年”，数值类可问“数值是多少”
- [图中: ...] 实体只能用位置描述引用，禁止写出真实名称
- 工具链：先识别视觉实体，再搜索该实体的目标属性""",

    "set_extremum": """\
**推理拓扑：集合极值（L2）**
图中多个实体拥有同类时间/数值属性，问题要求找出极值。
- 问题结构：「图中这些实体里，哪个[属性]最早/最高/最大？」
- 所有 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 工具链：先识别多个视觉实体 → 分别搜索属性 → 用 code_interpreter 比较后得出极值""",

    "same_target_relation": """\
**推理拓扑：同一目标判断（L2）**
两个图中实体分别通过各自关系指向同一个人物/组织/地点。
- 问题结构：「图中A的某关系对象，和图中B的某关系对象，是否是同一个人/组织/地点？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- answer 不要只写“是”，要写成“是，都是 XXX”""",

    "same_group_relation": """\
**推理拓扑：同组归属判断（L2）**
两个图中实体分别先连到各自的中间地点/组织，再归属于同一个上级地点/组织。
- 问题结构：「图中A和图中B，是否属于同一个国家/地区/集团/母组织？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 不要直接泄露中间节点名称，应该把它们当作隐藏的归属链
- answer 不要只写“是”，要写成“是，都属于 XXX”""",

    "delta_time": """\
**推理拓扑：时间差计算（L2）**
两个图中实体拥有可解析为年份的时间属性，问题要求算出差值。
- 问题结构：「图中A和图中B在某时间属性上相差多少年？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 工具链：识别两个视觉实体 → 分别搜索时间属性 → code_interpreter 计算差值""",

    "delta_quantity": """\
**推理拓扑：数值差计算（L2）**
两个图中实体拥有同类且可比较的数值属性，问题要求算出差值。
- 问题结构：「图中A和图中B在某数值属性上相差多少？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 如果骨架里带有单位，应自然体现在问题或答案中
- 工具链：识别两个视觉实体 → 分别搜索数值属性 → code_interpreter 解析数值并计算差值""",

    "sum_threshold_quantity": """\
**推理拓扑：数量求和阈值判断（L2）**
三个图中实体拥有同类且可比较的数值属性，问题要求判断前两个的和是否超过第三个。
- 问题结构：「图中A和图中B的某数值加起来，是否超过图中C的对应数值？」
- 三个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- answer 要写出明确判断，并给出关键比较结果
- 工具链：识别三个视觉实体 → 分别搜索数值属性 → code_interpreter 求和并判断阈值""",

    "bridge_intersection": """\
**推理拓扑：交集桥接（L2）**
两个图中实体通过各自的关系共同指向同一个隐藏答案。
- 问题应呈现为「满足条件一 AND 条件二的是什么」结构
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出名称
- 工具链：先识别两个视觉实体 → 分别搜索各自的关系 → 找出共同交集""",

    "multihop_path": """\
**推理拓扑：深度跳跃（L3）**
图中实体 → 隐藏中间节点 → 最终答案，中间节点有额外属性可用于限定。
- 骨架中 discriminator（限定词）可以嵌入问题作为定语从句，增加唯一性
  例："创立于1990年的[隐藏中间节点]的现任CEO是谁？"
- [图中: ...] 实体用位置描述，[隐藏中间节点] 用关系描述，禁止写出名称
- 工具链：识别视觉实体 → web_search 找中间节点 → visit 深读获取答案""",

    "path_reasoning_select_then_follow": """\
**推理拓扑：路径依赖选择后继续（L3）**
先在两个图中实体之间做一次比较/筛选，再对被选中的那个实体继续追问下一步属性。
- 题目必须体现“先选，再查”这两个步骤，不能退化成普通比较题或普通单跳题
- 问题结构类似：「图中A和图中B中，满足某条件的那个实体，其[下一步属性]是什么？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 不能直接泄露被选中的实体名称；必须让求解者先完成第一步选择
- 工具链：识别两个视觉实体 → 搜索并比较属性 → 确定被选中的实体 → 再搜索该实体的下一步属性""",

    "path_reasoning_join_then_follow": """\
**推理拓扑：路径依赖汇合后继续（L3）**
两个图中实体先分别连到同一个隐藏目标，再从这个共享目标继续追问下一步属性。
- 题目必须体现“先找到共同点，再继续往下查”这两个步骤，但第一步可以隐含在自然表达里
- 优先使用自然短语，如“共同关联的演出地点 / 共同关联的创始人 / 共同关联的组织”，不要写“隐藏目标”“共享目标”“那处地点”
- 问题结构类似：「图中A和图中B共同关联的演出地点，位于哪座城市？」或「图中A和图中B共同关联的创始人，出生于哪里？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出真实名称
- 不能直接泄露共享目标的名字；必须让求解者先完成汇合步骤
- 工具链：识别两个视觉实体 → 分别搜索各自关系 → 找出共享目标 → 再搜索共享目标的下一步属性""",

    "comparative": """\
**推理拓扑：平行对比（L2）**
两个图中实体拥有同类可比属性，问题要求比较并得出结论。
- 问题结构：「图中X和图中Y，哪个[关系]更[大/早/多]？」
- 两个 [图中: ...] 实体都用位置描述引用，禁止写出名称
- answer 字段：写出明确的比较结论（例："[Entity_A]，创立于1940年，早于[Entity_B]的1960年"）
- 工具链：识别两个视觉实体 → 分别搜索 → code_interpreter 做数值比较""",

    "L1": """\
**级别：L1（单跳知识题）**
- 骨架中标记 [图中: ...] 的实体用自然的位置描述引用，禁止还原为实体名
- 工具链已在骨架中预规划，严格按预规划填写""",
}

POLISH_PROMPT = """\
你需要将问题骨架润色为自然流畅的问题，并规划工具使用序列。

## 图片信息
- 描述：{image_description}
- 领域：{domain}

{tools_block}

## {motif_rules}

## 问题骨架（共 {n_skeletons} 个）

{skeletons_text}

## 通用要求
1. 每个问题**只能有一个问号**。用「铺垫句（陈述背景）+ 问句」或定语从句结构。
2. [图中: ...] 用自然位置描述引用，**绝对禁止写出真实名称**
3. [隐藏] / [隐藏中间节点] 用关系描述引用，**绝对禁止写出真实名称**
4. 严格按骨架预规划的工具链填写 tool_sequence，禁止添加额外步骤

{output_format}"""


VISION_PROMPT = """\
请基于这张图片生成4道纯视觉题（不需要外部知识），每道题只需一种工具即可回答。

## 图片信息
- 描述：{image_description}
- 领域：{domain}
- 图中实体：{entities_brief}

## 可用工具（只能使用以下4种，禁止使用其他名称）
1. code_interpreter(code) — 执行Python代码，包括OCR识别文字、裁剪放大图像区域、计数物体、颜色分析等一切图像处理操作
2. web_search(query) — 搜索网络获取文本信息
3. visit(url, goal) — 访问网页提取内容
4. image_search(search_type, query/image_url) — 以图搜图或文字搜图，用于识别图中未知的视觉元素（人物、品牌logo、地标等）

⚠️ 严禁使用 crop / ocr / OCR / reverse_image_search / count_objects / count / color_detection / Color Analysis / image_processing 等名称！
   所有图像处理操作（裁剪、OCR、计数、颜色检测等）统一写在 code_interpreter 的 action 中。

## 要求
- 2道 code_interpreter 题：需要OCR/裁剪/放大/计数才能回答，答案完全在图中
- 2道 image_search 题：需要对图中某个视觉元素做反向图搜来识别
- 答案必须具体明确，禁止使用"或""类似"等模糊表述
- 答案不能在问题文本中出现或被暗示

## 输出格式
请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "questions": [
        {{
            "question": "问题文本",
            "answer": "答案",
            "tool_type": "code_interpreter 或 image_search",
            "tool_sequence": [
                {{"step": 1, "tool": "code_interpreter 或 image_search（只能二选一）", "action": "操作描述", "expected_output": "预期输出"}}
            ],
            "entities_involved": ["相关实体ID"],
            "rationale": "简述推理过程"
        }}
    ]
}}"""


def _format_skeletons(skeletons: list) -> str:
    """将 motif 骨架列表格式化为 LLM 可读文本。"""
    lines = []
    for i, sk in enumerate(skeletons):
        lines.append(f"### 骨架 {i}")
        mtype = sk.get("motif_type", "unknown")
        lines.append(f"  类型: {mtype}  难度: {sk.get('difficulty', '?')}")
        program = sk.get("program", {})
        if program:
            lines.append(
                "  程序: "
                f"{program.get('program_type', '?')} / {program.get('operation', '?')} / "
                f"target={program.get('target_type', '?')}"
            )

        anchors = sk.get("visual_anchors", {})
        if anchors:
            for k, v in anchors.items():
                lines.append(f"  视觉锚点 {k}: {v}")

        rg = sk.get("reasoning_graph", {})
        for k, v in rg.items():
            if v:
                lines.append(f"  推理步骤 {k}: {v}")

        lines.append(f"  答案: {sk.get('target_answer', '?')}")

        plan = sk.get("tool_plan", [])
        if plan:
            steps = " → ".join(f"{s['tool']}({s['reason']})" for s in plan)
            lines.append(f"  工具链: {steps}")

        lines.append("")
    return "\n".join(lines)


def polish_level(
    skeletons: list,
    motif_type: str,
    image_b64: str,
    image_description: str,
    domain: str,
) -> list:
    """对一批同类型 motif 骨架调用 LLM 润色，返回问题列表。"""
    def _fallback_questions() -> list:
        if motif_type not in {"path_reasoning_select_then_follow", "path_reasoning_join_then_follow"}:
            return []

        fallback = []
        for idx, sk in enumerate(skeletons):
            anchors = sk.get("visual_anchors", {})
            program = sk.get("program", {}) or {}
            relation_keys = program.get("relation_keys", [])
            target_type = program.get("target_type", "OTHER")
            answer_hint = _path_answer_prompt(target_type)
            answer_text = sk.get("target_answer", "")
            if "路径推理答案:" in answer_text:
                answer_text = answer_text.split("路径推理答案:", 1)[1].strip()

            if motif_type == "path_reasoning_select_then_follow":
                compare_rel = relation_keys[0] if len(relation_keys) > 0 else "相关属性"
                follow_rel = relation_keys[1] if len(relation_keys) > 1 else "下一步属性"
                compare_phrase = _selection_phrase(program.get("selection_type", target_type), compare_rel)
                follow_clause = _path_follow_clause(follow_rel, target_type)
                question_text = (
                    f"图中{anchors.get('Entity_A', '[图中实体A]')}和{anchors.get('Entity_B', '[图中实体B]')}"
                    f"所对应的两个实体里，{compare_phrase}的那个，{follow_clause}？"
                )
                tool_sequence = [
                    {"step": 1, "tool": "code_interpreter", "action": "识别两个图中实体", "input": "图片中的两个位置锚点", "expected_output": "两个候选实体名称"},
                    {"step": 2, "tool": "web_search", "action": f"搜索两个实体的{_relation_natural_text(compare_rel, program.get('selection_type', 'OTHER'))}", "input": "两个实体名", "expected_output": "可比较的属性值"},
                    {"step": 3, "tool": "code_interpreter", "action": "比较两个属性值并确定满足条件的实体", "input": "上一步结果", "expected_output": "被选中的实体"},
                    {"step": 4, "tool": "web_search", "action": f"搜索被选中实体的{_relation_natural_text(follow_rel, target_type)}", "input": "被选中的实体名", "expected_output": "最终答案"},
                ]
                rationale = "先比较两个图中实体，确定满足条件的那个，再继续查询它的下一步属性。"
            else:
                rel_a = relation_keys[0] if len(relation_keys) > 0 else "关系A"
                rel_b = relation_keys[1] if len(relation_keys) > 1 else "关系B"
                follow_rel = relation_keys[2] if len(relation_keys) > 2 else "下一步属性"
                shared_type = program.get("shared_target_type", "OTHER")
                shared_phrase = _join_shared_target_phrase(rel_a, rel_b, shared_type)
                follow_clause = _join_follow_clause(rel_a, rel_b, shared_type, follow_rel, target_type, answer_text)
                question_text = (
                    f"图中{anchors.get('Entity_A', '[图中实体A]')}和{anchors.get('Entity_B', '[图中实体B]')}"
                    f"{shared_phrase}，{follow_clause}？"
                )
                tool_sequence = [
                    {"step": 1, "tool": "code_interpreter", "action": "识别两个图中实体", "input": "图片中的两个位置锚点", "expected_output": "两个候选实体名称"},
                    {"step": 2, "tool": "web_search", "action": f"搜索第一个实体的{_relation_natural_text(rel_a, shared_type)}和第二个实体的{_relation_natural_text(rel_b, shared_type)}", "input": "两个实体名", "expected_output": "共享目标"},
                    {"step": 3, "tool": "web_search", "action": f"搜索共享目标的{_relation_natural_text(follow_rel, target_type)}", "input": "共享目标", "expected_output": "最终答案"},
                ]
                rationale = "先找到两个图中实体共同关联的隐藏目标，再继续查询该共享目标的下一步属性。"

            fallback.append({
                "skeleton_index": idx,
                "question": question_text,
                "answer": answer_text,
                "tool_sequence": tool_sequence,
                "obfuscated_entities": list(anchors.values()),
                "rationale": rationale,
                "reasoning_path": {
                    "motif_type": sk.get("motif_type"),
                    "program": sk.get("program", {}),
                    "visual_anchors": sk.get("visual_anchors", {}),
                    "reasoning_graph": sk.get("reasoning_graph", {}),
                    "answer": sk.get("target_answer"),
                },
                "obfuscation_applied": True,
                "level": difficulty,
            })
        return fallback

    if not skeletons:
        return []

    difficulty = skeletons[0].get("difficulty", "L2")
    motif_rules = MOTIF_RULES.get(motif_type, MOTIF_RULES.get(difficulty, ""))

    prompt = POLISH_PROMPT.format(
        image_description=image_description,
        domain=domain,
        tools_block=TOOLS_BLOCK,
        motif_rules=motif_rules,
        n_skeletons=len(skeletons),
        skeletons_text=_format_skeletons(skeletons),
        output_format=OUTPUT_FORMAT,
    )

    result = call_vlm_json(
        prompt,
        f"请根据图片和问题骨架，生成 {difficulty} 级别的问题。严格按骨架的推理拓扑出题。",
        image_b64=image_b64,
        max_tokens=4096,
        temperature=0.5,
        max_attempts=3,
    )

    if result is None:
        logger.warning(f"  {motif_type} 润色失败")
        return _fallback_questions()

    questions = result.get("questions", [])
    if not questions:
        return _fallback_questions()
    for q in questions:
        idx = q.get("skeleton_index", 0)
        if idx < len(skeletons):
            sk = skeletons[idx]
            q["reasoning_path"] = {
                "motif_type": sk.get("motif_type"),
                "program": sk.get("program", {}),
                "visual_anchors": sk.get("visual_anchors", {}),
                "reasoning_graph": sk.get("reasoning_graph", {}),
                "answer": sk.get("target_answer"),
            }
            q["obfuscation_applied"] = True
        q["level"] = difficulty

    return questions


def generate_vision_questions(
    entities: list,
    image_b64: str,
    image_description: str,
    domain: str,
) -> list:
    """生成纯视觉L1题（不依赖知识链）。"""
    entities_brief = "\n".join(
        f"- {e['id']}: {e['name']} ({e.get('type', '?')}) @ {e.get('location_in_image', '?')}"
        for e in entities[:10]
    )

    prompt = VISION_PROMPT.format(
        image_description=image_description,
        domain=domain,
        entities_brief=entities_brief,
    )

    result = call_vlm_json(
        prompt,
        "请基于图片生成4道纯视觉题。",
        image_b64=image_b64,
        max_tokens=2048,
        temperature=0.5,
        max_attempts=2,
    )

    if result is None:
        logger.warning("  纯视觉题生成失败")
        return []

    questions = result.get("questions", [])
    for q in questions:
        q["level"] = "L1"
        q["obfuscation_applied"] = False
        q["reasoning_path"] = {}
    return questions


def _normalize_answer_text(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    return re.sub(r"\s+", " ", answer.strip().lower())


def _question_anchor_signature(question: dict) -> tuple[str, ...]:
    anchors = (
        question.get("reasoning_path", {})
        .get("visual_anchors", {})
    )
    if not isinstance(anchors, dict):
        return ()
    values = [v for _, v in sorted(anchors.items()) if isinstance(v, str) and v.strip()]
    return tuple(values)


def _question_program_anchors(question: dict) -> tuple[str, ...]:
    program = (question.get("reasoning_path", {}) or {}).get("program", {}) or {}
    anchors = program.get("anchors", [])
    if not isinstance(anchors, list):
        return ()
    values = [str(v).strip().lower() for v in anchors if str(v).strip()]
    return tuple(sorted(values))


def _question_pair_signature(question: dict) -> tuple[str, ...]:
    anchors = _question_program_anchors(question)
    if len(anchors) < 2:
        return ()
    return tuple(anchors)


def _selection_state_from_questions(selected_questions: list) -> dict:
    state = {
        "type_counts": defaultdict(int),
        "anchor_counts": defaultdict(int),
        "entity_counts": defaultdict(int),
        "pair_counts": defaultdict(int),
        "seen_answers": set(),
    }
    for question in selected_questions:
        group_name = question.get("_selection_group", "")
        if group_name:
            state["type_counts"][group_name] += 1

        answer_sig = _normalize_answer_text(question.get("answer", ""))
        if answer_sig:
            state["seen_answers"].add(answer_sig)

        anchor_sig = _question_anchor_signature(question)
        if anchor_sig:
            state["anchor_counts"][anchor_sig] += 1

        for entity in _question_program_anchors(question):
            state["entity_counts"][entity] += 1

        pair_sig = _question_pair_signature(question)
        if pair_sig:
            state["pair_counts"][pair_sig] += 1
    return state


def _select_questions(
    grouped_questions: list[tuple[str, list]],
    max_total: int,
    max_per_type: int,
    max_per_anchor: int,
    max_per_entity: int,
    max_per_pair: int,
    seed_questions: list | None = None,
) -> list:
    """跨题型做全局筛选，控制题型、锚点、实体和实体对重复度。"""
    def _question_allowed(question: dict, relaxed: bool = False) -> bool:
        answer_sig = _normalize_answer_text(question.get("answer", ""))
        anchor_sig = _question_anchor_signature(question)
        entity_sig = _question_program_anchors(question)
        pair_sig = _question_pair_signature(question)

        if answer_sig and answer_sig in state["seen_answers"]:
            return False
        if not relaxed and anchor_sig and state["anchor_counts"][anchor_sig] >= max_per_anchor:
            return False
        if entity_sig and any(state["entity_counts"][entity] >= max_per_entity for entity in entity_sig):
            return False
        if pair_sig and state["pair_counts"][pair_sig] >= max_per_pair:
            return False
        return True

    queues = [(group_name, list(qs)) for group_name, qs in grouped_questions if qs]
    if not queues:
        return []

    selected = []
    seed_questions = seed_questions or []
    state = _selection_state_from_questions(seed_questions)

    while queues and len(selected) < max_total:
        progressed = False
        next_round = []

        for group_name, qs in queues:
            if len(selected) >= max_total:
                break
            if state["type_counts"][group_name] >= max_per_type:
                continue

            chosen_idx = None
            for idx, question in enumerate(qs):
                if not _question_allowed(question):
                    continue
                chosen_idx = idx
                break

            if chosen_idx is None:
                if qs:
                    next_round.append((group_name, qs))
                continue

            question = qs.pop(chosen_idx)
            question["_selection_group"] = group_name
            answer_sig = _normalize_answer_text(question.get("answer", ""))
            anchor_sig = _question_anchor_signature(question)
            entity_sig = _question_program_anchors(question)
            pair_sig = _question_pair_signature(question)

            selected.append(question)
            state["type_counts"][group_name] += 1
            if answer_sig:
                state["seen_answers"].add(answer_sig)
            if anchor_sig:
                state["anchor_counts"][anchor_sig] += 1
            for entity in entity_sig:
                state["entity_counts"][entity] += 1
            if pair_sig:
                state["pair_counts"][pair_sig] += 1
            progressed = True

            if qs and state["type_counts"][group_name] < max_per_type:
                next_round.append((group_name, qs))

        if progressed:
            queues = next_round
            continue

        for group_name, qs in queues:
            if len(selected) >= max_total:
                break
            if state["type_counts"][group_name] >= max_per_type or not qs:
                continue
            chosen_idx = None
            for idx, question in enumerate(qs):
                if not _question_allowed(question, relaxed=True):
                    continue
                chosen_idx = idx
                break
            if chosen_idx is None:
                continue

            question = qs.pop(chosen_idx)
            question["_selection_group"] = group_name
            answer_sig = _normalize_answer_text(question.get("answer", ""))
            anchor_sig = _question_anchor_signature(question)
            entity_sig = _question_program_anchors(question)
            pair_sig = _question_pair_signature(question)
            selected.append(question)
            state["type_counts"][group_name] += 1
            if answer_sig:
                state["seen_answers"].add(answer_sig)
            if anchor_sig:
                state["anchor_counts"][anchor_sig] += 1
            for entity in entity_sig:
                state["entity_counts"][entity] += 1
            if pair_sig:
                state["pair_counts"][pair_sig] += 1

        break

    return selected


# ============================================================
# 兼容旧数据：从 knowledge_chains 提取三元组
# ============================================================
def triples_from_legacy_chains(knowledge_chains: list) -> list:
    """将旧格式的 knowledge_chains 转换为三元组列表。"""
    triples = []
    for group in knowledge_chains:
        root_name = group.get("root_entity", "")
        for chain in group.get("chains", []):
            prev_name = root_name
            for hop in chain.get("hops", []):
                target = hop.get("target_entity", "")
                if not target:
                    continue
                triples.append({
                    "head": prev_name,
                    "relation": hop.get("relation", ""),
                    "tail": target,
                    "tail_type": "OTHER",
                    "fact": hop.get("fact", ""),
                    "source_snippet": hop.get("source_snippet", ""),
                })
                prev_name = target
    return _sanitize_triples(triples)


# ============================================================
# 单张图片处理
# ============================================================
def generate_questions(entity_file: str) -> dict | None:
    with open(entity_file, encoding="utf-8") as f:
        entity_data = json.load(f)

    img_id = entity_data["img_id"]
    img_path = os.path.join(FILTERED_IMAGE_DIR, f"{img_id}.jpg")

    if is_done(3, img_id):
        logger.info(f"  [{img_id}] 已有检查点，跳过")
        return load_checkpoint(3, img_id)

    if not os.path.exists(img_path):
        for ext in [".jpeg", ".png", ".webp"]:
            alt = os.path.join(FILTERED_IMAGE_DIR, f"{img_id}{ext}")
            if os.path.exists(alt):
                img_path = alt
                break
        else:
            logger.warning(f"  [{img_id}] 图片不存在: {img_path}")
            return None

    logger.info(f"  [{img_id}] 开始生成问题...")
    try:
        image_b64 = file_to_b64(img_path)
    except Exception as e:
        logger.error(f"  [{img_id}] 图片读取失败: {e}")
        return None

    entities = entity_data.get("entities", [])

    # 获取三元组（兼容新旧格式）
    triples = entity_data.get("triples")
    if triples is None:
        legacy_chains = entity_data.get("knowledge_chains", [])
        if legacy_chains:
            triples = triples_from_legacy_chains(legacy_chains)
            logger.info(f"  [{img_id}] 从旧格式知识链转换出 {len(triples)} 个三元组")
        else:
            triples = []
    triples = _sanitize_triples(triples)

    if not triples:
        logger.warning(f"  [{img_id}] 无三元组，跳过")
        return None

    image_desc = entity_data.get("image_description", "")
    domain = entity_data.get("domain", "other")
    from step3_graphenv_runtime import (
        GraphEnv,
        build_question_programs,
        realize_programs,
        select_programs_for_levels,
    )
    from experimental.random_walk_step3.runtime import build_randomwalk_shadow_result

    # ---- 3a. GraphEnv 建图 + QuestionProgram 提议 ----
    G, nodes = _build_nx_graph(triples, entities)
    programs, program_meta = build_question_programs(G, nodes)
    l2_programs, l3_programs, level_meta = select_programs_for_levels(programs, G, nodes)
    env = GraphEnv(G, nodes)

    logger.info(
        f"  [{img_id}] GraphEnv 候选: raw={program_meta.get('raw_program_count', 0)} "
        f"accepted={program_meta.get('accepted_program_count', 0)} "
        f"L2={len(l2_programs)}/{level_meta.get('level_2_raw_count', 0)} "
        f"L3={len(l3_programs)}/{level_meta.get('level_3_raw_count', 0)}"
    )

    # ---- 3b. 并行语言化与 L1 ----
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_l2 = pool.submit(realize_programs, l2_programs, env, image_b64, image_desc, domain)
        fut_l3 = pool.submit(realize_programs, l3_programs, env, image_b64, image_desc, domain)
        fut_vis = pool.submit(generate_vision_questions, entities, image_b64, image_desc, domain)
        fut_shadow = pool.submit(
            build_randomwalk_shadow_result,
            G,
            nodes,
            image_b64,
            image_desc,
            domain,
            seed=13,
            walks_per_anchor=32,
            max_steps=10,
            temperature=0.95,
            top_p=0.9,
            epsilon=0.1,
            length_weights=(0.45, 0.35, 0.20),
            length_quota=(2, 2, 1),
            utility_threshold=4.8,
            long_boost_walks=16,
            long_boost_temperature=1.15,
            long_boost_top_p=0.95,
            long_boost_epsilon=0.2,
            max_questions=5,
            best_of_n=3,
            do_realize=True,
        )
        l2_qs = fut_l2.result()
        l3_qs = fut_l3.result()
        all_l1 = fut_vis.result()
        try:
            shadow_result = fut_shadow.result()
        except Exception as e:
            logger.warning(f"  [{img_id}] random walk shadow 生成失败: {e}")
            shadow_result = {"level_2": [], "level_3": [], "metadata": {"shadow_error": str(e)}}

    for i, q in enumerate(all_l1):
        q["question_id"] = f"L1_{i+1:02d}"
    for i, q in enumerate(l2_qs):
        q["question_id"] = f"L2_{i+1:02d}"
    for i, q in enumerate(l3_qs):
        q["question_id"] = f"L3_{i+1:02d}"

    logger.info(
        f"  [{img_id}] 生成完成: L1={len(all_l1)} "
        f"L2={len(l2_qs)}/{level_meta.get('level_2_raw_count', 0)} "
        f"L3={len(l3_qs)}/{level_meta.get('level_3_raw_count', 0)}"
    )

    family_counts = program_meta.get("family_counts", {})
    result = {
        "image_id": img_id,
        "level_1": all_l1,
        "level_2": l2_qs,
        "level_3": l3_qs,
        "metadata": {
            "total_questions": len(all_l1) + len(l2_qs) + len(l3_qs),
            "level_1_count": len(all_l1),
            "level_2_count": len(l2_qs),
            "level_3_count": len(l3_qs),
            "triples_count": len(triples),
            "chains_selected": len(programs),
            "raw_program_count": program_meta.get("raw_program_count", 0),
            "accepted_program_count": program_meta.get("accepted_program_count", 0),
            "rejected_program_count": program_meta.get("rejected_program_count", 0),
            "level_2_raw_count": level_meta.get("level_2_raw_count", 0),
            "level_3_raw_count": level_meta.get("level_3_raw_count", 0),
            "lookup_count": family_counts.get("lookup", 0),
            "compare_count": family_counts.get("compare", 0),
            "delta_count": family_counts.get("delta", 0),
            "same_target_count": family_counts.get("same_target", 0),
            "same_group_count": family_counts.get("same_group", 0),
            "extremum_count": family_counts.get("extremum", 0),
            "threshold_count": family_counts.get("threshold", 0),
            "multihop_count": family_counts.get("multihop", 0),
            "path_reasoning_count": family_counts.get("select_then_follow", 0) + family_counts.get("join_then_follow", 0),
            "selected_l2_families": level_meta.get("selected_l2_families", {}),
            "selected_l3_families": level_meta.get("selected_l3_families", {}),
            "shadow_random_walk_count": len(shadow_result.get("level_2", [])) + len(shadow_result.get("level_3", [])),
        },
    }

    save_checkpoint(3, img_id, result)
    os.makedirs(QUESTION_DIR, exist_ok=True)
    q_file = os.path.join(QUESTION_DIR, f"{img_id}.json")
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    shadow_level_2 = shadow_result.get("level_2", [])
    shadow_level_3 = shadow_result.get("level_3", [])
    for i, q in enumerate(shadow_level_2):
        q["question_id"] = f"L2_{i+1:02d}"
    for i, q in enumerate(shadow_level_3):
        q["question_id"] = f"L3_{i+1:02d}"
    shadow_payload = {
        "image_id": img_id,
        "level_1": all_l1,
        "level_2": shadow_level_2,
        "level_3": shadow_level_3,
        "metadata": {
            "total_questions": len(all_l1) + len(shadow_level_2) + len(shadow_level_3),
            "level_1_count": len(all_l1),
            "level_2_count": len(shadow_level_2),
            "level_3_count": len(shadow_level_3),
            "triples_count": len(triples),
            "shadow_mode": "random_walk_ab_shadow",
            **shadow_result.get("metadata", {}),
        },
    }
    shadow_q_file = os.path.join(QUESTION_DIR, f"{img_id}_randomwalk_shadow.json")
    with open(shadow_q_file, "w", encoding="utf-8") as f:
        json.dump(shadow_payload, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# 聚合最终输出
# ============================================================
def aggregate_final():
    """将所有问题文件聚合为最终 JSONL。"""
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    q_files = sorted(
        path
        for path in glob.glob(os.path.join(QUESTION_DIR, "*.json"))
        if not path.endswith("_randomwalk_shadow.json")
    )
    logger.info(f"聚合 {len(q_files)} 个问题文件...")

    all_l1, all_l2, all_l3 = [], [], []

    for qf in q_files:
        with open(qf, encoding="utf-8") as f:
            data = json.load(f)

        img_id = data.get("image_id", os.path.splitext(os.path.basename(qf))[0])
        img_path = f"output/images/{img_id}.jpg"

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
                    "reasoning_path": q.get("reasoning_path", {}),
                    "entities_involved": q.get("entities_involved", []),
                    "obfuscation_applied": q.get("obfuscation_applied", False),
                    "obfuscated_entities": q.get("obfuscated_entities", []),
                    "domain": data.get("domain", "other"),
                    "rationale": q.get("rationale", ""),
                    "metadata": {
                        "generation_model": MODEL_NAME,
                        "chain_based": bool(q.get("reasoning_path")),
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

    stats = {
        "total_questions": len(all_questions),
        "level_1_count": len(all_l1),
        "level_2_count": len(all_l2),
        "level_3_count": len(all_l3),
        "images_with_questions": len(q_files),
    }
    with open(os.path.join(STATS_DIR, "question_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"聚合完成: L1={len(all_l1)} L2={len(all_l2)} L3={len(all_l3)} 总计={len(all_questions)}")
    return stats


# ============================================================
# 主流程
# ============================================================
def main(workers: int = MAX_WORKERS):
    logger.info("=" * 60)
    logger.info("第三步：基于三元组的分层问题生成")
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
            r = fut.result()
            if r is not None:
                results.append(r)

    logger.info(f"问题生成完成：{len(results)}/{len(entity_files)} 张图片")

    stats = aggregate_final()

    logger.info("=" * 60)
    logger.info(f"第三步完成！总计 {stats['total_questions']} 道题")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第三步：基于三元组的分层问题生成")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    args = parser.parse_args()
    main(workers=args.workers)
