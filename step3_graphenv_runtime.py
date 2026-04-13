from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Any

import networkx as nx

from core.vlm import call_vlm_json

_BAD_ANSWERS = {"true", "false", "yes", "no", "none", "n/a", "unknown", "various", "multiple"}
_LOW_VALUE_RELATIONS = {
    "related_to", "associated_with", "connected_to",
    "located_left_of", "located_right_of", "located_above", "located_below",
}
_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}
_QUANTITY_SCALE = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}
_LOOKUP_TAIL_TYPE_BONUS = {
    "PERSON": 2.0,
    "ORG": 1.8,
    "LOCATION": 1.6,
    "TIME": 1.4,
    "QUANTITY": 1.4,
    "OTHER": 0.0,
}
_ONTOLOGYESE_BANLIST = {
    "对应的", "所对应的", "实体", "对象", "相关信息", "组织或机构是什么",
}
_EDGE_WHITELIST = {
    "founded_by", "founder", "ceo", "music_by", "book_and_lyrics_by", "features_songs_by",
    "director", "author", "writer", "lyrics", "headquartered_in", "performed_at",
    "premiered_at", "opened_at", "incorporated_as", "originally_named", "born_in",
    "born_on", "died_in", "died_on", "has_theaters_count", "revenue", "population",
}
_EDGE_BLACKLIST = {
    "related_to", "associated_with", "connected_to",
    "located_left_of", "located_right_of", "located_above", "located_below",
}
_FAMILY_PRIOR = {
    "lookup": -0.3,
    "compare": 0.4,
    "delta": 0.5,
    "same_target": 0.7,
    "same_group": 0.7,
    "extremum": 0.8,
    "threshold": 0.9,
    "multihop": 1.0,
    "select_then_follow": 1.3,
    "join_then_follow": 1.2,
}
_RHETORICAL_EXEMPLARS = {
    "select_then_follow": [
        "首演时间更早的那部音乐剧，作曲者是谁？",
        "成立时间更早的那个品牌，后来注册为什么公司名称？",
    ],
    "join_then_follow": [
        "共同关联的演出地点，位于哪座城市？",
        "共同关联的创始人，出生于哪里？",
    ],
}


@dataclass(frozen=True)
class RelationProfile:
    relation: str
    wh_type: str
    question_head_zh: str
    role_phrase_zh: str | None
    askability: float
    lexicalizability: float
    hideability: float
    generic_penalty: float
    tool_affordance: tuple[str, ...]


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


def _relation_slug(rel: str) -> str:
    return rel.strip().lower().replace(" ", "_")


def _relation_tokens(relation_key: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", relation_key or "")
        if token and token not in {"in", "on", "at", "by", "of", "the", "a", "an"}
    }


def _answer_ok(name: str) -> bool:
    s = str(name or "").strip()
    if not s or len(s) < 2:
        return False
    if s.lower() in _BAD_ANSWERS:
        return False
    return True


def _edge_evidence_score(data: dict) -> float:
    score = 0.0
    if data.get("fact"):
        score += 1.0
    if data.get("source_snippet"):
        score += 0.6
    if data.get("source") == "cross_entity":
        score += 0.3
    return score


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
    if _has_phrase("book_and_lyrics_by"):
        return "编剧和作词者"
    if _has_token("composer") or _has_phrase("music_by"):
        return "作曲者"
    if _has_phrase("features_songs_by"):
        return "歌曲创作组合"
    if _has_token("director") or _has_phrase("directed_by"):
        return "导演"
    if _has_phrase("has_theaters_count"):
        return "剧院数量"
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

    # 不再维护领域特定 relation 的中文映射表——换一个领域就爆炸。
    # 通用 fallback：直接返回 raw slug（去下划线）。
    # 这样 realize LLM 在 prompt 里能同时看到原始 relation 和它需要的翻译工作，
    # 不会被一个不准确的中文短语误导成语义不符的问法。
    return rel.replace("_", " ")


def _wh_type_for_relation(relation_key: str, target_type: str) -> str:
    rel = _relation_slug(relation_key)
    target_type = _normalize_tail_type(target_type)
    if target_type == "PERSON":
        return "who"
    if target_type == "TIME":
        return "when"
    if target_type == "QUANTITY":
        return "how_many"
    if target_type == "LOCATION":
        if any(k in rel for k in ("headquartered", "performed_at", "premiered_at", "opened_at", "located_in", "based_in", "born_in", "died_in")):
            return "where"
        return "which_place"
    if target_type == "ORG":
        if any(k in rel for k in ("owner", "parent", "brand_of", "publisher", "label", "subsidiary")):
            return "which_org"
        return "which_org"
    return "what"


def _tool_affordance_for_relation(relation_key: str, target_type: str) -> tuple[str, ...]:
    rel = _relation_slug(relation_key)
    tools = {"web_search"}
    if target_type in {"TIME", "QUANTITY"}:
        tools.add("code_interpreter")
    if any(k in rel for k in ("logo", "landmark", "person", "celebrity")):
        tools.add("image_search")
    return tuple(sorted(tools))


def _relation_profile(relation_key: str, target_type: str) -> RelationProfile:
    rel = _relation_slug(relation_key)
    head = _relation_natural_text(rel, target_type)
    wh_type = _wh_type_for_relation(rel, target_type)
    askability = 0.7 + max(0.0, _relation_question_value(rel, target_type)) * 0.25
    # 判断关系可信度：
    # 1. 命中已知映射（head 不是 raw slug，也不是"相关" 开头）→ 高可信
    # 2. raw slug 但 token 数 ≤ 3 → 仍算可信的 crisp relation（如 won_championship_in / plays_for）
    # 3. raw slug 且 token 数 > 4 → 句子片段（如 covered_the_pre_draft_workout_of）→ 低可信
    n_tokens = len(rel.split("_"))
    raw_fallback = head == rel.replace("_", " ")
    hit_known_map = (not raw_fallback) and (not head.startswith("相关"))
    is_crisp_raw = raw_fallback and n_tokens <= 3
    is_sentence_fragment = raw_fallback and n_tokens > 4

    if hit_known_map:
        lex_base = 0.85
    elif is_crisp_raw:
        lex_base = 0.65  # crisp 未知关系仍可用（LLM 能从英文翻译），但分数略低
    else:
        lex_base = 0.2   # 句子片段，几乎不可用
    lexicalizability = lex_base
    hideability = 0.8 if lex_base >= 0.6 else 0.2
    generic_penalty = 0.0 if lex_base >= 0.6 else 0.9
    if rel in _EDGE_WHITELIST:
        askability += 0.35
        lexicalizability += 0.25
        hideability += 0.15
    if rel in _EDGE_BLACKLIST:
        askability -= 1.0
        lexicalizability -= 0.5
        generic_penalty += 1.2
    if target_type == "OTHER":
        askability -= 0.3
        lexicalizability -= 0.2
    return RelationProfile(
        relation=rel,
        wh_type=wh_type,
        question_head_zh=head,
        role_phrase_zh=head if not head.startswith("相关") else None,
        askability=max(0.0, min(2.0, askability)),
        lexicalizability=max(0.0, min(1.5, lexicalizability)),
        hideability=max(0.0, min(1.5, hideability)),
        generic_penalty=max(0.0, generic_penalty),
        tool_affordance=_tool_affordance_for_relation(rel, target_type),
    )


def _path_answer_prompt(target_type: str) -> str:
    target_type = _normalize_tail_type(target_type)
    return {
        "PERSON": "是谁",
        "ORG": "是什么组织或机构",
        "LOCATION": "是哪里",
        "TIME": "是什么时间",
        "QUANTITY": "是多少",
    }.get(target_type, "是什么")


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
    if "book_and_lyrics_by" in rel:
        return "编剧和作词者是谁"
    if any(k in rel for k in ("author", "writer")):
        return "作者是谁"
    if "lyrics" in rel:
        return "作词者是谁"
    if any(k in rel for k in ("composer", "music_by")):
        return "作曲者是谁"
    if "features_songs_by" in rel:
        return "歌曲出自哪个组合"
    if any(k in rel for k in ("director", "directed_by")):
        return "导演是谁"
    if "has_theaters_count" in rel:
        return "拥有多少家剧院"
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
        "PERSON": "相关人物是谁",
        "ORG": "隶属于哪个机构",
        "LOCATION": "位于哪里",
        "TIME": "是什么时候",
        "QUANTITY": "是多少",
    }.get(target_type, "答案是什么")
    if fallback_phrase.startswith("相关"):
        return generic_clause
    return f"{fallback_phrase}{_path_answer_prompt(target_type)}"


def _answer_specificity_score(answer_text: str, target_type: str) -> float:
    answer = (answer_text or "").strip()
    if not answer:
        return -1.0
    score = 0.0
    length = len(answer)
    if length >= 6:
        score += 0.3
    if target_type in {"PERSON", "ORG"}:
        score += 0.5
    elif target_type == "LOCATION":
        score += 0.2
    if target_type == "LOCATION" and answer.lower() in {"new york city", "united states", "usa", "china", "japan"}:
        score -= 0.4
    if target_type == "ORG" and answer.lower() in {"organization", "company", "group"}:
        score -= 0.6
    return score


def _relation_question_value(relation_key: str, target_type: str) -> float:
    rel = _relation_slug(relation_key)
    score = 0.0
    if any(k in rel for k in ("founder", "founded_by", "ceo", "music_by", "book_and_lyrics_by", "features_songs_by", "director", "author", "writer", "lyrics")):
        score += 1.2
    if any(k in rel for k in ("incorporated", "renamed", "originally_named")):
        score += 1.0
    if any(k in rel for k in ("performed_at", "premiered_at", "opened_at", "headquartered")):
        score += 0.7
    if any(k in rel for k in ("located_in", "located_at", "has_theaters_count")):
        score += 0.25
    if any(k in rel for k in ("related_to", "associated_with", "connected_to")):
        score -= 1.0
    if target_type in {"PERSON", "ORG"}:
        score += 0.4
    return score


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
    try:
        number = float(match.group("number").replace(",", ""))
    except ValueError:
        return None, ""
    scale = match.group("scale").lower()
    if scale:
        number *= _QUANTITY_SCALE.get(scale, 1.0)
    prefix = match.group("prefix").strip().lower()
    unit = match.group("unit").strip().lower()
    return number, f"{prefix}{unit}"


def _extract_quantity_from_edge(nodes: dict, node_key: str, data: dict) -> tuple[float | None, str]:
    normalized = data.get("normalized_value")
    unit = _normalize_unit(data.get("unit"))
    if isinstance(normalized, (int, float)):
        return float(normalized), unit.lower()
    if isinstance(normalized, str):
        try:
            return float(normalized.strip()), unit.lower()
        except ValueError:
            parsed = _extract_quantity(normalized)
            if parsed[0] is not None:
                return parsed
    return _extract_quantity(nodes.get(node_key, {}).get("name", node_key))


def _format_compact_quantity(value: float | int | None, unit_signature: str = "") -> str:
    if value is None:
        return "未知"
    unit_signature = (unit_signature or "").strip()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    abs_value = abs(numeric)
    if abs_value >= 1_000_000_000:
        compact = f"{numeric / 1_000_000_000:.2f}".rstrip("0").rstrip(".")
        suffix = "B"
    elif abs_value >= 1_000_000:
        compact = f"{numeric / 1_000_000:.2f}".rstrip("0").rstrip(".")
        suffix = "M"
    elif abs_value >= 1_000:
        compact = f"{numeric / 1_000:.2f}".rstrip("0").rstrip(".")
        suffix = "K"
    else:
        compact = f"{numeric:.2f}".rstrip("0").rstrip(".")
        suffix = ""

    if suffix:
        if unit_signature in {"$", "€", "£", "¥"}:
            return f"{unit_signature}{compact}{suffix}"
        return f"{compact}{suffix}{unit_signature}"

    base = compact
    if unit_signature in {"$", "€", "£", "¥"}:
        return f"{unit_signature}{base}"
    return f"{base}{unit_signature}"


def _same_target_relation_compatible(rel_a: str, rel_b: str, target_type: str) -> tuple[bool, float]:
    if not rel_a or not rel_b:
        return False, 0.0
    if rel_a == rel_b:
        return True, 1.0

    target_type = _normalize_tail_type(target_type)
    def bucket(rel: str) -> str:
        rel = rel or ""
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
            if any(k in rel for k in ("performed_at", "premiered_at", "opened_at", "played_at", "staged_at", "held_at")):
                return "location_occurrence"
            if any(k in rel for k in ("formed_by_intersection", "intersection", "crosses", "runs_through", "borders", "connects")):
                return "location_structure"
            return "location_other"
        return "other"

    bucket_a = bucket(rel_a)
    bucket_b = bucket(rel_b)
    if bucket_a != bucket_b:
        return False, 0.0
    blocked = {"location_structure", "location_other", "org_other", "person_other", "other"}
    if bucket_a in blocked:
        return False, 0.0
    bonus = {
        "person_role": 0.8,
        "person_creator": 0.8,
        "person_biographical": 0.6,
        "org_ownership": 0.8,
        "org_affiliation": 0.7,
        "org_membership": 0.6,
        "location_membership": 0.7,
        "location_occurrence": 0.5,
    }.get(bucket_a, 0.0)
    return True, bonus


def _generic_answer_penalty(answer_text: str, target_type: str) -> float:
    answer = (answer_text or "").strip().lower()
    if not answer:
        return 2.0
    penalty = 0.0
    if answer in _BAD_ANSWERS:
        penalty += 2.0
    if target_type == "LOCATION":
        if answer in {"united states", "usa", "china", "japan", "europe", "asia"}:
            penalty += 0.8
    if target_type == "ORG":
        if answer in {"company", "organization", "group"}:
            penalty += 0.8
    if target_type == "PERSON":
        if answer in {"founder", "author", "ceo"}:
            penalty += 0.8
    return penalty


def _ontologyese_penalty(text: str) -> float:
    lowered = (text or "").strip()
    penalty = 0.0
    for phrase in _ONTOLOGYESE_BANLIST:
        if phrase in lowered:
            penalty += 0.9
    return penalty


def _recoverable_abstraction_score(visibility_plan: dict) -> float:
    score = 0.0
    for item in visibility_plan.get("soft_hidden_nodes", []):
        role = (item.get("role") or "").strip()
        if not role:
            continue
        if role.startswith("共同关联的") or role.endswith("创始人") or role.endswith("作曲者") or role.endswith("演出地点"):
            score += 0.9
        elif role.startswith("相关") or "组织或机构" in role or "对象" in role or "实体" in role:
            score -= 0.8
        else:
            score += 0.4
    return score


def _question_intent_specificity(surface_plan: dict, semantic_intent: str) -> float:
    text = " ".join(
        str(surface_plan.get(key, ""))
        for key in ("question_intent", "follow_clause", "shared_phrase", "selection_phrase", "role_phrase", "relation_phrase")
    )
    penalty = _ontologyese_penalty(text)
    score = 0.2
    if any(token in text for token in ("谁", "哪座城市", "哪个国家", "哪一年", "什么时候", "多少")):
        score += 0.8
    if semantic_intent in {"ask_person", "ask_location", "ask_time", "ask_org", "ask_quantity"}:
        score += 0.3
    if "是什么" in text and not any(token in text for token in ("公司名称", "作品")):
        score -= 0.4
    return max(-1.0, score - penalty)


def _anchor_referability(anchor_texts: list[str]) -> float:
    score = 0.0
    for text in anchor_texts:
        if "画面" in text:
            score += 0.3
        if any(token in text for token in ("偏左", "偏右", "最左侧", "最右侧", "下方", "上方", "中央")):
            score += 0.4
        if text in ("[图中实体]", "图中的标识"):
            score -= 0.5
    return score / max(len(anchor_texts), 1)


def _clue_minimality(program: "QuestionProgram") -> float:
    if program.family in {"select_then_follow", "join_then_follow"}:
        return 1.0
    if program.family in {"multihop", "threshold", "same_group"}:
        return 0.8
    if program.family in {"compare", "delta", "extremum"}:
        return 0.6
    return 0.3


def _counterfactual_drop(program: "QuestionProgram") -> float:
    return {
        "select_then_follow": 1.0,
        "join_then_follow": 0.9,
        "multihop": 0.8,
        "same_group": 0.7,
        "threshold": 0.7,
        "compare": 0.4,
        "delta": 0.4,
        "lookup": 0.1,
    }.get(program.family, 0.3)


@dataclass
class QuestionProgram:
    program_id: str
    family: str
    reasoning_family: str
    goal: str
    difficulty: str
    operations: list[dict]
    proof_graph: list[dict]
    anchors: list[str]
    answer_node: str | None
    answer_value: str
    target_type: str
    visibility_plan: dict
    surface_plan: dict
    tool_plan: list[dict]
    semantic_intent: str = ""
    realization_schema: str = ""
    semantic_pivot: str = ""
    tool_requirements: list[str] = field(default_factory=list)
    tool_irreducibility_score: float = 0.0
    answerability_certificate: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    legacy_label: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.semantic_intent:
            self.semantic_intent = {
                "PERSON": "ask_person",
                "ORG": "ask_org",
                "LOCATION": "ask_location",
                "TIME": "ask_time",
                "QUANTITY": "ask_quantity",
            }.get(self.target_type, "ask_other")
        if not self.realization_schema:
            self.realization_schema = {
                "lookup": "anchor_first",
                "compare": "criterion_first",
                "delta": "criterion_first",
                "same_target": "anchor_first",
                "same_group": "anchor_first",
                "extremum": "criterion_first",
                "threshold": "criterion_first",
                "multihop": "anchor_first",
                "select_then_follow": "criterion_first",
                "join_then_follow": "shared_target_first",
            }.get(self.family, "anchor_first")
        if not self.semantic_pivot:
            self.semantic_pivot = self.metadata.get("pivot_relation", "")
        if not self.tool_requirements:
            self.tool_requirements = sorted({step.get("tool") for step in self.tool_plan if step.get("tool")})

    @property
    def core_operation_count(self) -> int:
        return len([op for op in self.operations if op.get("op") not in {"start", "terminate"}])

    def to_reasoning_path(self) -> dict:
        return {
            "program": {
                "program_id": self.program_id,
                "family": self.family,
                "reasoning_family": self.reasoning_family,
                "goal": self.goal,
                "difficulty": self.difficulty,
                "anchors": self.anchors,
                "answer_node": self.answer_node,
                "answer_value": self.answer_value,
                "target_type": self.target_type,
                "operations": self.operations,
                "semantic_intent": self.semantic_intent,
                "realization_schema": self.realization_schema,
                "semantic_pivot": self.semantic_pivot,
                "tool_requirements": self.tool_requirements,
                "tool_irreducibility_score": self.tool_irreducibility_score,
                "legacy_label": self.legacy_label,
            },
            "proof_graph": self.proof_graph,
            "visibility_plan": self.visibility_plan,
            "surface_plan": self.surface_plan,
            "scores": self.scores,
        }


@dataclass
class SearchState:
    state_id: str
    kind: str
    anchors: tuple[str, ...]
    operations: list[dict]
    proof_edges: list[dict]
    frontier: tuple[str, ...]
    answer_candidate: str | None = None
    score: float = 0.0
    extra: dict = field(default_factory=dict)


class GraphEnv:
    def __init__(self, G: nx.DiGraph, nodes: dict[str, dict]):
        self.G = G
        self.nodes = nodes
        self.in_image_nodes = [k for k, v in nodes.items() if v.get("in_image")]
        self._relation_profiles: dict[tuple[str, str], RelationProfile] = {}

    def node_name(self, key: str) -> str:
        return self.nodes.get(key, {}).get("name", key)

    def node_type(self, key: str) -> str:
        return _normalize_tail_type(self.nodes.get(key, {}).get("tail_type", "OTHER"))

    def node_location(self, key: str) -> str:
        return self.nodes.get(key, {}).get("location", "")

    def node_entity_type(self, key: str) -> str:
        return self.nodes.get(key, {}).get("entity_type", "")

    def node_visual_descriptor(self, key: str) -> str:
        """根据 entity_type + 出边 relation 推断视觉描述词。

        目标：让读者仅凭方位+描述就能在图中定位到对应实体，
        比如 "音乐剧海报"、"品牌 logo"、"霓虹灯招牌"。
        """
        etype = self.node_entity_type(key)
        rels = set()
        for edge in self.unique_follows(key):
            rel = (edge.get("relation") or "").strip().lower().replace(" ", "_")
            if rel:
                rels.add(rel)

        # ---- product / 演出 / 影视类 ----
        musical_signals = {
            "performed_at", "music_by", "book_and_lyrics_by",
            "features_songs_by", "broadway_premiere_date",
            "broadway_closure_date", "choreographed_by",
        }
        movie_signals = {
            "directed_by", "starring", "produced_by",
            "box_office", "release_date", "film_studio",
        }
        if rels & musical_signals:
            return "音乐剧海报"
        if rels & movie_signals:
            return "电影海报"

        # ---- brand 细分 ----
        food_signals = {
            "first_restaurant_opened", "global_systemwide_sales",
            "menu_item", "restaurant_count",
        }
        electronics_signals = {
            "manufactures", "stock_exchange", "listed_on",
            "product_line", "semiconductor", "invented",
            "installed_led_screen_in_times_square",
            "led_screen_dimensions",
        }
        auto_signals = {
            "vehicle_model", "engine_type", "manufacturer",
        }
        if etype == "brand":
            if rels & electronics_signals:
                return "电子品牌广告牌"
            if rels & food_signals:
                return "餐饮品牌广告牌"
            if rels & auto_signals:
                return "汽车品牌标识"
            return "品牌广告牌"

        # ---- landmark ----
        landmark_signals = {
            "formed_by_intersection_of", "originally_named",
            "renamed_in", "hosted_first_ball_drop_in", "known_as",
            "architectural_style", "height", "opened_in",
        }
        if etype == "landmark" or rels & landmark_signals:
            return "地标建筑"

        # ---- person ----
        if etype == "person":
            return "人物照片"

        # ---- text / object ----
        if etype == "text":
            return "文字招牌"
        if etype == "object":
            return "标识"

        # ---- fallback by entity_type ----
        if etype == "product":
            return "广告海报"

        # ---- 没有 entity_type 时，用 tail_type 兜底 ----
        tail = self.node_type(key)
        return {
            "PERSON": "人物照片",
            "ORG": "机构标识",
            "LOCATION": "地点标识",
        }.get(tail, "标识")

    def obfuscated_anchor(self, key: str) -> str:
        loc = (self.node_location(key) or "").strip()
        desc = self.node_visual_descriptor(key)
        if loc:
            # 去掉 "细小的" 等前缀干扰，确保以 "画面" 开头
            if "画面" in loc and not loc.startswith("画面"):
                loc = loc[loc.index("画面"):]
            elif not loc.startswith("画面"):
                loc = f"画面{loc}"
            return f"{loc}的{desc}"
        return f"图中的{desc}"

    def edge_key(self, src: str, dst: str, data: dict) -> dict:
        tail_type = _normalize_tail_type(data.get("tail_type") or self.node_type(dst))
        profile = self.relation_profile(data.get("relation", ""), tail_type)
        return {
            "src": src,
            "dst": dst,
            "relation": data.get("relation", ""),
            "tail_type": tail_type,
            "normalized_value": _normalize_scalar_value(data.get("normalized_value")),
            "unit": _normalize_unit(data.get("unit")),
            "fact": data.get("fact", ""),
            "source_snippet": data.get("source_snippet", ""),
            "source": data.get("source", ""),
            "provenance": data.get("provenance") or data.get("source", ""),
            "evidence": _edge_evidence_score(data),
            "relation_profile": asdict(profile),
        }

    def relation_profile(self, relation_key: str, target_type: str) -> RelationProfile:
        key = (_relation_slug(relation_key), _normalize_tail_type(target_type))
        if key not in self._relation_profiles:
            self._relation_profiles[key] = _relation_profile(relation_key, target_type)
        return self._relation_profiles[key]

    def outgoing(self, node: str) -> list[dict]:
        rows = []
        for dst, data in self.G[node].items():
            rel = _relation_slug(data.get("relation", ""))
            if rel in _LOW_VALUE_RELATIONS:
                continue
            edge = self.edge_key(node, dst, data)
            profile = edge.get("relation_profile", {})
            if profile.get("askability", 0.0) <= 0.0:
                continue
            rows.append(edge)
        rows.sort(
            key=lambda x: (
                x.get("relation_profile", {}).get("askability", 0.0)
                + x.get("relation_profile", {}).get("lexicalizability", 0.0)
                + x["evidence"]
            ),
            reverse=True,
        )
        return rows

    def unique_follows(self, node: str) -> list[dict]:
        results = []
        seen = set()
        for edge in self.outgoing(node):
            signature = (edge["dst"], edge["relation"])
            if signature in seen:
                continue
            seen.add(signature)
            if not _answer_ok(self.node_name(edge["dst"])):
                continue
            results.append(edge)
        return results

    def pair_edges(self, a: str, b: str) -> tuple[list[dict], list[dict]]:
        return self.unique_follows(a), self.unique_follows(b)

    def shared_targets(self, a: str, b: str) -> list[tuple[dict, dict, str]]:
        out_a, out_b = self.pair_edges(a, b)
        by_dst_a = {edge["dst"]: edge for edge in out_a}
        shared = []
        for edge_b in out_b:
            edge_a = by_dst_a.get(edge_b["dst"])
            if not edge_a:
                continue
            target = edge_b["dst"]
            target_type = self.node_type(target)
            if target_type not in {"PERSON", "ORG", "LOCATION"}:
                continue
            ok, bonus = _same_target_relation_compatible(
                _relation_slug(edge_a["relation"]),
                _relation_slug(edge_b["relation"]),
                target_type,
            )
            if not ok:
                continue
            shared.append((edge_a, edge_b, target, bonus))
        return shared

    def same_group_candidates(self, a: str, b: str) -> list[dict]:
        results = []
        for edge_a1 in self.unique_follows(a):
            child_a = edge_a1["dst"]
            if self.node_type(child_a) not in {"LOCATION", "ORG"}:
                continue
            for edge_b1 in self.unique_follows(b):
                child_b = edge_b1["dst"]
                if self.node_type(child_b) != self.node_type(child_a):
                    continue
                for edge_a2 in self.unique_follows(child_a):
                    parent = edge_a2["dst"]
                    parent_type = self.node_type(parent)
                    if parent_type not in {"LOCATION", "ORG"}:
                        continue
                    for edge_b2 in self.unique_follows(child_b):
                        if edge_b2["dst"] != parent:
                            continue
                        results.append({
                            "entity_a": a,
                            "entity_b": b,
                            "child_a": child_a,
                            "child_b": child_b,
                            "parent": parent,
                            "edge_a1": edge_a1,
                            "edge_b1": edge_b1,
                            "edge_a2": edge_a2,
                            "edge_b2": edge_b2,
                            "parent_type": parent_type,
                            "score": edge_a1["evidence"] + edge_b1["evidence"] + edge_a2["evidence"] + edge_b2["evidence"],
                        })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


class BeamProgramProposer:
    def __init__(self, env: GraphEnv, beam_size: int = 24, max_depth: int = 3):
        self.env = env
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.counter = 0
        self.seed_kind_quota = {
            "single": 8,
            "pair": 12,
            "triple": 4,
        }
        self.next_kind_quota = {
            "followed": 8,
            "compared": 10,
            "joined": 8,
            "single": 4,
            "pair": 6,
            "triple": 4,
        }

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def _seed_single_score(self, anchor: str) -> float:
        outgoing = self.env.unique_follows(anchor)
        high_value = sum(1 for edge in outgoing if edge["tail_type"] in {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"})
        evidence = sum(edge["evidence"] for edge in outgoing[:4])
        return 1.0 + 0.15 * high_value + 0.1 * evidence

    def _seed_pair_score(self, a: str, b: str) -> float:
        shared = len(self.env.shared_targets(a, b))
        same_group = len(self.env.same_group_candidates(a, b))
        comparable = 0
        out_a, out_b = self.env.pair_edges(a, b)
        for edge_a in out_a[:6]:
            for edge_b in out_b[:6]:
                if edge_a["dst"] == edge_b["dst"]:
                    continue
                target_type = edge_a["tail_type"]
                if target_type not in {"TIME", "QUANTITY"} or edge_b["tail_type"] != target_type:
                    continue
                ok, compat_bonus, _ = _value_relations_compatible(
                    _relation_slug(edge_a["relation"]),
                    _relation_slug(edge_b["relation"]),
                    target_type,
                )
                if ok:
                    comparable += 1 + compat_bonus
        return 1.2 + 0.7 * shared + 0.5 * same_group + 0.18 * comparable

    def _seed_triple_score(self, anchors: tuple[str, ...]) -> float:
        bucket_entries: dict[tuple[str, str], int] = defaultdict(int)
        for anchor in anchors:
            for edge in self.env.unique_follows(anchor)[:6]:
                if edge["tail_type"] not in {"TIME", "QUANTITY"}:
                    continue
                bucket_entries[(edge["tail_type"], _relation_slug(edge["relation"]))] += 1
        best_bucket = max(bucket_entries.values(), default=0)
        quantity_bonus = sum(1 for (tt, _), count in bucket_entries.items() if tt == "QUANTITY" and count >= 3)
        return 1.4 + 0.25 * best_bucket + 0.2 * quantity_bonus

    def seed_states(self) -> list[SearchState]:
        states: list[SearchState] = []
        for anchor in self.env.in_image_nodes:
            states.append(SearchState(
                state_id=self.next_id("state"),
                kind="single",
                anchors=(anchor,),
                operations=[{"op": "start", "anchor": anchor}],
                proof_edges=[],
                frontier=(anchor,),
                score=self._seed_single_score(anchor),
            ))
        for a, b in combinations(self.env.in_image_nodes, 2):
            states.append(SearchState(
                state_id=self.next_id("state"),
                kind="pair",
                anchors=(a, b),
                operations=[{"op": "start", "anchors": [a, b]}],
                proof_edges=[],
                frontier=(a, b),
                score=self._seed_pair_score(a, b),
            ))
        for triple in combinations(self.env.in_image_nodes, 3):
            states.append(SearchState(
                state_id=self.next_id("state"),
                kind="triple",
                anchors=tuple(triple),
                operations=[{"op": "start", "anchors": list(triple)}],
                proof_edges=[],
                frontier=tuple(triple),
                score=self._seed_triple_score(tuple(triple)),
            ))
        return states

    def _trim_beam(self, states: list[SearchState], quotas: dict[str, int]) -> list[SearchState]:
        if not states:
            return []
        grouped: dict[str, list[SearchState]] = defaultdict(list)
        for state in states:
            grouped[state.kind].append(state)

        selected: list[SearchState] = []
        seen = set()
        for kind, quota in quotas.items():
            bucket = sorted(grouped.get(kind, []), key=lambda s: s.score, reverse=True)
            for state in bucket[:quota]:
                if state.state_id in seen:
                    continue
                selected.append(state)
                seen.add(state.state_id)

        if len(selected) < self.beam_size:
            remainder = sorted(states, key=lambda s: s.score, reverse=True)
            for state in remainder:
                if state.state_id in seen:
                    continue
                selected.append(state)
                seen.add(state.state_id)
                if len(selected) >= self.beam_size:
                    break

        return selected[: self.beam_size]

    def propose(self) -> list[QuestionProgram]:
        programs: list[QuestionProgram] = []
        beam = self._trim_beam(self.seed_states(), self.seed_kind_quota)

        for _depth in range(self.max_depth):
            next_beam: list[SearchState] = []
            for state in beam:
                terminal, expanded = self.expand_state(state)
                programs.extend(terminal)
                next_beam.extend(expanded)
            if not next_beam:
                break
            beam = self._trim_beam(next_beam, self.next_kind_quota)

        deduped = {}
        for program in programs:
            signature = (
                program.family,
                tuple(program.anchors),
                program.answer_value,
                tuple((edge.get("src"), edge.get("dst"), edge.get("relation")) for edge in program.proof_graph),
            )
            if signature not in deduped or deduped[signature].scores.get("overall", 0.0) < program.scores.get("overall", 0.0):
                deduped[signature] = program
        return list(deduped.values())

    def expand_state(self, state: SearchState) -> tuple[list[QuestionProgram], list[SearchState]]:
        if state.kind == "single":
            return self._expand_single(state)
        if state.kind == "pair":
            return self._expand_pair(state)
        if state.kind == "triple":
            return self._expand_triple(state)
        if state.kind == "followed":
            return self._expand_followed(state)
        if state.kind == "compared":
            return self._expand_compared(state)
        if state.kind == "joined":
            return self._expand_joined(state)
        return [], []

    def _expand_single(self, state: SearchState):
        anchor = state.anchors[0]
        terminals: list[QuestionProgram] = []
        expanded: list[SearchState] = []
        for edge in self.env.unique_follows(anchor)[:6]:
            target_type = edge["tail_type"]
            if target_type in {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"}:
                terminals.append(self._make_lookup_program(anchor, edge))
            if target_type in {"PERSON", "ORG", "LOCATION", "OTHER"}:
                expanded.append(SearchState(
                    state_id=self.next_id("state"),
                    kind="followed",
                    anchors=state.anchors,
                    operations=state.operations + [{"op": "follow", "src": anchor, "dst": edge["dst"], "relation": edge["relation"]}],
                    proof_edges=state.proof_edges + [edge],
                    frontier=(edge["dst"],),
                    score=state.score + edge["evidence"] + _LOOKUP_TAIL_TYPE_BONUS.get(target_type, 0.0),
                    extra={"mid": edge["dst"], "edge_1": edge},
                ))
        return terminals, expanded

    def _expand_followed(self, state: SearchState):
        terminals: list[QuestionProgram] = []
        mid = state.extra["mid"]
        edge_1 = state.extra["edge_1"]
        for edge_2 in self.env.unique_follows(mid)[:6]:
            target_type = edge_2["tail_type"]
            if target_type not in {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"}:
                continue
            terminals.append(self._make_multihop_program(state.anchors[0], edge_1, edge_2))
        return terminals, []

    def _expand_pair(self, state: SearchState):
        a, b = state.anchors
        terminals: list[QuestionProgram] = []
        expanded: list[SearchState] = []

        out_a, out_b = self.env.pair_edges(a, b)

        # compare candidates
        for edge_a in out_a:
            for edge_b in out_b:
                if edge_a["dst"] == edge_b["dst"]:
                    continue
                target_type = edge_a["tail_type"]
                if target_type not in {"TIME", "QUANTITY"} or edge_b["tail_type"] != target_type:
                    continue
                ok, compat_bonus, generic_rel = _value_relations_compatible(
                    _relation_slug(edge_a["relation"]),
                    _relation_slug(edge_b["relation"]),
                    target_type,
                )
                if not ok:
                    continue
                if target_type == "TIME":
                    va = _extract_year_from_edge(self.env.nodes, edge_a["dst"], edge_a)
                    vb = _extract_year_from_edge(self.env.nodes, edge_b["dst"], edge_b)
                    if va is None or vb is None or va == vb:
                        continue
                else:
                    va, unit_a = _extract_quantity_from_edge(self.env.nodes, edge_a["dst"], edge_a)
                    vb, unit_b = _extract_quantity_from_edge(self.env.nodes, edge_b["dst"], edge_b)
                    if va is None or vb is None or va == vb or unit_a != unit_b:
                        continue
                compare_state = SearchState(
                    state_id=self.next_id("state"),
                    kind="compared",
                    anchors=state.anchors,
                    operations=state.operations + [{
                        "op": "compare",
                        "left": a,
                        "right": b,
                        "edge_left": edge_a,
                        "edge_right": edge_b,
                        "target_type": target_type,
                        "relation_label": generic_rel or _relation_slug(edge_a["relation"]),
                    }],
                    proof_edges=state.proof_edges + [edge_a, edge_b],
                    frontier=(a, b),
                    score=state.score + edge_a["evidence"] + edge_b["evidence"] + compat_bonus + 1.2,
                    extra={"edge_a": edge_a, "edge_b": edge_b, "target_type": target_type, "compat_bonus": compat_bonus},
                )
                terminals.extend(self._make_compare_programs(compare_state))
                expanded.append(compare_state)

        # join candidates
        for edge_a, edge_b, shared, bonus in self.env.shared_targets(a, b):
            joined_state = SearchState(
                state_id=self.next_id("state"),
                kind="joined",
                anchors=state.anchors,
                operations=state.operations + [{
                    "op": "join",
                    "left": a,
                    "right": b,
                    "shared": shared,
                    "edge_left": edge_a,
                    "edge_right": edge_b,
                    "shared_type": self.env.node_type(shared),
                }],
                proof_edges=state.proof_edges + [edge_a, edge_b],
                frontier=(shared,),
                score=state.score + edge_a["evidence"] + edge_b["evidence"] + bonus + 1.1,
                extra={"shared": shared, "edge_a": edge_a, "edge_b": edge_b},
            )
            terminals.append(self._make_same_target_program(joined_state))
            expanded.append(joined_state)

        # same group candidates
        for group in self.env.same_group_candidates(a, b)[:3]:
            terminals.append(self._make_same_group_program(group))

        return terminals, expanded

    def _expand_compared(self, state: SearchState):
        terminals: list[QuestionProgram] = []
        edge_a = state.extra["edge_a"]
        edge_b = state.extra["edge_b"]
        target_type = state.extra["target_type"]
        if target_type == "TIME":
            va = _extract_year_from_edge(self.env.nodes, edge_a["dst"], edge_a)
            vb = _extract_year_from_edge(self.env.nodes, edge_b["dst"], edge_b)
            winner = state.anchors[0] if va < vb else state.anchors[1]
        else:
            va, _ = _extract_quantity_from_edge(self.env.nodes, edge_a["dst"], edge_a)
            vb, unit_sig = _extract_quantity_from_edge(self.env.nodes, edge_b["dst"], edge_b)
            winner = state.anchors[0] if va > vb else state.anchors[1]

        follow_edges = sorted(
            self.env.unique_follows(winner),
            key=lambda edge: (
                _relation_question_value(edge["relation"], edge["tail_type"])
                + edge["evidence"]
                + _answer_specificity_score(self.env.node_name(edge["dst"]), edge["tail_type"])
            ),
            reverse=True,
        )
        for follow_edge in follow_edges[:6]:
            follow_type = follow_edge["tail_type"]
            if follow_type not in {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"}:
                continue
            profile = follow_edge.get("relation_profile", {})
            if profile.get("askability", 0.0) < 0.45 or profile.get("lexicalizability", 0.0) < 0.15:
                continue
            terminals.append(self._make_select_then_follow_program(state, winner, follow_edge))
        return terminals, []

    def _expand_joined(self, state: SearchState):
        terminals: list[QuestionProgram] = []
        shared = state.extra["shared"]
        follow_edges = sorted(
            self.env.unique_follows(shared),
            key=lambda edge: (
                _relation_question_value(edge["relation"], edge["tail_type"])
                + edge["evidence"]
                + _answer_specificity_score(self.env.node_name(edge["dst"]), edge["tail_type"])
            ),
            reverse=True,
        )
        for follow_edge in follow_edges[:6]:
            follow_type = follow_edge["tail_type"]
            if follow_type not in {"PERSON", "ORG", "LOCATION", "TIME", "QUANTITY"}:
                continue
            profile = follow_edge.get("relation_profile", {})
            if profile.get("askability", 0.0) < 0.45 or profile.get("lexicalizability", 0.0) < 0.15:
                continue
            terminals.append(self._make_join_then_follow_program(state, follow_edge))
        return terminals, []

    def _expand_triple(self, state: SearchState):
        terminals: list[QuestionProgram] = []
        anchors = state.anchors
        # extremum and threshold from shared comparable attributes
        bucket_entries: dict[tuple[str, str], list[tuple[str, dict, Any, str]]] = defaultdict(list)
        for anchor in anchors:
            for edge in self.env.unique_follows(anchor):
                tt = edge["tail_type"]
                if tt == "TIME":
                    value = _extract_year_from_edge(self.env.nodes, edge["dst"], edge)
                    unit_sig = "year"
                elif tt == "QUANTITY":
                    value, unit_sig = _extract_quantity_from_edge(self.env.nodes, edge["dst"], edge)
                else:
                    continue
                if value is None:
                    continue
                bucket_key = (tt, _relation_slug(edge["relation"]))
                bucket_entries[bucket_key].append((anchor, edge, value, unit_sig))

        for (tt, rel_slug), entries in bucket_entries.items():
            if len(entries) < 3:
                continue
            entries = entries[:3]
            terminals.append(self._make_extremum_program(entries, tt, rel_slug))
            if tt == "QUANTITY":
                terminals.append(self._make_threshold_program(entries, rel_slug))
        return terminals, []

    def _make_lookup_program(self, anchor: str, edge: dict) -> QuestionProgram:
        target = edge["dst"]
        answer = self.env.node_name(target)
        target_type = edge["tail_type"]
        role = _relation_natural_text(edge["relation"], target_type)
        visibility = {
            "visible_anchors": [anchor],
            "hard_hidden_nodes": [],
            "soft_hidden_nodes": [{"node": target, "role": role, "kind": "answer_target"}],
            "answer_node": target,
        }
        surface = {
            "family": "lookup",
            "anchor_texts": [self.env.obfuscated_anchor(anchor)],
            "role_phrase": role,
            "question_intent": _path_follow_clause(edge["relation"], target_type),
            "forbidden_names": [],
        }
        proof_graph = [edge]
        scores = self._score_program("lookup", [anchor], proof_graph, answer, target_type, 1, target)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="lookup",
            reasoning_family="lookup",
            goal=f"lookup_{target_type.lower()}",
            difficulty="L2",
            operations=[{"op": "follow", "src": anchor, "dst": target, "relation": edge["relation"]}],
            proof_graph=proof_graph,
            anchors=[anchor],
            answer_node=target,
            answer_value=answer,
            target_type=target_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(anchor)}"},
                {"tool": "web_search", "reason": f"搜索该实体的{role}"},
            ],
            semantic_pivot=_relation_slug(edge["relation"]),
            scores=scores,
            legacy_label="lookup",
            tool_irreducibility_score=0.7,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True},
        )

    def _make_multihop_program(self, anchor: str, edge_1: dict, edge_2: dict) -> QuestionProgram:
        mid = edge_1["dst"]
        target = edge_2["dst"]
        answer = self.env.node_name(target)
        target_type = edge_2["tail_type"]
        mid_role = _relation_natural_text(edge_1["relation"], self.env.node_type(mid))
        follow_clause = _path_follow_clause(edge_2["relation"], target_type)
        visibility = {
            "visible_anchors": [anchor],
            "hard_hidden_nodes": [mid],
            "soft_hidden_nodes": [{"node": mid, "role": mid_role, "kind": "intermediate"}],
            "answer_node": target,
        }
        surface = {
            "family": "multihop",
            "anchor_texts": [self.env.obfuscated_anchor(anchor)],
            "mid_role": mid_role,
            "question_intent": follow_clause,
            "forbidden_names": [self.env.node_name(mid)],
        }
        proof_graph = [edge_1, edge_2]
        scores = self._score_program("multihop", [anchor], proof_graph, answer, target_type, 2, target)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="multihop",
            reasoning_family="multihop",
            goal=f"multihop_{target_type.lower()}",
            difficulty="L3",
            operations=[
                {"op": "follow", "src": anchor, "dst": mid, "relation": edge_1["relation"]},
                {"op": "follow", "src": mid, "dst": target, "relation": edge_2["relation"]},
            ],
            proof_graph=proof_graph,
            anchors=[anchor],
            answer_node=target,
            answer_value=answer,
            target_type=target_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(anchor)}"},
                {"tool": "web_search", "reason": f"先找到该实体对应的{mid_role}"},
                {"tool": "web_search", "reason": f"继续查询该{mid_role}的{_relation_natural_text(edge_2['relation'], target_type)}"},
            ],
            semantic_pivot=_relation_slug(edge_2["relation"]),
            scores=scores,
            legacy_label="multihop",
            tool_irreducibility_score=1.0,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_multistep": True},
        )

    def _make_compare_programs(self, state: SearchState) -> list[QuestionProgram]:
        a, b = state.anchors
        edge_a = state.extra["edge_a"]
        edge_b = state.extra["edge_b"]
        target_type = state.extra["target_type"]
        proof_graph = [edge_a, edge_b]
        relation_label = _relation_natural_text(edge_a["relation"], target_type)
        if target_type == "TIME":
            va = _extract_year_from_edge(self.env.nodes, edge_a["dst"], edge_a)
            vb = _extract_year_from_edge(self.env.nodes, edge_b["dst"], edge_b)
            winner = a if va < vb else b
            winner_val = va if winner == a else vb
            loser_val = vb if winner == a else va
            answer_text = self.env.node_name(winner)
            delta_answer = f"时间差: {abs(va - vb)} 年"
            compare_goal = "compare_time"
            delta_goal = "delta_time"
        else:
            va, unit_sig = _extract_quantity_from_edge(self.env.nodes, edge_a["dst"], edge_a)
            vb, _ = _extract_quantity_from_edge(self.env.nodes, edge_b["dst"], edge_b)
            winner = a if va > vb else b
            answer_text = self.env.node_name(winner)
            winner_val = va if winner == a else vb
            loser_val = vb if winner == a else va
            delta_answer = f"数值差: {_format_compact_quantity(abs(va - vb), unit_sig)}"
            compare_goal = "compare_quantity"
            delta_goal = "delta_quantity"

        common_visibility = {
            "visible_anchors": [a, b],
            "hard_hidden_nodes": [edge_a["dst"], edge_b["dst"]],
            "soft_hidden_nodes": [],
            "answer_node": winner,
        }
        compare_surface = {
            "family": "compare",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "compare_phrase": _selection_phrase(target_type, edge_a["relation"]),
            "relation_phrase": relation_label,
            "forbidden_names": [],
        }
        compare_scores = self._score_program("compare", [a, b], proof_graph, answer_text, target_type, 1, winner)
        compare_prog = QuestionProgram(
            program_id=self.next_id("prog"),
            family="compare",
            reasoning_family="compare",
            goal=compare_goal,
            difficulty="L2",
            operations=[asdict_safe(state.operations[-1])],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=winner,
            answer_value=answer_text,
            target_type=target_type,
            visibility_plan=common_visibility,
            surface_plan=compare_surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": f"分别搜索两个实体的{relation_label}"},
                {"tool": "code_interpreter", "reason": "比较两个值并确定满足条件的实体"},
            ],
            semantic_pivot=_relation_slug(edge_a["relation"]),
            scores=compare_scores,
            legacy_label="comparative",
            metadata={"winner_value": winner_val, "loser_value": loser_val, "relation": edge_a["relation"]},
            tool_irreducibility_score=0.9,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_comparison": True},
        )

        delta_surface = {
            "family": "delta",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "relation_phrase": relation_label,
            "target_type": target_type,
            "forbidden_names": [],
        }
        delta_scores = self._score_program("delta", [a, b], proof_graph, delta_answer, target_type, 1, None)
        delta_prog = QuestionProgram(
            program_id=self.next_id("prog"),
            family="delta",
            reasoning_family="delta",
            goal=delta_goal,
            difficulty="L2",
            operations=[asdict_safe(state.operations[-1])],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=None,
            answer_value=delta_answer,
            target_type=target_type,
            visibility_plan=common_visibility,
            surface_plan=delta_surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": f"分别搜索两个实体的{relation_label}"},
                {"tool": "code_interpreter", "reason": "计算两个值之间的差"},
            ],
            semantic_pivot=_relation_slug(edge_a["relation"]),
            scores=delta_scores,
            legacy_label="delta",
            metadata={"left_value": winner_val if winner == a else loser_val, "right_value": winner_val if winner == b else loser_val, "relation": edge_a["relation"]},
            tool_irreducibility_score=1.0,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_comparison": True},
        )
        return [compare_prog, delta_prog]

    def _make_same_target_program(self, state: SearchState) -> QuestionProgram:
        a, b = state.anchors
        shared = state.extra["shared"]
        edge_a = state.extra["edge_a"]
        edge_b = state.extra["edge_b"]
        shared_type = self.env.node_type(shared)
        answer = f"是，都是 {self.env.node_name(shared)}"
        role_a = _relation_natural_text(edge_a["relation"], shared_type)
        role_b = _relation_natural_text(edge_b["relation"], shared_type)
        visibility = {
            "visible_anchors": [a, b],
            "hard_hidden_nodes": [shared],
            "soft_hidden_nodes": [
                {"node": shared, "role": role_a, "kind": "shared_role_a"},
                {"node": shared, "role": role_b, "kind": "shared_role_b"},
            ],
            "answer_node": shared,
        }
        surface = {
            "family": "same_target",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "role_a": role_a,
            "role_b": role_b,
            "entity_type": shared_type,
            "forbidden_names": [self.env.node_name(shared)],
        }
        proof_graph = [edge_a, edge_b]
        scores = self._score_program("same_target", [a, b], proof_graph, answer, shared_type, 1, shared)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="same_target",
            reasoning_family="join",
            goal=f"same_{shared_type.lower()}",
            difficulty="L2",
            operations=[asdict_safe(state.operations[-1])],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=shared,
            answer_value=answer,
            target_type=shared_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": f"分别搜索两个实体的{role_a}/{role_b}，判断是否指向同一目标"},
            ],
            semantic_pivot=_relation_slug(edge_a["relation"]),
            scores=scores,
            legacy_label="same_target",
            tool_irreducibility_score=0.8,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True},
        )

    def _make_same_group_program(self, group: dict) -> QuestionProgram:
        a = group["entity_a"]
        b = group["entity_b"]
        parent = group["parent"]
        parent_type = group["parent_type"]
        answer = f"是，都属于 {self.env.node_name(parent)}"
        visibility = {
            "visible_anchors": [a, b],
            "hard_hidden_nodes": [group["child_a"], group["child_b"]],
            "soft_hidden_nodes": [
                {"node": group["child_a"], "role": _relation_natural_text(group["edge_a1"]["relation"], self.env.node_type(group["child_a"])), "kind": "child_a"},
                {"node": group["child_b"], "role": _relation_natural_text(group["edge_b1"]["relation"], self.env.node_type(group["child_b"])), "kind": "child_b"},
            ],
            "answer_node": parent,
        }
        surface = {
            "family": "same_group",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "parent_type": parent_type,
            "forbidden_names": [self.env.node_name(group["child_a"]), self.env.node_name(group["child_b"])],
        }
        proof_graph = [group["edge_a1"], group["edge_a2"], group["edge_b1"], group["edge_b2"]]
        scores = self._score_program("same_group", [a, b], proof_graph, answer, parent_type, 2, parent)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="same_group",
            reasoning_family="group",
            goal=f"same_group_{parent_type.lower()}",
            difficulty="L2",
            operations=[
                {"op": "follow", "src": a, "dst": group["child_a"], "relation": group["edge_a1"]["relation"]},
                {"op": "follow", "src": group["child_a"], "dst": parent, "relation": group["edge_a2"]["relation"]},
                {"op": "follow", "src": b, "dst": group["child_b"], "relation": group["edge_b1"]["relation"]},
                {"op": "follow", "src": group["child_b"], "dst": parent, "relation": group["edge_b2"]["relation"]},
            ],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=parent,
            answer_value=answer,
            target_type=parent_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": "分别追溯两个实体的归属链"},
                {"tool": "code_interpreter", "reason": "整理两条归属链并判断是否归于同一上级"},
            ],
            semantic_pivot=_relation_slug(group["edge_a2"]["relation"]),
            scores=scores,
            legacy_label="same_group",
            tool_irreducibility_score=0.95,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_multistep": True},
        )

    def _make_extremum_program(self, entries: list[tuple[str, dict, Any, str]], target_type: str, rel_slug: str) -> QuestionProgram:
        anchors = [node for node, *_ in entries]
        answer_anchor = None
        if target_type == "TIME":
            answer_anchor = min(entries, key=lambda item: item[2])[0]
        else:
            answer_anchor = max(entries, key=lambda item: item[2])[0]
        answer = self.env.node_name(answer_anchor)
        proof_graph = [edge for _, edge, _, _ in entries]
        visibility = {
            "visible_anchors": anchors,
            "hard_hidden_nodes": [edge["dst"] for _, edge, _, _ in entries],
            "soft_hidden_nodes": [],
            "answer_node": answer_anchor,
        }
        surface = {
            "family": "extremum",
            "anchor_texts": [self.env.obfuscated_anchor(a) for a in anchors],
            "relation_phrase": _relation_natural_text(rel_slug, target_type),
            "target_type": target_type,
            "forbidden_names": [],
        }
        scores = self._score_program("extremum", anchors, proof_graph, answer, target_type, 1, answer_anchor)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="extremum",
            reasoning_family="aggregate",
            goal=f"extremum_{target_type.lower()}",
            difficulty="L2",
            operations=[{"op": "aggregate", "mode": "extremum", "target_type": target_type, "relation": rel_slug}],
            proof_graph=proof_graph,
            anchors=anchors,
            answer_node=answer_anchor,
            answer_value=answer,
            target_type=target_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": "识别这些图中实体"},
                {"tool": "web_search", "reason": f"分别搜索这些实体的{_relation_natural_text(rel_slug, target_type)}"},
                {"tool": "code_interpreter", "reason": "比较这些值并确定极值"},
            ],
            semantic_pivot=rel_slug,
            scores=scores,
            legacy_label="extremum",
            tool_irreducibility_score=0.95,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_aggregate": True},
        )

    def _make_threshold_program(self, entries: list[tuple[str, dict, Any, str]], rel_slug: str) -> QuestionProgram:
        (a, edge_a, va, unit_sig), (b, edge_b, vb, _), (c, edge_c, vc, _) = entries
        sum_value = va + vb
        exceeds = sum_value > vc
        answer = f"{'是' if exceeds else '否'}，{_format_compact_quantity(sum_value, unit_sig)} {'>' if exceeds else '<='} {_format_compact_quantity(vc, unit_sig)}"
        proof_graph = [edge_a, edge_b, edge_c]
        visibility = {
            "visible_anchors": [a, b, c],
            "hard_hidden_nodes": [edge_a["dst"], edge_b["dst"], edge_c["dst"]],
            "soft_hidden_nodes": [],
            "answer_node": None,
        }
        surface = {
            "family": "threshold",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b), self.env.obfuscated_anchor(c)],
            "relation_phrase": _relation_natural_text(rel_slug, "QUANTITY"),
            "forbidden_names": [],
            "unit": unit_sig,
        }
        scores = self._score_program("threshold", [a, b, c], proof_graph, answer, "QUANTITY", 2, None)
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="threshold",
            reasoning_family="aggregate",
            goal="sum_threshold_quantity",
            difficulty="L2",
            operations=[{"op": "aggregate", "mode": "sum_then_compare", "relation": rel_slug, "unit": unit_sig}],
            proof_graph=proof_graph,
            anchors=[a, b, c],
            answer_node=None,
            answer_value=answer,
            target_type="QUANTITY",
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": "识别这三个图中实体"},
                {"tool": "web_search", "reason": f"分别搜索这三个实体的{_relation_natural_text(rel_slug, 'QUANTITY')}"},
                {"tool": "code_interpreter", "reason": "求和前两个值并与第三个比较"},
            ],
            semantic_pivot=rel_slug,
            scores=scores,
            legacy_label="sum_threshold_quantity",
            tool_irreducibility_score=1.0,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_aggregate": True},
        )

    def _make_select_then_follow_program(self, state: SearchState, winner: str, follow_edge: dict) -> QuestionProgram:
        a, b = state.anchors
        edge_a = state.extra["edge_a"]
        edge_b = state.extra["edge_b"]
        target_type = follow_edge["tail_type"]
        answer = self.env.node_name(follow_edge["dst"])
        selection_type = state.extra["target_type"]
        compare_edge = edge_a if winner == a else edge_b
        visibility = {
            "visible_anchors": [a, b],
            "hard_hidden_nodes": [winner, edge_a["dst"], edge_b["dst"]],
            "soft_hidden_nodes": [
                {"node": winner, "role": _selection_phrase(selection_type, compare_edge["relation"]), "kind": "selected_entity"},
                {"node": follow_edge["dst"], "role": _relation_natural_text(follow_edge["relation"], target_type), "kind": "follow_role"},
            ],
            "answer_node": follow_edge["dst"],
        }
        surface = {
            "family": "select_then_follow",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "selection_phrase": _selection_phrase(selection_type, compare_edge["relation"]),
            "follow_clause": _path_follow_clause(follow_edge["relation"], target_type),
            "forbidden_names": [self.env.node_name(winner)],
        }
        proof_graph = [edge_a, edge_b, follow_edge]
        scores = self._score_program("select_then_follow", [a, b], proof_graph, answer, target_type, 3, follow_edge["dst"])
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="select_then_follow",
            reasoning_family="select_follow",
            goal=f"select_then_follow_{target_type.lower()}",
            difficulty="L3",
            operations=[
                asdict_safe(state.operations[-1]),
                {"op": "select", "winner": winner, "criterion": surface["selection_phrase"]},
                {"op": "follow", "src": winner, "dst": follow_edge["dst"], "relation": follow_edge["relation"]},
            ],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=follow_edge["dst"],
            answer_value=answer,
            target_type=target_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": f"分别搜索两个实体的{_relation_natural_text(compare_edge['relation'], selection_type)}"},
                {"tool": "code_interpreter", "reason": "比较两个值并确定满足条件的实体"},
                {"tool": "web_search", "reason": f"继续搜索被选中实体的{_relation_natural_text(follow_edge['relation'], target_type)}"},
            ],
            semantic_pivot=_relation_slug(follow_edge["relation"]),
            scores=scores,
            legacy_label="path_reasoning_select_then_follow",
            tool_irreducibility_score=1.0,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_multistep": True},
        )

    def _make_join_then_follow_program(self, state: SearchState, follow_edge: dict) -> QuestionProgram:
        a, b = state.anchors
        shared = state.extra["shared"]
        edge_a = state.extra["edge_a"]
        edge_b = state.extra["edge_b"]
        shared_type = self.env.node_type(shared)
        target_type = follow_edge["tail_type"]
        answer = self.env.node_name(follow_edge["dst"])
        visibility = {
            "visible_anchors": [a, b],
            "hard_hidden_nodes": [shared],
            "soft_hidden_nodes": [
                {"node": shared, "role": _join_shared_target_phrase(edge_a["relation"], edge_b["relation"], shared_type), "kind": "shared_target"},
            ],
            "answer_node": follow_edge["dst"],
        }
        surface = {
            "family": "join_then_follow",
            "anchor_texts": [self.env.obfuscated_anchor(a), self.env.obfuscated_anchor(b)],
            "shared_phrase": _join_shared_target_phrase(edge_a["relation"], edge_b["relation"], shared_type),
            "follow_clause": _join_follow_clause(edge_a["relation"], edge_b["relation"], shared_type, follow_edge["relation"], target_type, answer),
            "forbidden_names": [self.env.node_name(shared)],
        }
        proof_graph = [edge_a, edge_b, follow_edge]
        scores = self._score_program("join_then_follow", [a, b], proof_graph, answer, target_type, 3, follow_edge["dst"])
        return QuestionProgram(
            program_id=self.next_id("prog"),
            family="join_then_follow",
            reasoning_family="join_follow",
            goal=f"join_then_follow_{target_type.lower()}",
            difficulty="L3",
            operations=[
                asdict_safe(state.operations[-1]),
                {"op": "follow", "src": shared, "dst": follow_edge["dst"], "relation": follow_edge["relation"]},
            ],
            proof_graph=proof_graph,
            anchors=[a, b],
            answer_node=follow_edge["dst"],
            answer_value=answer,
            target_type=target_type,
            visibility_plan=visibility,
            surface_plan=surface,
            tool_plan=[
                {"tool": "code_interpreter", "reason": f"裁剪识别 {self.env.obfuscated_anchor(a)} 和 {self.env.obfuscated_anchor(b)}"},
                {"tool": "web_search", "reason": f"分别搜索两个实体的{_relation_natural_text(edge_a['relation'], shared_type)}和{_relation_natural_text(edge_b['relation'], shared_type)}，找出共享目标"},
                {"tool": "web_search", "reason": f"继续搜索共享目标的{_relation_natural_text(follow_edge['relation'], target_type)}"},
            ],
            semantic_pivot=_relation_slug(follow_edge["relation"]),
            scores=scores,
            legacy_label="path_reasoning_join_then_follow",
            tool_irreducibility_score=1.0,
            answerability_certificate={"requires_external_fact": True, "requires_visual_anchor": True, "requires_multistep": True},
        )

    def _score_program(self, family: str, anchors: list[str], proof_graph: list[dict], answer_text: str, target_type: str, reasoning_depth: int, answer_node: str | None) -> dict:
        evidence = sum(edge.get("evidence", 0.0) for edge in proof_graph) / max(len(proof_graph), 1)
        visual_dependency = 1.0 if any(anchor in self.env.in_image_nodes for anchor in anchors) else 0.0
        tool_dependency = 1.0 if reasoning_depth >= 2 else 0.6
        novelty = 1.0 if len(set(anchors)) == len(anchors) else 0.5
        info_gain = 1.0 + (0.5 if target_type in {"PERSON", "ORG"} else 0.3 if target_type == "LOCATION" else 0.2)
        generic_penalty = _generic_answer_penalty(answer_text, target_type)
        relation_value = sum(_relation_question_value(edge.get("relation", ""), edge.get("tail_type", target_type)) for edge in proof_graph) / max(len(proof_graph), 1)
        lexicalizability = sum(
            edge.get("relation_profile", {}).get("lexicalizability", 0.0)
            for edge in proof_graph
        ) / max(len(proof_graph), 1)
        askability = sum(
            edge.get("relation_profile", {}).get("askability", 0.0)
            for edge in proof_graph
        ) / max(len(proof_graph), 1)
        answer_specificity = _answer_specificity_score(answer_text, target_type)
        family_bonus = _FAMILY_PRIOR.get(family, 0.5)
        overall = (
            2.0 * visual_dependency
            + 1.8 * tool_dependency
            + 1.6 * evidence
            + 1.2 * info_gain
            + 1.0 * reasoning_depth
            + 0.8 * novelty
            + 0.9 * relation_value
            + 0.7 * lexicalizability
            + 0.6 * askability
            + 0.8 * answer_specificity
            + family_bonus
            - generic_penalty
        )
        return {
            "evidence": round(evidence, 3),
            "visual_dependency": visual_dependency,
            "tool_dependency": tool_dependency,
            "reasoning_depth": reasoning_depth,
            "info_gain": info_gain,
            "relation_value": round(relation_value, 3),
            "askability": round(askability, 3),
            "lexicalizability": round(lexicalizability, 3),
            "answer_specificity": round(answer_specificity, 3),
            "generic_penalty": generic_penalty,
            "overall": round(overall, 3),
        }


def asdict_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return dict(obj)
    return obj


class ProgramVerifier:
    def __init__(self, env: GraphEnv):
        self.env = env

    def _has_direct_answer_edge(self, src: str, answer_node: str | None, relation: str | None = None) -> bool:
        if not answer_node:
            return False
        for edge in self.env.unique_follows(src):
            if edge["dst"] != answer_node:
                continue
            if relation and _relation_slug(edge["relation"]) != _relation_slug(relation):
                continue
            return True
        return False

    def verify(self, program: QuestionProgram) -> tuple[bool, str]:
        if not program.proof_graph:
            return False, "empty_proof"
        if not _answer_ok(program.answer_value):
            return False, "bad_answer"
        if not any(anchor in self.env.in_image_nodes for anchor in program.anchors):
            return False, "no_visual_anchor"
        if any(edge.get("src") == edge.get("dst") for edge in program.proof_graph):
            return False, "self_loop"
        if program.difficulty == "L3" and program.core_operation_count < 3:
            if program.family not in {"join_then_follow", "select_then_follow", "multihop"}:
                return False, "insufficient_depth"
        if program.family == "select_then_follow":
            ops = [op.get("op") for op in program.operations]
            if "compare" not in ops or "select" not in ops or ops.count("follow") < 1:
                return False, "invalid_select_flow"
        if program.family == "join_then_follow":
            ops = [op.get("op") for op in program.operations]
            if "join" not in ops or ops.count("follow") < 1:
                return False, "invalid_join_flow"
        if program.family == "multihop":
            ops = [op.get("op") for op in program.operations]
            if ops.count("follow") < 2:
                return False, "invalid_multihop"
        if program.family == "select_then_follow":
            select_op = next((op for op in program.operations if op.get("op") == "select"), {})
            follow_op = next((op for op in reversed(program.operations) if op.get("op") == "follow"), {})
            winner = select_op.get("winner")
            relation = follow_op.get("relation")
            if not winner or not program.answer_node or not relation:
                return False, "incomplete_select_flow"
            other_anchors = [anchor for anchor in program.anchors if anchor != winner]
            if any(self._has_direct_answer_edge(anchor, program.answer_node, relation) for anchor in other_anchors):
                return False, "redundant_select_step"
        if program.family == "join_then_follow":
            join_op = next((op for op in program.operations if op.get("op") == "join"), {})
            follow_op = next((op for op in reversed(program.operations) if op.get("op") == "follow"), {})
            shared = join_op.get("shared")
            relation = follow_op.get("relation")
            if not shared or not program.answer_node or not relation:
                return False, "incomplete_join_flow"
            if any(self._has_direct_answer_edge(anchor, program.answer_node, relation) for anchor in program.anchors):
                return False, "redundant_join_step"
        if program.family == "same_group":
            if program.answer_node and any(self._has_direct_answer_edge(anchor, program.answer_node) for anchor in program.anchors):
                return False, "redundant_group_step"
        return True, "pass"


def _surface_ontology_penalty(surface_plan: dict) -> float:
    fields = []
    for key in ("role_phrase", "question_intent", "follow_clause", "shared_phrase", "selection_phrase", "relation_phrase"):
        value = surface_plan.get(key)
        if isinstance(value, str):
            fields.append(value)
    return _ontologyese_penalty(" ".join(fields))


def _build_visibility_variants(program: QuestionProgram, env: GraphEnv) -> list[dict]:
    base = json.loads(json.dumps(program.visibility_plan, ensure_ascii=False))
    variants = [base]
    if program.family == "join_then_follow":
        soft = json.loads(json.dumps(base, ensure_ascii=False))
        shared = next((item.get("node") for item in soft.get("soft_hidden_nodes", []) if item.get("kind") == "shared_target"), None)
        if shared:
            soft["hard_hidden_nodes"] = [node for node in soft.get("hard_hidden_nodes", []) if node != shared]
            variants.append(soft)
    if program.family == "select_then_follow":
        soft = json.loads(json.dumps(base, ensure_ascii=False))
        winner = next((item.get("node") for item in soft.get("soft_hidden_nodes", []) if item.get("kind") == "selected_entity"), None)
        if winner:
            soft["hard_hidden_nodes"] = [node for node in soft.get("hard_hidden_nodes", []) if node != winner]
            variants.append(soft)
    deduped = []
    seen = set()
    for variant in variants:
        signature = (
            tuple(sorted(variant.get("hard_hidden_nodes", []))),
            tuple(sorted((item.get("node"), item.get("role"), item.get("kind")) for item in variant.get("soft_hidden_nodes", []))),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(variant)
    return deduped


def _compile_surface_plan(program: QuestionProgram, env: GraphEnv, visibility_plan: dict) -> dict:
    surface = json.loads(json.dumps(program.surface_plan, ensure_ascii=False))
    anchor_texts = [env.obfuscated_anchor(a) for a in program.anchors]
    surface["anchor_texts"] = anchor_texts
    soft_roles = [item.get("role", "") for item in visibility_plan.get("soft_hidden_nodes", []) if item.get("role")]
    surface["soft_roles"] = soft_roles
    surface["forbidden_names"] = [env.node_name(node) for node in visibility_plan.get("hard_hidden_nodes", []) if isinstance(node, str)]
    if program.family == "join_then_follow" and not surface.get("shared_phrase"):
        surface["shared_phrase"] = next((role for role in soft_roles if role.startswith("共同关联")), "共同关联的目标")
    if program.family == "select_then_follow" and not surface.get("selection_phrase"):
        surface["selection_phrase"] = next((role for role in soft_roles if "更早" in role or "更高" in role), "满足条件的那个")
    return surface


def _build_microplans(program: QuestionProgram, surface_plan: dict) -> list[dict]:
    anchors = surface_plan.get("anchor_texts", [])
    head_noun = surface_plan.get("shared_phrase") or surface_plan.get("role_phrase") or surface_plan.get("relation_phrase") or surface_plan.get("question_intent") or "答案"
    wh_form = {
        "ask_person": "谁",
        "ask_location": "哪座城市" if "城市" in surface_plan.get("follow_clause", "") else "哪里",
        "ask_time": "什么时候",
        "ask_org": "哪个机构",
        "ask_quantity": "多少",
        "ask_work": "哪部作品",
    }.get(program.semantic_intent, "什么")

    microplans = []
    if program.family == "select_then_follow":
        selection = surface_plan.get("selection_phrase", "满足条件的那个")
        follow_clause = surface_plan.get("follow_clause", "答案是什么")
        microplans.append({
            "viewpoint": "criterion_first",
            "head_noun": head_noun,
            "selection_form": selection,
            "refer_style": "anchor_pair",
            "wh_form": wh_form,
            "template_hint": f"{anchors[0]}和{anchors[1]}，{selection}的那个，{follow_clause}？" if len(anchors) >= 2 else f"{selection}的那个，{follow_clause}？",
        })
        microplans.append({
            "viewpoint": "anchor_first",
            "head_noun": head_noun,
            "selection_form": selection,
            "refer_style": "anchor_pair",
            "wh_form": wh_form,
            "template_hint": f"{anchors[0]}与{anchors[1]}相比，{selection}的一方，{follow_clause}？" if len(anchors) >= 2 else f"{selection}的一方，{follow_clause}？",
        })
    elif program.family == "join_then_follow":
        shared_phrase = surface_plan.get("shared_phrase", "共同关联的目标")
        follow_clause = surface_plan.get("follow_clause", "答案是什么")
        microplans.append({
            "viewpoint": "shared_target_first",
            "head_noun": shared_phrase,
            "selection_form": "",
            "refer_style": "anchor_pair",
            "wh_form": wh_form,
            "template_hint": f"{anchors[0]}和{anchors[1]}{shared_phrase}，{follow_clause}？" if len(anchors) >= 2 else f"{shared_phrase}，{follow_clause}？",
        })
        microplans.append({
            "viewpoint": "anchor_first",
            "head_noun": shared_phrase,
            "selection_form": "",
            "refer_style": "anchor_pair",
            "wh_form": wh_form,
            "template_hint": f"这两处{shared_phrase}，{follow_clause}？",
        })
    else:
        intent = surface_plan.get("question_intent") or surface_plan.get("follow_clause") or surface_plan.get("relation_phrase") or "答案是什么"
        base_hint = f"{'、'.join(anchors)}，{intent}？" if anchors else f"{intent}？"
        microplans.append({
            "viewpoint": program.realization_schema,
            "head_noun": head_noun,
            "selection_form": surface_plan.get("selection_phrase", ""),
            "refer_style": "anchor_first",
            "wh_form": wh_form,
            "template_hint": base_hint,
        })
    return microplans


def _compute_utility_scores(program: QuestionProgram, env: GraphEnv) -> dict:
    surface = program.surface_plan
    answer_uniqueness = 1.0 if _answer_ok(program.answer_value) and _generic_answer_penalty(program.answer_value, program.target_type) < 1.0 else 0.2
    clue_minimality = _clue_minimality(program)
    reasoning_compactness = max(0.1, 1.1 - 0.12 * max(program.core_operation_count - 3, 0))
    tool_irreducibility = max(program.tool_irreducibility_score, 0.2)
    counterfactual_drop = _counterfactual_drop(program)
    lexicalizability = sum(
        edge.get("relation_profile", {}).get("lexicalizability", 0.0)
        for edge in program.proof_graph
    ) / max(len(program.proof_graph), 1)
    anchor_referability = _anchor_referability(surface.get("anchor_texts", []))
    question_intent_specificity = _question_intent_specificity(surface, program.semantic_intent)
    recoverable_abstraction = _recoverable_abstraction_score(program.visibility_plan)
    ontologyese_penalty = _surface_ontology_penalty(surface)
    ambiguity_penalty = 0.5 if program.target_type == "OTHER" else 0.0
    hidden_unrecoverability_penalty = 0.8 if recoverable_abstraction < 0 else 0.0
    utility = (
        answer_uniqueness
        + clue_minimality
        + reasoning_compactness
        + tool_irreducibility
        + counterfactual_drop
        + lexicalizability
        + anchor_referability
        + question_intent_specificity
        + recoverable_abstraction
        - ontologyese_penalty
        - ambiguity_penalty
        - hidden_unrecoverability_penalty
    )
    return {
        "answer_uniqueness": round(answer_uniqueness, 3),
        "clue_minimality": round(clue_minimality, 3),
        "reasoning_compactness": round(reasoning_compactness, 3),
        "tool_irreducibility": round(tool_irreducibility, 3),
        "counterfactual_drop": round(counterfactual_drop, 3),
        "lexicalizability": round(lexicalizability, 3),
        "anchor_referability": round(anchor_referability, 3),
        "question_intent_specificity": round(question_intent_specificity, 3),
        "recoverable_abstraction": round(recoverable_abstraction, 3),
        "ontologyese_penalty": round(ontologyese_penalty, 3),
        "utility": round(utility, 3),
    }


def _apply_editor_enrichment(program: QuestionProgram, env: GraphEnv) -> QuestionProgram:
    variants = []
    for visibility in _build_visibility_variants(program, env):
        surface = _compile_surface_plan(program, env, visibility)
        utility_seed = (
            _recoverable_abstraction_score(visibility)
            + _question_intent_specificity(surface, program.semantic_intent)
            - _surface_ontology_penalty(surface)
        )
        variants.append((utility_seed, visibility, surface))
    variants.sort(key=lambda item: item[0], reverse=True)
    _, best_visibility, best_surface = variants[0]
    program.visibility_plan = best_visibility
    program.surface_plan = best_surface
    program.surface_plan["microplans"] = _build_microplans(program, program.surface_plan)
    utility_scores = _compute_utility_scores(program, env)
    program.scores.update(utility_scores)
    program.scores["overall"] = round(program.scores.get("overall", 0.0) + 0.9 * utility_scores["utility"], 3)
    return program


def _pairwise_rerank(programs: list[QuestionProgram], env: GraphEnv) -> list[QuestionProgram]:
    if not programs:
        return []
    wins = defaultdict(float)
    for i, left in enumerate(programs):
        for right in programs[i + 1:]:
            same_answer = _program_answer_signature(left) == _program_answer_signature(right)
            same_anchor = _program_anchor_signature(left, env) == _program_anchor_signature(right, env)
            if not (same_answer or same_anchor):
                continue
            left_score = (
                left.scores.get("utility", 0.0)
                + left.scores.get("lexicalizability", 0.0)
                + left.scores.get("question_intent_specificity", 0.0)
                - left.scores.get("ontologyese_penalty", 0.0)
            )
            right_score = (
                right.scores.get("utility", 0.0)
                + right.scores.get("lexicalizability", 0.0)
                + right.scores.get("question_intent_specificity", 0.0)
                - right.scores.get("ontologyese_penalty", 0.0)
            )
            if left_score >= right_score:
                wins[left.program_id] += 1.0
            else:
                wins[right.program_id] += 1.0
    for program in programs:
        program.scores["pairwise_preference"] = round(wins.get(program.program_id, 0.0), 3)
        program.scores["selection_score"] = round(
            program.scores.get("overall", 0.0)
            + 0.8 * program.scores.get("utility", 0.0)
            + 0.4 * wins.get(program.program_id, 0.0),
            3,
        )
    return sorted(programs, key=lambda p: p.scores.get("selection_score", p.scores.get("overall", 0.0)), reverse=True)


REALIZATION_RULES = {
    "lookup": "单实体查找题。问题必须直接围绕一个图中实体的一跳属性，用位置描述指代图中实体。禁止使用“对应的组织或机构”“相关属性”这类泛化表达，优先把 relation 具体化成自然问法。",
    "compare": "平行比较题。问题必须询问两个图中实体里谁更早/更高。",
    "delta": "差值题。问题必须询问两个图中实体之间相差多少。",
    "same_target": "同一目标判断题。问题必须问两个图中实体的相关对象是否相同。",
    "same_group": "同组归属判断题。问题必须问两个图中实体是否归属同一国家/地区/组织。",
    "extremum": "集合极值题。问题必须在多个图中实体中找最值。",
    "threshold": "阈值判断题。问题必须问前两个实体的数值之和是否超过第三个。",
    "multihop": "多跳题。问题必须保留中间角色描述，不得泄露中间节点名。禁止使用“对应的地点/组织是什么”这类生硬表达，必须把 relation 具体化。",
    "select_then_follow": "路径依赖选择后继续题。题面必须体现先选出满足条件的实体，再继续查它的下一步属性。禁止使用“对应的组织或机构是什么”“所对应的两个实体里”这类程序味表达，应直接写成“首演时间更早的那部音乐剧”“成立更早的那个品牌”等自然短语。",
    "join_then_follow": "路径依赖汇合后继续题。题面必须体现两个图中实体共同关联的角色，再追问共享目标的下一步属性。禁止出现隐藏目标/共享节点/那处地点，也不要用“对应的地点是什么”这类泛化句式。",
}

REALIZATION_OUTPUT = """请输出严格 JSON：\n{\n  \"questions\": [\n    {\n      \"program_id\": \"...\",\n      \"question\": \"...\",\n      \"answer\": \"...\",\n      \"tool_sequence\": [{\"step\":1,\"tool\":\"...\",\"action\":\"...\",\"input\":\"...\",\"expected_output\":\"...\"}],\n      \"rationale\": \"...\"\n    }\n  ]\n}\n"""


def _program_family_prompt(family: str) -> str:
    return REALIZATION_RULES.get(family, "根据给定语义计划，生成自然、单问号的问题。")


def _format_programs(programs: list[QuestionProgram], env: GraphEnv) -> str:
    lines = []
    for idx, program in enumerate(programs):
        lines.append(f"### Program {idx}")
        lines.append(f"ID: {program.program_id}")
        lines.append(f"Family: {program.family}")
        lines.append(f"Goal: {program.goal}")
        lines.append(f"Difficulty: {program.difficulty}")
        lines.append(f"Anchors: {', '.join(env.obfuscated_anchor(a) for a in program.anchors)}")
        lines.append(f"SurfacePlan: {json.dumps(program.surface_plan, ensure_ascii=False)}")
        lines.append(f"VisibilityPlan: {json.dumps(program.visibility_plan, ensure_ascii=False)}")
        lines.append(f"Answer: {program.answer_value}")
        lines.append(f"ToolPlan: {json.dumps(program.tool_plan, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def _single_program_brief(program: QuestionProgram, env: GraphEnv) -> str:
    anchors = "、".join(env.obfuscated_anchor(a) for a in program.anchors)
    hidden_roles = []
    for item in program.visibility_plan.get("soft_hidden_nodes", []):
        role = item.get("role")
        if role:
            hidden_roles.append(role)
    proof_lines = []
    for edge in program.proof_graph:
        src = env.node_name(edge.get("src", ""))
        dst = env.node_name(edge.get("dst", ""))
        rel_text = _relation_natural_text(edge.get("relation", ""), edge.get("tail_type", "OTHER"))
        proof_lines.append(f"- {src} --{rel_text}--> {dst}")
    return "\n".join([
        f"ProgramID: {program.program_id}",
        f"Family: {program.family}",
        f"ReasoningFamily: {program.reasoning_family}",
        f"Difficulty: {program.difficulty}",
        f"Anchors: {anchors}",
        f"Goal: {program.goal}",
        f"TargetType: {program.target_type}",
        f"SemanticIntent: {program.semantic_intent}",
        f"RealizationSchema: {program.realization_schema}",
        f"SemanticPivot: {program.semantic_pivot}",
        f"Operations: {json.dumps(program.operations, ensure_ascii=False)}",
        f"SurfacePlan: {json.dumps(program.surface_plan, ensure_ascii=False)}",
        f"VisibilityPlan: {json.dumps(program.visibility_plan, ensure_ascii=False)}",
        f"HiddenRoles: {json.dumps(hidden_roles, ensure_ascii=False)}",
        f"Microplans: {json.dumps(program.surface_plan.get('microplans', []), ensure_ascii=False)}",
        f"Answer: {program.answer_value}",
        "ProofGraph:",
        *proof_lines,
    ])


def _postcheck_question_text(question: str, program: QuestionProgram) -> bool:
    if not question or question.count("?") + question.count("？") != 1:
        return False
    bad_phrases = ["隐藏目标", "共享节点", "那处地点"]
    if any(bp in question for bp in bad_phrases):
        return False
    forbidden = set(program.surface_plan.get("forbidden_names", []))
    forbidden.update(program.visibility_plan.get("hard_hidden_nodes", []))
    forbidden_names = {
        name for name in forbidden if isinstance(name, str) and len(name) >= 2
    }
    for name in forbidden_names:
        if name and name in question:
            return False
    if program.family == "join_then_follow":
        shared_phrase = (program.surface_plan.get("shared_phrase") or "").strip()
        if shared_phrase and shared_phrase not in question:
            return False
    if program.family == "select_then_follow":
        selection_phrase = (program.surface_plan.get("selection_phrase") or "").strip()
        if selection_phrase and selection_phrase not in question:
            return False
    if _ontologyese_penalty(question) > 0:
        return False
    return True


def _roundtrip_semantic_check(question: str, program: QuestionProgram, microplan: dict) -> bool:
    if not _postcheck_question_text(question, program):
        return False
    wh_form = (microplan.get("wh_form") or "").strip()
    if wh_form and wh_form in {"谁", "哪座城市", "哪里", "什么时候", "哪个机构", "多少", "哪部作品"}:
        if wh_form not in question:
            if wh_form == "哪里" and "哪座城市" in question:
                pass
            elif wh_form == "哪个机构" and any(token in question for token in ("哪个公司", "哪家机构", "哪一机构")):
                pass
            elif wh_form == "什么时候" and any(token in question for token in ("哪一年", "何时")):
                pass
            else:
                return False
    if program.family == "join_then_follow":
        shared_phrase = (program.surface_plan.get("shared_phrase") or "").strip()
        if shared_phrase and shared_phrase not in question:
            return False
    if program.family == "select_then_follow":
        selection_phrase = (program.surface_plan.get("selection_phrase") or "").strip()
        if selection_phrase and selection_phrase not in question:
            return False
    return True


def _fallback_question(program: QuestionProgram, env: GraphEnv) -> dict:
    anchors = [env.obfuscated_anchor(a) for a in program.anchors]
    surface = program.surface_plan
    family = program.family
    if family == "lookup":
        q = f"{anchors[0]}，{surface['question_intent']}？"
    elif family == "compare":
        q = f"{anchors[0]}和{anchors[1]}，{surface['compare_phrase']}的是哪一个？"
    elif family == "delta":
        noun = "时间属性" if program.target_type == "TIME" else f"{surface.get('relation_phrase','数值属性')}"
        q = f"{anchors[0]}和{anchors[1]}，它们的{noun}相差多少？"
    elif family == "same_target":
        q = f"{anchors[0]}的{surface['role_a']}和{anchors[1]}的{surface['role_b']}，是否指向同一个目标？"
    elif family == "same_group":
        noun = "国家或地区" if program.target_type == "LOCATION" else "上级组织"
        q = f"{anchors[0]}和{anchors[1]}，是否归属于同一个{noun}？"
    elif family == "extremum":
        q = f"{'、'.join(anchors)}，哪个的{surface['relation_phrase']}最{'早' if program.target_type == 'TIME' else '高'}？"
    elif family == "threshold":
        q = f"{anchors[0]}和{anchors[1]}的{surface['relation_phrase']}加起来，是否超过{anchors[2]}的对应数值？"
    elif family == "multihop":
        q = f"{anchors[0]}，其{surface['mid_role']}的{surface['question_intent']}？"
    elif family == "select_then_follow":
        q = f"{anchors[0]}和{anchors[1]}，{surface['selection_phrase']}的那个，{surface['follow_clause']}？"
    elif family == "join_then_follow":
        q = f"{anchors[0]}和{anchors[1]}{surface['shared_phrase']}，{surface['follow_clause']}？"
    else:
        q = f"{anchors[0]}，相关答案是什么？"

    tool_sequence = []
    for idx, step in enumerate(program.tool_plan, start=1):
        tool_sequence.append({
            "step": idx,
            "tool": step["tool"],
            "action": step["reason"],
            "input": "图片与中间结果",
            "expected_output": "下一步所需信息",
        })
    return {
        "question": q,
        "answer": program.answer_value,
        "tool_sequence": tool_sequence,
        "rationale": f"基于 {program.family} 程序与 proof graph 生成问题。",
    }

def _draft_paraphrases(program: QuestionProgram, microplan: dict, env: GraphEnv, image_b64: str, image_description: str, domain: str) -> list[dict]:
    exemplars = "\n".join(f"- {item}" for item in _RHETORICAL_EXEMPLARS.get(program.family, []))
    prompt = (
        "你需要把一个结构化 QuestionProgram 和一个 microplan 变成自然中文问题。\n"
        f"图片描述：{image_description}\n领域：{domain}\n"
        f"规则：{_program_family_prompt(program.family)}\n"
        "要求：\n"
        "- 只改写表达，不要改变语义条件。\n"
        "- 不得泄露 hard_hidden 节点名。\n"
        "- 必须保留 microplan 指定的句法重心和疑问类型。\n"
        "- 输出两种不同但语义等价的自然问法。\n"
        "- 强禁用这些表达：对应的、所对应的、实体、对象、相关信息、组织或机构是什么。\n"
        f"- 可参考的优质问法风格：\n{exemplars or '- 无'}\n"
        "输出严格 JSON：\n"
        "{\n"
        "  \"drafts\": [\n"
        "    {\"question\": \"...\", \"answer\": \"...\", \"tool_sequence\": [{\"step\":1,\"tool\":\"...\",\"action\":\"...\",\"input\":\"...\",\"expected_output\":\"...\"}], \"rationale\": \"...\"},\n"
        "    {\"question\": \"...\", \"answer\": \"...\", \"tool_sequence\": [{\"step\":1,\"tool\":\"...\",\"action\":\"...\",\"input\":\"...\",\"expected_output\":\"...\"}], \"rationale\": \"...\"}\n"
        "  ]\n"
        "}\n"
        f"Program:\n{_single_program_brief(program, env)}\n"
        f"Microplan: {json.dumps(microplan, ensure_ascii=False)}"
    )
    result = call_vlm_json(
        prompt,
        f"请为 {program.family} 程序生成自然中文问题草稿。",
        image_b64=image_b64,
        max_tokens=2048,
        temperature=0.8,
        max_attempts=1,
    )
    if isinstance(result, dict):
        drafts = result.get("drafts", [])
        if isinstance(drafts, list):
            return [d for d in drafts if isinstance(d, dict)]
    return []


def _correct_draft(program: QuestionProgram, microplan: dict, draft: dict, image_b64: str) -> dict | None:
    prompt = (
        "你是中文问题编辑器。请在不改变逻辑条件的前提下，把这道问题修得更像真人提问。\n"
        "要求：\n"
        "- 不得增删语义条件。\n"
        "- 不得泄露 hard_hidden 节点。\n"
        "- 只允许做局部词法、指代、语序优化。\n"
        "- 禁止使用“对应的/实体/对象/相关信息/组织或机构是什么”。\n"
        "输出严格 JSON：\n"
        "{\"question\":\"...\",\"answer\":\"...\",\"tool_sequence\":[...],\"rationale\":\"...\"}\n"
        f"ProgramID: {program.program_id}\n"
        f"Family: {program.family}\n"
        f"Microplan: {json.dumps(microplan, ensure_ascii=False)}\n"
        f"Draft: {json.dumps(draft, ensure_ascii=False)}"
    )
    result = call_vlm_json(
        prompt,
        "请做局部改写，不要改变语义。",
        image_b64=image_b64,
        max_tokens=1024,
        temperature=0.4,
        max_attempts=1,
    )
    return result if isinstance(result, dict) else None


def realize_programs(programs: list[QuestionProgram], env: GraphEnv, image_b64: str, image_description: str, domain: str) -> list[dict]:
    if not programs:
        return []
    realized: list[dict] = []
    for program in programs:
        candidates = []
        microplans = program.surface_plan.get("microplans", []) or _build_microplans(program, program.surface_plan)
        for microplan in microplans[:2]:
            drafts = _draft_paraphrases(program, microplan, env, image_b64, image_description, domain)
            for draft in drafts[:2]:
                question = draft.get("question", "")
                if not _roundtrip_semantic_check(question, program, microplan):
                    continue
                corrected = _correct_draft(program, microplan, draft, image_b64) or draft
                corrected_q = corrected.get("question", "")
                if not _roundtrip_semantic_check(corrected_q, program, microplan):
                    corrected = draft
                    corrected_q = question
                if not _roundtrip_semantic_check(corrected_q, program, microplan):
                    continue
                local_quality = (
                    program.scores.get("utility", 0.0)
                    + _question_intent_specificity(program.surface_plan, program.semantic_intent)
                    - _ontologyese_penalty(corrected_q)
                )
                candidates.append((local_quality, corrected))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            result = candidates[0][1]
            answer = result.get("answer") or program.answer_value
            tool_sequence = result.get("tool_sequence") or _fallback_question(program, env)["tool_sequence"]
            realized.append({
                "question": result.get("question", ""),
                "answer": answer,
                "tool_sequence": tool_sequence,
                "rationale": result.get("rationale", f"基于 {program.family} 程序生成。"),
                "reasoning_path": program.to_reasoning_path(),
                "obfuscation_applied": True,
                "obfuscated_entities": [env.obfuscated_anchor(a) for a in program.anchors],
                "level": program.difficulty,
            })
            continue
        fallback = _fallback_question(program, env)
        realized.append({
            **fallback,
            "reasoning_path": program.to_reasoning_path(),
            "obfuscation_applied": True,
            "obfuscated_entities": [env.obfuscated_anchor(a) for a in program.anchors],
            "level": program.difficulty,
        })
    return realized


def _program_answer_signature(program: QuestionProgram) -> str:
    return re.sub(r"\s+", " ", program.answer_value.strip().lower())


def _program_anchor_signature(program: QuestionProgram, env: GraphEnv) -> tuple[str, ...]:
    return tuple(sorted(env.obfuscated_anchor(a) for a in program.anchors))


def _program_pair_signature(program: QuestionProgram) -> tuple[str, ...]:
    anchors = tuple(sorted(program.anchors))
    return anchors if len(anchors) >= 2 else ()


def _select_programs(
    programs: list[QuestionProgram],
    env: GraphEnv,
    max_total: int,
    max_per_family: int,
    max_per_anchor: int,
    max_per_entity: int,
    max_per_pair: int,
    seed_programs: list[QuestionProgram] | None = None,
) -> list[QuestionProgram]:
    if not programs:
        return []
    seed_programs = seed_programs or []
    family_counts = defaultdict(int)
    anchor_counts = defaultdict(int)
    entity_counts = defaultdict(int)
    pair_counts = defaultdict(int)
    seen_answers = set()

    for program in seed_programs:
        family_counts[program.family] += 1
        anchor_counts[_program_anchor_signature(program, env)] += 1
        for anchor in program.anchors:
            entity_counts[anchor] += 1
        pair = _program_pair_signature(program)
        if pair:
            pair_counts[pair] += 1
        answer_sig = _program_answer_signature(program)
        if answer_sig:
            seen_answers.add(answer_sig)

    ranked = sorted(
        programs,
        key=lambda p: (
            p.scores.get("selection_score", p.scores.get("overall", 0.0)),
            p.scores.get("utility", 0.0),
        ),
        reverse=True,
    )
    selected: list[QuestionProgram] = []
    for program in ranked:
        if len(selected) >= max_total:
            break
        if family_counts[program.family] >= max_per_family:
            continue
        answer_sig = _program_answer_signature(program)
        if answer_sig and answer_sig in seen_answers:
            continue
        anchor_sig = _program_anchor_signature(program, env)
        if anchor_counts[anchor_sig] >= max_per_anchor:
            continue
        if any(entity_counts[a] >= max_per_entity for a in program.anchors):
            continue
        pair_sig = _program_pair_signature(program)
        if pair_sig and pair_counts[pair_sig] >= max_per_pair:
            continue

        selected.append(program)
        family_counts[program.family] += 1
        anchor_counts[anchor_sig] += 1
        for anchor in program.anchors:
            entity_counts[anchor] += 1
        if pair_sig:
            pair_counts[pair_sig] += 1
        if answer_sig:
            seen_answers.add(answer_sig)
    return selected


def build_question_programs(G: nx.DiGraph, nodes: dict) -> tuple[list[QuestionProgram], dict]:
    env = GraphEnv(G, nodes)
    proposer = BeamProgramProposer(env)
    programs = proposer.propose()
    verifier = ProgramVerifier(env)
    accepted = []
    rejected = []
    for program in programs:
        ok, reason = verifier.verify(program)
        if ok:
            accepted.append(_apply_editor_enrichment(program, env))
        else:
            rejected.append((program.program_id, reason))

    accepted = _pairwise_rerank(accepted, env)

    meta = {
        "raw_program_count": len(programs),
        "accepted_program_count": len(accepted),
        "rejected_program_count": len(rejected),
        "family_counts": dict(_count_by_family(accepted)),
        "top_utility_programs": [
            {
                "program_id": p.program_id,
                "family": p.family,
                "utility": p.scores.get("utility", 0.0),
                "selection_score": p.scores.get("selection_score", 0.0),
            }
            for p in accepted[:10]
        ],
        "rejected_reasons": rejected[:20],
    }
    return accepted, meta


def _count_by_family(programs: list[QuestionProgram]) -> defaultdict[str, int]:
    counts = defaultdict(int)
    for program in programs:
        counts[program.family] += 1
    return counts


def select_programs_for_levels(programs: list[QuestionProgram], G: nx.DiGraph, nodes: dict) -> tuple[list[QuestionProgram], list[QuestionProgram], dict]:
    env = GraphEnv(G, nodes)
    l2_raw = [p for p in programs if p.difficulty == "L2" and p.scores.get("utility", 0.0) >= 1.8]
    l3_raw = [p for p in programs if p.difficulty == "L3" and p.scores.get("utility", 0.0) >= 2.6]

    l2_non_lookup = _select_programs(
        [p for p in l2_raw if p.family != "lookup"],
        env,
        max_total=8,
        max_per_family=2,
        max_per_anchor=2,
        max_per_entity=3,
        max_per_pair=2,
    )
    remaining_l2 = [p for p in l2_raw if p not in l2_non_lookup]
    l2 = l2_non_lookup + _select_programs(
        remaining_l2,
        env,
        max_total=max(0, 8 - len(l2_non_lookup)),
        max_per_family=1,
        max_per_anchor=2,
        max_per_entity=3,
        max_per_pair=2,
        seed_programs=l2_non_lookup,
    )

    preferred_l3 = []
    for family in ("select_then_follow", "join_then_follow"):
        picked = _select_programs(
            [p for p in l3_raw if p.family == family],
            env,
            max_total=1,
            max_per_family=1,
            max_per_anchor=99,
            max_per_entity=99,
            max_per_pair=99,
            seed_programs=preferred_l3,
        )
        preferred_l3.extend(picked)

    remaining = [p for p in l3_raw if p not in preferred_l3]
    l3 = preferred_l3 + _select_programs(
        remaining,
        env,
        max_total=max(0, 3 - len(preferred_l3)),
        max_per_family=2,
        max_per_anchor=2,
        max_per_entity=2,
        max_per_pair=1,
        seed_programs=l2 + preferred_l3,
    )

    meta = {
        "level_2_raw_count": len(l2_raw),
        "level_3_raw_count": len(l3_raw),
        "selected_l2_families": dict(_count_by_family(l2)),
        "selected_l3_families": dict(_count_by_family(l3)),
    }
    return l2, l3, meta
