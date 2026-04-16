from __future__ import annotations

import re
from dataclasses import dataclass

_BAD_ANSWERS = {"true", "false", "yes", "no", "none", "n/a", "unknown", "various", "multiple"}
_LOW_VALUE_RELATIONS = {
    "related_to", "associated_with", "connected_to",
    "located_left_of", "located_right_of", "located_above", "located_below",
}
_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}
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
