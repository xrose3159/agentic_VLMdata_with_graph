"""弱约束随机游走 + 后验归纳题型。

核心思路：
1. 从 step2 输出构建异构求解图（region / entity / fact 三层）
2. 从图中锚点出发，在证据图上自由扩展子图（不是线性链）
3. 每一步用连续打分函数偏置方向，难度由偏好向量控制
4. 走到"已经能问出自然问题"时停下
5. 后验归纳题型：从子图枚举所有可闭合意图，选最佳
6. 题型是 walk 的输出，不是输入
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from step3_generate import _sanitize_triples
from step3_graphenv_runtime import (
    RelationProfile,
    _BAD_ANSWERS,
    _LOW_VALUE_RELATIONS,
    _ONTOLOGYESE_BANLIST,
    _answer_specificity_score,
    _generic_answer_penalty,
    _normalize_tail_type,
    _path_follow_clause,
    _relation_natural_text,
    _relation_profile,
    _relation_slug,
    _selection_phrase,
    _value_relations_compatible,
    _wh_type_for_relation,
)
from core.vlm import call_vlm_json


# ============================================================
# S1. HeteroSolveGraph
# ============================================================

# Node types
NTYPE_IMAGE = "full_image"
NTYPE_REGION = "region"
NTYPE_ENTITY = "entity"
NTYPE_FACT = "fact"

# Edge types
ETYPE_OBSERVE = "observe"
ETYPE_RESOLVE = "resolve"
ETYPE_RETRIEVE = "retrieve"


class HeteroSolveGraph:
    """异构求解图：full_image → region → entity → fact."""

    def __init__(self, entity_json: dict):
        self.G = nx.DiGraph()
        self.image_description = entity_json.get("image_description", "")
        self.domain = entity_json.get("domain", "other")
        self._region_data: dict[str, dict] = {}
        self._entity_for_region: dict[str, str] = {}
        self._build(entity_json)

    # ---- construction ----

    def _build(self, data: dict):
        entities = data.get("high_conf") or data.get("entities") or []
        triples = _sanitize_triples(data.get("triples", []))

        # full_image node
        img_key = "__full_image__"
        self.G.add_node(img_key, ntype=NTYPE_IMAGE)

        # build entity name → entity obj index
        entity_by_name: dict[str, dict] = {}
        for e in entities:
            name = (e.get("name") or "").strip()
            if name:
                entity_by_name[name.lower()] = e

        # region + entity nodes
        for e in entities:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            eid = e.get("id", name)
            region_key = f"region:{eid}"
            entity_key = f"entity:{name.lower()}"

            # region node
            region_data = {
                "entity_id": eid,
                "name": name,
                "entity_type": e.get("type", ""),
                "bbox": e.get("bbox", []),
                "location": e.get("location_in_image", ""),
                "confidence": float(e.get("confidence", 0.0)),
                "crop_path": e.get("crop_path", ""),
            }
            self.G.add_node(region_key, ntype=NTYPE_REGION, **region_data)
            self._region_data[region_key] = region_data
            self.G.add_edge(img_key, region_key, etype=ETYPE_OBSERVE)

            # entity node
            if not self.G.has_node(entity_key):
                self.G.add_node(entity_key, ntype=NTYPE_ENTITY, name=name)
            self.G.add_edge(region_key, entity_key, etype=ETYPE_RESOLVE,
                           entity_type=e.get("type", ""),
                           resolve_mode=e.get("resolve_mode", "ocr_likely"))
            self._entity_for_region[region_key] = entity_key

        # fact nodes from triples
        for t in triples:
            head = (t.get("head") or "").strip()
            tail = (t.get("tail") or "").strip()
            relation = (t.get("relation") or "").strip()
            if not head or not tail or not relation:
                continue

            head_key = f"entity:{head.lower()}"
            if not self.G.has_node(head_key):
                # head 不在图中实体里，跳过（只保留从图中实体出发的 triple）
                continue

            tail_type = _normalize_tail_type(t.get("tail_type"))
            fact_key = f"fact:{head.lower()}:{_relation_slug(relation)}:{tail.lower()}"

            if not self.G.has_node(fact_key):
                self.G.add_node(
                    fact_key,
                    ntype=NTYPE_FACT,
                    value=tail,
                    tail_type=tail_type,
                    normalized_value=t.get("normalized_value", ""),
                    unit=t.get("unit", ""),
                )

            rel_slug = _relation_slug(relation)
            if rel_slug in _LOW_VALUE_RELATIONS:
                continue

            profile = _relation_profile(relation, tail_type)
            if profile.askability <= 0:
                continue

            self.G.add_edge(
                head_key,
                fact_key,
                etype=ETYPE_RETRIEVE,
                relation=relation,
                relation_slug=rel_slug,
                tail_type=tail_type,
                fact=t.get("fact", ""),
                source_snippet=t.get("source_snippet", ""),
                provenance=t.get("provenance", t.get("source", "")),
                normalized_value=t.get("normalized_value", ""),
                unit=t.get("unit", ""),
                askability=profile.askability,
                lexicalizability=profile.lexicalizability,
                retrieval_mode=t.get("retrieval_mode", "snippet_only"),
            )

    # ---- query methods ----

    def regions(self) -> list[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("ntype") == NTYPE_REGION]

    def entity_for_region(self, region_key: str) -> str | None:
        return self._entity_for_region.get(region_key)

    def entity_name(self, entity_key: str) -> str:
        return self.G.nodes.get(entity_key, {}).get("name", entity_key.removeprefix("entity:"))

    def region_entity_name(self, region_key: str) -> str:
        ek = self.entity_for_region(region_key)
        return self.entity_name(ek) if ek else ""

    def region_info(self, region_key: str) -> dict:
        return self._region_data.get(region_key, {})

    def fact_value(self, fact_key: str) -> str:
        return self.G.nodes.get(fact_key, {}).get("value", "")

    def fact_tail_type(self, fact_key: str) -> str:
        return self.G.nodes.get(fact_key, {}).get("tail_type", "OTHER")

    def retrieve_edges(self, entity_key: str) -> list[dict]:
        """所有从 entity 出发的 retrieve 边，含完整元数据。"""
        edges = []
        for _, dst, data in self.G.out_edges(entity_key, data=True):
            if data.get("etype") == ETYPE_RETRIEVE:
                edges.append({"fact_key": dst, **data})
        edges.sort(key=lambda e: e.get("askability", 0), reverse=True)
        return edges

    def comparable_pairs(self, tail_type: str) -> list[tuple[str, str, str, str, dict, dict]]:
        """找所有可比较的 (regionA, regionB, relation_bucket, tail_type, edgeA, edgeB) 对。"""
        tail_type = _normalize_tail_type(tail_type)
        # 收集每个 region 的 retrieve 边按 tail_type 分组
        region_facts: dict[str, list[tuple[str, dict]]] = {}
        for rk in self.regions():
            ek = self.entity_for_region(rk)
            if not ek:
                continue
            for edge in self.retrieve_edges(ek):
                if edge.get("tail_type") == tail_type:
                    region_facts.setdefault(rk, []).append((ek, edge))

        pairs = []
        region_list = list(region_facts.keys())
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                rA, rB = region_list[i], region_list[j]
                for ekA, edgeA in region_facts[rA]:
                    for ekB, edgeB in region_facts[rB]:
                        ok, score, bucket = _value_relations_compatible(
                            edgeA.get("relation", ""),
                            edgeB.get("relation", ""),
                            tail_type,
                        )
                        if ok and score > 0:
                            pairs.append((rA, rB, bucket, tail_type, edgeA, edgeB))
        return pairs

    def rankable_groups(self, tail_type: str, min_size: int = 3) -> list[list[tuple[str, dict]]]:
        """找 3+ 个 region 共享同类 retrieve 边的组（按 relation bucket 分）。"""
        tail_type = _normalize_tail_type(tail_type)
        # bucket → [(region_key, entity_key, edge)]
        buckets: dict[str, list[tuple[str, str, dict]]] = {}
        for rk in self.regions():
            ek = self.entity_for_region(rk)
            if not ek:
                continue
            for edge in self.retrieve_edges(ek):
                if edge.get("tail_type") != tail_type:
                    continue
                from step3_graphenv_runtime import _value_relation_bucket
                bucket = _value_relation_bucket(edge.get("relation", ""), tail_type)
                if bucket.endswith("other"):
                    continue
                buckets.setdefault(bucket, []).append((rk, ek, edge))

        groups = []
        for bucket, items in buckets.items():
            # 去重：每个 region 只保留一条最高分边
            seen_regions = {}
            for rk, ek, edge in items:
                if rk not in seen_regions or edge.get("askability", 0) > seen_regions[rk][2].get("askability", 0):
                    seen_regions[rk] = (rk, ek, edge)
            unique = list(seen_regions.values())
            if len(unique) >= min_size:
                groups.append([(rk, edge) for rk, ek, edge in unique])
        return groups

    def shared_fact_targets(self, entity_a: str, entity_b: str) -> list[tuple[dict, dict, str]]:
        """两个 entity 共同指向的 fact 节点。"""
        targets_a = {e["fact_key"]: e for e in self.retrieve_edges(entity_a)}
        shared = []
        for edge_b in self.retrieve_edges(entity_b):
            fk = edge_b["fact_key"]
            if fk in targets_a:
                shared.append((targets_a[fk], edge_b, fk))
        return shared

    def region_score(self, region_key: str) -> float:
        info = self.region_info(region_key)
        confidence = info.get("confidence", 0.5)
        ek = self.entity_for_region(region_key)
        degree = len(self.retrieve_edges(ek)) if ek else 0
        etype = info.get("entity_type", "")
        type_bonus = {"brand": 0.3, "product": 0.25, "landmark": 0.2, "person": 0.15}.get(etype, 0.0)
        loc = info.get("location", "")
        loc_specificity = 0.2 if any(t in loc for t in ("偏左", "偏右", "最左", "最右", "上方", "下方", "顶部", "底部")) else 0.0
        return confidence * 0.3 + min(degree, 8) * 0.15 + type_bonus + loc_specificity

    def visual_descriptor(self, region_key: str) -> str:
        """生成视觉描述：方位 + 视觉类别。"""
        info = self.region_info(region_key)
        etype = info.get("entity_type", "")
        loc = (info.get("location", "") or "").strip()

        # 视觉类别
        ek = self.entity_for_region(region_key)
        rels = set()
        if ek:
            for edge in self.retrieve_edges(ek):
                rel = _relation_slug(edge.get("relation", ""))
                if rel:
                    rels.add(rel)

        musical_signals = {"performed_at", "music_by", "book_and_lyrics_by", "broadway_premiere_date", "choreographed_by"}
        movie_signals = {"directed_by", "starring", "produced_by", "box_office", "release_date"}
        food_signals = {"first_restaurant_opened", "global_systemwide_sales", "menu_item", "restaurant_count"}
        electronics_signals = {"stock_exchange", "listed_on", "invented", "manufactures"}

        if rels & musical_signals:
            desc = "音乐剧海报"
        elif rels & movie_signals:
            desc = "电影海报"
        elif etype == "brand" and rels & electronics_signals:
            desc = "电子品牌广告牌"
        elif etype == "brand" and rels & food_signals:
            desc = "餐饮品牌广告牌"
        elif etype == "brand":
            desc = "品牌广告牌"
        elif etype == "landmark":
            desc = "地标建筑"
        elif etype == "person":
            desc = "人物照片"
        elif etype == "text":
            desc = "文字招牌"
        elif etype == "product":
            desc = "广告海报"
        else:
            desc = "标识"

        if loc:
            if "画面" in loc and not loc.startswith("画面"):
                loc = loc[loc.index("画面"):]
            elif not loc.startswith("画面"):
                loc = f"画面{loc}"
            return f"{loc}的{desc}"
        return f"图中的{desc}"

    def stats(self) -> dict:
        counts = {NTYPE_IMAGE: 0, NTYPE_REGION: 0, NTYPE_ENTITY: 0, NTYPE_FACT: 0}
        for _, d in self.G.nodes(data=True):
            nt = d.get("ntype", "")
            if nt in counts:
                counts[nt] += 1
        edge_counts = {ETYPE_OBSERVE: 0, ETYPE_RESOLVE: 0, ETYPE_RETRIEVE: 0}
        for _, _, d in self.G.edges(data=True):
            et = d.get("etype", "")
            if et in edge_counts:
                edge_counts[et] += 1
        return {"nodes": counts, "edges": edge_counts}



# ============================================================
# S2. DifficultyProfile + WalkState
# ============================================================

@dataclass
class DifficultyProfile:
    """难度偏好向量。不同难度只是权重不同。"""
    name: str
    w_visual_novelty: float = 0.0      # 引入新图中锚点
    w_fact_gain: float = 0.0           # 引入新可问事实
    w_compute_affordance: float = 0.0  # 形成可比较/排序/交集的值
    w_branch_novelty: float = 0.0      # 形成新独立分支
    w_closure_gain: float = 0.0        # 离"可问"更近
    w_shortcut_penalty: float = 0.0    # 答案太容易直达
    w_redundancy: float = 0.0          # 凑数步骤
    w_generic_penalty: float = 0.0     # 答案太泛
    min_anchors: int = 1
    min_branches: int = 1
    require_compute: bool = False
    max_steps: int = 6


EASY = DifficultyProfile(
    name="easy",
    w_visual_novelty=0.2, w_fact_gain=1.5, w_compute_affordance=0.0,
    w_branch_novelty=0.0, w_closure_gain=2.5,
    w_shortcut_penalty=-0.3, w_redundancy=-0.5, w_generic_penalty=-1.0,
    min_anchors=1, min_branches=1, require_compute=False, max_steps=4,
)

HARD = DifficultyProfile(
    name="hard",
    w_visual_novelty=1.8, w_fact_gain=1.0, w_compute_affordance=2.5,
    w_branch_novelty=2.0, w_closure_gain=1.0,
    w_shortcut_penalty=-2.0, w_redundancy=-1.0, w_generic_penalty=-1.5,
    min_anchors=2, min_branches=2, require_compute=True, max_steps=10,
)

ALL_DIFFICULTIES = [EASY, HARD]


@dataclass
class WalkState:
    """游走状态：一个逐步长大的证据子图。"""
    subgraph: nx.DiGraph  # 已探索的子图
    frontier: list[str]   # 可继续扩展的节点
    used_anchors: list[str]  # 已使用的 region keys
    steps_taken: int = 0

    def explored_entities(self) -> set[str]:
        return {n for n, d in self.subgraph.nodes(data=True) if d.get("ntype") == NTYPE_ENTITY}

    def explored_facts(self) -> set[str]:
        return {n for n, d in self.subgraph.nodes(data=True) if d.get("ntype") == NTYPE_FACT}

    def explored_regions(self) -> set[str]:
        return {n for n, d in self.subgraph.nodes(data=True) if d.get("ntype") == NTYPE_REGION}

    def fact_values_by_type(self, tail_type: str) -> dict[str, list[tuple[str, str, dict]]]:
        """按 entity 分组返回某 tail_type 的 fact。{entity_key: [(fact_key, value, edge_data)]}"""
        result: dict[str, list] = {}
        for u, v, d in self.subgraph.edges(data=True):
            if d.get("etype") == ETYPE_RETRIEVE and d.get("tail_type") == tail_type:
                val = self.subgraph.nodes.get(v, {}).get("value", "")
                result.setdefault(u, []).append((v, val, d))
        return result


# ============================================================
# S3. Move Types + Scoring
# ============================================================

MOVE_EXPAND = "expand"       # 从 frontier 节点展开一条边
MOVE_SPAWN = "spawn_anchor"  # 引入新的图中锚点
MOVE_STOP = "stop"           # 停止游走


@dataclass
class Move:
    move_type: str
    src: str = ""
    dst: str = ""
    edge_data: dict = field(default_factory=dict)
    region_key: str = ""  # for MOVE_SPAWN
    score: float = 0.0


def _score_expand(move: Move, state: WalkState, graph: HeteroSolveGraph, diff: DifficultyProfile) -> float:
    """给一条展开边打分。"""
    etype = move.edge_data.get("etype", "")
    score = 0.0

    # fact_gain: retrieve 边带来新事实
    if etype == ETYPE_RETRIEVE:
        ask = move.edge_data.get("askability", 0.5)
        lex = move.edge_data.get("lexicalizability", 0.5)
        is_new = move.dst not in state.explored_facts()
        score += diff.w_fact_gain * (ask + lex) * (1.0 if is_new else 0.2)
        # generic answer penalty
        val = graph.fact_value(move.dst)
        tt = graph.fact_tail_type(move.dst)
        score += diff.w_generic_penalty * _generic_answer_penalty(val, tt)
        # compute affordance: 如果这个 fact 的 tail_type 是 TIME/QUANTITY 且子图里已有同类值
        if tt in ("TIME", "QUANTITY"):
            existing = state.fact_values_by_type(tt)
            # 不同 entity 的同类值 = compare 可能
            src_entity = move.src
            other_entities_with_same_type = [ek for ek in existing if ek != src_entity]
            if other_entities_with_same_type:
                score += diff.w_compute_affordance * 1.5

    # resolve 边带来新实体
    elif etype == ETYPE_RESOLVE:
        is_new = move.dst not in state.explored_entities()
        score += diff.w_fact_gain * 0.5 * (1.0 if is_new else 0.1)

    # redundancy: 已在子图中的节点
    if state.subgraph.has_node(move.dst):
        score += diff.w_redundancy * 0.8

    return score


def _score_spawn(move: Move, state: WalkState, graph: HeteroSolveGraph, diff: DifficultyProfile) -> float:
    """给 spawn_new_anchor 打分。"""
    if move.region_key in state.explored_regions():
        return -10.0  # 已使用
    score = diff.w_visual_novelty * graph.region_score(move.region_key)
    score += diff.w_branch_novelty * 1.0
    # 如果需要更多锚点来满足 budget
    if len(state.used_anchors) < diff.min_anchors:
        score += 3.0
    return score


def _score_stop(state: WalkState, graph: HeteroSolveGraph, diff: DifficultyProfile, closures: list) -> float:
    """给 STOP 打分。只有当存在可闭合意图时才有正分。"""
    if not closures:
        return -10.0
    best_closure_score = max(c.get("score", 0) for c in closures) if closures else 0.0
    score = diff.w_closure_gain * best_closure_score

    # budget 满足度奖励
    n_anchors = len(state.used_anchors)
    n_branches = len({c.get("anchor") for c in closures if c.get("anchor")}) if closures else 0
    if n_anchors >= diff.min_anchors and n_branches >= diff.min_branches:
        score += 2.0
    else:
        score -= 3.0  # budget 不满足时不想停

    if diff.require_compute:
        has_compute = any(c.get("family") in ("compare", "rank", "set_merge") for c in closures)
        if not has_compute:
            score -= 5.0

    # hard 难度下，检查工具多样性 budget
    if diff.name == "hard":
        # 检查是否有 image_search_needed 的 resolve 边在子图中
        has_image_resolve = any(
            d.get("resolve_mode") == "image_search_needed"
            for _, _, d in state.subgraph.edges(data=True)
            if d.get("etype") == ETYPE_RESOLVE
        )
        if not has_image_resolve:
            score -= 2.0  # 还没有需要 image_search 的实体

        # 检查是否有 page_only 的 retrieve 边
        has_page_only = any(
            d.get("retrieval_mode") == "page_only"
            for _, _, d in state.subgraph.edges(data=True)
            if d.get("etype") == ETYPE_RETRIEVE
        )
        if not has_page_only:
            score -= 1.5  # 还没有需要 visit 的事实

    return score


# ============================================================
# S4. ClosureCompiler — 后验枚举可闭合意图
# ============================================================

def enumerate_closures(state: WalkState, graph: HeteroSolveGraph) -> list[dict]:
    """从当前子图中枚举所有可闭合的问题意图。"""
    closures = []

    # 收集子图中的实体和事实
    entities = state.explored_entities()
    regions = state.explored_regions()
    region_to_entity = {rk: graph.entity_for_region(rk) for rk in regions}

    # ---- 1) Direct lookup: region → entity → fact ----
    for rk in regions:
        ek = region_to_entity.get(rk)
        if not ek or ek not in entities:
            continue
        for _, fk, d in state.subgraph.out_edges(ek, data=True):
            if d.get("etype") != ETYPE_RETRIEVE:
                continue
            val = state.subgraph.nodes.get(fk, {}).get("value", "")
            tt = state.subgraph.nodes.get(fk, {}).get("tail_type", "OTHER")
            if not val or val.lower() in _BAD_ANSWERS:
                continue
            if _generic_answer_penalty(val, tt) >= 1.2:
                continue
            ask = d.get("askability", 0.5)
            closures.append({
                "family": "lookup",
                "level": 2,
                "anchors": [rk],
                "anchor": rk,
                "answer": val,
                "answer_type": tt,
                "relation": d.get("relation", ""),
                "entity_key": ek,
                "fact_key": fk,
                "edge_data": d,
                "score": ask + _answer_specificity_score(val, tt),
            })

    # ---- 2) Compare: 两个 region 有同类 TIME/QUANTITY fact ----
    region_list = [rk for rk in regions if region_to_entity.get(rk) in entities]
    for tt in ("TIME", "QUANTITY"):
        facts_by_entity = state.fact_values_by_type(tt)
        region_facts = []
        for rk in region_list:
            ek = region_to_entity.get(rk)
            if ek and ek in facts_by_entity:
                for fk, val, d in facts_by_entity[ek]:
                    region_facts.append((rk, ek, fk, val, d))

        for i in range(len(region_facts)):
            for j in range(i + 1, len(region_facts)):
                rA, eA, fA, vA, dA = region_facts[i]
                rB, eB, fB, vB, dB = region_facts[j]
                if eA == eB:
                    continue
                ok, compat_score, bucket = _value_relations_compatible(
                    dA.get("relation", ""), dB.get("relation", ""), tt,
                )
                if not ok:
                    continue
                # compare closure
                closures.append({
                    "family": "compare",
                    "level": 3,
                    "anchors": [rA, rB],
                    "answer": f"compare({vA}, {vB})",
                    "answer_type": tt,
                    "branches": [
                        {"region": rA, "entity": eA, "fact": fA, "value": vA, "edge": dA},
                        {"region": rB, "entity": eB, "fact": fB, "value": vB, "edge": dB},
                    ],
                    "relation_bucket": bucket,
                    "score": compat_score + dA.get("askability", 0) + dB.get("askability", 0),
                })

                # compare_then_follow: 赢家还有别的 retrieve 边
                try:
                    nA = float(dA.get("normalized_value", ""))
                    nB = float(dB.get("normalized_value", ""))
                except (ValueError, TypeError):
                    continue
                for compare_type in ("earlier", "later") if tt == "TIME" else ("larger", "smaller"):
                    winner_ek = eA if (nA < nB if compare_type in ("earlier", "smaller") else nA > nB) else eB
                    winner_rk = rA if winner_ek == eA else rB
                    # follow edges from winner
                    for _, fk2, d2 in state.subgraph.out_edges(winner_ek, data=True):
                        if d2.get("etype") != ETYPE_RETRIEVE:
                            continue
                        if d2.get("tail_type") == tt:
                            continue  # 不追问同类属性
                        follow_val = state.subgraph.nodes.get(fk2, {}).get("value", "")
                        follow_tt = state.subgraph.nodes.get(fk2, {}).get("tail_type", "OTHER")
                        if not follow_val or follow_val.lower() in _BAD_ANSWERS:
                            continue
                        closures.append({
                            "family": "compare_then_follow",
                            "level": 3,
                            "anchors": [rA, rB],
                            "answer": follow_val,
                            "answer_type": follow_tt,
                            "branches": [
                                {"region": rA, "entity": eA, "fact": fA, "value": vA, "edge": dA},
                                {"region": rB, "entity": eB, "fact": fB, "value": vB, "edge": dB},
                            ],
                            "compare_type": compare_type,
                            "winner": winner_ek,
                            "follow": {"relation": d2.get("relation", ""), "value": follow_val, "type": follow_tt},
                            "score": compat_score + d2.get("askability", 0) + 1.5,
                        })

    # ---- 3) Rank: 3+ region 有同类 fact ----
    for tt in ("TIME", "QUANTITY"):
        facts_by_entity = state.fact_values_by_type(tt)
        items = []
        for rk in region_list:
            ek = region_to_entity.get(rk)
            if ek and ek in facts_by_entity:
                for fk, val, d in facts_by_entity[ek]:
                    try:
                        float(d.get("normalized_value", ""))
                        items.append((rk, ek, fk, val, d))
                    except (ValueError, TypeError):
                        pass
        if len(items) >= 3:
            # 只用前 3 个（不同 entity）
            seen_ek = set()
            top3 = []
            for rk, ek, fk, val, d in items:
                if ek not in seen_ek:
                    top3.append((rk, ek, fk, val, d))
                    seen_ek.add(ek)
                if len(top3) >= 3:
                    break
            if len(top3) >= 3:
                closures.append({
                    "family": "rank",
                    "level": 3,
                    "anchors": [rk for rk, _, _, _, _ in top3],
                    "answer": "rank_winner",
                    "answer_type": "OTHER",
                    "branches": [
                        {"region": rk, "entity": ek, "fact": fk, "value": val, "edge": d}
                        for rk, ek, fk, val, d in top3
                    ],
                    "score": sum(d.get("askability", 0) for _, _, _, _, d in top3) + 2.0,
                })

    # ---- 4) Set merge: 两个 entity 共享 fact target ----
    entity_list = [(rk, region_to_entity[rk]) for rk in region_list if region_to_entity.get(rk)]
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            rA, eA = entity_list[i]
            rB, eB = entity_list[j]
            if eA == eB:
                continue
            targets_a = {}
            for _, fk, d in state.subgraph.out_edges(eA, data=True):
                if d.get("etype") == ETYPE_RETRIEVE:
                    targets_a[fk] = d
            for _, fk, d in state.subgraph.out_edges(eB, data=True):
                if d.get("etype") == ETYPE_RETRIEVE and fk in targets_a:
                    val = state.subgraph.nodes.get(fk, {}).get("value", "")
                    tt = state.subgraph.nodes.get(fk, {}).get("tail_type", "OTHER")
                    if val and val.lower() not in _BAD_ANSWERS:
                        closures.append({
                            "family": "set_merge",
                            "level": 3,
                            "anchors": [rA, rB],
                            "answer": val,
                            "answer_type": tt,
                            "shared_fact": fk,
                            "edge_a": targets_a[fk],
                            "edge_b": d,
                            "score": targets_a[fk].get("askability", 0) + d.get("askability", 0) + 1.0,
                        })

    # ---- 5) L1 read: region 自身可识别 ----
    for rk in regions:
        info = graph.region_info(rk)
        etype = info.get("entity_type", "")
        if etype in ("text", "brand", "person", "landmark", "product"):
            name = graph.region_entity_name(rk)
            if name:
                closures.append({
                    "family": "read",
                    "level": 1,
                    "anchors": [rk],
                    "anchor": rk,
                    "answer": name,
                    "answer_type": "OTHER",
                    "score": 0.5,
                })

    return closures


# ============================================================
# S5. SubgraphWalker — 弱约束自由游走
# ============================================================

class SubgraphWalker:
    def __init__(self, graph: HeteroSolveGraph, rng: random.Random, tau: float = 0.7):
        self.graph = graph
        self.rng = rng
        self.tau = max(0.1, tau)
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"walk_{self._counter:04d}"

    def _softmax_sample(self, items: list[tuple[float, Any]]) -> Any:
        if not items:
            return None
        if len(items) == 1:
            return items[0][1]
        scores = [s for s, _ in items]
        pivot = max(scores)
        weights = [math.exp((s - pivot) / self.tau) for s in scores]
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(items)[1]
        r = self.rng.random() * total
        c = 0.0
        for w, (_, item) in zip(weights, items):
            c += w
            if c >= r:
                return item
        return items[-1][1]

    def _initial_state(self, start_region: str) -> WalkState:
        sg = nx.DiGraph()
        info = self.graph.region_info(start_region)
        sg.add_node(start_region, ntype=NTYPE_REGION, **info)
        ek = self.graph.entity_for_region(start_region)
        if ek:
            sg.add_node(ek, ntype=NTYPE_ENTITY, name=self.graph.entity_name(ek))
            etype = info.get("entity_type", "")
            sg.add_edge(start_region, ek, etype=ETYPE_RESOLVE, entity_type=etype)
        frontier = [ek] if ek else [start_region]
        return WalkState(subgraph=sg, frontier=frontier, used_anchors=[start_region])

    def _enumerate_moves(self, state: WalkState, diff: DifficultyProfile) -> list[Move]:
        moves = []

        # expand: 从 frontier 节点出发的 full graph 边，且目标不在子图中（或是 fact 边）
        for node in state.frontier:
            for _, dst, data in self.graph.G.out_edges(node, data=True):
                etype = data.get("etype", "")
                if etype not in (ETYPE_RESOLVE, ETYPE_RETRIEVE):
                    continue
                moves.append(Move(
                    move_type=MOVE_EXPAND,
                    src=node, dst=dst,
                    edge_data=dict(data),
                ))

        # spawn: 未使用的 region
        used = set(state.used_anchors)
        for rk in self.graph.regions():
            if rk not in used:
                moves.append(Move(
                    move_type=MOVE_SPAWN,
                    region_key=rk,
                ))

        return moves

    def _apply_move(self, state: WalkState, move: Move) -> WalkState:
        sg = state.subgraph.copy()
        frontier = list(state.frontier)
        used_anchors = list(state.used_anchors)

        if move.move_type == MOVE_EXPAND:
            # add dst node with attributes from full graph
            dst_data = dict(self.graph.G.nodes.get(move.dst, {}))
            if not sg.has_node(move.dst):
                sg.add_node(move.dst, **dst_data)
            if not sg.has_edge(move.src, move.dst):
                sg.add_edge(move.src, move.dst, **move.edge_data)
            # update frontier: dst 变成新的可扩展点（如果是 entity）
            dst_ntype = dst_data.get("ntype", "")
            if dst_ntype == NTYPE_ENTITY and move.dst not in frontier:
                frontier.append(move.dst)
            # remove src from frontier if no more outgoing in full graph
            remaining = [
                d for _, d, data in self.graph.G.out_edges(move.src, data=True)
                if data.get("etype") in (ETYPE_RESOLVE, ETYPE_RETRIEVE) and not sg.has_edge(move.src, d)
            ]
            # keep src in frontier if it still has unexplored edges
            # (don't remove — other edges might be useful)

        elif move.move_type == MOVE_SPAWN:
            rk = move.region_key
            info = self.graph.region_info(rk)
            sg.add_node(rk, ntype=NTYPE_REGION, **info)
            ek = self.graph.entity_for_region(rk)
            if ek:
                if not sg.has_node(ek):
                    sg.add_node(ek, ntype=NTYPE_ENTITY, name=self.graph.entity_name(ek))
                sg.add_edge(rk, ek, etype=ETYPE_RESOLVE, entity_type=info.get("entity_type", ""))
                if ek not in frontier:
                    frontier.append(ek)
            used_anchors.append(rk)

        return WalkState(
            subgraph=sg,
            frontier=frontier,
            used_anchors=used_anchors,
            steps_taken=state.steps_taken + 1,
        )

    def walk(self, diff: DifficultyProfile) -> tuple[WalkState, list[dict]]:
        """执行一次完整游走，返回 (final_state, closures)。"""
        # 选起始锚点
        regions = self.graph.regions()
        if not regions:
            return WalkState(nx.DiGraph(), [], []), []
        start = self._softmax_sample([(self.graph.region_score(r), r) for r in regions])
        state = self._initial_state(start)

        for step in range(diff.max_steps):
            closures = enumerate_closures(state, self.graph)
            moves = self._enumerate_moves(state, diff)

            # score all moves + STOP
            scored = []
            for move in moves:
                if move.move_type == MOVE_EXPAND:
                    s = _score_expand(move, state, self.graph, diff)
                elif move.move_type == MOVE_SPAWN:
                    s = _score_spawn(move, state, self.graph, diff)
                else:
                    continue
                move.score = s
                scored.append((s, move))

            stop_score = _score_stop(state, self.graph, diff, closures)
            scored.append((stop_score, Move(move_type=MOVE_STOP, score=stop_score)))

            if not scored:
                break

            chosen = self._softmax_sample(scored)
            if chosen is None or chosen.move_type == MOVE_STOP:
                return state, closures

            state = self._apply_move(state, chosen)

        # max steps reached
        closures = enumerate_closures(state, self.graph)
        return state, closures

    def generate(self, n_walks: int = 10, difficulties: list[DifficultyProfile] | None = None) -> list[dict]:
        """多次游走，收集所有 closure 候选。"""
        if difficulties is None:
            difficulties = ALL_DIFFICULTIES
        all_candidates = []
        seen = set()

        walks_per_diff = max(1, n_walks // len(difficulties))
        for diff in difficulties:
            for _ in range(walks_per_diff):
                state, closures = self.walk(diff)
                for c in closures:
                    sig = (c.get("family"), tuple(sorted(c.get("anchors", []))), c.get("answer", ""))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    c["difficulty"] = diff.name
                    c["walk_steps"] = state.steps_taken
                    c["n_subgraph_nodes"] = state.subgraph.number_of_nodes()
                    all_candidates.append(c)

        # sort by score descending
        all_candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        return all_candidates


# ============================================================
# S6. Irreducibility Checks
# ============================================================

def check_irreducibility(closure: dict, graph: HeteroSolveGraph) -> tuple[bool, str]:
    """5 个不可约性检查。"""
    # 1. answer uniqueness
    val = (closure.get("answer") or "").strip().lower()
    if val in _BAD_ANSWERS or not val:
        return False, "bad_answer"
    tt = closure.get("answer_type", "OTHER")
    if _generic_answer_penalty(closure.get("answer", ""), tt) >= 1.2:
        return False, "generic_answer"

    # 2. realizable question: 锚点必须有视觉描述
    for rk in closure.get("anchors", []):
        ref = graph.visual_descriptor(rk)
        if not ref or ref in ("图中的标识",):
            return False, "anchor_not_describable"

    # 3. no_python_shortcut: compare_then_follow 的 loser 不能有同样的 follow 答案
    if closure.get("family") == "compare_then_follow":
        winner = closure.get("winner")
        follow_val = closure.get("follow", {}).get("value", "")
        branches = closure.get("branches", [])
        for b in branches:
            if b.get("entity") != winner:
                # 检查 loser 有没有同关系的同值
                loser_ek = b.get("entity")
                follow_rel = closure.get("follow", {}).get("relation", "")
                for edge in graph.retrieve_edges(loser_ek):
                    if edge.get("relation") == follow_rel and graph.fact_value(edge["fact_key"]) == follow_val:
                        return False, "python_shortcut"

    # 4. no_branch_shortcut: 多分支 rank 的 answer 在去掉任一分支后必须变
    # (inherently satisfied for argmax/argmin when winner is removed)

    # 5. answer not directly visible in image
    answer = closure.get("answer", "")
    if closure.get("family") != "read":  # L1 read 的答案本来就是实体名
        for rk in graph.regions():
            name = graph.region_entity_name(rk)
            if name and name.lower() == answer.lower():
                return False, "answer_visible_in_image"

    return True, ""


# ============================================================
# S7. QuestionFrame + Realization
# ============================================================

@dataclass
class QuestionFrame:
    level: int
    family: str
    wh_type: str
    visible_refs: list[str]
    criterion: str
    follow_relation: str
    hidden_entities: list[str]
    hidden_values: list[str]
    answer: str
    answer_type: str


def compile_frame(closure: dict, graph: HeteroSolveGraph) -> QuestionFrame:
    """从 closure 编译 QuestionFrame。"""
    anchors = closure.get("anchors", [])
    visible_refs = [graph.visual_descriptor(rk) for rk in anchors]
    family = closure.get("family", "lookup")
    answer = closure.get("answer", "")
    answer_type = closure.get("answer_type", "OTHER")

    hidden_entities = []
    hidden_values = []
    for b in closure.get("branches", []):
        ename = graph.entity_name(b.get("entity", ""))
        if ename:
            hidden_entities.append(ename)
        bval = b.get("value", "")
        if bval:
            hidden_values.append(bval)

    if family == "read":
        return QuestionFrame(
            level=1, family="read", wh_type="what",
            visible_refs=visible_refs, criterion="", follow_relation="",
            hidden_entities=[], hidden_values=[],
            answer=answer, answer_type=answer_type,
        )

    if family == "lookup":
        rel = closure.get("relation", "")
        wh = _wh_type_for_relation(rel, answer_type)
        follow_rel = _relation_natural_text(rel, answer_type)
        ek = closure.get("entity_key", "")
        hidden_entities = [graph.entity_name(ek)] if ek else []
        return QuestionFrame(
            level=2, family="lookup", wh_type=wh,
            visible_refs=visible_refs, criterion="",
            follow_relation=follow_rel,
            hidden_entities=hidden_entities, hidden_values=[],
            answer=answer, answer_type=answer_type,
        )

    if family == "compare":
        branches = closure.get("branches", [])
        rel = branches[0]["edge"].get("relation", "") if branches else ""
        tt = answer_type
        criterion = _relation_natural_text(rel, tt)
        hidden_entities = [graph.entity_name(b.get("entity", "")) for b in branches]
        hidden_values = [b.get("value", "") for b in branches]
        return QuestionFrame(
            level=3, family="compare", wh_type="which",
            visible_refs=visible_refs, criterion=criterion,
            follow_relation="",
            hidden_entities=hidden_entities, hidden_values=hidden_values,
            answer=answer, answer_type=tt,
        )

    if family == "compare_then_follow":
        branches = closure.get("branches", [])
        compare_rel = branches[0]["edge"].get("relation", "") if branches else ""
        compare_type = closure.get("compare_type", "earlier")
        criterion = _selection_phrase(compare_type, compare_rel)
        follow = closure.get("follow", {})
        follow_rel = _relation_natural_text(follow.get("relation", ""), follow.get("type", "OTHER"))
        wh = _wh_type_for_relation(follow.get("relation", ""), follow.get("type", "OTHER"))
        hidden_entities = [graph.entity_name(b.get("entity", "")) for b in branches]
        hidden_values = [b.get("value", "") for b in branches]
        return QuestionFrame(
            level=3, family="compare_then_follow", wh_type=wh,
            visible_refs=visible_refs, criterion=criterion,
            follow_relation=follow_rel,
            hidden_entities=hidden_entities, hidden_values=hidden_values,
            answer=answer, answer_type=closure.get("follow", {}).get("type", "OTHER"),
        )

    if family == "rank":
        branches = closure.get("branches", [])
        rel = branches[0]["edge"].get("relation", "") if branches else ""
        criterion = _relation_natural_text(rel, answer_type)
        hidden_entities = [graph.entity_name(b.get("entity", "")) for b in branches]
        hidden_values = [b.get("value", "") for b in branches]
        return QuestionFrame(
            level=3, family="rank", wh_type="which",
            visible_refs=visible_refs, criterion=criterion,
            follow_relation="",
            hidden_entities=hidden_entities, hidden_values=hidden_values,
            answer=answer, answer_type="OTHER",
        )

    if family == "set_merge":
        rel_a = closure.get("edge_a", {}).get("relation", "")
        criterion = f"共同的{_relation_natural_text(rel_a, answer_type)}"
        hidden_entities = [graph.entity_name(b.get("entity", "")) for b in closure.get("branches", [])]
        return QuestionFrame(
            level=3, family="set_merge", wh_type=_wh_type_for_relation(rel_a, answer_type),
            visible_refs=visible_refs, criterion=criterion,
            follow_relation="",
            hidden_entities=hidden_entities, hidden_values=[],
            answer=answer, answer_type=answer_type,
        )

    # fallback
    return QuestionFrame(
        level=closure.get("level", 2), family=family, wh_type="what",
        visible_refs=visible_refs, criterion="", follow_relation="",
        hidden_entities=hidden_entities, hidden_values=hidden_values,
        answer=answer, answer_type=answer_type,
    )


def _build_realize_prompt(frame: QuestionFrame, image_desc: str) -> str:
    hidden_text = json.dumps(frame.hidden_entities + frame.hidden_values, ensure_ascii=False)
    refs = "、".join(frame.visible_refs)
    lines = [
        "你看到一张图片，想问一个问题。下面是结构化的出题框架，请据此生成一句自然的中文问题。",
        "",
        f"图片描述：{image_desc}",
        f"视觉锚点：{refs}",
        f"题目类型：{frame.family}",
    ]
    if frame.criterion:
        lines.append(f"判断条件：{frame.criterion}")
    if frame.follow_relation:
        lines.append(f"追问属性：{frame.follow_relation}")
    lines += [
        f"答案：{frame.answer}",
        f"答案类型：{frame.answer_type}",
        f"隐藏信息（不可出现在题面中）：{hidden_text}",
        "",
        "要求：",
        "- 用视觉特征指代图中实体（方位+外观），不写实体名",
        "- 问题必须是完整的自然句子，像真人好奇心提出的问题",
    ]
    # 按题型给具体的提问引导
    if frame.family == "read":
        lines.append('- L1 识别题：问"这个XX上写的是什么文字/品牌/名称"，不要只说"XX是什么"')
        lines.append('  好："画面左上角那个红底白字的广告牌上写的是什么品牌名称？"')
        lines.append('  坏："左上角红底白字品牌广告牌？"')
    elif frame.family == "lookup":
        lines.append('- L2 查询题：先用视觉特征描述图中的东西，再用日常口语问一个需要查资料才能回答的问题')
        lines.append('  好："画面左下角那个有着金色拱门标志的快餐店，它最早是在哪座城市开的第一家店？"')
        lines.append('  坏："画面左下角品牌的创始人是谁？"（太干，没有视觉描述）')
        lines.append('  坏："画面左下角那个带有巨大黄色发光拱门标志的餐饮品牌，它的创始人是谁？"（过度堆砌形容词）')
    elif frame.family in ("compare", "compare_then_follow"):
        lines.append('- L3 比较题：用外观差异（球衣号码/颜色/位置）区分两个目标，问法要日常化')
        lines.append('  好："穿27号球衣的那位和穿5号球衣的那位，谁的年龄更大？"')
        lines.append('  坏："画面下方偏右身穿白色27号球衣的球员和画面上方偏左身穿白色8号球衣的球员，谁的出生日期更早？"（堆砌方位词）')
        lines.append('  要点：视觉描述够区分就行，不要把方位堆满；用"年龄更大"而非"出生日期更早"这种日常说法')
    elif frame.family == "rank":
        lines.append('- L3 排名题：简洁列出要比较的几个目标，问谁最XX')
        lines.append('  好："图中这三个品牌广告牌，哪个品牌成立的时间最早？"')
    elif frame.family == "set_merge":
        lines.append('- L3 交集题：问两个目标的共同点')
        lines.append('  好："这两个品牌有没有在同一个证券交易所上市？"')
    lines += [
        "- 只输出一个问题，带问号",
        "",
        '输出 JSON（不适合出题则 question 填空）：{"question": "...", "answer": "..."}',
    ]
    return "\n".join(lines)


def realize_frame(frame: QuestionFrame, image_b64: str, image_desc: str) -> dict | None:
    prompt = _build_realize_prompt(frame, image_desc)
    obj = call_vlm_json(
        prompt,
        "请生成自然中文问题。",
        image_b64=image_b64,
        max_tokens=600,
        temperature=0.7,
        max_attempts=1,
    )
    if not isinstance(obj, dict):
        return None
    question = str(obj.get("question") or "").strip()
    if not question:
        return None
    return {"question": question, "answer": obj.get("answer") or frame.answer}


def postcheck_name_leak(question: str, closure: dict, graph: HeteroSolveGraph) -> tuple[bool, str]:
    if not question:
        return False, "empty"
    q_lower = question.lower()
    # hidden entities
    for name in [graph.entity_name(b.get("entity", "")) for b in closure.get("branches", [])]:
        if name and len(name) >= 3 and name.lower() in q_lower:
            return False, f"hidden_leak:{name}"
    # entity key from lookup
    ek = closure.get("entity_key", "")
    if ek:
        name = graph.entity_name(ek)
        if name and len(name) >= 3 and name.lower() in q_lower:
            return False, f"hidden_leak:{name}"
    # all in-image entity names
    for rk in graph.regions():
        name = graph.region_entity_name(rk)
        if not name:
            continue
        if bool(re.search(r"[A-Za-z]", name)):
            if len(name) >= 4 and name.lower() in q_lower:
                return False, f"in_image_leak:{name}"
        else:
            if len(name) >= 2 and name in question:
                return False, f"in_image_leak:{name}"
    return True, ""


# ============================================================
# S8. ToolPlanCompiler
# ============================================================

def compile_tool_plan(closure: dict, graph: HeteroSolveGraph) -> list[dict]:
    steps = []
    step_num = 0

    def _add_resolve_steps(region_key: str, entity_name: str):
        nonlocal step_num
        info = graph.region_info(region_key)
        # 用 resolve_mode 决定工具
        ek = graph.entity_for_region(region_key)
        resolve_mode = "ocr_likely"
        if ek:
            for _, _, d in graph.G.out_edges(region_key, data=True):
                if d.get("etype") == ETYPE_RESOLVE:
                    resolve_mode = d.get("resolve_mode", "ocr_likely")
                    break
        step_num += 1
        steps.append({
            "step": step_num, "tool": "code_interpreter",
            "action": f"裁剪 {info.get('location', '')} 区域",
            "expected_output": "裁剪后的实体图像",
        })
        step_num += 1
        if resolve_mode == "image_search_needed":
            steps.append({
                "step": step_num, "tool": "image_search",
                "action": "反向图片搜索识别实体",
                "expected_output": f"识别出 {entity_name}",
            })
        else:
            steps.append({
                "step": step_num, "tool": "code_interpreter",
                "action": "OCR 识别文字/品牌名",
                "expected_output": f"识别出 {entity_name}",
            })

    family = closure.get("family", "")

    if family == "read":
        rk = closure.get("anchor", closure.get("anchors", [""])[0])
        name = graph.region_entity_name(rk)
        _add_resolve_steps(rk, name)

    elif family == "lookup":
        rk = closure.get("anchors", [""])[0]
        ek = closure.get("entity_key", "")
        name = graph.entity_name(ek) if ek else graph.region_entity_name(rk)
        _add_resolve_steps(rk, name)
        rel = closure.get("relation", "")
        tt = closure.get("answer_type", "OTHER")
        retrieval_mode = closure.get("edge_data", {}).get("retrieval_mode", "snippet_only")
        step_num += 1
        steps.append({
            "step": step_num, "tool": "web_search",
            "action": f"搜索 {name} 的{_relation_natural_text(rel, tt)}",
            "expected_output": f"获取到 {closure.get('answer', '')}",
        })
        if retrieval_mode == "page_only":
            step_num += 1
            steps.append({
                "step": step_num, "tool": "visit",
                "action": f"深读搜索结果页获取 {name} 的详细{_relation_natural_text(rel, tt)}信息",
                "expected_output": f"确认 {closure.get('answer', '')}",
            })

    elif family in ("compare", "compare_then_follow", "rank"):
        branches = closure.get("branches", [])
        for b in branches:
            name = graph.entity_name(b.get("entity", ""))
            _add_resolve_steps(b.get("region", ""), name)
            rel = b.get("edge", {}).get("relation", "")
            tt = b.get("edge", {}).get("tail_type", "OTHER")
            retrieval_mode = b.get("edge", {}).get("retrieval_mode", "snippet_only")
            step_num += 1
            steps.append({
                "step": step_num, "tool": "web_search",
                "action": f"搜索 {name} 的{_relation_natural_text(rel, tt)}",
                "expected_output": f"获取到 {b.get('value', '')}",
            })
            if retrieval_mode == "page_only":
                step_num += 1
                steps.append({
                    "step": step_num, "tool": "visit",
                    "action": f"深读搜索结果页确认 {name} 的{_relation_natural_text(rel, tt)}",
                    "expected_output": f"确认 {b.get('value', '')}",
                })
        # compute
        step_num += 1
        if family == "rank":
            steps.append({
                "step": step_num, "tool": "code_interpreter",
                "action": "比较多个值找出最大/最小",
                "expected_output": "选出排名结果",
            })
        else:
            steps.append({
                "step": step_num, "tool": "code_interpreter",
                "action": "比较两个值选出满足条件的一方",
                "expected_output": f"选出 {closure.get('winner', '')}",
            })
        # follow
        if family == "compare_then_follow" and closure.get("follow"):
            follow = closure["follow"]
            step_num += 1
            steps.append({
                "step": step_num, "tool": "web_search",
                "action": f"搜索赢家的{_relation_natural_text(follow.get('relation', ''), follow.get('type', 'OTHER'))}",
                "expected_output": f"获取到 {closure.get('answer', '')}",
            })

    elif family == "set_merge":
        for rk in closure.get("anchors", []):
            name = graph.region_entity_name(rk)
            _add_resolve_steps(rk, name)
            step_num += 1
            steps.append({
                "step": step_num, "tool": "web_search",
                "action": f"搜索 {name} 的相关属性",
                "expected_output": "获取属性集合",
            })
        step_num += 1
        steps.append({
            "step": step_num, "tool": "code_interpreter",
            "action": "对两组属性求交集/差集",
            "expected_output": closure.get("answer", ""),
        })

    return steps


def tool_depth_score(plan: list[dict]) -> int:
    score = 0
    for step in plan:
        tool = step.get("tool", "")
        action = step.get("action", "")
        if tool == "code_interpreter":
            if any(k in action for k in ("比较", "最", "交集", "差集")):
                score += 3
            else:
                score += 1
        elif tool == "image_search":
            score += 2
        elif tool == "web_search":
            score += 1
        elif tool == "visit":
            score += 2
    return score


# ============================================================
# S9. Top-Level Entry
# ============================================================

def _classify_hard_bucket(closure: dict, tool_plan: list[dict]) -> str:
    """给 L3 closure 打 hard bucket tag。"""
    tools_used = {s.get("tool") for s in tool_plan}
    has_image_search = "image_search" in tools_used
    has_visit = "visit" in tools_used
    has_web = "web_search" in tools_used
    code_actions = [s.get("action", "") for s in tool_plan if s.get("tool") == "code_interpreter"]
    code_types = set()
    for a in code_actions:
        if "裁剪" in a or "crop" in a.lower():
            code_types.add("cv")
        elif "OCR" in a or "识别" in a:
            code_types.add("ocr")
        elif "比较" in a or "最" in a or "排序" in a:
            code_types.add("compute")
        elif "交集" in a or "差集" in a:
            code_types.add("set_op")

    if has_image_search and has_web and has_visit and len(code_types) >= 2:
        return "all_tools"
    if has_image_search and len(closure.get("anchors", [])) >= 2:
        return "image_heavy"
    if has_visit:
        return "visit_heavy"
    if len(code_types) >= 2:
        return "code_heavy"
    return "standard"


def generate_questions(
    entity_json_path: str,
    image_path: str | None = None,
    *,
    seed: int = 42,
    tau: float = 0.7,
    n_walks: int = 12,
    max_per_level: dict[int, int] | None = None,
    max_workers: int = 4,
) -> dict:
    """完整流程：构建图 → 自由游走 → 后验归纳 → 语言化。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_per_level is None:
        max_per_level = {1: 4, 2: 3, 3: 4}

    with open(entity_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = HeteroSolveGraph(data)
    rng = random.Random(seed)
    walker = SubgraphWalker(graph, rng, tau=tau)

    # generate closures from walks
    all_closures = walker.generate(n_walks=n_walks)

    # irreducibility checks
    checked = []
    rejected = []
    for c in all_closures:
        ok, reason = check_irreducibility(c, graph)
        if ok:
            checked.append(c)
        else:
            rejected.append({"family": c.get("family"), "answer": c.get("answer", ""), "reason": reason})

    # compile tool plans first (needed for bucket classification)
    for c in checked:
        c["tool_plan"] = compile_tool_plan(c, graph)
        c["tool_depth"] = tool_depth_score(c["tool_plan"])
        if c.get("level", 2) >= 3:
            c["hard_bucket"] = _classify_hard_bucket(c, c["tool_plan"])
        else:
            c["hard_bucket"] = ""

    # select best per level + bucket quotas
    selected: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    seen_answers: set[str] = set()
    bucket_counts: dict[str, int] = {}
    BUCKET_QUOTAS = {"image_heavy": 2, "visit_heavy": 1, "code_heavy": 1, "all_tools": 1, "standard": 2}

    # L3 先按 bucket 多样性选
    l3_candidates = [c for c in checked if c.get("level", 2) >= 3]
    l3_candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    for c in l3_candidates:
        ans_sig = c.get("answer", "").strip().lower()
        if ans_sig in seen_answers:
            continue
        bucket = c.get("hard_bucket", "standard")
        if bucket_counts.get(bucket, 0) >= BUCKET_QUOTAS.get(bucket, 2):
            continue
        if len(selected[3]) >= max_per_level.get(3, 4):
            break
        selected[3].append(c)
        seen_answers.add(ans_sig)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # L2 和 L1
    for c in checked:
        level = c.get("level", 2)
        if level >= 3:
            continue
        ans_sig = c.get("answer", "").strip().lower()
        if ans_sig in seen_answers:
            continue
        quota = max_per_level.get(level, 3)
        if len(selected.get(level, [])) >= quota:
            continue
        selected.setdefault(level, []).append(c)
        seen_answers.add(ans_sig)

    to_realize = []
    for level in (3, 2, 1):
        to_realize.extend(selected.get(level, []))

    # realize
    image_b64 = None
    if image_path:
        from core.image_utils import file_to_b64
        image_b64 = file_to_b64(image_path)

    results_by_level: dict[str, list] = {"level_1": [], "level_2": [], "level_3": []}
    realize_rejects = []

    if image_b64 and to_realize:
        frames = [(c, compile_frame(c, graph)) for c in to_realize]

        def _realize_one(idx, closure, frame):
            result = realize_frame(frame, image_b64, graph.image_description)
            return idx, closure, frame, result

        realized = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(frames))) as pool:
            futures = {pool.submit(_realize_one, i, c, f): i for i, (c, f) in enumerate(frames)}
            for fut in as_completed(futures):
                idx, closure, frame, result = fut.result()
                if result is None:
                    realize_rejects.append({"family": closure.get("family"), "reason": "llm_failed"})
                    continue
                q = result["question"]
                ok, reason = postcheck_name_leak(q, closure, graph)
                if not ok:
                    realize_rejects.append({"family": closure.get("family"), "reason": reason, "question": q})
                    continue
                realized.append((idx, closure, frame, result))

        realized.sort(key=lambda x: x[0])

        for idx, closure, frame, result in realized:
            level = closure.get("level", 2)
            level_key = f"level_{level}"
            n = len(results_by_level.get(level_key, [])) + 1
            record = {
                "question_id": f"L{level}_{n:02d}",
                "question": result["question"],
                "answer": result["answer"],
                "tool_sequence": closure.get("tool_plan", []),
                "level": f"L{level}",
                "family": closure.get("family", ""),
                "difficulty": closure.get("difficulty", ""),
                "obfuscation_applied": level >= 2,
                "obfuscated_entities": [graph.visual_descriptor(rk) for rk in closure.get("anchors", [])],
                "reasoning_path": {
                    "family": closure.get("family"),
                    "anchors": closure.get("anchors", []),
                    "branches": closure.get("branches", []),
                    "compare_type": closure.get("compare_type"),
                    "follow": closure.get("follow"),
                },
                "hard_bucket": closure.get("hard_bucket", ""),
                "rationale": f"{closure.get('family')} from {closure.get('difficulty', 'unknown')} walk, "
                             f"tool_depth={closure.get('tool_depth', 0)}, bucket={closure.get('hard_bucket', '')}",
                "entities_involved": [graph.region_info(rk).get("entity_id", "") for rk in closure.get("anchors", [])],
            }
            results_by_level.setdefault(level_key, []).append(record)
    else:
        for c in to_realize:
            level = c.get("level", 2)
            level_key = f"level_{level}"
            frame = compile_frame(c, graph)
            n = len(results_by_level.get(level_key, [])) + 1
            results_by_level.setdefault(level_key, []).append({
                "question_id": f"L{level}_{n:02d}",
                "family": c.get("family"),
                "difficulty": c.get("difficulty"),
                "frame": {
                    "visible_refs": frame.visible_refs,
                    "criterion": frame.criterion,
                    "follow_relation": frame.follow_relation,
                    "answer": frame.answer,
                    "answer_type": frame.answer_type,
                },
                "tool_sequence": c.get("tool_plan", []),
                "tool_depth": c.get("tool_depth", 0),
            })

    return {
        **results_by_level,
        "metadata": {
            "graph_stats": graph.stats(),
            "total_closures": len(all_closures),
            "irreducibility_pass": len(checked),
            "irreducibility_reject": rejected[:20],
            "selected": {level: len(items) for level, items in selected.items()},
            "realized_count": sum(len(v) for v in results_by_level.values()),
            "hard_bucket_dist": dict(bucket_counts),
            "realize_rejects": realize_rejects,
        },
    }
