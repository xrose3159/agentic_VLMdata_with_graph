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


def _canon(s: str) -> str:
    """canonical form of entity/value string: strip, lowercase, collapse whitespace."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip()).lower()


def _set_merge_relations_compatible(rel_a: str, rel_b: str, tail_type: str) -> bool:
    """set_merge 的两个 retrieve 关系是否语义兼容。

    允许：
    - 完全相同的 relation（两人都 born_in 某城市）
    - 同一语义 bucket（headquartered_in 和 located_in 都是"城市"类）
    """
    rel_a = (rel_a or "").strip().lower()
    rel_b = (rel_b or "").strip().lower()
    if not rel_a or not rel_b:
        return False
    if rel_a == rel_b:
        return True

    # 同 bucket 的 relation 语义组
    buckets = [
        # "出生/来源" 城市
        {"born_in", "birthplace", "from", "hometown", "origin", "birth_city"},
        # "总部/位置" 城市
        {"headquartered_in", "based_in", "located_in", "hq", "headquarters"},
        # "所属团队/组织"
        {"plays_for", "plays for", "played_for", "played for", "member_of", "signed_with"},
        # "所属联赛/联盟"
        {"competes_in", "plays_in", "part_of", "league"},
        # "合作/赞助"
        {"sponsored_by", "partner_of", "endorses", "endorsed_by"},
        # "职业/角色"
        {"occupation", "profession", "role", "position"},
        # "国籍"
        {"nationality", "citizen_of", "country"},
    ]
    for bucket in buckets:
        if rel_a in bucket and rel_b in bucket:
            return True
    return False


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
        self.local_artifacts = entity_json.get("local_artifacts", {})
        self._region_data: dict[str, dict] = {}
        self._entity_for_region: dict[str, str] = {}
        self._build(entity_json)
        # 预缓存 canonical name → entity_key 查表，供 multi_hop 和 walker 复用
        self._name_to_entity_canon_map: dict[str, str] = {}
        for n, nd in self.G.nodes(data=True):
            if nd.get("ntype") == NTYPE_ENTITY:
                canon = _canon(nd.get("name", "") or n.removeprefix("entity:"))
                if canon and canon not in self._name_to_entity_canon_map:
                    self._name_to_entity_canon_map[canon] = n

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
        # 桥接提升：对于 head 不在 in-image 实体里的 triple，创建一个合成 entity 节点，
        # 这样 multi_hop 可以把它当作中间跳。合成节点没有 region，不会被 walker spawn 选中。
        # 先尝试用 canonical 形式折叠到已有实体，避免 "Nuggets" vs "Denver Nuggets" 这种别名重复。
        canon_to_existing_entity: dict[str, str] = {}
        for ek, nd in list(self.G.nodes(data=True)):
            if nd.get("ntype") == NTYPE_ENTITY:
                ename = nd.get("name", "") or ek.removeprefix("entity:")
                canon_to_existing_entity[_canon(ename)] = ek

        for t in triples:
            head = (t.get("head") or "").strip()
            tail = (t.get("tail") or "").strip()
            relation = (t.get("relation") or "").strip()
            if not head or not tail or not relation:
                continue

            head_key = f"entity:{head.lower()}"
            if not self.G.has_node(head_key):
                # 先尝试通过 canonical 形式折叠到已有 in-image 实体
                canon_head = _canon(head)
                matched = canon_to_existing_entity.get(canon_head)
                if matched:
                    head_key = matched
                else:
                    # 创建合成桥接实体（不是 in-image，没有 region）
                    head_key = f"entity:{canon_head}" if canon_head else f"entity:{head.lower()}"
                    if not self.G.has_node(head_key):
                        self.G.add_node(
                            head_key,
                            ntype=NTYPE_ENTITY,
                            name=head,
                            synthetic=True,
                            in_image=False,
                            confidence=0.0,
                        )
                        canon_to_existing_entity[canon_head] = head_key

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
            src_entity = move.src
            other_entities_with_same_type = [ek for ek in existing if ek != src_entity]
            if other_entities_with_same_type:
                score += diff.w_compute_affordance * 1.5

        # bridge affordance: 这条边的 tail value 是图里另一个 entity 的名字 → multi_hop 桥接点
        canon_val = _canon(val)
        if canon_val and canon_val in getattr(graph, "_name_to_entity_canon_map", {}):
            score += 1.5 if diff.name == "hard" else 0.3

        # page_only 边奖励（HARD 难度下）— evidenced 比 semantic 更值钱
        if diff.name == "hard":
            rm = move.edge_data.get("retrieval_mode", "")
            if rm == "page_only_evidenced" or rm == "page_only":
                score += 1.5  # 有真证据
            elif rm == "page_only_semantic":
                score += 0.7  # 语义推的，值钱但不如 evidenced

        # 合成源节点折扣（避免 walker 被合成实体的大扇出淹没）
        src_node_data = graph.G.nodes.get(move.src, {})
        if src_node_data.get("synthetic"):
            score *= 0.5

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
    # HARD 难度下，偏好 resolve_mode=image_search_needed 的锚点（工具链至少 +1 image_search）
    if diff.name == "hard":
        for _, _, d in graph.G.out_edges(move.region_key, data=True):
            if d.get("etype") == ETYPE_RESOLVE and d.get("resolve_mode") == "image_search_needed":
                score += 1.0
                break
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

    # hard 难度下，硬预算检查（不是软偏好）
    if diff.name == "hard":
        budget_deficit = 0

        # 必须有 image_search_needed 的实体
        has_image_resolve = any(
            d.get("resolve_mode") == "image_search_needed"
            for _, _, d in state.subgraph.edges(data=True)
            if d.get("etype") == ETYPE_RESOLVE
        )
        if not has_image_resolve:
            budget_deficit += 1

        # 必须有 page_only 的事实（evidenced 或 semantic 都算，walker 阶段宽松）
        has_page_only = any(
            d.get("retrieval_mode") in ("page_only_evidenced", "page_only_semantic", "page_only")
            for _, _, d in state.subgraph.edges(data=True)
            if d.get("etype") == ETYPE_RETRIEVE
        )
        if not has_page_only:
            budget_deficit += 1

        # 必须有 ≥2 个独立分支（不同锚点的 resolve 链）
        if len(state.used_anchors) < 2:
            budget_deficit += 1

        # 必须有 compute closure
        has_compute = any(c.get("family") in ("compare", "rank", "set_merge", "compare_then_follow") for c in closures)
        if not has_compute:
            budget_deficit += 1

        # budget_deficit 越大，STOP 越不应该
        score -= budget_deficit * 3.0

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
                # 按 rank_type 真正计算 winner（argmin/argmax），不再用 rank_winner 占位
                for rank_type in ("earliest", "latest") if tt == "TIME" else ("largest", "smallest"):
                    try:
                        sortable = [
                            (float(d.get("normalized_value", "")), rk, ek, fk, val, d)
                            for rk, ek, fk, val, d in top3
                        ]
                    except (ValueError, TypeError):
                        continue
                    if rank_type in ("earliest", "smallest"):
                        sortable.sort(key=lambda x: x[0])
                    else:
                        sortable.sort(key=lambda x: x[0], reverse=True)
                    winner_norm, winner_rk, winner_ek, winner_fk, winner_val, winner_d = sortable[0]
                    # 验证 winner 唯一（否则这道题不可答）
                    if len(sortable) >= 2 and sortable[0][0] == sortable[1][0]:
                        continue  # tie，答案不唯一
                    # 答案是 winner 对应的 entity name
                    winner_name = graph.entity_name(winner_ek)
                    if not winner_name:
                        continue
                    closures.append({
                        "family": "rank",
                        "level": 3,
                        "anchors": [rk for rk, _, _, _, _ in top3],
                        "answer": winner_name,                    # 真 winner 名字
                        "answer_type": "OTHER",
                        "rank_type": rank_type,                   # earliest/latest/largest/smallest
                        "winner_entity": winner_ek,
                        "winner_value": winner_val,
                        "winner_fact": winner_fk,
                        "branches": [
                            {"region": rk, "entity": ek, "fact": fk, "value": val, "edge": d}
                            for rk, ek, fk, val, d in top3
                        ],
                        "score": sum(d.get("askability", 0) for _, _, _, _, d in top3) + 2.0,
                    })

    # ---- 4) Set merge: 两个 entity 共享 fact 的 tail value ----
    # 注意：fact_key 里嵌了 head，同一个 tail 在不同 head 下是不同 node
    #       所以用 (normalized_tail_value, tail_type) 做匹配键
    entity_list = [(rk, region_to_entity[rk]) for rk in region_list if region_to_entity.get(rk)]
    seen_merge_keys: set = set()
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            rA, eA = entity_list[i]
            rB, eB = entity_list[j]
            if eA == eB:
                continue
            # 收集 A 的所有 retrieve fact tail：{(norm_val, tt): (fk, d, rel)}
            targets_a: dict = {}
            for _, fk_a, da in state.subgraph.out_edges(eA, data=True):
                if da.get("etype") != ETYPE_RETRIEVE:
                    continue
                val_a = state.subgraph.nodes.get(fk_a, {}).get("value", "") or ""
                tt_a = state.subgraph.nodes.get(fk_a, {}).get("tail_type", "OTHER")
                key = (val_a.strip().lower(), tt_a)
                if key[0]:
                    targets_a.setdefault(key, (fk_a, da))
            # 遍历 B 的 retrieve fact tail，寻找匹配
            for _, fk_b, db in state.subgraph.out_edges(eB, data=True):
                if db.get("etype") != ETYPE_RETRIEVE:
                    continue
                val_b = state.subgraph.nodes.get(fk_b, {}).get("value", "") or ""
                tt_b = state.subgraph.nodes.get(fk_b, {}).get("tail_type", "OTHER")
                key = (val_b.strip().lower(), tt_b)
                if not key[0] or key not in targets_a:
                    continue
                fk_a, da = targets_a[key]
                val = val_b  # 两者 value 等价，取 B 的原始字符串
                if val.lower() in _BAD_ANSWERS:
                    continue
                # set_merge 要求两条边的 relation 语义兼容
                # （否则会产出 "born_in + exploring_expansion_to" 这种怪问题）
                rel_a = da.get("relation", "")
                rel_b = db.get("relation", "")
                if not _set_merge_relations_compatible(rel_a, rel_b, tt_b):
                    continue
                # 去除"shared value 就是另一个 in-image 实体名"的退化情况
                # 这类会被 irreducibility 里的 answer_visible_in_image 挡掉，但先过滤省工
                is_in_image_name = any(
                    val.strip().lower() == graph.entity_name(ek).strip().lower()
                    for _, ek in entity_list if ek
                )
                if is_in_image_name:
                    continue
                dedup_key = (eA, eB, key)
                if dedup_key in seen_merge_keys:
                    continue
                seen_merge_keys.add(dedup_key)
                closures.append({
                    "family": "set_merge",
                    "level": 3,
                    "anchors": [rA, rB],
                    "answer": val,
                    "answer_type": tt_b,
                    "shared_fact": fk_b,
                    "shared_value": val,
                    "edge_a": da,
                    "edge_b": db,
                    "relation_a": da.get("relation", ""),
                    "relation_b": db.get("relation", ""),
                    "branches": [
                        {"region": rA, "entity": eA, "fact": fk_a, "value": val, "edge": da},
                        {"region": rB, "entity": eB, "fact": fk_b, "value": val, "edge": db},
                    ],
                    "score": da.get("askability", 0) + db.get("askability", 0) + 1.5,
                })

    # ---- 6) Multi-hop lookup: N-hop (N ∈ {2, 3}) ----
    # 桥接提升后，中间实体可以是 in-image 或 synthetic（合成）。
    # 用全图缓存的 canonical name → entity_key 查表定位桥接点。
    name_to_entity: dict[str, str] = dict(getattr(graph, "_name_to_entity_canon_map", {}))
    # 合并本次 walk 已探索的 entity（确保枚举不漏本轮 walk 新带进来的实体）
    for ek in entities:
        canon_name = _canon(graph.entity_name(ek))
        if canon_name and canon_name not in name_to_entity:
            name_to_entity[canon_name] = ek

    seen_multi_hop: set = set()
    THREE_HOP_CAP = 8  # 每个锚点最多 8 条 3-hop，防组合爆炸

    def _emit_multi_hop(hops: list, rA: str):
        """hops: [(eX, fkX, dX), ...]，长度 2 或 3。"""
        if len(hops) < 2 or len(hops) > 3:
            return
        # 末端事实
        last_fk = hops[-1][1]
        last_d = hops[-1][2]
        final_val = state.subgraph.nodes.get(last_fk, {}).get("value", "") or ""
        if not final_val or final_val.lower() in _BAD_ANSWERS:
            return
        tt_last = last_d.get("tail_type", "OTHER")
        if tt_last == "OTHER" and _generic_answer_penalty(final_val, tt_last) >= 1.2:
            return

        # 循环 / 别名坍缩检测：所有 hop 实体的 canonical name 互不相等
        canon_entities = [_canon(graph.entity_name(h[0])) for h in hops]
        if len(set(canon_entities)) < len(canon_entities):
            return
        if _canon(final_val) in canon_entities:
            return  # 答案回指到链上实体

        # 关系语义合理性：中间跳 (0..N-2) 的 tail_type 必须是 entity-like
        # 末端跳可以是 TIME/QUANTITY/LOCATION/PERSON/ORG/OTHER
        for mid_hop in hops[:-1]:
            mid_tt = mid_hop[2].get("tail_type", "OTHER")
            if mid_tt in ("TIME", "QUANTITY"):
                return  # 不能以数值作为桥接

        # 关系词汇化可信度：每一跳的 relation 必须有足够高的 lexicalizability，
        # 否则"covered the pre-draft workout of" 这种 Step2 LLM 从句子切出来的长关系碎片
        # 会被当成正经 relation 参与 multi_hop 枚举。
        # 判据：_relation_profile 给 hit_known_map 0.85 / crisp_raw 0.65 / sentence_fragment 0.2
        # threshold 0.5 刚好过滤掉句子片段，保留通用短语。
        for hop in hops:
            lex = hop[2].get("lexicalizability", 0)
            if lex < 0.5:
                return  # 句子片段型 relation，不适合作为 multi_hop 的桥

        hop_chain_data = []
        for ek_h, fk_h, d_h in hops:
            hop_chain_data.append({
                "entity": ek_h,
                "relation": d_h.get("relation", ""),
                "value": state.subgraph.nodes.get(fk_h, {}).get("value", "") or "",
                "edge": d_h,
                "fact": fk_h,
            })

        dedup_key = (rA, tuple(canon_entities),
                     tuple(h[2].get("relation", "") for h in hops),
                     _canon(final_val))
        if dedup_key in seen_multi_hop:
            return
        seen_multi_hop.add(dedup_key)

        score_bonus = 2.0 if len(hops) == 2 else 3.5  # 3-hop 更值钱
        total_ask = sum(h[2].get("askability", 0) for h in hops)
        closures.append({
            "family": "multi_hop",
            "level": 3,
            "anchors": [rA],
            "bridge_entity": hops[1][0],  # 第一个桥
            "bridge_value": hop_chain_data[0].get("value", ""),
            "answer": final_val,
            "answer_type": tt_last,
            "hop_chain": hop_chain_data,
            "n_hops": len(hops),
            "score": total_ask + score_bonus,
        })

    for rk_A in regions:
        eA = region_to_entity.get(rk_A)
        if not eA or eA not in entities:
            continue
        # 锚点本身不能是合成实体（锚点必须在图里可视指代）
        if graph.G.nodes.get(eA, {}).get("synthetic"):
            continue

        # Hop 1 枚举：收集所有可以桥接到另一个 entity 的边
        hop1_edges_bridged = []
        for _, fk1, d1 in state.subgraph.out_edges(eA, data=True):
            if d1.get("etype") != ETYPE_RETRIEVE:
                continue
            tail_val_1 = state.subgraph.nodes.get(fk1, {}).get("value", "") or ""
            bridge_key = _canon(tail_val_1)
            if not bridge_key:
                continue
            eB = name_to_entity.get(bridge_key)
            if not eB or eB == eA:
                continue
            hop1_edges_bridged.append((eB, fk1, d1))

            # 2-hop: 直接从 eB 展开末端
            for _, fk2, d2 in graph.G.out_edges(eB, data=True):
                if d2.get("etype") != ETYPE_RETRIEVE:
                    continue
                _emit_multi_hop([(eA, fk1, d1), (eB, fk2, d2)], rk_A)

        # 3-hop：复用 hop1 的 bridged 边找 eC
        three_hop_count = 0
        for eB, fk1, d1 in hop1_edges_bridged:
            if three_hop_count >= THREE_HOP_CAP:
                break
            for _, fk2, d2 in graph.G.out_edges(eB, data=True):
                if three_hop_count >= THREE_HOP_CAP:
                    break
                if d2.get("etype") != ETYPE_RETRIEVE:
                    continue
                tail_val_2 = state.subgraph.nodes.get(fk2, {}).get("value", "") or ""
                bridge_key_2 = _canon(tail_val_2)
                if not bridge_key_2:
                    continue
                eC = name_to_entity.get(bridge_key_2)
                if not eC or eC == eA or eC == eB:
                    continue
                # 3-hop 末端
                for _, fk3, d3 in graph.G.out_edges(eC, data=True):
                    if three_hop_count >= THREE_HOP_CAP:
                        break
                    if d3.get("etype") != ETYPE_RETRIEVE:
                        continue
                    _emit_multi_hop([(eA, fk1, d1), (eB, fk2, d2), (eC, fk3, d3)], rk_A)
                    three_hop_count += 1

    # ---- 5) L1 read: region 自身能被 VLM 直接视读 ----
    # 只对 resolve_mode=ocr_likely 的 region 生成 L1 read closure。
    # image_search_needed 的 region 需要工具识别，不属于 L1。
    for rk in regions:
        info = graph.region_info(rk)
        etype = info.get("entity_type", "")
        if etype not in ("text", "brand", "person", "landmark", "product"):
            continue
        resolve_mode = "ocr_likely"
        for _, _, d in graph.G.out_edges(rk, data=True):
            if d.get("etype") == ETYPE_RESOLVE:
                resolve_mode = d.get("resolve_mode", "ocr_likely")
                break
        if resolve_mode != "ocr_likely":
            continue
        name = graph.region_entity_name(rk)
        if not name:
            continue
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

        # 补充：从全图枚举 L3 closure（walker 子图受限时的 fallback）
        full_state = self._full_graph_state()
        full_closures = enumerate_closures(full_state, self.graph)
        for c in full_closures:
            if c.get("level", 2) < 3:
                continue  # L1/L2 由 walker 提供
            sig = (c.get("family"), tuple(sorted(c.get("anchors", []))), c.get("answer", ""))
            if sig in seen:
                continue
            seen.add(sig)
            c["difficulty"] = "hard"
            c["walk_steps"] = 0
            c["n_subgraph_nodes"] = full_state.subgraph.number_of_nodes()
            c["from_full_graph"] = True
            all_candidates.append(c)

        # sort by score descending
        all_candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        return all_candidates

    def _full_graph_state(self) -> WalkState:
        """构造一个包含全图所有 region/entity/fact 及边的 WalkState，用于 L3 兜底枚举。"""
        sg = self.graph.G.copy()
        regions = [n for n, d in sg.nodes(data=True) if d.get("ntype") == NTYPE_REGION]
        return WalkState(subgraph=sg, frontier=[], used_anchors=regions)


# ============================================================
# S6. Irreducibility Checks
# ============================================================

def check_irreducibility(closure: dict, graph: HeteroSolveGraph) -> tuple[bool, str]:
    """不可约性检查。基础 5 项 + 工具不可约 3 项。"""
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
                loser_ek = b.get("entity")
                follow_rel = closure.get("follow", {}).get("relation", "")
                for edge in graph.retrieve_edges(loser_ek):
                    if edge.get("relation") == follow_rel and graph.fact_value(edge["fact_key"]) == follow_val:
                        return False, "python_shortcut"

    # 4. no_branch_shortcut
    # (inherently satisfied for argmax/argmin when winner is removed)

    # 5. answer not directly visible in image
    answer = closure.get("answer", "")
    if closure.get("family") != "read":
        for rk in graph.regions():
            name = graph.region_entity_name(rk)
            if name and name.lower() == answer.lower():
                return False, "answer_visible_in_image"

    # 6. multi_hop 短路检查：严格 irreducibility
    #    禁止任何能绕开桥的捷径，对 2-hop 和 3-hop 都生效
    if closure.get("family") == "multi_hop":
        hop_chain = closure.get("hop_chain", [])
        if not hop_chain:
            return False, "multi_hop_empty"

        eA = hop_chain[0].get("entity", "") if hop_chain else ""
        final_canon = _canon(closure.get("answer", ""))

        # 检查 1：eA 不能有直达 final_val 的 retrieve 边
        if eA and final_canon:
            for edge in graph.retrieve_edges(eA):
                if _canon(graph.fact_value(edge["fact_key"])) == final_canon:
                    return False, "multi_hop_direct_shortcut"

        # 检查 2 (3-hop only)：eA 不能有直达最后桥实体 eC 的边
        # 否则可以跳过 eB 直接 eA→eC→answer
        if len(hop_chain) >= 3:
            eC = hop_chain[2].get("entity", "")
            eC_canon = _canon(graph.entity_name(eC)) if eC else ""
            if eA and eC_canon:
                for edge in graph.retrieve_edges(eA):
                    if _canon(graph.fact_value(edge["fact_key"])) == eC_canon:
                        return False, "multi_hop_skip_bridge"

        # 检查 3 (3-hop only)：eB 不能有直达 final_val 的边
        # 否则 hop2 + hop3 可以合并为 1 步
        if len(hop_chain) >= 3:
            eB = hop_chain[1].get("entity", "")
            if eB and final_canon:
                for edge in graph.retrieve_edges(eB):
                    if _canon(graph.fact_value(edge["fact_key"])) == final_canon:
                        return False, "multi_hop_hop2_direct_shortcut"

    return True, ""


def check_tool_irreducibility(closure: dict, tool_plan: list[dict], bucket: str, graph: HeteroSolveGraph = None) -> tuple[bool, str]:
    """工具不可约性检查：某个工具从序列里删掉后，题目还能答 → reject 该 bucket。

    只对 L3 hard bucket 题目执行。L1/L2 不检查。
    """
    if closure.get("level", 2) < 3:
        return True, ""
    if bucket == "standard" or not bucket:
        return True, ""

    tools_used = {s.get("tool") for s in tool_plan}

    # no_image_search_shortcut: image_heavy/all_tools 必须真正依赖 image_search
    if bucket in ("image_heavy", "all_tools"):
        if "image_search" not in tools_used:
            return False, f"bucket_{bucket}_but_no_image_search"
        # 至少有一个锚点的 entity 需要 image_search resolve
        if graph is not None:
            has_image_needed = False
            # multi_hop / lookup 用 anchors（region key）直接查 resolve 边
            for rk in closure.get("anchors", []):
                for _, _, d in graph.G.out_edges(rk, data=True):
                    if d.get("etype") == ETYPE_RESOLVE and d.get("resolve_mode") == "image_search_needed":
                        has_image_needed = True
                        break
                if has_image_needed:
                    break
            # compare/rank/set_merge 还要看 branches
            if not has_image_needed:
                for b in closure.get("branches", []):
                    rk_b = b.get("region", "")
                    if rk_b:
                        for _, _, d in graph.G.out_edges(rk_b, data=True):
                            if d.get("etype") == ETYPE_RESOLVE and d.get("resolve_mode") == "image_search_needed":
                                has_image_needed = True
                                break
                    if has_image_needed:
                        break
            if not has_image_needed:
                return False, "image_search_replaceable_by_ocr"

    # ---- prior_answerable 检查（visit_heavy 专用） ----
    # Tier B 数据显示 visit_heavy 44% 被 no_tool 秒答，但 no_visit 0%。
    # 说明很多"Jina 读过的事实"其实是模型先验已知的百科级常识。
    # 检查：最终 hop 的 relation 是否是"强先验"类型（总部/创始人/创立时间等）
    # 是 → 降级到 standard，而不是等到 Step5 才发现。
    # false positive 代价低（standard 仍然在 hard split），false negative 代价高。
    _STRONG_PRIOR_RELATIONS = {
        "headquarters", "headquartered_in", "headquartered in",
        "founded_by", "founded by", "founder",
        "founded_in", "founded in",
        "ceo", "owner", "owned_by", "owned by",
        "parent_company", "parent company",
        "country", "nationality",
        "born_in", "born in", "birthplace",
        "capital", "located_in", "located in", "located_at", "located at",
        "based_in", "based in",
    }

    def _is_prior_answerable_visit(closure_inner: dict) -> bool:
        """visit_heavy 候选的最终答案是否大概率靠先验就能推出。

        条件：最终 hop 的 relation（slug 归一后）在 STRONG_PRIOR_RELATIONS 里，
        且 head entity 看起来像百科命名实体（非纯数字/日期）。
        """
        hops = closure_inner.get("hop_chain") or []
        if not hops:
            # set_merge / compare 结构：检查 follow relation 或 branch edge relation
            follow = closure_inner.get("follow")
            if isinstance(follow, dict):
                rel = (follow.get("relation") or "").strip().lower()
                return rel in _STRONG_PRIOR_RELATIONS
            return False
        last_hop = hops[-1] if isinstance(hops[-1], dict) else {}
        rel = (last_hop.get("relation") or "").strip().lower()
        # slug 归一：去下划线
        rel_alt = rel.replace("_", " ")
        if rel not in _STRONG_PRIOR_RELATIONS and rel_alt not in _STRONG_PRIOR_RELATIONS:
            return False
        # 检查喂入该 relation 的 head entity 是命名实体（非纯数字）
        # hop_chain 结构：entity 是链的当前节点名，value 是答案
        head_raw = (last_hop.get("entity") or "").strip()
        # entity key 格式是 "entity:xxx"，去前缀
        head_entity = head_raw.split(":", 1)[-1].strip() if ":" in head_raw else head_raw
        if not head_entity or head_entity.replace(",", "").replace(".", "").isdigit():
            return False
        return True

    # no_visit_shortcut: visit_heavy/all_tools 必须真正依赖 visit
    # ★ 严格门槛：visit_heavy 只吃 page_only_evidenced（真证据）
    #   ultra_long / all_tools 可以接受 page_only_semantic（语义推的）
    #   旧数据的 "page_only" 当作 evidenced 向后兼容
    strict_evidenced_modes = {"page_only_evidenced", "page_only"}
    loose_page_modes = {"page_only_evidenced", "page_only_semantic", "page_only"}

    if bucket in ("visit_heavy", "all_tools"):
        if "visit" not in tools_used:
            return False, f"bucket_{bucket}_but_no_visit"
        # visit_heavy 用严格；all_tools 也用严格（要求真证据）
        accepted_modes = strict_evidenced_modes
        branches = closure.get("branches", [])
        has_page_only = False
        edge_data = closure.get("edge_data", {})
        if edge_data.get("retrieval_mode") in accepted_modes:
            has_page_only = True
        for b in branches:
            if b.get("edge", {}).get("retrieval_mode") in accepted_modes:
                has_page_only = True
                break
        if closure.get("follow", {}).get("retrieval_mode") in accepted_modes:
            has_page_only = True
        # multi_hop 的 edges 在 hop_chain 里
        for hop in closure.get("hop_chain", []):
            if hop.get("edge", {}).get("retrieval_mode") in accepted_modes:
                has_page_only = True
                break
        if not has_page_only:
            return False, "visit_heavy_needs_evidenced_page_only"
        # ★ 先验可答检查：最终 hop 的 relation 是强先验 → 模型靠常识就能答
        if bucket == "visit_heavy" and _is_prior_answerable_visit(closure):
            return False, "visit_heavy_prior_answerable"

    # ultra_long bucket：可以接受 semantic 的 page_only（宽松）
    # 因为 ultra_long 本身就是复合（≥7 步 + ≥2 工具种），不需要纯 visit 依赖
    if bucket == "ultra_long":
        if "visit" not in tools_used:
            return False, f"bucket_ultra_long_but_no_visit"
        accepted_modes = loose_page_modes
        branches = closure.get("branches", [])
        has_any_page = False
        edge_data = closure.get("edge_data", {})
        if edge_data.get("retrieval_mode") in accepted_modes:
            has_any_page = True
        for b in branches:
            if b.get("edge", {}).get("retrieval_mode") in accepted_modes:
                has_any_page = True
                break
        if closure.get("follow", {}).get("retrieval_mode") in accepted_modes:
            has_any_page = True
        for hop in closure.get("hop_chain", []):
            if hop.get("edge", {}).get("retrieval_mode") in accepted_modes:
                has_any_page = True
                break
        if not has_any_page:
            return False, "ultra_long_needs_any_page_only"

    # no_code_shortcut: code_heavy/all_tools 必须有 ≥2 个非平凡 code skill tag
    if bucket in ("code_heavy", "all_tools"):
        skill_tags = set(_extract_code_skill_tags(tool_plan))
        nontrivial = skill_tags - {"cv_preprocess", "ocr_parse"}
        if len(nontrivial) < 2:
            return False, f"code_nontrivial_skills_only_{len(nontrivial)}"

    # ultra_long: 必须真的 ≥7 步，且工具种类 ≥3（避免 7 个 web_search 凑数）
    if bucket == "ultra_long":
        if len(tool_plan) < 7:
            return False, "ultra_long_too_short"
        if len(tools_used) < 3:
            return False, "ultra_long_not_diverse"

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
    chain_trace: str = ""  # multi_hop 专用：完整 hop 关系链，防 LLM 翻译漂移


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
        rel_natural = _relation_natural_text(rel, branches[0]["edge"].get("tail_type", "OTHER") if branches else "OTHER")
        # 用 rank_type 构造 criterion（"成立时间最早" / "数值最大" 等）
        rank_type = closure.get("rank_type", "earliest")
        rank_phrase_map = {
            "earliest": "最早", "latest": "最晚",
            "largest": "最大", "smallest": "最小",
        }
        criterion = f"{rel_natural}{rank_phrase_map.get(rank_type, '最早')}"
        hidden_entities = [graph.entity_name(b.get("entity", "")) for b in branches]
        hidden_values = [b.get("value", "") for b in branches]
        # answer 已经是 winner 的真 entity name，必须隐藏不让 LLM 泄露
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

    if family == "multi_hop":
        hop_chain = closure.get("hop_chain", [])
        if len(hop_chain) >= 2:
            # 隐藏所有 hop 实体名（锚点 + 所有桥）
            hidden_entity_names = []
            for hop in hop_chain:
                ek = hop.get("entity", "")
                ename = graph.entity_name(ek) if ek else ""
                if ename:
                    hidden_entity_names.append(ename)
            # 隐藏所有 bridge value（除最终答案外的中间值都不能出现）
            hidden_value_list = [h.get("value", "") for h in hop_chain[:-1]]
            hidden_value_list.append(answer)

            # criterion = 中间所有跳的关系链（用 "的" 连接），并附带原始 relation slug 供 prompt
            mid_relations = [
                _relation_natural_text(h.get("relation", ""), "OTHER")
                for h in hop_chain[:-1]
            ]
            criterion = "的".join(mid_relations)

            # 构造完整的 "真实关系链描述"（给 realize prompt 用，防 LLM 翻译漂移）
            chain_trace_parts = []
            for i, h in enumerate(hop_chain):
                rel_raw = h.get("relation", "")
                rel_natural = _relation_natural_text(rel_raw, h.get("edge", {}).get("tail_type", "OTHER"))
                if i < len(hop_chain) - 1:
                    # 中间跳：只给出关系，不给值（值是隐藏桥接名）
                    chain_trace_parts.append(f"hop{i+1}[{rel_raw} → {rel_natural}]")
                else:
                    chain_trace_parts.append(f"hop{i+1}[{rel_raw} → {rel_natural}]（末端）")
            chain_trace = " → ".join(chain_trace_parts)

            # follow_relation = 最后一跳关系（最终问的属性）
            last = hop_chain[-1]
            tt_last = last.get("edge", {}).get("tail_type", answer_type)
            follow_rel = _relation_natural_text(last.get("relation", ""), tt_last)
            wh = _wh_type_for_relation(last.get("relation", ""), tt_last)

            frame = QuestionFrame(
                level=3, family="multi_hop", wh_type=wh,
                visible_refs=visible_refs,
                criterion=criterion,
                follow_relation=follow_rel,
                hidden_entities=hidden_entity_names,
                hidden_values=hidden_value_list,
                answer=answer, answer_type=tt_last,
            )
            # 附加字段：供 realize prompt 展示完整关系链
            frame.chain_trace = chain_trace  # type: ignore[attr-defined]
            return frame

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
        "- ★ 严格只问一个问题，整段话只能有一个问号 ？",
        "- ★ 禁止用 \"…？XX 是…？\" 这种连问两个的写法",
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
    elif frame.family == "compare":
        lines.append('- L3 比较题：用外观差异（颜色/位置/号码）区分两个目标，问法日常化')
        lines.append('  好："穿27号球衣的那位和穿5号球衣的那位，谁的年龄更大？"')
        lines.append('  坏："画面下方偏右身穿白色27号球衣的球员和画面上方偏左身穿白色8号球衣的球员，谁的出生日期更早？"（堆砌方位词）')
    elif frame.family == "compare_then_follow":
        lines.append('- L3 比较+追问题：先比较选出 winner，再追问 winner 的另一属性。')
        lines.append('  ★ 必须合并成一个问句，把比较条件作为定语从句嵌入。整段话只能有 1 个问号。')
        lines.append('  好："这两个广告牌对应的品牌中，成立时间更早的那个的创始人是谁？"')
        lines.append('  好："穿 27 号和 8 号球衣的两位球员里，年龄更大的那位的出生城市是哪里？"')
        lines.append('  坏："这两个里哪一个成立时间更早？它的创始人是谁？"（连问两个，禁止）')
        lines.append('  坏："哪一个的创立时间更晚？它现在归属于哪家公司？"（连问两个，禁止）')
        lines.append(f'  注意：最终答案是 follow 属性（{frame.follow_relation}），不要把 compare 的中间结果当答案')
    elif frame.family == "rank":
        lines.append('- L3 排名题：简洁列出要比较的几个目标，问谁最XX')
        lines.append('  好："图中这三个品牌广告牌，哪个品牌成立的时间最早？"')
    elif frame.family == "set_merge":
        lines.append('- L3 交集题：问两个目标的共同点')
        lines.append('  好："这两个品牌有没有在同一个证券交易所上市？"')
    elif frame.family == "multi_hop":
        # 展示真实关系链，防 LLM 翻译漂移
        chain_trace = getattr(frame, "chain_trace", "")
        if chain_trace:
            lines.append(f'- 真实关系链（必须严格遵守）：{chain_trace}')
        lines.append('- L3 多跳题：用"这个XX的YY的ZZ"结构把 2-3 跳串起来，中间节点不能写名字')
        lines.append('  2-hop 好："画面里穿白色27号球衣那位球员，他效力的那支球队的主场所在城市人口是多少？"')
        lines.append('  3-hop 好："画面里穿白色8号球衣那位球员，他所在的那支球队的母公司，曾经收购过的那家公司总部位于哪座城市？"')
        lines.append('  坏："Jamal Murray 所在的 Denver Nuggets 主场城市人口？"（泄露锚点名+桥接名）')
        lines.append('  要点：')
        lines.append('    1. 锚点：只用方位+外观/号码/颜色指代，绝不写名字')
        lines.append('    2. 每个中间跳都用"它的/其/那支/那家..."代词串联，桥接实体的名字一个都不能出现')
        lines.append('    3. 隐藏值列表里的所有字符串都禁止写进问题')
        lines.append('    4. 问句最后问末端属性（追问属性那一列）')
        lines.append('    5. 3-hop 时代词可以叠加："那位的...的...的..."')
        lines.append('    ★ 6. 问题里每一跳的语义必须严格对应下面的"真实关系链"，不能换关系')
        lines.append('       例：真实关系是 "plays_for" → 必须用"效力的/所在的"')
        lines.append('       例：真实关系是 "won_championship_in" → 必须用"夺冠年份"，不能说成"加入联盟年份"')
        lines.append('       例：真实关系是 "covered_pre_draft_workout_of" → 这种太特殊的关系**不要出题**，改用描述性的问法')
    lines += [
        "- 只输出一个问题，带问号",
        "",
        '输出 JSON（不适合出题则 question 填空）：{"question": "...", "answer": "..."}',
    ]
    return "\n".join(lines)


def realize_frame(frame: QuestionFrame, image_b64: str | None, image_desc: str) -> dict | None:
    """根据 QuestionFrame 调 LLM 生成自然语言问题。

    注意：不再传 image_b64 给 LLM。
    QuestionFrame 已经包含 visible_refs（文字方位描述）+ image_desc（Step2 提取的图片描述），
    LLM 不需要看图就能润色问题。传图只会浪费 token + 拖慢响应。
    `image_b64` 参数保留只为兼容签名，函数内不使用。
    """
    prompt = _build_realize_prompt(frame, image_desc)
    obj = call_vlm_json(
        prompt,
        "请生成自然中文问题。",
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
    # 单问号检查：整段话只能有 1 个 ? 或 ？，否则是连问多个
    n_q_marks = question.count("？") + question.count("?")
    if n_q_marks > 1:
        return False, f"multiple_questions:{n_q_marks}_marks"
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
    # multi_hop: anchor + 所有 bridge 名字 + 中间 bridge value 都必须隐藏
    if closure.get("family") == "multi_hop":
        hop_chain = closure.get("hop_chain", [])
        ans_lower = (closure.get("answer") or "").lower()
        for hop in hop_chain:
            eh = hop.get("entity", "")
            name = graph.entity_name(eh) if eh else ""
            if name and len(name) >= 3 and name.lower() in q_lower:
                return False, f"multi_hop_entity_leak:{name}"
        # 中间跳的 value（除最后一跳的答案外）也是 bridge name 的字面，禁止出现
        for hop in hop_chain[:-1]:
            val = (hop.get("value") or "").strip()
            if val and len(val) >= 3 and val.lower() != ans_lower and val.lower() in q_lower:
                return False, f"multi_hop_value_leak:{val}"
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

    def _needs_visit(retrieval_mode: str) -> bool:
        """判断 retrieval_mode 是否需要 visit 步骤（兼容新旧 label）。"""
        return retrieval_mode in ("page_only", "page_only_evidenced", "page_only_semantic")

    def _resolve_mode_of(region_key: str) -> str:
        for _, _, d in graph.G.out_edges(region_key, data=True):
            if d.get("etype") == ETYPE_RESOLVE:
                return d.get("resolve_mode", "ocr_likely")
        return "ocr_likely"

    def _add_resolve_steps(region_key: str, entity_name: str):
        """只在确实需要工具的情况下加 resolve 步骤。
        - ocr_likely：VLM 能直接读，不加任何工具
        - image_search_needed：只加 image_search 一步（crop 是隐式的，不展开为显式 code 步骤）
        """
        nonlocal step_num
        resolve_mode = _resolve_mode_of(region_key)
        if resolve_mode == "image_search_needed":
            step_num += 1
            info = graph.region_info(region_key)
            loc = info.get("location", "") or ""
            steps.append({
                "step": step_num, "tool": "image_search",
                "action": f"对{loc}区域做图像检索以识别实体" if loc else "对目标区域做图像检索以识别实体",
                "expected_output": f"识别出 {entity_name}",
            })
        # ocr_likely → 不加任何步骤（直接视读）

    family = closure.get("family", "")

    if family == "read":
        # L1 read：VLM 应该直接视读，不走任何工具链
        # 即使 resolve_mode=image_search_needed，也不算 L1 read（应落到 L2 lookup 去）
        return []

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
        if _needs_visit(retrieval_mode):
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
            if _needs_visit(retrieval_mode):
                step_num += 1
                steps.append({
                    "step": step_num, "tool": "visit",
                    "action": f"深读搜索结果页确认 {name} 的{_relation_natural_text(rel, tt)}",
                    "expected_output": f"确认 {b.get('value', '')}",
                })
        # compute: 拆成两步 — 先标准化，再比较/排序
        step_num += 1
        # Step A: normalize_compare —— 日期/单位/货币标准化
        first_tail_type = branches[0].get("fact_tail_type", "OTHER") if branches else "OTHER"
        if first_tail_type == "TIME":
            normalize_action = "用 datetime 把日期字符串标准化为可比较的 ISO 格式"
        elif first_tail_type == "QUANTITY":
            normalize_action = "用 re 解析数值并标准化单位（百万/亿 → 浮点数）"
        else:
            normalize_action = "对收集到的值做标准化清洗"
        steps.append({
            "step": step_num, "tool": "code_interpreter",
            "action": normalize_action,
            "expected_output": "标准化后的数值列表",
        })
        # Step B: compute —— 比较或排序
        step_num += 1
        if family == "rank":
            steps.append({
                "step": step_num, "tool": "code_interpreter",
                "action": "排序多个标准化值，找出最大/最小",
                "expected_output": "选出排名结果",
            })
        else:
            steps.append({
                "step": step_num, "tool": "code_interpreter",
                "action": "比较两个标准化后的值，选出满足条件的一方",
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

    elif family == "multi_hop":
        hop_chain = closure.get("hop_chain", [])
        rA = closure.get("anchors", [""])[0]
        name_A = graph.region_entity_name(rA)
        _add_resolve_steps(rA, name_A)  # 可能加 image_search

        if len(hop_chain) >= 2:
            # 泛化循环：对每个 hop 生成 web_search + (可选) visit
            for hop_idx, hop in enumerate(hop_chain):
                rel = hop.get("relation", "")
                tt = hop.get("edge", {}).get("tail_type", "OTHER")
                retrieval = hop.get("edge", {}).get("retrieval_mode", "snippet_only")
                hop_val = hop.get("value", "")

                step_num += 1
                if hop_idx == 0:
                    action = f"搜索 {name_A} 的{_relation_natural_text(rel, tt)}"
                elif hop_idx == 1:
                    action = f"在上一步结果基础上，搜索其{_relation_natural_text(rel, tt)}"
                else:
                    action = f"在前 {hop_idx} 步结果基础上，继续搜索其{_relation_natural_text(rel, tt)}"
                steps.append({
                    "step": step_num, "tool": "web_search",
                    "action": action,
                    "expected_output": f"获取到 {hop_val}",
                })
                if _needs_visit(retrieval):
                    step_num += 1
                    steps.append({
                        "step": step_num, "tool": "visit",
                        "action": f"深读搜索结果页确认{_relation_natural_text(rel, tt)}",
                        "expected_output": f"确认 {hop_val}",
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
        # 先 evidence_vote_merge 去重
        step_num += 1
        steps.append({
            "step": step_num, "tool": "code_interpreter",
            "action": "对多源搜索结果做去重与投票，合并候选",
            "expected_output": "归一化的属性集合",
        })
        # 再求交集/差集
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

def _extract_code_skill_tags(tool_plan: list[dict]) -> set[str]:
    """从工具序列里提取 code skill tags。

    6 类 skill：
    - cv_preprocess：裁剪、缩放、二值化
    - ocr_parse：OCR 识别 + 文本清洗
    - layout_geometry：bbox 面积、距离、位置排序
    - tabular_aggregate：多值整理、groupby、merge
    - normalize_compare：日期/单位/货币标准化后比较
    - evidence_vote_merge：多源候选投票、去重、交集
    """
    skills = set()
    for s in tool_plan:
        if s.get("tool") != "code_interpreter":
            continue
        action = s.get("action", "").lower()
        if any(k in action for k in ("裁剪", "crop", "缩放", "resize", "二值化")):
            skills.add("cv_preprocess")
        if any(k in action for k in ("ocr", "识别文字", "识别品牌")):
            skills.add("ocr_parse")
        if any(k in action for k in ("面积", "距离", "位置", "bbox", "最大", "最小", "排序")):
            skills.add("layout_geometry")
        if any(k in action for k in ("groupby", "聚合", "表", "合并", "merge")):
            skills.add("tabular_aggregate")
        if any(k in action for k in ("比较", "标准化", "归一化", "日期", "单位", "货币")):
            skills.add("normalize_compare")
        if any(k in action for k in ("交集", "差集", "投票", "去重", "evidence", "intersection")):
            skills.add("evidence_vote_merge")
    return skills


def _classify_hard_bucket(closure: dict, tool_plan: list[dict]) -> str:
    """给 L3 closure 打 hard bucket tag。

    新优先级（工具依赖类 > 长度类）：
      all_tools > image_heavy > visit_heavy > code_heavy > ultra_long > chain_heavy > standard

    image_heavy 拆两个 subtype（对外仍是同一个 bucket）：
      image_compare:        ≥2 visual anchors + image_search（保留旧定义）
      image_resolve_follow: ≥1 visual anchor + image_search + follow chain ≥ 2 hops

    ultra_long 降为次级标签（`is_ultra_long` flag），不再抢主 bucket。
    """
    tools_used = {s.get("tool") for s in tool_plan}
    has_image_search = "image_search" in tools_used
    has_visit = "visit" in tools_used
    has_web = "web_search" in tools_used

    # 用 code skill tags 替代旧的 code_types
    skill_tags = _extract_code_skill_tags(tool_plan)
    closure["code_skill_tags"] = list(skill_tags)
    nontrivial_skills = skill_tags - {"cv_preprocess", "ocr_parse"}

    # ---- 次级标签：is_ultra_long（≥7 步 + ≥2 工具种）----
    # 不再作为主 bucket，改为 flag。诊断数据证明 ultra_long 抢走了大量
    # 含 image_search 的闭合（8 张 sports 图诊断：8 个被抢走）。
    is_ultra_long = len(tool_plan) >= 7 and len(tools_used) >= 2
    closure["is_ultra_long"] = is_ultra_long

    # ---- 主 bucket 分类（工具依赖优先）----

    # 1. all_tools：4 种工具全出现 + 非平凡 code
    if has_image_search and has_web and has_visit and len(nontrivial_skills) >= 2:
        closure["hard_bucket_subtype"] = ""
        return "all_tools"

    # 2. image_heavy：分两个 subtype
    #    但如果 image_search 只是链的第一步且其余全是 web/visit，
    #    不应把一条 7 步纯 visit 链标成 image_heavy。
    #    判据：image_search 步数占总步数的比例 ≥ 某阈值 OR 是核心锚点 resolve。
    n_anchors = len(closure.get("anchors", []))
    n_hops = len(closure.get("hop_chain", []))
    n_branches = len(closure.get("branches", []))
    follow_depth = n_hops + n_branches  # follow chain 长度
    n_image_steps = sum(1 for s in tool_plan if s.get("tool") == "image_search")

    if has_image_search:
        if n_anchors >= 2:
            closure["hard_bucket_subtype"] = "image_compare"
            return "image_heavy"
        if n_anchors >= 1 and follow_depth >= 2:
            # 对 ultra_long 级别的链（≥7 步），只有 image_search 占比 ≥ 15% 才算 image_heavy
            # 否则 image_search 只是 1 步 resolve，链的主体是 visit/web → 归 ultra_long
            if is_ultra_long and len(tool_plan) > 0 and n_image_steps / len(tool_plan) < 0.15:
                pass  # 不归 image_heavy，继续往下判 visit_heavy 或 ultra_long
            else:
                closure["hard_bucket_subtype"] = "image_resolve_follow"
                return "image_heavy"

    # 3. visit_heavy
    if has_visit:
        closure["hard_bucket_subtype"] = ""
        return "visit_heavy"

    # 4. code_heavy：≥2 类非平凡 skill
    if len(nontrivial_skills) >= 2:
        closure["hard_bucket_subtype"] = ""
        return "code_heavy"

    # 5. ultra_long（只有当主类都不命中时才用长度做主 bucket）
    if is_ultra_long:
        closure["hard_bucket_subtype"] = ""
        return "ultra_long"

    # 6. chain_heavy：multi_hop 裸链
    if closure.get("family") == "multi_hop":
        closure["hard_bucket_subtype"] = ""
        return "chain_heavy"

    # 7. standard
    closure["hard_bucket_subtype"] = ""
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
        max_per_level = {1: 4, 2: 3, 3: 8}

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
    BUCKET_QUOTAS = {"image_heavy": 2, "visit_heavy": 1, "code_heavy": 1, "all_tools": 1,
                     "chain_heavy": 2, "standard": 1, "ultra_long": 2}

    # L3 先按 bucket 多样性选（bucket 是硬门槛，不只是标签）
    l3_candidates = [c for c in checked if c.get("level", 2) >= 3]
    l3_candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
    tool_irr_rejects = []
    for c in l3_candidates:
        ans_sig = c.get("answer", "").strip().lower()
        if ans_sig in seen_answers:
            continue
        bucket = c.get("hard_bucket", "standard")
        # 工具不可约性硬检查：bucket 不是标签，是门槛
        tok, treason = check_tool_irreducibility(c, c.get("tool_plan", []), bucket, graph)
        if not tok:
            tool_irr_rejects.append({"family": c.get("family"), "bucket": bucket, "reason": treason})
            # 降级到 standard
            c["hard_bucket"] = "standard"
            bucket = "standard"
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
    # 注：不再传图给 LLM。realize 只用 frame + image_desc 文字润色。
    # `image_path` 仍然作为"是否走 realize 路径"的开关（None 时返回 frame skeleton）。
    results_by_level: dict[str, list] = {"level_1": [], "level_2": [], "level_3": []}
    realize_rejects = []

    if image_path and to_realize:
        frames = [(c, compile_frame(c, graph)) for c in to_realize]

        def _realize_one(idx, closure, frame):
            result = realize_frame(frame, None, graph.image_description)
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
                    "hop_chain": closure.get("hop_chain", []),
                    "n_hops": closure.get("n_hops"),
                },
                "hard_bucket": closure.get("hard_bucket", ""),
                "hard_bucket_subtype": closure.get("hard_bucket_subtype", ""),
                "is_ultra_long": closure.get("is_ultra_long", False),
                "code_skill_tags": closure.get("code_skill_tags", []),
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
                "hard_bucket": c.get("hard_bucket", ""),
                "code_skill_tags": c.get("code_skill_tags", []),
                "n_hops": c.get("n_hops"),
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
            "tool_irreducibility_rejects": tool_irr_rejects,
            "realize_rejects": realize_rejects,
        },
    }
