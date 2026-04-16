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
import re

import logging
import os
import sys
import networkx as nx

logger = logging.getLogger("step3")

from step2_generate import _sanitize_triples
from step2_graphenv_runtime import (
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
from core.vlm import call_vlm, extract_json


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
                           entity_type=e.get("type", ""))
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
                normalized_value=t.get("normalized_value", ""),
                unit=t.get("unit", ""),
                askability=profile.askability,
                lexicalizability=profile.lexicalizability,
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
                from step2_graphenv_runtime import _value_relation_bucket
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
# 出题入口
# ============================================================

# placeholder to find the cut point
def generate_questions(
    entity_json_path: str,
    image_path: str | None = None,
    *,
    seed: int = 42,
    n_questions: int = 9,
    **_kwargs,
) -> dict:
    """构建异构图 → 格式化图谱 → 一次 LLM 调用出题。"""

    with open(entity_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = HeteroSolveGraph(data)
    entities = data.get("high_conf") or data.get("entities") or []
    triples = _sanitize_triples(data.get("triples", []))

    # ---- 1. 直接格式化图谱 ----
    def _format_graph() -> str:
        parts = []

        # 实体列表
        parts.append("【图中实体】")
        for e in entities:
            name = e.get("name", "")
            etype = e.get("type", "")
            loc = e.get("location_in_image", "")
            bbox = e.get("bbox", [])
            parts.append(f"  - {name}  类型={etype}  位置={loc}  bbox={bbox}")

        # 三元组（按 head 分组）
        parts.append(f"\n【知识三元组】（共 {len(triples)} 条）")
        by_head: dict[str, list] = {}
        for t in triples:
            by_head.setdefault(t.get("head", ""), []).append(t)
        for head, ts in by_head.items():
            for t in ts:
                rel = t.get("relation", "")
                tail = t.get("tail", "")
                tt = t.get("tail_type", "")
                nv = t.get("normalized_value", "")
                unit = t.get("unit", "")
                val_str = f" ({nv} {unit})" if nv else ""
                parts.append(f"  {head} —[{rel}]→ {tail} [{tt}]{val_str}")

        # 空间细节
        regions = graph.regions()
        if len(regions) >= 2:
            parts.append("\n【空间细节】")
            for rk in regions:
                info = graph.region_info(rk)
                bbox = info.get("bbox", [])
                if bbox and len(bbox) == 4:
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    parts.append(f"  {info.get('name','')}: 中心=({cx},{cy}) 尺寸={w}×{h} 面积={w*h}px²")

        parts.append(f"\n图片描述: {graph.image_description[:300]}")
        return "\n".join(parts)

    context = _format_graph()
    logger.info(f"图谱: {len(entities)} 实体, {len(triples)} 三元组, context={len(context)} 字符")

    _BATCH_PROMPT = """你是一个多模态智能体（Agentic MLLM）训练数据生成专家。你正在为一个学术研究项目生成高质量的 agentic 训练数据，这些数据将用于训练和评测多模态智能体的工具调用能力。请保持严谨，每道题都必须经得起验证。

下面是一张图片的知识图谱和实体信息。请严格按以下结构生成 9 道中文问题，分 3 个方向，每个方向 3 个难度各 1 道：

## 智能体的 4 种工具

1. web_search(query) → 搜索网络获取文本信息
2. image_search(query/image) → 文字搜图或反向图搜识别实体
3. visit(url) → 访问网页深度阅读
4. code_interpreter(code) → 执行 Python 代码（PIL/OpenCV/easyocr 等），内置丰富的图像处理能力：
   - 几何变换：crop 裁剪、rotate 旋转、flip 翻转、resize 缩放、zoom_in 局部放大
   - 特征检测：canny_edge 边缘检测、hough_line 直线检测、hough_circle 圆检测、contour 轮廓提取（面积/周长）
   - 分割与过滤：grabcut 前景分割、inrange_color 颜色过滤（HSV 阈值）
   - 标注绘制：draw_line 画线、draw_circle 画圆、draw_bbox 画框
   - 颜色分析：color_analysis 主色调/HSV 统计、hist_eq 直方图均衡化
   - 文字识别：OCR（easyocr）
   - 测量计算：面积、距离、像素统计、坐标定位

## 难度定义（严格按工具步数区分！）

- **easy**：tools 列表恰好 1-2 个工具
- **medium**：tools 列表恰好 3-4 个工具
- **hard**：tools 列表恰好 5 个或更多工具

请严格遵守步数要求！easy 题的 tools 不能超过 2 个，hard 题的 tools 不能少于 5 个。

## 3 个方向 × 3 个难度 = 9 题

### 方向 A：检索型（retrieval）— 主要靠 web_search / image_search / visit
easy（1 道，tools 1-2 个）：单步检索即可。
  例："画面中央那个红色标志的品牌是哪一年成立的？" → tools: [web_search]
medium（1 道，tools 3-4 个）：多步检索。
  例："画面左侧品牌的创始人出生在哪个城市？" → tools: [image_search, web_search, visit]
hard（1 道，tools 5+ 个）：多跳推理 + 多实体关联 + 深读。
  例："画面左侧品牌的创始人出生城市的现任市长，他是哪一年上任的？" → tools: [CROP, image_search, web_search, visit, web_search]

### 方向 B：代码型（code）— 主要靠 code_interpreter 做计算/图像处理
easy（1 道，tools 1-2 个）：单步计算或读取。
  例："画面右下角那个标志的面积是多少像素？" → tools: [code_interpreter]
  例："图中有多少个圆形 logo？" → tools: [code_interpreter]（用 HoughCircle 检测）
medium（1 道，tools 3-4 个）：多步代码操作，鼓励用 OpenCV 图像处理。
  例："对画面中央的 logo 做边缘检测后，轮廓的周长是多少像素？" → tools: [code_interpreter, code_interpreter, code_interpreter]（crop → canny_edge → contour 测量）
  例："图中所有红色区域的总面积占整幅图的百分比是多少？" → tools: [code_interpreter, code_interpreter, code_interpreter]（HSV 转换 → inrange_color → 面积统计）
hard（1 道，tools 5+ 个）：复杂的多步视觉操作链。
  例："把左上角 logo 裁剪出来，水平翻转，做边缘检测，计算轮廓周长，与右下角 logo 的轮廓周长对比" → tools: [code_interpreter, code_interpreter, code_interpreter, code_interpreter, code_interpreter]
  例："用 GrabCut 分割出中央人物，统计前景的主色调 HSV 值，再与背景的主色调对比" → tools: [code_interpreter, code_interpreter, code_interpreter, code_interpreter, code_interpreter]

### 方向 C：混合型（hybrid）— 既要检索又要写代码
easy（1 道，tools 1-2 个）：简单搜索+简单计算。
  例："这两个品牌成立年份差多少年？" → tools: [web_search, code_interpreter]
medium（1 道，tools 3-4 个）：检索知识后做计算/图像处理。
  例："画面中两个品牌的年营收差值是多少？" → tools: [web_search, visit, code_interpreter]
  例："裁剪出画面中的品牌 logo，用颜色过滤提取其主色调，搜索该品牌官方色的 RGB 值是否一致" → tools: [code_interpreter, code_interpreter, web_search]
hard（1 道，tools 5+ 个）：完整工具链，结合检索与图像处理。
  例："裁剪出画面中模糊 logo，做直方图均衡化增强后 OCR 识别文字，再搜索该品牌年营收" → tools: [code_interpreter, code_interpreter, code_interpreter, web_search, visit]

## 规则

### 基本格式
1. 用日常口语，简短自然
2. 用位置+外观描述指代图中实体，不写实体真名
3. 每道题只问一个问题
4. **严格控制 tools 列表长度来匹配难度等级**

### 答案质量（最重要！生成的数据将用于学术研究，必须严谨）
5. **答案必须唯一确定**：只能有一个正确答案，不能有多种合理答案。这是最核心的要求。
   坏："这个品牌有什么特色产品？"（答案不唯一）
   坏："它成立的那一年，还有哪些著名公司成立？"（答案不唯一，可以列很多公司）
   坏："这个城市有哪些著名景点？"（答案是开放列表）
   好："这个品牌是哪一年成立的？"（唯一确切值：1919）
   好："这个品牌的总部在哪个城市？"（唯一确切值：Chicago）
   好："这两个广告牌的面积差是多少像素？"（唯一确切值：12345px²）
6. **答案必须可验证**：通过工具调用能得到确定性的结果，不允许主观判断。
   坏："这两个广告牌哪个更好看？"（主观）
   好："这两个广告牌哪个面积更大？"（客观可算）
7. **答案必须基于图谱或图片**：不要凭空引入图谱里没有的外部知识来出题。
   坏："这个大楼的高度相当于几枚土星五号？"（土星五号和图无关）
   好："这个大楼有多少层？"（图谱里有或搜索可得）
8. **禁止开放性问题**：禁止"有哪些""列举""描述""还有什么"类问题。
   坏："这一年还发生了什么大事？"（无限多答案）
   好："这一年美国的总统是谁？"（唯一答案）
9. **short_answer 必须是一个精确的值**：数字、人名、地名、年份等。不能是一个列表或一段话。

### 字段说明
10. **evidence 字段**：列出这道题依赖的图谱子图（用到了哪些实体和三元组），直接从上面的【知识三元组】和【空间细节】里摘抄
11. **answer 字段**：完整回答，可以包含推理过程
12. **short_answer 字段**：精确简短的最终答案（如 "1919"、"日本东京"、"3531px²"）。如果需要搜索才能确定，写 "需搜索"

输出严格 JSON 数组（9 道题，每个方向 easy/medium/hard 各 1 道）：
[
  {{
    "question": "...",
    "answer": "KitchenAid 成立于 1919 年，由 Whirlpool 旗下的 Hobart 公司创立。",
    "short_answer": "1919",
    "tools": ["web_search"],
    "direction": "retrieval",
    "evidence": [
      "KitchenAid —[founded_in]→ 1919 [TIME]"
    ],
    "reasoning": "..."
  }},
  ...
]

{context}
"""

    results_by_cat: dict[str, list] = {}

    prompt = _BATCH_PROMPT.format(context=context)
    logger.info(f"LLM 出题: 目标 9 题 (3方向 × 3题)")

    # 一次 LLM 调用出所有题
    raw = call_vlm(prompt, "请生成问题数组。", max_tokens=4096, temperature=0.6)
    # 解析 JSON 数组
    questions = None
    if raw:
        # 尝试解析 JSON 数组
        import re
        # 找 [ ... ] 块
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                questions = json.loads(match.group())
            except json.JSONDecodeError:
                pass
    if not questions:
        # fallback: 尝试解析单个 JSON
        obj = extract_json(raw) if raw else None
        if obj and isinstance(obj, list):
            questions = obj
        elif obj and isinstance(obj, dict) and obj.get("question"):
            questions = [obj]

    if questions:
        logger.info(f"LLM 返回 {len(questions)} 道题")
        for qi in questions:
            if not isinstance(qi, dict) or not qi.get("question"):
                continue
            # 分类：retrieval / code / hybrid
            # CROP/OCR 是辅助检索的（裁剪方便识别、读文字方便搜索），不算 code
            # 真正的 code：code_interpreter 做计算/几何/图像操作、POINT/INSERTIMAGE/DRAW2DPATH 等
            raw_tools = qi.get("tools", [])
            _RETRIEVAL_HELPERS = {"crop", "ocr"}  # 辅助检索，归入 retrieval
            _REAL_CODE = {"code_interpreter", "point", "insertimage", "draw2dpath", "detectblackarea", "astar"}
            _RETRIEVAL_TOOLS = {"web_search", "image_search", "visit"}
            has_real_code = False
            has_retrieval = False
            for t in raw_tools:
                tn = t.strip().lower() if isinstance(t, str) else ""
                if tn in _REAL_CODE:
                    has_real_code = True
                elif tn in _RETRIEVAL_TOOLS or tn in _RETRIEVAL_HELPERS:
                    has_retrieval = True
            if has_real_code and has_retrieval:
                category = "hybrid"
            elif has_real_code:
                category = "code"
            else:
                category = "retrieval"
            # 难度：按工具步数客观判定（不依赖 LLM 标注）
            n_steps = len(raw_tools)
            if n_steps >= 5:
                difficulty = "hard"
            elif n_steps >= 3:
                difficulty = "medium"
            else:
                difficulty = "easy"
            n = len(results_by_cat.get(category, [])) + 1
            record = {
                "question_id": f"{category}_{difficulty}_{n:02d}",
                "question": qi["question"],
                "answer": qi.get("answer", ""),
                "short_answer": qi.get("short_answer", ""),
                "tools": raw_tools,
                "n_steps": n_steps,
                "category": category,
                "difficulty": difficulty,
                "evidence": qi.get("evidence", []),
                "reasoning": qi.get("reasoning", ""),
            }
            results_by_cat.setdefault(category, []).append(record)
    else:
        logger.warning("LLM 出题失败，返回为空")

    for cat in ("retrieval", "code", "hybrid"):
        results_by_cat.setdefault(cat, [])


    n_ret = len(results_by_cat.get("retrieval", []))
    n_code = len(results_by_cat.get("code", []))
    n_hyb = len(results_by_cat.get("hybrid", []))
    logger.info(f"分类: retrieval={n_ret} code={n_code} hybrid={n_hyb}")

    return {
        **results_by_cat,
        "metadata": {
            "graph_stats": graph.stats(),
            "category_counts": {"retrieval": n_ret, "code": n_code, "hybrid": n_hyb},
            "total_questions": n_ret + n_code + n_hyb,
        },
    }
