"""
第三步：基于三元组的分层问题生成。

流程：
  3a. 从三元组建 NetworkX 有向图
  3b. 三种推理拓扑探测器：Bridge / MultiHop / Comparative
  3c. 每个 Motif 转为结构化骨架 JSON
  3d. LLM 按 Motif 类型润色骨架 → 自然语言问题 + 工具序列

用法：
    python step3_generate.py
    python step3_generate.py --workers 4
"""

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

from core.config import ENTITY_DIR, QUESTION_DIR, FINAL_DIR, STATS_DIR, FILTERED_IMAGE_DIR, MAX_WORKERS
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


def _sanitize_triples(triples: list) -> list:
    """兼容旧数据：缺失 tail_type 时默认 OTHER。"""
    normalized = []
    for t in triples or []:
        if not isinstance(t, dict):
            continue
        item = dict(t)
        item["tail_type"] = _normalize_tail_type(item.get("tail_type"))
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
    """返回 (is_in_image, location)。"""
    if key in exact_map:
        e = exact_map[key]
        return True, e.get("location_in_image", "")
    for kw, e in keyword_map.items():
        if e.get("type") == "brand":
            if key.startswith(kw):
                return True, e.get("location_in_image", "")
        else:
            if kw in set(key.split()) or kw in key:
                return True, e.get("location_in_image", "")
    for e in entities:
        e_name = e["name"].strip().lower()
        if len(e_name) < 3:
            continue
        if e_name in key or key in e_name:
            return True, e.get("location_in_image", "")
    return False, ""


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
            is_img, loc = _match_entity(key, exact_map, keyword_map, entities)
            nodes[key] = {
                "name": name,
                "in_image": is_img,
                "location": loc,
                "tail_type": normalized_tail_type,
            }
            G.add_node(
                key,
                in_image=is_img,
                location=loc,
                name=name,
                tail_type=normalized_tail_type,
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
                       tail_type=tail_type)
        elif (
            G[h_key][t_key].get("tail_type", "OTHER") == "OTHER"
            and tail_type != "OTHER"
        ):
            G[h_key][t_key]["tail_type"] = tail_type

    logger.info(f"    图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G, nodes


# ============================================================
# 3c. 三种推理拓扑探测器
# ============================================================

_BAD_ANSWERS = {"true", "false", "yes", "no", "none", "n/a", "unknown", "various", "multiple"}


def _answer_ok(name: str) -> bool:
    s = name.strip()
    if not s or len(s) < 3:
        return False
    if s.lower() in _BAD_ANSWERS:
        return False
    return True


def _find_bridge_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """交集桥接：两个图中实体共同指向同一个图外节点。
    形态: [A in_image] →rel_1→ [T] ←rel_2← [B in_image]
    """
    bridges = []
    seen_targets = set()
    for target in G.nodes():
        if G.nodes[target].get("in_image"):
            continue
        if not _answer_ok(nodes.get(target, {}).get("name", "")):
            continue
        preds = list(G.predecessors(target))
        img_preds = [p for p in preds if p in in_image_nodes]
        if len(img_preds) >= 2 and target not in seen_targets:
            a, b = img_preds[0], img_preds[1]
            bridges.append({
                "motif_type": "bridge",
                "difficulty": "L2",
                "source_a": a,
                "source_b": b,
                "target": target,
                "rel_a": G[a][target]["relation"],
                "rel_b": G[b][target]["relation"],
                "fact_a": G[a][target].get("fact", ""),
                "fact_b": G[b][target].get("fact", ""),
            })
            seen_targets.add(target)
        if len(bridges) >= max_n:
            break
    return bridges


def _find_multihop_motifs(G: nx.DiGraph, in_image_nodes: set, nodes: dict, max_n: int = 3) -> list:
    """深度跳跃：A → X → T，要求 X 有其他出边（用于生成限定语）。
    形态: [A in_image] →rel_1→ [X] →rel_2→ [T]，X 还有额外属性 →rel_3→ [Y]
    """
    multihops = []
    seen_answers = set()
    for start in in_image_nodes:
        for mid in G.successors(start):
            if mid in in_image_nodes:
                continue
            out_edges = list(G.out_edges(mid, data=True))
            if len(out_edges) < 1:
                continue
            for _, end, end_data in out_edges:
                if end == start or end in in_image_nodes or end in seen_answers:
                    continue
                if not _answer_ok(nodes.get(end, {}).get("name", "")):
                    continue
                # 找 X 上不指向 end 的第一条出边作为限定属性
                disc_edges = [(u, v, d) for u, v, d in out_edges if v != end]
                disc = None
                if disc_edges:
                    _, dv, dd = disc_edges[0]
                    disc = {
                        "node": dv,
                        "node_name": nodes.get(dv, {}).get("name", dv),
                        "relation": dd.get("relation", ""),
                        "fact": dd.get("fact", ""),
                    }
                multihops.append({
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
                })
                seen_answers.add(end)
                if len(multihops) >= max_n:
                    return multihops
    return multihops


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

                a_rel = a_data.get("relation", "").strip()
                b_rel = b_data.get("relation", "").strip()
                a_rel_norm = a_rel.lower().replace(" ", "_")
                b_rel_norm = b_rel.lower().replace(" ", "_")
                same_relation = int(bool(a_rel_norm) and a_rel_norm == b_rel_norm)
                distinct_values = int(a_val != b_val)

                candidate_pairs.append({
                    "a_node": a_node,
                    "a_val": a_val,
                    "a_data": a_data,
                    "a_rel_norm": a_rel_norm,
                    "b_node": b_node,
                    "b_val": b_val,
                    "b_data": b_data,
                    "b_rel_norm": b_rel_norm,
                    "same_relation": same_relation,
                    "distinct_values": distinct_values,
                })

        candidate_pairs.sort(
            key=lambda x: (x["same_relation"], x["distinct_values"]),
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
                "time_value" if bucket_type == "TIME" else "quantity_value"
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
            })
            if len(comparatives) >= max_n:
                return comparatives
    return comparatives


# ============================================================
# 3d. Motif 探测入口
# ============================================================

def find_motifs(triples: list, entities: list,
                max_bridge: int = 3, max_multihop: int = 3, max_comparative: int = 3):
    """建图并运行三个探测器，返回 (motifs_dict, nodes_dict)。"""
    G, nodes = _build_nx_graph(triples, entities)
    in_image_nodes = {k for k, v in nodes.items() if v["in_image"]}

    bridges     = _find_bridge_motifs(G, in_image_nodes, nodes, max_bridge)
    multihops   = _find_multihop_motifs(G, in_image_nodes, nodes, max_multihop)
    comparatives = _find_comparative_motifs(G, in_image_nodes, nodes, max_comparative)

    logger.info(
        f"    Motif 探测: Bridge={len(bridges)}  MultiHop={len(multihops)}  Comparative={len(comparatives)}"
    )
    return {"bridge": bridges, "multihop": multihops, "comparative": comparatives}, nodes


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

    if mtype == "bridge":
        a, b, t = motif["source_a"], motif["source_b"], motif["target"]
        return {
            "motif_type": "bridge_intersection",
            "difficulty": "L2",
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
        return []

    questions = result.get("questions", [])
    for q in questions:
        idx = q.get("skeleton_index", 0)
        if idx < len(skeletons):
            sk = skeletons[idx]
            q["reasoning_path"] = {
                "motif_type": sk.get("motif_type"),
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

    # ---- 3a. Motif 探测 ----
    motifs, nodes = find_motifs(triples, entities)
    bridge_skeletons     = [motif_to_skeleton(m, nodes) for m in motifs["bridge"]]
    multihop_skeletons   = [motif_to_skeleton(m, nodes) for m in motifs["multihop"]]
    comparative_skeletons = [motif_to_skeleton(m, nodes) for m in motifs["comparative"]]

    logger.info(
        f"  [{img_id}] 骨架数: Bridge={len(bridge_skeletons)}"
        f"  MultiHop={len(multihop_skeletons)}"
        f"  Comparative={len(comparative_skeletons)}"
    )

    # ---- 3b. 并行润色（4路并发VLM调用） ----
    image_desc = entity_data.get("image_description", "")
    domain = entity_data.get("domain", "other")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_bridge = pool.submit(
            polish_level, bridge_skeletons, "bridge_intersection", image_b64, image_desc, domain)
        fut_multihop = pool.submit(
            polish_level, multihop_skeletons, "multihop_path", image_b64, image_desc, domain)
        fut_comparative = pool.submit(
            polish_level, comparative_skeletons, "comparative", image_b64, image_desc, domain)
        fut_vis = pool.submit(
            generate_vision_questions, entities, image_b64, image_desc, domain)

        bridge_qs      = fut_bridge.result()
        multihop_qs    = fut_multihop.result()
        comparative_qs = fut_comparative.result()
        vision_qs      = fut_vis.result()

    # L1 = 纯视觉题；L2 = Bridge + Comparative；L3 = MultiHop
    all_l1 = vision_qs
    l2_qs  = bridge_qs + comparative_qs
    l3_qs  = multihop_qs

    for i, q in enumerate(all_l1):
        q["question_id"] = f"L1_{i+1:02d}"
    for i, q in enumerate(l2_qs):
        q["question_id"] = f"L2_{i+1:02d}"
    for i, q in enumerate(l3_qs):
        q["question_id"] = f"L3_{i+1:02d}"

    logger.info(
        f"  [{img_id}] 生成完成: L1={len(all_l1)} L2={len(l2_qs)} L3={len(l3_qs)}"
    )

    total_selected = len(bridge_skeletons) + len(multihop_skeletons) + len(comparative_skeletons)
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
            "chains_selected": total_selected,
        },
    }

    save_checkpoint(3, img_id, result)
    os.makedirs(QUESTION_DIR, exist_ok=True)
    q_file = os.path.join(QUESTION_DIR, f"{img_id}.json")
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ============================================================
# 聚合最终输出
# ============================================================
def aggregate_final():
    """将所有问题文件聚合为最终 JSONL。"""
    os.makedirs(FINAL_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    q_files = sorted(glob.glob(os.path.join(QUESTION_DIR, "*.json")))
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
                        "generation_model": "qwen3.5-397b",
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
