"""
第二步：实体提取与知识扩展。

对第一步筛选出的图片：
  2a. VLM 直接输出实体名称 + Bounding Box + 位置描述（纯 VLM 驱动，无 SAM3）
  2b. LLM 为每个实体生成搜索计划 → 执行真实搜索 → 从搜索结果提取事实三元组

用法：
    python step2_enrich.py
    python step2_enrich.py --workers 4
"""

import argparse
import base64
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

# 搜索请求绕过代理（Serper 直连）
_http = httpx.Client(trust_env=False, timeout=30)
# Jina Reader 需要系统证书，不能禁 trust_env
_jina_http = httpx.Client(timeout=30)

from core.config import (
    FILTERED_IMAGE_DIR, ENTITY_DIR, MAX_WORKERS,
    SERPER_KEY, JINA_API_KEY, JINA_READER_URL, JINA_TIMEOUT,
)
from core.vlm import call_vlm, call_vlm_json
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step2", "step2_enrich.log")

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

ENTITY_CROP_QUALITY = int(os.environ.get("ENTITY_CROP_QUALITY", "90"))

# ============================================================
# Serper / Jina 调用缓存（disk-based memoization）
# 同一 query/url 不重复调 API，直接返回缓存结果。
# 缓存目录：output/.cache/serper/ 和 output/.cache/jina/
# 缓存 key：sha256(api_type + 请求参数)[:16].json
# 跳过缓存：设置环境变量 DISABLE_API_CACHE=1
# ============================================================
import hashlib as _hl

_API_CACHE_DIR = os.path.join("output", ".cache")
_DISABLE_CACHE = os.environ.get("DISABLE_API_CACHE", "") == "1"


def _cache_key(api_type: str, **params) -> str:
    payload = json.dumps({"_t": api_type, **params}, sort_keys=True, ensure_ascii=False)
    return _hl.sha256(payload.encode()).hexdigest()[:16]


def _cache_get(api_type: str, key: str) -> dict | None:
    if _DISABLE_CACHE:
        return None
    path = os.path.join(_API_CACHE_DIR, api_type, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_set(api_type: str, key: str, data: dict) -> None:
    if _DISABLE_CACHE:
        return
    d = os.path.join(_API_CACHE_DIR, api_type)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass  # 缓存写失败不影响主流程


# ============================================================
# Prompt：搜索计划生成
# ============================================================
SEARCH_PLAN_PROMPT = """\
你是一个知识图谱搜索规划师。给定从图片中提取的实体列表，你需要为每个实体规划搜索查询，目的是获取足够信息来构建多跳知识链。

## 图片描述
{image_description}

## 实体列表
{entities_json}

## 任务
为每个实体规划 2-3 条搜索查询。每条查询要：
1. 有明确的搜索目的（了解什么信息）
2. 查询词要自然、适合搜索引擎（不要太长，不要堆砌关键词）
3. 不同查询要覆盖不同方向（如：基本信息、关联公司/人物、历史/地理背景）
4. 考虑实体之间的交叉关联——如果多个实体可能有关联，设计能发现这种关联的查询

## 注意
- 对于通用词（如"琴弦"、"红色窗帘"），搜索意义不大，可以标记 skip=true
- 对于有具体名称的实体（品牌、型号、地名、人名等），才值得搜索
- 查询词用实体相关的语言（英文实体用英文搜，中文实体用中文搜）

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "search_plans": [
        {{
            "entity_id": "E1",
            "entity_name": "McDonald's",
            "skip": false,
            "skip_reason": null,
            "queries": [
                {{
                    "query": "McDonald's history founding",
                    "purpose": "了解麦当劳的创立历史"
                }},
                {{
                    "query": "McDonald's Corporation revenue headquarters",
                    "purpose": "了解公司规模和总部位置"
                }},
                {{
                    "query": "McDonald's Times Square New York",
                    "purpose": "了解麦当劳在时代广场的店铺信息（与图片场景关联）"
                }}
            ]
        }},
        {{
            "entity_id": "E4",
            "entity_name": "红色窗帘",
            "skip": true,
            "skip_reason": "通用物品描述，搜索无法获取有价值的知识链信息",
            "queries": []
        }}
    ]
}}"""


# ============================================================
# Prompt：从搜索结果中提取事实三元组
# ============================================================
TRIPLE_EXTRACTION_PROMPT = """\
你需要从搜索结果中提取事实三元组。每个三元组描述两个实体之间的一个事实关系。

## 图中实体（标记为 in_image）
{entities_summary}

## 各实体的搜索结果
{search_results}

## 任务
请逐条阅读搜索结果，提取所有有价值的事实三元组。

要求：
1. 每个三元组的 fact 必须有搜索结果中的原文佐证（source_snippet），不要编造
2. 实体名要具体、准确（如"Imperial Theatre"而非"某剧院"，"1528 Broadway"而非"某地址"）
3. 【命名一致性】如果某个实体在"图中实体"列表中已有名称，head/tail 必须完全使用该名称，禁止加地点后缀、店名变体或其他修饰（例如图中实体是"McDonald's"，则不得写成"McDonald's Times Square"或"McDonald's Corporation"）
4. 如果搜索结果中提到了两个图中实体的关联，务必提取
5. 尽量多提取，覆盖不同方向的关系（地理、历史、商业、文化等）
6. head 和 tail 不能是同一个实体
7. 不要只提取"图中实体→外部知识"的关系，也要提取搜索结果中提到的"外部知识→外部知识"的关系，这样才能形成多跳链
8. 每个三元组必须包含字段 "tail_type"，并且只能从以下枚举中选择一个值：
   ["TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"]
   - TIME: 年份、日期、时间段
   - QUANTITY: 带单位或可比较的数值（金额、人数、面积、长度、重量等）
   - LOCATION: 国家/城市/地址/地理位置
   - PERSON: 具体人物
   - ORG: 公司/机构/组织
   - OTHER: 以上都不满足
9. 如果 "tail_type" 是 TIME 或 QUANTITY，请尽量补充：
   - "normalized_value": 便于计算/比较的标准化值。TIME 优先写年份或 ISO 日期字符串；QUANTITY 优先写纯数字
   - "unit": TIME 可写 year/date；QUANTITY 写 $, people, km2, kg, percent 等。如果无法可靠判断，可填空字符串
10. 每个三元组请补充 "provenance"，只能从以下枚举中选择一个：
   - text_exact
   - text_rewrite
   - image_resolved
   - cross_entity

## 示例
假设搜索 Billy Elliot 的结果提到"Billy Elliot premiered at the Imperial Theatre, 249 West 45th Street"，应提取两条三元组：
- (Billy Elliot, premiered_at, Imperial Theatre)
- (Imperial Theatre, located_at, 249 West 45th Street)
而不是只提取第一条。搜索结果中出现的每一对实体关系都值得提取。

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "triples": [
        {{
            "head": "头实体名",
            "relation": "关系类型（如 located_at, founded_by, performed_at, headquartered_in 等）",
            "tail": "尾实体名",
            "tail_type": "TIME/QUANTITY/LOCATION/PERSON/ORG/OTHER 之一",
            "normalized_value": "可选；TIME/QUANTITY 时尽量填写，如 1990 / 2024-01-01 / 500000000",
            "unit": "可选；TIME/QUANTITY 时尽量填写，如 year / date / $ / people / km2",
            "provenance": "text_exact/text_rewrite/image_resolved/cross_entity 之一",
            "fact": "一句话描述这个事实",
            "source_snippet": "搜索结果中的佐证原文片段"
        }}
    ]
}}"""


# ============================================================
# Prompt：第二轮搜索计划
# ============================================================
ROUND2_SEARCH_PLAN_PROMPT = """\
你需要为以下实体各生成一条搜索查询词，目的是发现这些实体的**更多关联信息**，以延伸知识链。

## 待搜索的实体及其上下文
{tail_entities_with_context}

## 任务
为每个实体生成一条高质量的搜索查询词。要求：
1. 查询词要能发现该实体的**新的关联**（地理、历史、人物、事件等），而不是重复已有信息
2. **优先挖掘数值/日期/位置/人名类事实**，这些事实能直接用于 compare/rank：
   - 时间类：founded_year, established_date, draft_year, debut_year, release_date
   - 数值类：revenue, brand_value, arena_capacity, population, box_office, attendance
   - 位置类：headquartered_in, arena_location, home_city, born_in
   - 人物类：owner, CEO, founder, head_coach, general_manager
   例："Denver Nuggets" → 搜 "Denver Nuggets owner arena capacity founded year"
3. 如果实体本身太泛（如纯年份"1960"、通用概念"jazz musicians"），要加上方向性的词使搜索有意义，或者标记 skip
4. 查询词要自然、适合搜索引擎
5. 如果某个实体确实不值得搜索（太泛、太通用、搜不出有价值的信息），标记 skip=true

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "queries": [
        {{
            "entity": "实体名",
            "skip": false,
            "query": "搜索查询词"
        }}
    ]
}}"""


# ============================================================
# Prompt：跨实体关联发现 — LLM 为每对实体生成搜索计划
# ============================================================
CROSS_ENTITY_PROMPT = """\
你是一个知识图谱搜索规划师。以下是同一张图片中出现的若干实体对，你需要为每一对规划搜索查询，目的是发现它们之间的真实世界关联。

## 图片描述
{image_description}

## 图中实体
{entities_summary}

## 需要生成搜索计划的实体对
{pairs_list}

## 任务
为每对实体规划 1-2 条搜索查询。每条查询要：
1. 有明确的搜索目的（了解什么关联信息）
2. 查询词要自然、适合搜索引擎（不要太长，不要堆砌关键词）
3. 查询词要直接针对两者之间可能存在的关联：同一母公司、同一地点、竞争关系、合作关系、历史渊源、同一行业等
4. 不要简单拼接两个实体名——设计能真正揭示它们关联的搜索词
5. 不同查询要覆盖不同方向（如：地理关联、商业关系、历史渊源）
6. 结合图片场景上下文推测可能的关联方向

## 注意
- 如果两个实体明显没有关联可能（如一个品牌logo和一个毫无关系的通用物品），可以标记 skip=true
- 查询词用实体相关的语言（英文实体用英文搜，中文实体用中文搜）

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "search_plans": [
        {{
            "entity_a": "McDonald's",
            "entity_b": "TGI Fridays",
            "skip": false,
            "skip_reason": null,
            "queries": [
                {{
                    "query": "Times Square fast food restaurants competition",
                    "purpose": "了解两家餐厅在时代广场的竞争或邻近关系"
                }},
                {{
                    "query": "McDonald's TGI Fridays parent company restaurant industry",
                    "purpose": "了解两个餐饮品牌是否有商业或母公司层面的关联"
                }}
            ]
        }},
        {{
            "entity_a": "红色窗帘",
            "entity_b": "Sony",
            "skip": true,
            "skip_reason": "红色窗帘是通用物品描述，与Sony品牌之间不太可能有有价值的关联",
            "queries": []
        }}
    ]
}}"""


# ============================================================
# Prompt：从跨实体搜索结果中提取桥接三元组
# ============================================================
CROSS_ENTITY_TRIPLE_PROMPT = """\
你需要从跨实体搜索结果中提取桥接三元组。重点是发现图中不同实体之间的直接或间接关系。

## 图中实体
{entities_summary}

## 跨实体搜索结果
{cross_search_results}

## 任务
请从搜索结果中提取事实三元组，重点关注：
1. 两个图中实体之间的**直接关系** (A, relation, B)，如"TDK sponsors Times Square billboard"
2. 两个图中实体通过**桥节点**产生的间接关联 (A, rel, X) + (B, rel, X)，如两个品牌都属于同一母公司
3. 每个三元组的 fact 必须有搜索结果中的原文佐证（source_snippet），不要编造
4. 【命名一致性】图中实体的 head/tail 必须完全使用"图中实体"列表中的原始名称，禁止加地点后缀或任何变体
5. 实体名要具体、准确
6. head 和 tail 不能是同一个实体
7. 每个三元组必须包含字段 "tail_type"，并且只能从以下枚举中选择一个值：
   ["TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"]
8. 如果 tail_type 是 TIME 或 QUANTITY，也尽量补充 "normalized_value" 和 "unit"
9. provenance 固定填写 "cross_entity"

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{{{
    "triples": [
        {{{{
            "head": "头实体名",
            "relation": "关系类型",
            "tail": "尾实体名",
            "tail_type": "TIME/QUANTITY/LOCATION/PERSON/ORG/OTHER 之一",
            "normalized_value": "可选；TIME/QUANTITY 时尽量填写",
            "unit": "可选；TIME/QUANTITY 时尽量填写",
            "provenance": "cross_entity",
            "fact": "一句话描述这个事实",
            "source_snippet": "搜索结果中的佐证原文片段"
        }}}}
    ]
}}}}"""


# ============================================================
# 三元组字段规范化
# ============================================================

_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}
_PROVENANCE_ENUM = {"text_exact", "text_rewrite", "image_resolved", "cross_entity"}


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


def _normalize_provenance(value, default: str = "text_rewrite") -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized if normalized in _PROVENANCE_ENUM else default


def _sanitize_triples(triples: list[dict]) -> list[dict]:
    """规范化三元组字段，确保 tail_type 始终存在且合法。"""
    normalized = []
    for t in triples or []:
        if not isinstance(t, dict):
            continue
        item = dict(t)
        item["tail_type"] = _normalize_tail_type(item.get("tail_type"))
        item["normalized_value"] = _normalize_scalar_value(item.get("normalized_value"))
        item["unit"] = _normalize_unit(item.get("unit"))
        item["provenance"] = _normalize_provenance(item.get("provenance"), default="text_rewrite")
        item.pop("relation_family", None)
        normalized.append(item)
    return normalized


def _entity_type_alias(entity: dict) -> str:
    etype = str(entity.get("type", "")).strip().lower()
    return {
        "brand": "brand",
        "person": "person",
        "landmark": "landmark",
        "product": "product",
        "text": "title",
    }.get(etype, "")


def _exact_name_queries_for_entity(entity: dict) -> list[dict]:
    name = str(entity.get("name", "")).strip()
    if not name:
        return []
    queries = [{"query": name, "origin": "text_exact", "purpose": "实体精确名基线检索"}]
    alias = _entity_type_alias(entity)
    if alias:
        queries.append({"query": f"{name} {alias}", "origin": "text_exact", "purpose": "实体精确名 + 类型别名检索"})
    return queries


def _exact_name_queries_for_tail(tail_name: str) -> list[dict]:
    tail_name = str(tail_name or "").strip()
    if not tail_name:
        return []
    return [{"query": tail_name, "origin": "text_exact", "purpose": "扩展实体精确名基线检索"}]


# ============================================================
# 工具函数
# ============================================================

def _bbox_to_location(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    """将 bbox 转换为自然语言位置描述，用于 step3 模糊化。

    使用连续比例而非硬切三等分，输出更自然的中文描述。
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    rx = cx / max(width, 1)
    ry = cy / max(height, 1)

    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))

    if rx < 0.2:
        hpos = "最左侧"
    elif rx < 0.4:
        hpos = "偏左"
    elif rx > 0.8:
        hpos = "最右侧"
    elif rx > 0.6:
        hpos = "偏右"
    else:
        hpos = ""

    if ry < 0.2:
        vpos = "顶部"
    elif ry < 0.4:
        vpos = "上方"
    elif ry > 0.8:
        vpos = "底部"
    elif ry > 0.6:
        vpos = "下方"
    else:
        vpos = ""

    if not hpos and not vpos:
        return "画面中央"

    size_hint = ""
    if area_ratio > 0.15:
        size_hint = "大幅"
    elif area_ratio < 0.005:
        size_hint = "细小的"

    parts = []
    if size_hint:
        parts.append(size_hint)
    parts.append("画面")
    if vpos:
        parts.append(vpos)
    if hpos:
        parts.append(hpos)

    return "".join(parts)


def _extract_value(name: str):
    m = re.search(
        r"(\$\s*\d+(?:[\.,]\d+)?)|(¥\s*\d+(?:[\.,]\d+)?)|"
        r"(\d+(?:[\.,]\d+)?\s*(?:kg|g|cm|mm|inch|in|hz|mhz|ghz|gb|tb|w|kw|v|mah))",
        name, flags=re.IGNORECASE,
    )
    return m.group(0).strip() if m else None


def _confidence_bucket(conf: float) -> str:
    if conf >= 0.75:
        return "high"
    if conf >= 0.45:
        return "medium"
    return "low"


def _guess_domain_from_labels(labels: list[str]) -> str:
    norm = " ".join(labels).lower()
    if any(k in norm for k in ("burger", "pizza", "sandwich", "hot dog", "donut", "apple", "banana", "cake")):
        return "food"
    if any(k in norm for k in ("football", "basketball", "tennis", "baseball", "skateboard", "surfboard")):
        return "sports"
    if any(k in norm for k in ("computer", "laptop", "phone", "keyboard", "monitor", "mouse", "tv")):
        return "electronics"
    if any(k in norm for k in ("store", "shop", "cashier", "shelf", "product", "cart")):
        return "retail"
    if any(k in norm for k in ("map", "street", "road", "building", "bridge", "tower")):
        return "geography"
    return "other"


def _describe_spatial_relation(cx, cy, ox, oy, other_name: str) -> str:
    """描述 (cx,cy) 相对于 (ox,oy) 的空间方向关系。"""
    dx = cx - ox
    dy = cy - oy

    angle_deg = 0
    import math
    if abs(dx) > 1 or abs(dy) > 1:
        angle_deg = math.degrees(math.atan2(-dy, dx)) % 360

    if angle_deg < 22.5 or angle_deg >= 337.5:
        direction = "右方"
    elif angle_deg < 67.5:
        direction = "右上方"
    elif angle_deg < 112.5:
        direction = "上方"
    elif angle_deg < 157.5:
        direction = "左上方"
    elif angle_deg < 202.5:
        direction = "左方"
    elif angle_deg < 247.5:
        direction = "左下方"
    elif angle_deg < 292.5:
        direction = "下方"
    else:
        direction = "右下方"

    return f"{other_name}的{direction}"


def _attach_nearby_entities(entities: list[dict], width: int, height: int) -> None:
    """为每个实体附加空间邻近关系，包含方向描述。"""
    if not entities:
        return
    diag = (width**2 + height**2) ** 0.5
    threshold = 0.28 * diag
    centers = []
    for e in entities:
        bbox = e.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = bbox
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    for i, e in enumerate(entities):
        cx, cy = centers[i]
        neighbors = []
        for j, other in enumerate(entities):
            if i == j:
                continue
            ox, oy = centers[j]
            d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if d <= threshold:
                rel = _describe_spatial_relation(cx, cy, ox, oy, other["name"])
                neighbors.append({
                    "id": other["id"],
                    "name": other["name"],
                    "relation": rel,
                    "distance_ratio": round(d / diag, 3),
                })
        neighbors.sort(key=lambda n: n["distance_ratio"])
        e["nearby_entities"] = neighbors[:6]




# ============================================================
# 三元组实体名规范化（方案 C：启发式合并）
# ============================================================

def _llm_canonicalize_aliases(
    triples: list[dict],
    entities: list[dict],
    img_id: str,
) -> tuple[list[dict], list[dict], dict[str, str]]:
    """LLM-based 实体别名归一化。

    场景：启发式 _normalize_triple_entities 只做字串 substring 匹配，管不住
    "Nuggets / Denver Nuggets / the Denver Nuggets" 这类同一实体的多种写法。
    多跳题的 bridge node 依赖干净的实体名，否则同一个桥节点会被拆成多个。

    算法：
      1. 收集所有出现在 entities[] + triples[].head/tail 里的 unique 字符串
      2. 一次 LLM 调用，要求它把同义变体分组，为每组选一个 canonical form
      3. 按 alias_map 重写所有 triples head/tail + entities name
      4. 去重：归一化后可能产生重复 triples / entities

    Returns: (new_triples, new_entities, alias_map)
      alias_map: 原字符串 → canonical 字符串
      如果 LLM 调用失败 / 返回空，返回原始 triples + entities + 空 map。
    """
    # 1. 收集所有候选字符串
    names: set[str] = set()
    for e in entities:
        n = (e.get("name") or "").strip()
        if n:
            names.add(n)
    for t in triples:
        for side in ("head", "tail"):
            v = (t.get(side) or "").strip()
            if v:
                names.add(v)

    # 过滤掉太短 / 纯数字 / 非实体的（LLM 不用管这些）
    def _meaningful(s: str) -> bool:
        s = s.strip()
        if len(s) < 2 or len(s) > 80:
            return False
        # 纯数字、年份、金额等不参与归一（它们是 value，不是实体）
        if s.replace(",", "").replace(".", "").replace("%", "").replace(" ", "").isdigit():
            return False
        return True

    candidates = sorted(n for n in names if _meaningful(n))
    if len(candidates) < 2:
        return triples, entities, {}

    # 2. 调 LLM 做分组
    prompt = """你是实体名归一化助手。给你一份字符串列表，里面混杂了同一实体的多种写法。
你的任务：找出所有**指向同一实体**的变体，选一个最规范的形式作为 canonical。

判据：
- 只归一真正同义的变体（大小写、冠词、缩写、全称 vs 简称、翻译、空格、标点差异）
- 不归一语义相似但不同的实体（例如 "Apple" 公司 和 "Apple" 水果不合并）
- 不归一上下位关系（例如 "NBA" 和 "Brooklyn Nets" 不合并）
- canonical 选择优先级：最具体、最常用的正式全称（例如选 "Denver Nuggets" 而不是 "Nuggets"）
- 严格：没有明显变体的字符串不出现在输出里，不要硬凑

输出严格 JSON：
{
  "groups": [
    {"canonical": "Denver Nuggets", "aliases": ["Nuggets", "the Denver Nuggets", "DEN"]},
    {"canonical": "Google Pixel 9 Pro", "aliases": ["Pixel 9 Pro", "Google Pixel"]}
  ]
}

规则：
- canonical 必须是输入列表里的一个字符串（不要自创）
- aliases 必须是输入列表里的字符串（不要自创）
- aliases 不含 canonical 自身
- 只输出需要归一的组，不包含孤立实体
- 严格输出 JSON，不加 markdown 代码块标记"""

    user = f"图片 {img_id} 的候选实体/三元组字符串列表（共 {len(candidates)} 个）：\n\n" + "\n".join(
        f"- {n}" for n in candidates
    )

    try:
        resp = call_vlm_json(
            prompt, user,
            max_tokens=2000,
            temperature=0.2,
            max_attempts=2,
        )
    except Exception as exc:
        logger.warning(f"  [{img_id}] alias LLM call failed: {exc}")
        return triples, entities, {}

    if not isinstance(resp, dict):
        return triples, entities, {}

    groups = resp.get("groups") or []
    if not isinstance(groups, list):
        return triples, entities, {}

    # 3. 构建 alias_map
    alias_map: dict[str, str] = {}
    candidates_set = set(candidates)
    for g in groups:
        if not isinstance(g, dict):
            continue
        canon = (g.get("canonical") or "").strip()
        aliases = g.get("aliases") or []
        if not canon or canon not in candidates_set:
            continue
        if not isinstance(aliases, list):
            continue
        for a in aliases:
            if not isinstance(a, str):
                continue
            a = a.strip()
            if a and a != canon and a in candidates_set:
                alias_map[a] = canon

    if not alias_map:
        return triples, entities, {}

    # 4. 应用归一化到 triples
    new_triples: list[dict] = []
    seen_triples: set[tuple] = set()
    merged_count = 0
    for t in triples:
        h = (t.get("head") or "").strip()
        ta = (t.get("tail") or "").strip()
        new_h = alias_map.get(h, h)
        new_ta = alias_map.get(ta, ta)
        if new_h != h or new_ta != ta:
            merged_count += 1
        if not new_h or not new_ta or new_h.lower() == new_ta.lower():
            continue  # 归一后变成自环
        key = (new_h.lower(), (t.get("relation") or "").lower(), new_ta.lower())
        if key in seen_triples:
            continue
        seen_triples.add(key)
        new_triples.append({**t, "head": new_h, "tail": new_ta})

    # 5. 应用归一化到 entities：被归为 alias 的 entity 从列表里删除
    alias_set = set(alias_map.keys())
    new_entities: list[dict] = []
    dropped_entity_count = 0
    for e in entities:
        n = (e.get("name") or "").strip()
        if n in alias_set:
            dropped_entity_count += 1
            continue
        new_entities.append(e)

    logger.info(
        f"  [{img_id}] alias LLM: {len(groups)} groups, {len(alias_map)} aliases merged, "
        f"triples {len(triples)} → {len(new_triples)} ({merged_count} rewritten), "
        f"entities {len(entities)} → {len(new_entities)} (dropped {dropped_entity_count})"
    )
    return new_triples, new_entities, alias_map


def _normalize_triple_entities(triples: list[dict], entities: list[dict]) -> list[dict]:
    """将三元组中漂移的实体名统一回图中实体的 canonical name。

    规则：对每个 head/tail，若它包含某个图中实体名（或被其包含），
    且该图中实体名长度 >= 3，则替换为图中实体名。
    替换后 head == tail 的三元组丢弃。
    """
    # 按名称长度降序排列，优先匹配更长的实体名，避免"McDonald's"误匹配"McDonald's Corporation"
    canonical = sorted(
        [e["name"].strip() for e in entities if len(e["name"].strip()) >= 3],
        key=len,
        reverse=True,
    )

    def _resolve(name: str) -> str:
        name_lower = name.lower()
        for canon in canonical:
            canon_lower = canon.lower()
            if canon_lower in name_lower or name_lower in canon_lower:
                return canon
        return name

    normalized = []
    for t in triples:
        head = _resolve(t.get("head", "").strip())
        tail = _resolve(t.get("tail", "").strip())
        if not head or not tail or head.lower() == tail.lower():
            continue
        normalized.append({**t, "head": head, "tail": tail})

    # 去重（规范化后可能产生重复）
    seen: set[tuple] = set()
    deduped = []
    for t in normalized:
        key = (t["head"].lower(), t.get("relation", "").lower(), t["tail"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


# ============================================================
# VLM 实体提取（纯 VLM 驱动，直接输出 BBox）
# ============================================================

VLM_ENTITY_PROMPT = """\
请仔细观察这张图片（使用 0-1000 的归一化坐标系，左上角为原点）。列出所有你能识别的具体实体。

要求：
1. 只列有具体名称的实体（品牌logo、地标建筑、文字招牌、人名、产品名、公司名等），不要列泛泛的物体（如"汽车""行人""建筑"）
2. name：用于后续网络搜索的实体具体名称（如"McDonald's", "索尼 WH-1000XM4"）
3. bbox：该实体在图中的边界框 [x_min, y_min, x_max, y_max]，坐标范围 0-1000。请尽量精确框住实体的实际范围
4. 每个实体只列一次，同一实体在图中出现多次时只取最清晰的那个，不得重复
5. 同时给出一段详细的图片整体描述（100字以上）

输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "image_description": "图片整体描述",
    "entities": [
        {{
            "name": "实体名",
            "type": "brand/landmark/text/person/product/object",
            "bbox": [100, 200, 300, 400]
        }}
    ]
}}"""


def extract_entities_vlm(img_path: str, img_id: str | None = None) -> dict:
    """纯 VLM 驱动的实体提取：VLM 直接输出 BBox 和位置描述。

    流程：
      1. 读取图片 → 调用 VLM → 解析返回 JSON
      2. 将 0-1000 归一化 bbox 转换为实际像素坐标
      3. 面积占全图 >80% → 幻觉，跳过
      4. 裁剪实体图片并组装返回数据
    """
    if Image is None:
        raise RuntimeError("未安装 Pillow。")

    if img_id is None:
        img_id = os.path.splitext(os.path.basename(img_path))[0]

    rgb_image = Image.open(img_path).convert("RGB")
    width, height = rgb_image.size
    crop_dir = os.path.join(ENTITY_DIR, "crops", img_id)
    os.makedirs(crop_dir, exist_ok=True)

    logger.info(f"    [{img_id}] VLM 识别实体（名称 + BBox）...")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    image_data_url = f"data:image/jpeg;base64,{b64}"

    vlm_result = call_vlm_json(
        "你是一个图片实体识别专家。请仔细观察图片中的所有细节，尽可能多地识别具体实体，为每个实体提供精确的边界框坐标。",
        [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": VLM_ENTITY_PROMPT},
        ],
        max_tokens=2000,
        temperature=0.3,
    )

    if vlm_result is None:
        logger.warning(f"    [{img_id}] VLM 实体识别失败")
        return {"entities": [], "image_description": "", "domain": "other"}

    image_desc = vlm_result.get("image_description", "")
    vlm_entities = vlm_result.get("entities", [])

    # 去重：名称完全相同（忽略大小写）的保留第一个
    seen_names: set[str] = set()
    deduped_vlm: list[dict] = []
    for e in vlm_entities:
        key = e.get("name", "").strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            deduped_vlm.append(e)
    if len(deduped_vlm) < len(vlm_entities):
        logger.info(f"    [{img_id}] 实体去重: {len(vlm_entities)} → {len(deduped_vlm)} 个")
    vlm_entities = deduped_vlm

    logger.info(f"    [{img_id}] VLM 识别出 {len(vlm_entities)} 个候选实体")

    entities = []
    hallucination_count = 0
    idx = 1
    for e in vlm_entities:
        name = e.get("name", "").strip()
        if not name or len(name) < 2:
            continue

        raw_bbox = e.get("bbox")
        if not raw_bbox or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            logger.info(f"    [{img_id}] 跳过 '{name}': 无有效 bbox")
            continue

        # 归一化坐标 → 实际像素坐标
        ix1 = max(0, int(float(raw_bbox[0]) / 1000.0 * width))
        iy1 = max(0, int(float(raw_bbox[1]) / 1000.0 * height))
        ix2 = min(width, max(ix1 + 1, int(float(raw_bbox[2]) / 1000.0 * width)))
        iy2 = min(height, max(iy1 + 1, int(float(raw_bbox[3]) / 1000.0 * height)))

        area_ratio = ((ix2 - ix1) * (iy2 - iy1)) / max(1.0, float(width * height))

        # 幻觉拦截：bbox 面积占全图 >80%
        if area_ratio > 0.8:
            hallucination_count += 1
            logger.info(f"    [{img_id}] 移除幻觉实体 '{name}': bbox 占全图 {area_ratio:.0%}")
            continue

        eid = f"E{idx}"

        # 五档搜索视图：tight / pad20 / pad40 / context / full_scene
        # tight  : 原 bbox 紧裁 —— 给 OCR 和精细 face/logo 识别
        # pad20  : bbox 扩 20% —— 保留一点周边上下文，Lens 反查默认用这档
        # pad40  : bbox 扩 40% —— 更大周边上下文，对难命中的实体更有效
        # context: 以实体为中心的 4× 面积 —— 捕获场景级周边（球场、货架、招牌墙）
        # full_scene: 整图 —— Lens 场景匹配后备（单张图共享一份，减少磁盘）
        search_views = {}
        # tight
        tight_crop = rgb_image.crop((ix1, iy1, ix2, iy2))
        tight_path = os.path.join(crop_dir, f"{eid}.jpg")
        tight_crop.save(tight_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
        search_views["tight"] = tight_path

        # pad20
        pw20 = int((ix2 - ix1) * 0.2)
        ph20 = int((iy2 - iy1) * 0.2)
        pad20_crop = rgb_image.crop((
            max(0, ix1 - pw20), max(0, iy1 - ph20),
            min(width, ix2 + pw20), min(height, iy2 + ph20),
        ))
        pad20_path = os.path.join(crop_dir, f"{eid}_pad20.jpg")
        pad20_crop.save(pad20_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
        search_views["pad20"] = pad20_path

        # pad40
        pw40 = int((ix2 - ix1) * 0.4)
        ph40 = int((iy2 - iy1) * 0.4)
        pad40_crop = rgb_image.crop((
            max(0, ix1 - pw40), max(0, iy1 - ph40),
            min(width, ix2 + pw40), min(height, iy2 + ph40),
        ))
        pad40_path = os.path.join(crop_dir, f"{eid}_pad40.jpg")
        pad40_crop.save(pad40_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
        search_views["pad40"] = pad40_path

        # context
        cx, cy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
        cw, ch = (ix2 - ix1) * 2, (iy2 - iy1) * 2
        ctx_crop = rgb_image.crop((
            max(0, cx - cw), max(0, cy - ch),
            min(width, cx + cw), min(height, cy + ch),
        ))
        ctx_path = os.path.join(crop_dir, f"{eid}_context.jpg")
        ctx_crop.save(ctx_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
        search_views["context"] = ctx_path

        # full_scene: 共享单张，每个实体都引用同一路径（减少磁盘 I/O）
        # 用固定 filename 实现单图幂等——循环里第一次写入后续直接引用
        full_scene_path = os.path.join(crop_dir, "full_scene.jpg")
        if not os.path.exists(full_scene_path):
            rgb_image.save(full_scene_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
        search_views["full_scene"] = full_scene_path

        pixel_bbox = [ix1, iy1, ix2, iy2]
        entities.append({
            "id": eid,
            "name": name,
            "type": e.get("type", "other"),
            "value": _extract_value(name),
            "bbox": pixel_bbox,
            "location_in_image": _bbox_to_location(ix1, iy1, ix2, iy2, width, height),
            "confidence": 0.9,
            "confidence_level": "high",
            "crop_path": tight_path,  # 向后兼容
            "search_views": search_views,
            "source": "vlm_only",
        })
        idx += 1
    domain = _guess_domain_from_labels([e["name"] for e in entities])

    # ---- local_artifacts：从 bbox 和实体名计算本地图像特征 ----
    local_artifacts = _compute_local_artifacts(entities, width, height)

    logger.info(
        f"    [{img_id}] 最终: {len(entities)} 个实体 "
        f"(VLM 识别 {len(vlm_entities)}, 幻觉移除 {hallucination_count}) | "
        f"local_artifacts: {len(local_artifacts.get('numeric_labels', []))} 数值, "
        f"{len(local_artifacts.get('layout_relations', []))} 空间关系"
    )

    # ---- Person proposal 第二遍（场景自适应） ----
    # 触发条件：(1) 没有 person 实体 (2) 图里可能有人但第一遍没框到
    # 场景不限于 sports——演唱会、颁奖礼、会议、工厂、任何有人但 VLM 主 prompt 没抓到的场景都适用
    # 做法：把整图切成 2×2 tiles，每个 tile 单独问"只标出人物"
    n_person = sum(1 for e in entities if e.get("type") == "person")

    # "图里可能有人"的启发式：
    #   条件 A：图片描述里含人物相关关键词（不限于 sports）
    #   条件 B：已有实体 ≥ 3（说明图不空，只是没检测到 person）
    # 满足 A + B + person=0 → 触发
    _PERSON_LIKELY_KEYWORDS = {
        # sports
        "nba", "nfl", "fifa", "mlb", "nhl", "arena", "court", "stadium", "scoreboard",
        "jersey", "basketball", "football", "soccer", "hockey", "baseball",
        "球场", "球员", "球队", "比赛", "记分牌", "得分",
        # entertainment / events
        "concert", "stage", "performer", "singer", "band", "演唱会", "舞台", "表演",
        "ceremony", "award", "颁奖", "典礼",
        # general scenes with people
        "crowd", "audience", "spectator", "人群", "观众",
        "conference", "meeting", "会议", "论坛",
        "factory", "worker", "工厂", "工人",
        "restaurant", "chef", "waiter", "餐厅",
        "classroom", "teacher", "student", "教室",
        # street / public space
        "pedestrian", "tourist", "行人", "游客",
        "protest", "parade", "游行",
        # 通用人物暗示
        "player", "coach", "referee", "athlete", "运动员", "教练", "裁判",
        "actor", "actress", "演员", "导演",
        "politician", "president", "minister", "总统", "部长",
    }
    desc_lower = image_desc.lower()
    has_person_context = any(kw in desc_lower for kw in _PERSON_LIKELY_KEYWORDS)

    if n_person == 0 and has_person_context and len(entities) >= 3:
        logger.info(f"    [{img_id}] 触发 person proposal 第二遍 (person=0, person_context=True)")

        _PERSON_PROPOSAL_PROMPT = """你是一个图片人物识别专家。这张图片是一个裁剪区域（整图的一部分）。
请**只标出人物**（运动员、表演者、路人、工作人员等）。不要标品牌、logo、文字、标志。

要求：
1. 只标能看清个体的人物（不要标远景模糊的人群）
2. name：如果你能认出是谁就写真名；否则用描述性名称（如 "white jersey #27", "man in blue suit", "woman at podium"）
3. bbox：该人物在这张裁剪图中的边界框 [x_min, y_min, x_max, y_max]，坐标范围 0-1000
4. 最多标 3 个最显眼的人物

输出严格 JSON（不加 markdown 代码块）：
{
    "persons": [
        {"name": "描述或真名", "bbox": [100, 200, 300, 600], "jersey_number": "27或null"}
    ]
}
没有人物就输出 {"persons": []}"""

        # 2×2 tile 切割
        tile_w, tile_h = width // 2, height // 2
        tiles = [
            (0, 0, tile_w, tile_h),           # 左上
            (tile_w, 0, width, tile_h),        # 右上
            (0, tile_h, tile_w, height),       # 左下
            (tile_w, tile_h, width, height),   # 右下
        ]

        person_proposals: list[dict] = []
        for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile_img = rgb_image.crop((tx1, ty1, tx2, ty2))
            import io
            buf = io.BytesIO()
            tile_img.save(buf, format="JPEG", quality=85)
            tile_b64 = base64.b64encode(buf.getvalue()).decode()
            tile_data_url = f"data:image/jpeg;base64,{tile_b64}"

            tile_result = call_vlm_json(
                "你是体育图片人物识别专家。只标出人物/球员，不要标 logo 或文字。",
                [
                    {"type": "image_url", "image_url": {"url": tile_data_url}},
                    {"type": "text", "text": _PERSON_PROPOSAL_PROMPT},
                ],
                max_tokens=500,
                temperature=0.3,
                max_attempts=1,
            )
            if not isinstance(tile_result, dict):
                continue
            for p in tile_result.get("persons", [])[:3]:
                pname = (p.get("name") or "").strip()
                pbbox = p.get("bbox")
                if not pname or not pbbox or not isinstance(pbbox, list) or len(pbbox) != 4:
                    continue
                # tile 坐标 → 全图坐标
                pw = tx2 - tx1
                ph = ty2 - ty1
                gx1 = tx1 + int(float(pbbox[0]) / 1000.0 * pw)
                gy1 = ty1 + int(float(pbbox[1]) / 1000.0 * ph)
                gx2 = tx1 + int(float(pbbox[2]) / 1000.0 * pw)
                gy2 = ty1 + int(float(pbbox[3]) / 1000.0 * ph)
                # 面积过滤
                area = (gx2 - gx1) * (gy2 - gy1)
                area_ratio = area / max(1, width * height)
                if area_ratio < 0.003 or area_ratio > 0.3:
                    continue
                person_proposals.append({
                    "name": pname[:60],
                    "bbox": [gx1, gy1, gx2, gy2],
                    "jersey_number": p.get("jersey_number"),
                    "tile_index": ti,
                    "source": "second_pass_proposal",
                })

        # 去重（同一 person 可能出现在相邻 tile 的重叠区）
        final_proposals: list[dict] = []
        for pp in person_proposals:
            # 简单 IoU 去重：和已有 proposal 重叠 > 50% 就跳过
            dup = False
            for fp in final_proposals:
                # IoU
                x1 = max(pp["bbox"][0], fp["bbox"][0])
                y1 = max(pp["bbox"][1], fp["bbox"][1])
                x2 = min(pp["bbox"][2], fp["bbox"][2])
                y2 = min(pp["bbox"][3], fp["bbox"][3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (pp["bbox"][2] - pp["bbox"][0]) * (pp["bbox"][3] - pp["bbox"][1])
                area2 = (fp["bbox"][2] - fp["bbox"][0]) * (fp["bbox"][3] - fp["bbox"][1])
                union = area1 + area2 - inter
                if union > 0 and inter / union > 0.5:
                    dup = True
                    break
            if not dup:
                final_proposals.append(pp)

        # 把 proposals 加入 entities（做 crop + search_views）
        n_added = 0
        for pp in final_proposals[:6]:  # 最多加 6 个 person（recall 保留，promotion gate 控预算）
            eid = f"E{idx}"
            ix1, iy1, ix2, iy2 = pp["bbox"]
            # tight crop
            search_views_pp = {}
            tight_crop = rgb_image.crop((ix1, iy1, ix2, iy2))
            tight_path = os.path.join(crop_dir, f"{eid}.jpg")
            tight_crop.save(tight_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
            search_views_pp["tight"] = tight_path
            # pad20
            pw20 = int((ix2 - ix1) * 0.2)
            ph20 = int((iy2 - iy1) * 0.2)
            pad20_crop = rgb_image.crop((
                max(0, ix1 - pw20), max(0, iy1 - ph20),
                min(width, ix2 + pw20), min(height, iy2 + ph20),
            ))
            pad20_path = os.path.join(crop_dir, f"{eid}_pad20.jpg")
            pad20_crop.save(pad20_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
            search_views_pp["pad20"] = pad20_path
            # pad40
            pw40 = int((ix2 - ix1) * 0.4)
            ph40 = int((iy2 - iy1) * 0.4)
            pad40_crop = rgb_image.crop((
                max(0, ix1 - pw40), max(0, iy1 - ph40),
                min(width, ix2 + pw40), min(height, iy2 + ph40),
            ))
            pad40_path = os.path.join(crop_dir, f"{eid}_pad40.jpg")
            pad40_crop.save(pad40_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
            search_views_pp["pad40"] = pad40_path
            # context
            cx, cy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            cw, ch = (ix2 - ix1) * 2, (iy2 - iy1) * 2
            ctx_crop = rgb_image.crop((
                max(0, cx - cw), max(0, cy - ch),
                min(width, cx + cw), min(height, cy + ch),
            ))
            ctx_path = os.path.join(crop_dir, f"{eid}_context.jpg")
            ctx_crop.save(ctx_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
            search_views_pp["context"] = ctx_path
            # full_scene（复用已有）
            full_scene_path = os.path.join(crop_dir, "full_scene.jpg")
            if not os.path.exists(full_scene_path):
                rgb_image.save(full_scene_path, format="JPEG", quality=ENTITY_CROP_QUALITY)
            search_views_pp["full_scene"] = full_scene_path

            entities.append({
                "id": eid,
                "name": pp["name"],
                "type": "person",
                "value": pp.get("jersey_number") or "",
                "bbox": [ix1, iy1, ix2, iy2],
                "location_in_image": _bbox_to_location(ix1, iy1, ix2, iy2, width, height),
                "confidence": 0.7,
                "confidence_level": "medium",
                "crop_path": tight_path,
                "search_views": search_views_pp,
                "source": "second_pass_proposal",
                "proposal_type": "human_gap_tile",
                "trigger_reason": "person_gap+scene_prior",
            })
            idx += 1
            n_added += 1
            logger.info(f"      person proposal: {eid} '{pp['name']}' bbox={pp['bbox']}")

        if n_added:
            logger.info(f"    [{img_id}] person proposal 第二遍: 添加 {n_added} 个 person 实体")
            # 重算 local_artifacts（新实体加入）
            local_artifacts = _compute_local_artifacts(entities, width, height)

    return {
        "entities": entities,
        "image_description": image_desc,
        "domain": domain,
        "local_artifacts": local_artifacts,
    }


def _compute_local_artifacts(entities: list, image_width: int, image_height: int) -> dict:
    """从 bbox + 实体名计算本地图像特征，给 Step3 code_heavy 题提供素材。

    不调 LLM/OCR，纯代码。
    """
    import re
    artifacts = {
        "numeric_labels": [],        # 从实体名提取的数值（球衣号码、价格、年份等）
        "bbox_areas": [],            # 每个实体的 bbox 面积和相对面积
        "layout_relations": [],      # 实体之间的空间关系
        "image_size": [image_width, image_height],
    }

    # 1. numeric_labels：从实体名里抽数字
    for e in entities:
        name = str(e.get("name", ""))
        # 匹配纯数字、年份、价格、球衣号码
        numbers = re.findall(r"\b(\d{1,4})\b", name)
        if numbers:
            artifacts["numeric_labels"].append({
                "entity_id": e.get("id", ""),
                "entity_name": name,
                "numbers": [int(n) for n in numbers],
            })

    # 2. bbox_areas：面积 + 占全图比例
    total_area = max(1, image_width * image_height)
    for e in entities:
        bbox = e.get("bbox", [])
        if len(bbox) == 4:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h
            artifacts["bbox_areas"].append({
                "entity_id": e.get("id", ""),
                "entity_name": e.get("name", ""),
                "bbox": bbox,
                "area": area,
                "relative_area": round(area / total_area, 4),
                "width": w,
                "height": h,
            })

    # 3. layout_relations：两两空间关系
    for i, ea in enumerate(entities):
        bba = ea.get("bbox", [])
        if len(bba) != 4:
            continue
        cx_a = (bba[0] + bba[2]) / 2
        cy_a = (bba[1] + bba[3]) / 2
        for eb in entities[i+1:]:
            bbb = eb.get("bbox", [])
            if len(bbb) != 4:
                continue
            cx_b = (bbb[0] + bbb[2]) / 2
            cy_b = (bbb[1] + bbb[3]) / 2
            dx = cx_b - cx_a
            dy = cy_b - cy_a
            distance = (dx * dx + dy * dy) ** 0.5
            # 判断方位
            if abs(dx) > abs(dy):
                direction = "right_of" if dx > 0 else "left_of"
            else:
                direction = "below" if dy > 0 else "above"
            artifacts["layout_relations"].append({
                "a_id": ea.get("id", ""),
                "b_id": eb.get("id", ""),
                "direction": direction,  # b 相对 a 的方位
                "distance": round(distance, 1),
            })

    # 4. extremum：最大/最小面积的实体
    if artifacts["bbox_areas"]:
        sorted_by_area = sorted(artifacts["bbox_areas"], key=lambda x: x["area"], reverse=True)
        artifacts["largest_entity"] = {"entity_id": sorted_by_area[0]["entity_id"],
                                       "entity_name": sorted_by_area[0]["entity_name"]}
        artifacts["smallest_entity"] = {"entity_id": sorted_by_area[-1]["entity_id"],
                                        "entity_name": sorted_by_area[-1]["entity_name"]}

    return artifacts


# ============================================================
# 搜索：SerpAPI (Google)
# ============================================================

def web_search(query: str, max_results: int = 5) -> dict:
    """调用 Serper 搜索 Google，返回结构化结果。带 disk cache。"""
    if not SERPER_KEY:
        return {"query": query, "error": "未配置 SERPER_KEY", "results": []}
    ck = _cache_key("serper_web", q=query, n=max_results)
    cached = _cache_get("serper_web", ck)
    if cached is not None:
        return cached
    try:
        resp = _http.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        data = resp.json()
        results = []

        kg = data.get("knowledgeGraph", {})
        if kg:
            kg_text_parts = []
            if kg.get("title"):
                kg_text_parts.append(kg["title"])
            if kg.get("description"):
                kg_text_parts.append(kg["description"])
            for attr in kg.get("attributes", {}).items():
                kg_text_parts.append(f"{attr[0]}: {attr[1]}")
            if kg_text_parts:
                results.append({"type": "knowledge_graph", "content": " | ".join(kg_text_parts)})

        answer_box = data.get("answerBox", {})
        if answer_box:
            snippet_text = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("snippetHighlighted", "")
            if isinstance(snippet_text, list):
                snippet_text = " ".join(snippet_text)
            if snippet_text:
                results.append({"type": "answer", "content": str(snippet_text)[:500]})

        for r in data.get("organic", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")[:500],
            })

        result = {"query": query, "results": results}
        _cache_set("serper_web", ck, result)
        return result
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


# ============================================================
# 网页深度读取：Jina Reader
# ============================================================

# Run-local Jina fast-fail: 一旦付费 key 402 一次，本 run 后续全切 free mode
# 避免每个 URL 先失败一次再 fallback（双倍延迟）
_jina_paid_failed = False  # module-level flag，本 run 内有效


def visit_url(url: str, max_chars: int = 3000, source_stage: str = "unknown") -> dict:
    """Jina Reader 读取网页。带 disk cache + run-local fast-fail。

    fast-fail: 付费 key 首次 402 后，本 run 后续全部直接走 free mode，不再逐 URL 试付费。
    """
    global _jina_paid_failed
    ck = _cache_key("jina", url=url, max_chars=max_chars)
    cached = _cache_get("jina", ck)
    if cached is not None:
        return cached
    import time as _t
    full_url = f"{JINA_READER_URL}{url}"

    # ---- 尝试 1: 付费 key（如果有且没 fast-fail）----
    if JINA_API_KEY and not _jina_paid_failed:
        t0 = _t.time()
        try:
            resp = _jina_http.get(
                full_url,
                headers={"Accept": "text/plain", "Authorization": f"Bearer {JINA_API_KEY}"},
            )
            elapsed_ms = int((_t.time() - t0) * 1000)
            if resp.status_code == 200:
                logger.debug(f"visit_url[{source_stage}] paid OK {resp.status_code} {elapsed_ms}ms {url[:60]}")
                r = {"url": url, "content": resp.text[:max_chars], "via": "paid"}
                _cache_set("jina", ck, r)
                return r
            # 402 余额耗尽 → 设 fast-fail flag + fall through 到免费模式
            if resp.status_code == 402:
                if not _jina_paid_failed:
                    _jina_paid_failed = True
                    logger.warning(
                        f"visit_url[{source_stage}] paid 402 → fast-fail activated, all subsequent visits use free mode"
                    )
                logger.debug(f"visit_url[{source_stage}] paid 402 {url[:60]}")
            else:
                logger.debug(
                    f"visit_url[{source_stage}] paid HTTP {resp.status_code} {elapsed_ms}ms {url[:60]} err={resp.text[:100]}"
                )
        except Exception as e:
            elapsed_ms = int((_t.time() - t0) * 1000)
            logger.debug(
                f"visit_url[{source_stage}] paid EXCEPTION {type(e).__name__} {elapsed_ms}ms {url[:60]}"
            )
            # fall through to free mode

    # ---- 尝试 2: 免费模式 fallback ----
    t0 = _t.time()
    try:
        resp = _jina_http.get(full_url, headers={"Accept": "text/plain"})
        elapsed_ms = int((_t.time() - t0) * 1000)
        if resp.status_code == 200:
            logger.debug(f"visit_url[{source_stage}] free OK {resp.status_code} {elapsed_ms}ms {url[:60]}")
            r = {"url": url, "content": resp.text[:max_chars], "via": "free"}
            _cache_set("jina", ck, r)
            return r
        err_body = (resp.text or "")[:150]
        logger.debug(
            f"visit_url[{source_stage}] free HTTP {resp.status_code} {elapsed_ms}ms {url[:60]} err={err_body}"
        )
        return {"url": url, "error": f"HTTP {resp.status_code}", "content": "", "via": "free_failed"}
    except Exception as e:
        elapsed_ms = int((_t.time() - t0) * 1000)
        logger.debug(
            f"visit_url[{source_stage}] free EXCEPTION {type(e).__name__} {elapsed_ms}ms {url[:60]}"
        )
        return {"url": url, "error": str(e), "content": "", "via": "free_exception"}


def _deep_read_top_urls(search_result: dict, max_urls: int = 2, max_chars: int = 3000,
                        source_stage: str = "web_search") -> list:
    # 只跳过明确读不动的社交媒体域名 + Jina free mode 确认 451 的法律限制域名
    _SKIP_DOMAINS = (
        "youtube.com", "facebook.com", "instagram.com", "twitter.com", "tiktok.com",
        # 经 canary 确认 Jina free mode 返回 451：
        "basketball-reference.com", "forbes.com",
    )
    candidates = []
    for r in search_result.get("results", []):
        if len(candidates) >= max_urls:
            break
        url = r.get("url", "")
        if not url or r.get("type") in ("knowledge_graph", "answer"):
            continue
        if any(skip in url for skip in _SKIP_DOMAINS):
            continue
        candidates.append((url, r.get("title", "")))

    if not candidates:
        return []

    def _fetch(item):
        url, title = item
        content = visit_url(url, max_chars=max_chars, source_stage=source_stage)
        if content.get("content"):
            return {"url": url, "title": title, "content": content["content"], "via": content.get("via", "")}
        return None

    read_results = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        for result in pool.map(_fetch, candidates):
            if result:
                read_results.append(result)
    return read_results


# ============================================================
# image_search：Serper 图片搜索
# ============================================================

def image_text_search(query: str, max_results: int = 5) -> dict:
    """文字搜图：用实体名搜图片结果，验证识别 + 找 alias。带 disk cache。"""
    if not SERPER_KEY:
        return {"query": query, "error": "未配置 SERPER_KEY", "results": []}
    ck = _cache_key("serper_img", q=query, n=max_results)
    cached = _cache_get("serper_img", ck)
    if cached is not None:
        return cached
    try:
        resp = _http.post(
            "https://google.serper.dev/images",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        data = resp.json()
        results = []
        for r in data.get("images", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "image_url": r.get("imageUrl", ""),
                "source_url": r.get("link", ""),
                "source_domain": r.get("source", ""),
            })
        result = {"query": query, "results": results}
        _cache_set("serper_img", ck, result)
        return result
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


# ============================================================
# 实体 resolution：用 image_search 验证识别 + 增加实体信息
# ============================================================

def _canonicalize_discovered_entities(discoveries: list[dict], existing_names: set) -> list[dict]:
    """从 reverse search 结果的标题碎片中提取 canonical entity name。

    discoveries 形如 [{name: "Jamal Murray 你可能不知道十件事...", source_url, source_entity_id}, ...]
    返回 [{name: "canonical name", type, source_url, source_entity_id, confidence_level}, ...]
    """
    if not discoveries:
        return []

    # 用 LLM 从标题碎片里提取 canonical entity name
    titles_text = "\n".join(f"{i+1}. {d.get('name', '')}" for i, d in enumerate(discoveries))
    prompt = (
        "下面是一些搜索结果的标题片段，每个片段来自对一张图像做反向搜索的结果。"
        "请从每个片段中提取一个最可能的 canonical entity name（人名/品牌/地标/作品名），"
        "如果无法提取则输出 null。\n\n"
        f"片段：\n{titles_text}\n\n"
        "严格输出 JSON：{\"canonical_names\": [\"name1\" 或 null, \"name2\" 或 null, ...]}\n"
        "要求：\n"
        "- 只输出实体名，不要加书名号、引号、多余描述\n"
        "- 如果是球员/人物，输出全名（如 Jamal Murray，不是 Murray）\n"
        "- 标题是文章标题或商品链接时，提取其中的主角名，不要把整段文本当实体"
    )
    result = call_vlm_json(
        prompt,
        "请提取 canonical entity name。",
        max_tokens=500,
        temperature=0.1,
        max_attempts=1,
    )
    if not isinstance(result, dict):
        return []
    names = result.get("canonical_names", [])
    if not isinstance(names, list):
        return []

    canonicalized = []
    seen = set()
    for i, name in enumerate(names):
        if i >= len(discoveries):
            break
        if not name or not isinstance(name, str):
            continue
        name = name.strip().strip("《》\"'")
        if len(name) < 2 or len(name) > 60:
            continue
        name_key = name.lower()
        if name_key in existing_names or name_key in seen:
            continue
        seen.add(name_key)
        disc = discoveries[i]
        canonicalized.append({
            "name": name,
            "source": "image_reverse_discovery",
            "source_entity_id": disc.get("source_entity_id", ""),
            "source_url": disc.get("source_url", ""),
            "confidence_level": "medium",
        })
    return canonicalized


def _resolve_entities(entities: list, img_id: str) -> tuple[list, list]:
    """对实体做 resolution：image_search(text) 验证名称 + 增加上下文信息。

    返回 (updated_entities, canonicalized_discoveries)：
    - updated_entities：带 resolution 字段的实体列表
    - canonicalized_discoveries：从 reverse search 里发现的新实体（已 canonicalize）
    """
    # 只对 person/landmark/product 做 image_text_search（品牌/文字 OCR 就够，不用图搜验证）
    # 这一改动把 image_text_search 调用量从 n_entities 降到 ~30%
    _IMG_SEARCH_TYPES = {"person", "landmark", "product"}
    img_tasks = []
    for e in entities:
        name = e.get("name", "").strip()
        if name and len(name) >= 2 and e.get("type", "") in _IMG_SEARCH_TYPES:
            img_tasks.append((e, name))

    # ---- Phase 1: image_search(text) 验证 person/landmark/product 实体 ----
    img_results = {}
    if img_tasks:
        def _do_img(task):
            entity, query = task
            return entity.get("id"), image_text_search(query, max_results=5)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for eid, result in pool.map(_do_img, img_tasks):
                if result.get("results"):
                    img_results[eid] = result

    # ---- Phase 2: 对 image_search_needed 实体做 reverse image search ----
    # 优先用真 reverse backend（Serper Google Lens），
    # 没 SERPER_KEY 时自动 fallback 到 VLM describe workaround
    from core.lens import reverse_search_entity
    from core.config import SERPER_KEY as _SERPER_KEY
    use_real_reverse = bool(_SERPER_KEY)

    reverse_tasks = []
    for e in entities:
        etype = e.get("type", "")
        if etype in ("person", "landmark", "product"):
            views = e.get("search_views", {})
            # 优先顺序：pad20 > tight > legacy crop_path
            # 经验观察：pad40 带进太多无关背景，Lens 命中反而下降（canary 对比）
            # pad40 仍在 search_views 里，供 Step3 未来使用（比如问题 prompt 的视觉参考图）
            # 不用 context / full_scene：它们周边太宽，返回场景级噪音页
            crop_path = (
                views.get("pad20")
                or views.get("tight")
                or e.get("crop_path", "")
            )
            if crop_path:
                reverse_tasks.append((e, crop_path))

    reverse_results: dict[str, dict] = {}
    discovered_entities: list[dict] = []  # 新发现的实体
    lens_source_pages: list[tuple] = []   # (eid, url, title) → 加入 visit queue

    if reverse_tasks and use_real_reverse:
        # === 真 reverse path: SerpApi Lens ===
        def _do_lens(task):
            entity, crop_path = task
            eid = entity.get("id", "")
            vlm_name = entity.get("name", "")
            try:
                lens_result = reverse_search_entity(crop_path, use_reverse_image=True)
            except Exception as exc:
                logger.debug(f"lens reverse failed for {eid}: {exc}")
                return eid, vlm_name, None
            return eid, vlm_name, lens_result

        with ThreadPoolExecutor(max_workers=3) as pool:  # SerpApi 限速保守
            for eid, vlm_name, lens_result in pool.map(_do_lens, reverse_tasks):
                if not lens_result or not lens_result.get("available"):
                    continue
                reverse_results[eid] = {
                    "lens_url": lens_result.get("lens_url", ""),
                    "candidate_titles": lens_result.get("candidate_titles", []),
                    "knowledge_graph": lens_result.get("knowledge_graph"),
                    "n_visual": lens_result.get("n_visual_matches", 0),
                    "n_exact": lens_result.get("n_exact_matches", 0),
                    "provenance": "lens_reverse",
                }
                # 候选实体：直接用 candidate_titles（后续 _canonicalize_discovered_entities 会用 LLM 清洗）
                for cand_title in lens_result.get("candidate_titles", [])[:5]:
                    if not cand_title or cand_title.lower() == vlm_name.lower():
                        continue
                    discovered_entities.append({
                        "name": cand_title.split(" - ")[0].split(" | ")[0].strip()[:60],
                        "source": "lens_reverse",
                        "source_entity_id": eid,
                        "source_url": "",  # lens 候选 title 本身没 url，用 source_pages 单独追
                    })
                # 来源页加入 visit queue
                for sp in lens_result.get("source_pages", [])[:5]:
                    url = sp.get("url", "")
                    if url:
                        lens_source_pages.append((eid, url, sp.get("title", "")))
                # knowledge_graph 里如果有 entity name 也加入候选
                kg = lens_result.get("knowledge_graph") or {}
                kg_title = (kg.get("title") or "").strip()
                if kg_title and kg_title.lower() != vlm_name.lower():
                    discovered_entities.append({
                        "name": kg_title[:60],
                        "source": "lens_kg",
                        "source_entity_id": eid,
                        "source_url": kg.get("source", {}).get("link", "") if isinstance(kg.get("source"), dict) else "",
                    })

    elif reverse_tasks:
        # === Fallback: VLM describe workaround（旧逻辑保留，无 SERPAPI_KEY 时用） ===
        def _do_reverse(task):
            entity, crop_path = task
            eid = entity.get("id", "")
            vlm_name = entity.get("name", "")
            import base64 as b64
            visual_desc = ""
            try:
                with open(crop_path, "rb") as f:
                    crop_b64 = b64.b64encode(f.read()).decode()
                raw = call_vlm(
                    "你是视觉识别助手。请用一句话描述这张图片中看到的内容，重点识别这是谁/什么。"
                    "只输出描述本身，不要说其他话。不要凭空猜测名字。",
                    "描述这张图片",
                    image_b64=crop_b64,
                    max_tokens=150,
                    temperature=0.2,
                    max_retries=1,
                )
                visual_desc = str(raw or "").strip()
            except Exception as exc:
                logger.debug(f"reverse describe failed for {eid}: {exc}")
            search_result = None
            if visual_desc and len(visual_desc) > 10:
                search_result = image_text_search(visual_desc[:100], max_results=3)
            return eid, vlm_name, visual_desc, search_result

        with ThreadPoolExecutor(max_workers=4) as pool:
            for eid, vlm_name, visual_desc, search_result in pool.map(_do_reverse, reverse_tasks):
                reverse_results[eid] = {
                    "visual_description": visual_desc,
                    "search_result": search_result,
                    "provenance": "vlm_describe_workaround",
                }
                if search_result and search_result.get("results"):
                    for r in search_result["results"][:2]:
                        title = r.get("title", "")
                        if title and len(title) > 3 and title.lower() != vlm_name.lower():
                            discovered_entities.append({
                                "name": title.split(" - ")[0].split(" | ")[0].strip()[:50],
                                "source": "image_reverse_discovery",
                                "source_entity_id": eid,
                                "source_url": r.get("source_url", ""),
                            })

    # ---- Phase 2.5: Type-repair for brand/product that might be person ----
    # sports 类图 VLM 把球员分成 brand（84 实体仅 1 person）。
    # 对 bbox 宽高比像人形的 brand/product 实体，试一次 Lens：
    # Lens 命中 → 升级 type=person + 纳入 Lens 结果路径。
    # 不命中 → 保持原 type（无副作用）。
    def _might_be_person(entity_inner: dict, w_img: int, h_img: int) -> bool:
        """bbox 几何：宽高比 0.3-0.9（人形竖直），面积 0.5%-30%。"""
        bbox = entity_inner.get("bbox", [0, 0, 1, 1])
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bh == 0:
            return False
        ratio = bw / bh
        area_ratio = (bw * bh) / max(1, w_img * h_img)
        return 0.3 <= ratio <= 0.9 and 0.005 <= area_ratio <= 0.30

    type_repair_count = 0
    if use_real_reverse:
        # 获取 image 尺寸（从 entity 的 search_views 里反推 width/height 不如直接从第一个 entity bbox）
        # 实际 width / height 在 extract_entities_vlm 里用 rgb_image.size 获得，
        # 但 _resolve_entities 没有这个参数。用 entity bbox 粗估整图尺寸即可。
        # 更好的做法：直接从实体 crop_path 推或传入。这里简化：从所有 bbox 的最大值反推。
        max_x = max((e.get("bbox", [0, 0, 0, 0])[2] for e in entities), default=1)
        max_y = max((e.get("bbox", [0, 0, 0, 0])[3] for e in entities), default=1)
        # 宽容估算：max_x / max_y 大约是图的宽高（bbox 是像素坐标）
        est_width = max(max_x, 100)
        est_height = max(max_y, 100)

        for e in entities:
            if e.get("type") not in ("brand", "product"):
                continue
            if not _might_be_person(e, est_width, est_height):
                continue
            eid = e.get("id", "")
            if eid in reverse_results:
                continue  # 已经跑过 Lens
            views = e.get("search_views", {})
            crop_path = views.get("pad20") or views.get("tight") or e.get("crop_path", "")
            if not crop_path:
                continue
            try:
                lens_result = reverse_search_entity(crop_path, use_reverse_image=True)
            except Exception as exc:
                logger.debug(f"type-repair lens failed for {eid}: {exc}")
                continue
            if not lens_result or not lens_result.get("available"):
                continue
            n_vis = lens_result.get("n_visual_matches", 0)
            if n_vis > 0:
                # Lens 命中 → 升级为 person
                old_type = e.get("type")
                e["type"] = "person"
                e["type_repair"] = f"lens_confirmed_from_{old_type}"
                type_repair_count += 1
                # 塞进 reverse_results 路径（和 Phase 2 真 lens 路径同结构）
                reverse_results[eid] = {
                    "lens_url": lens_result.get("lens_url", ""),
                    "candidate_titles": lens_result.get("candidate_titles", []),
                    "knowledge_graph": lens_result.get("knowledge_graph"),
                    "n_visual": n_vis,
                    "n_exact": lens_result.get("n_exact_matches", 0),
                    "provenance": "lens_reverse",
                }
                # 候选标题 → discovered_entities
                vlm_name = e.get("name", "")
                for cand_title in lens_result.get("candidate_titles", [])[:5]:
                    if not cand_title or cand_title.lower() == vlm_name.lower():
                        continue
                    discovered_entities.append({
                        "name": cand_title.split(" - ")[0].split(" | ")[0].strip()[:60],
                        "source": "lens_type_repair",
                        "source_entity_id": eid,
                        "source_url": "",
                    })
                # 来源页 → lens_source_pages
                for sp in lens_result.get("source_pages", [])[:3]:
                    url = sp.get("url", "")
                    if url:
                        lens_source_pages.append((eid, url, sp.get("title", "")))
                logger.info(f"    type-repair: {eid} '{e.get('name','')}' {old_type}→person (lens visual={n_vis})")

    if type_repair_count:
        logger.info(f"  [{img_id}] Phase 2.5 type-repair: {type_repair_count} entities upgraded to person")

    # ---- Phase 3: visit 来源页 ----
    visit_tasks = []
    for eid, result in img_results.items():
        for r in result.get("results", [])[:2]:
            source_url = r.get("source_url", "")
            if source_url:
                visit_tasks.append((eid, source_url, r.get("title", "")))
            if len(visit_tasks) >= 20:
                break
    # 也 visit reverse search 发现的来源页
    for disc in discovered_entities[:5]:
        url = disc.get("source_url", "")
        if url:
            visit_tasks.append((disc.get("source_entity_id", ""), url, disc.get("name", "")))
    # ★ 真 lens reverse 的 source pages 进 visit queue（最多 8 个，质量比 fallback 高）
    for eid, url, title in lens_source_pages[:8]:
        visit_tasks.append((eid, url, title))

    visit_results: dict[str, list] = {}
    if visit_tasks:
        # 记每个 task 的来源：lens_source 优先标，否则 image_search_source
        lens_url_set = {url for _, url, _ in lens_source_pages}

        def _do_visit(task):
            eid, url, title = task
            stage = "lens_source" if url in lens_url_set else "image_search_source"
            content = visit_url(url, max_chars=2000, source_stage=stage)
            return eid, {"url": url, "title": title, "content": content.get("content", ""), "via": content.get("via", "")}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for eid, result in pool.map(_do_visit, visit_tasks):
                if result.get("content"):
                    visit_results.setdefault(eid, []).append(result)

    # ---- Phase 4: 合并 resolution 信息 ----
    for e in entities:
        eid = e.get("id", "")
        resolution = {
            "image_search_titles": [],
            "image_search_sources": [],
            "reverse_visual_desc": "",
            "reverse_candidates": [],
            "visited_pages": [],
            # lens 真 reverse 字段（无 SERPAPI_KEY 时为空）
            "lens_provenance": "",
            "lens_url": "",
            "lens_candidate_titles": [],
            "lens_knowledge_graph": None,
            "lens_n_visual": 0,
            "lens_n_exact": 0,
        }

        if eid in img_results:
            for r in img_results[eid].get("results", [])[:5]:
                if r.get("title"):
                    resolution["image_search_titles"].append(r["title"])
                if r.get("source_url"):
                    resolution["image_search_sources"].append({
                        "url": r["source_url"],
                        "domain": r.get("source_domain", ""),
                    })

        if eid in reverse_results:
            rr = reverse_results[eid]
            prov = rr.get("provenance", "")
            resolution["lens_provenance"] = prov
            if prov == "lens_reverse":
                # 真 lens 路径
                resolution["lens_url"] = rr.get("lens_url", "")
                resolution["lens_candidate_titles"] = rr.get("candidate_titles", [])
                resolution["lens_knowledge_graph"] = rr.get("knowledge_graph")
                resolution["lens_n_visual"] = rr.get("n_visual", 0)
                resolution["lens_n_exact"] = rr.get("n_exact", 0)
                # 复用 reverse_candidates 字段供下游 LLM 用
                resolution["reverse_candidates"] = rr.get("candidate_titles", [])[:5]
            else:
                # VLM workaround 路径
                resolution["reverse_visual_desc"] = rr.get("visual_description", "")
            sr = rr.get("search_result") or {}
            if sr.get("results"):
                resolution["reverse_candidates"] = [
                    r.get("title", "") for r in sr["results"][:3]
                ]

        if eid in visit_results:
            resolution["visited_pages"] = [
                {"url": v["url"], "title": v["title"], "content_preview": v["content"][:200]}
                for v in visit_results[eid]
            ]

        e["resolution"] = resolution

        # resolve_mode：基于实际 resolution 结果，不只是 entity type 规则
        etype = e.get("type", "")
        vlm_name = e.get("name", "").lower()
        if etype in ("text", "brand"):
            e["resolve_mode"] = "ocr_likely"
        elif etype in ("person", "landmark", "product"):
            e["resolve_mode"] = "image_search_needed"
        else:
            e["resolve_mode"] = "ocr_likely"

        # 如果 image_search 结果和 VLM 名称差异大，提升为 image_search_needed
        if resolution.get("image_search_titles"):
            vlm_name = e.get("name", "").lower()
            titles_text = " ".join(resolution["image_search_titles"]).lower()
            if vlm_name and len(vlm_name) >= 3 and vlm_name not in titles_text:
                e["resolve_mode"] = "image_search_needed"

        # 真 lens reverse 命中（visual_matches >0 或 exact_matches >0）→ 强烈信号
        # 这是真证据，比规则打标可信
        if resolution.get("lens_provenance") == "lens_reverse":
            if resolution.get("lens_n_visual", 0) > 0 or resolution.get("lens_n_exact", 0) > 0:
                e["resolve_mode"] = "image_search_needed"
                # 也记一下"用了真 lens 验证过"
                e["resolve_mode_evidence"] = "lens_visual_matches"

    n_img = len(img_results)
    n_reverse = len(reverse_results)
    n_visit = sum(len(v) for v in visit_results.values())
    n_ocr = sum(1 for e in entities if e.get("resolve_mode") == "ocr_likely")
    n_isearch = sum(1 for e in entities if e.get("resolve_mode") == "image_search_needed")
    n_lens_real = sum(1 for rr in reverse_results.values() if rr.get("provenance") == "lens_reverse")
    n_lens_pages = len(lens_source_pages)

    # Canonicalize discovered_entities
    existing_names = {e.get("name", "").strip().lower() for e in entities}
    canonical_discoveries = _canonicalize_discovered_entities(discovered_entities, existing_names)

    reverse_mode = "real_lens" if use_real_reverse else "vlm_workaround"
    logger.info(f"  [{img_id}] 实体 resolution ({reverse_mode}): img_text={n_img}, reverse={n_reverse}, "
                f"lens_real={n_lens_real}, lens_pages={n_lens_pages}, "
                f"raw_discovered={len(discovered_entities)}, canonical_discovered={len(canonical_discoveries)}, "
                f"visit={n_visit}, resolve_mode: ocr={n_ocr} image_search={n_isearch}")
    for disc in canonical_discoveries[:5]:
        logger.info(f"    canonical 新实体: {disc.get('name', '?')} (来自 {disc.get('source_entity_id', '?')})")

    return entities, canonical_discoveries


# ============================================================
# 检查是否存在 L3 链（length >= 3）
# ============================================================

def _has_l3_chains(triples: list, entities: list) -> bool:
    from collections import defaultdict

    nodes = set()
    adjacency = defaultdict(list)
    for t in triples:
        h = t.get("head", "").strip().lower()
        tail = t.get("tail", "").strip().lower()
        if not h or not tail or h == tail:
            continue
        nodes.add(h)
        nodes.add(tail)
        adjacency[h].append(tail)

    for start in nodes:
        stack = [(start, {start}, 0)]
        while stack:
            curr, visited, depth = stack.pop()
            if depth >= 3:
                return True
            for nxt in adjacency.get(curr, []):
                if nxt not in visited:
                    stack.append((nxt, visited | {nxt}, depth + 1))
    return False


def _closure_richness_ok(triples: list, entities: list) -> tuple[bool, dict]:
    """Closure richness 检查：图里是否有足够多能闭合 hard motif 的原材料。

    满足任意 3 项即停。返回 (是否达标, 指标详情)。
    """
    from collections import defaultdict
    checks = {}

    # 1. image_resolved_anchors：有 resolve_mode=image_search_needed 的实体
    image_resolved = sum(1 for e in entities if e.get("resolve_mode") == "image_search_needed")
    checks["image_resolved_anchors"] = image_resolved
    checks["image_resolved_ok"] = image_resolved >= 3

    # 2. page_only_facts：必须 visit 才能拿到的事实
    # 支持两种细分：evidenced（有真证据）和 semantic（语义推的）
    # 兼容旧标签 "page_only" → 计入 total
    page_only_evidenced = sum(1 for t in triples if t.get("retrieval_mode") == "page_only_evidenced")
    page_only_semantic = sum(1 for t in triples if t.get("retrieval_mode") == "page_only_semantic")
    page_only_legacy = sum(1 for t in triples if t.get("retrieval_mode") == "page_only")
    page_only = page_only_evidenced + page_only_semantic + page_only_legacy
    checks["page_only_facts"] = page_only
    checks["page_only_evidenced"] = page_only_evidenced
    checks["page_only_semantic"] = page_only_semantic
    checks["page_only_ok"] = page_only >= 6

    # 3. compare_ready_pairs：按 tail_type + relation bucket 分组，≥2 个同组 entity
    type_groups: dict[tuple, set] = defaultdict(set)
    for t in triples:
        tt = t.get("tail_type", "OTHER")
        if tt in ("TIME", "QUANTITY"):
            rel = t.get("relation", "")
            # 简化：按 tail_type + 前缀分组
            bucket = f"{tt}:{rel[:20]}"
            type_groups[bucket].add(t.get("head", "").lower())
    compare_pairs = sum(1 for heads in type_groups.values() if len(heads) >= 2)
    checks["compare_ready_pairs"] = compare_pairs
    checks["compare_ready_ok"] = compare_pairs >= 4

    # 4. rank_ready_triplets：同组 ≥3 个 entity
    rank_triplets = sum(1 for heads in type_groups.values() if len(heads) >= 3)
    checks["rank_ready_triplets"] = rank_triplets
    checks["rank_ready_ok"] = rank_triplets >= 2

    # 5. cross_anchor_shared_nodes：同一 tail 被 ≥2 个不同 head 指向（bridge 节点）
    tail_heads: dict[str, set] = defaultdict(set)
    for t in triples:
        head = t.get("head", "").lower()
        tail = t.get("tail", "").lower()
        if head and tail:
            tail_heads[tail].add(head)
    shared_nodes = sum(1 for heads in tail_heads.values() if len(heads) >= 2)
    checks["cross_anchor_shared_nodes"] = shared_nodes
    checks["cross_shared_ok"] = shared_nodes >= 3

    # 6. fact_graph_size 兜底（避免前几项都不满足时无限扩）
    checks["total_triples"] = len(triples)
    checks["total_ok"] = len(triples) >= 40

    passed = sum([
        checks["image_resolved_ok"],
        checks["page_only_ok"],
        checks["compare_ready_ok"],
        checks["rank_ready_ok"],
        checks["cross_shared_ok"],
        checks["total_ok"],
    ])
    checks["passed_count"] = passed
    return passed >= 3, checks


def _graph_richness_ok(triples: list, entities: list) -> bool:
    """向后兼容的旧接口，内部调用新的 closure richness 检查。"""
    ok, _ = _closure_richness_ok(triples, entities)
    return ok


# ============================================================
# 扩展搜索：搜索尾实体以增加链深度
# ============================================================

def _extend_search(
    triples: list,
    entities: list,
    img_id: str,
    all_previous_search_results: list,
    round_idx: int,
    max_tail_entities: int = 6,
) -> tuple[list, list]:
    in_image_names = {e.get("name", "").lower().strip() for e in entities}
    already_searched = {sr.get("entity_name", "").lower().strip() for sr in all_previous_search_results}
    # round 1 过滤 in-image tail（避免重复已有图中实体的事实）
    # round ≥2 放开 in-image 过滤：目的是对 Denver Nuggets / NBA 这类桥接实体做二跳扩展
    block_in_image = round_idx < 2

    tail_counts: dict[str, int] = {}
    for t in triples:
        tail = t.get("tail", "").strip()
        tail_lower = tail.lower()
        if not tail or len(tail) < 3:
            continue
        if block_in_image and tail_lower in in_image_names:
            continue
        if tail_lower in already_searched:
            continue
        tail_counts[tail] = tail_counts.get(tail, 0) + 1

    if not tail_counts:
        logger.info(f"  [{img_id}] 第 {round_idx} 轮没有可扩展的尾实体，跳过")
        return triples, []

    sorted_tails = sorted(tail_counts.items(), key=lambda x: -x[1])
    selected_tails = [name for name, _ in sorted_tails[:max_tail_entities]]
    logger.info(f"  [{img_id}] 第 {round_idx} 轮候选尾实体: {len(selected_tails)} 个 → {selected_tails}")

    tail_context_lines = []
    for tail_name in selected_tails:
        sources = [f"({t['head']} → {t['relation']} → {t['tail']})"
                   for t in triples if t.get("tail", "").strip() == tail_name]
        tail_context_lines.append(f"- \"{tail_name}\"  来源: {'; '.join(sources[:3])}")

    logger.info(f"  [{img_id}] LLM 生成第 {round_idx} 轮搜索词...")
    search_plan_data = call_vlm_json(
        ROUND2_SEARCH_PLAN_PROMPT.format(tail_entities_with_context="\n".join(tail_context_lines)),
        "请为每个实体生成一条搜索查询词。",
        max_tokens=1024,
        temperature=0.3,
    )

    if search_plan_data is None:
        logger.warning(f"  [{img_id}] 第 {round_idx} 轮搜索计划生成失败，跳过")
        return triples, []

    ext_tasks = []
    for item in search_plan_data.get("queries", []):
        entity_name = item.get("entity", "")
        if item.get("skip", False):
            logger.info(f"    [R{round_idx+1}] 跳过 \"{entity_name}\"")
            continue
        seen_queries = set()
        for q_item in _exact_name_queries_for_tail(entity_name):
            query = q_item.get("query", "")
            if query and query.lower() not in seen_queries:
                ext_tasks.append((entity_name, query, q_item.get("origin", "text_exact")))
                seen_queries.add(query.lower())
        query = item.get("query", "")
        if query and query.lower() not in seen_queries:
            ext_tasks.append((entity_name, query, "text_rewrite"))

    round2_search_results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(web_search, q, 5): (name, q, origin) for name, q, origin in ext_tasks}
        for fut in as_completed(futures):
            entity_name, query, origin = futures[fut]
            sr = fut.result()
            sr["purpose"] = f"第{round_idx}轮扩展搜索: {entity_name}"
            sr["query_origin"] = origin
            round2_search_results.append({
                "entity_id": f"R{round_idx+1}_{entity_name[:20]}",
                "entity_name": entity_name,
                "searches": [sr],
                "round": round_idx + 1,
            })
            logger.info(f"    [R{round_idx+1}] \"{entity_name}\" → \"{query}\" [{origin}] → {len(sr.get('results', []))} 条结果")

    def _jina_ext(er):
        for s in er.get("searches", []):
            s["deep_reads"] = _deep_read_top_urls(s, max_urls=5, max_chars=2000, source_stage="round2_expand")
        return er

    with ThreadPoolExecutor(max_workers=6) as pool:
        round2_search_results = list(pool.map(_jina_ext, round2_search_results))

    round2_text = ""
    for sr in round2_search_results:
        round2_text += f"\n### {sr['entity_name']} (第 {round_idx} 轮扩展)\n"
        for s in sr.get("searches", []):
            round2_text += f"\n搜索词: \"{s['query']}\" (provenance: {s.get('query_origin', 'text_rewrite')})\n"
            for r in s.get("results", []):
                if r.get("type") == "knowledge_graph":
                    round2_text += f"  [知识面板] {r['content']}\n"
                elif r.get("type") == "answer":
                    round2_text += f"  [精选摘要] {r['content']}\n"
                else:
                    round2_text += f"  [{r.get('title', '')}] {r.get('snippet', '')[:300]}\n"
            for d in s.get("deep_reads", []):
                round2_text += f"\n  [深度读取: {d['title']}]\n  {d['content'][:1000]}\n"

    if not round2_text.strip():
        logger.info(f"  [{img_id}] 第 {round_idx} 轮搜索无结果")
        return triples, round2_search_results

    entities_summary = "\n".join(
        f"- {e.get('id', '?')}: {e['name']} (类型={e.get('type', '?')}, 位置={e.get('location_in_image', '?')})"
        for e in entities[:8]
    )

    existing_entities = set()
    for t in triples:
        existing_entities.add(t.get("head", ""))
        existing_entities.add(t.get("tail", ""))

    round2_prompt = TRIPLE_EXTRACTION_PROMPT.format(
        entities_summary=entities_summary,
        search_results=round2_text,
    )
    round2_prompt += (
        f"\n\n## 补充说明\n这是第 {round_idx} 轮扩展搜索。"
        f"之前已提取的实体包括：{', '.join(sorted(existing_entities))}\n"
        "请尽量复用这些实体名（保持一致），以便新三元组能与已有三元组构成更长的链。"
    )

    logger.info(f"  [{img_id}] 从第 {round_idx} 轮搜索结果提取三元组...")
    round2_triple_data = call_vlm_json(
        round2_prompt,
        "请严格基于搜索结果提取事实三元组。不要编造搜索结果中没有的信息。",
        max_tokens=4096,
        temperature=0.3,
    )

    if round2_triple_data is None:
        logger.warning(f"  [{img_id}] 第 {round_idx} 轮三元组提取失败")
        return triples, round2_search_results

    new_triples = _sanitize_triples(round2_triple_data.get("triples", []))
    logger.info(f"  [{img_id}] 第 {round_idx} 轮提取到 {len(new_triples)} 个新三元组")

    existing_keys = {
        (t.get("head", "").lower(), t.get("relation", "").lower(), t.get("tail", "").lower())
        for t in triples
    }

    added = 0
    for t in new_triples:
        key = (t.get("head", "").lower(), t.get("relation", "").lower(), t.get("tail", "").lower())
        if key not in existing_keys:
            triples.append(t)
            existing_keys.add(key)
            added += 1

    logger.info(f"  [{img_id}] 第 {round_idx} 轮合并后共 {len(triples)} 个三元组 (新增 {added})")
    return triples, round2_search_results


# ============================================================
# 空间关系兜底
# ============================================================

def _spatial_triples_from_bboxes(entities: list[dict]) -> list[dict]:
    """基于 bbox 像素坐标，为每对实体生成一条空间关系三元组。

    关系类型（主方向优先）：
      located_left_of / located_right_of / located_above / located_below
    """
    from itertools import combinations
    triples = []
    for a, b in combinations(entities, 2):
        a_name = a.get("name", "")
        b_name = b.get("name", "")
        a_bbox = a.get("bbox")
        b_bbox = b.get("bbox")
        if not a_name or not b_name or not a_bbox or not b_bbox:
            continue

        cx_a = (a_bbox[0] + a_bbox[2]) / 2
        cy_a = (a_bbox[1] + a_bbox[3]) / 2
        cx_b = (b_bbox[0] + b_bbox[2]) / 2
        cy_b = (b_bbox[1] + b_bbox[3]) / 2

        dx = cx_b - cx_a
        dy = cy_b - cy_a

        if abs(dx) >= abs(dy):
            if dx > 0:
                rel  = "located_left_of"
                fact = f"{a_name} 在图中位于 {b_name} 的左侧"
            else:
                rel  = "located_right_of"
                fact = f"{a_name} 在图中位于 {b_name} 的右侧"
        else:
            if dy > 0:
                rel  = "located_above"
                fact = f"{a_name} 在图中位于 {b_name} 的上方"
            else:
                rel  = "located_below"
                fact = f"{a_name} 在图中位于 {b_name} 的下方"

        triples.append({
            "head": a_name,
            "relation": rel,
            "tail": b_name,
            "tail_type": "OTHER",
            "fact": fact,
            "source_snippet": f"bbox {a_name}={a_bbox} vs {b_name}={b_bbox}",
            "source": "spatial",
        })
    return triples


def _add_spatial_fallback(triples: list[dict], entities: list[dict]) -> list[dict]:
    """为在三元组图中没有任何直接连接的实体对，补充空间关系三元组。

    "连接"定义：三元组中存在 head==A & tail==B 或 head==B & tail==A 的记录。
    只补充尚未连接的实体对，避免重复。
    返回新增的空间三元组列表（已合并到 triples 中）。
    """
    # 收集已有直接连接对（无向，用规范化小写判断）
    connected_pairs: set[tuple] = set()
    for t in triples:
        h = t.get("head", "").lower().strip()
        ta = t.get("tail", "").lower().strip()
        if h and ta:
            connected_pairs.add((min(h, ta), max(h, ta)))

    spatial_all = _spatial_triples_from_bboxes(entities)
    added = []
    seen_keys = {
        (t.get("head","").lower(), t.get("relation","").lower(), t.get("tail","").lower())
        for t in triples
    }
    for st in spatial_all:
        h  = st["head"].lower().strip()
        ta = st["tail"].lower().strip()
        pair = (min(h, ta), max(h, ta))
        if pair not in connected_pairs:
            key = (h, st["relation"].lower(), ta)
            if key not in seen_keys:
                triples.append(st)
                added.append(st)
                seen_keys.add(key)
                connected_pairs.add(pair)   # 标记已连接，同一对不重复补充

    return added


# ============================================================
# 跨实体关联发现
# ============================================================

def _find_cross_entity_relations(
    img_path: str,
    entities: list[dict],
    img_id: str,
    image_desc: str = "",
) -> tuple[list[dict], list[dict]]:
    """返回 (cross_triples, pair_search_results)。"""
    """发现图中不同实体之间的真实世界关联，返回带 source='cross_entity' 标记的三元组列表。

    对 high_conf 实体两两枚举（C(n,2) 对），由 LLM 为每对生成有针对性的搜索计划，再并行搜索。
    """
    if len(entities) < 2:
        return []

    try:
        from itertools import combinations

        # ── 枚举实体对（上限 10 对，从 20 降到 10 以控 fan-out）──
        MAX_CROSS_PAIRS = 10
        _SEARCHABLE_TYPES = {"brand", "product", "landmark", "person"}
        # 优先选 searchable 类型的实体，按 confidence 降序
        ranked = sorted(entities, key=lambda e: (
            1 if e.get("type", "") in _SEARCHABLE_TYPES else 0,
            float(e.get("confidence", 0)),
        ), reverse=True)
        all_pairs = []
        for a, b in combinations(ranked, 2):
            a_name = a.get("name", "")
            b_name = b.get("name", "")
            if a_name and b_name:
                all_pairs.append({"entity_a": a_name, "entity_b": b_name})
            if len(all_pairs) >= MAX_CROSS_PAIRS:
                break

        # ── LLM 为每对生成搜索计划 ──
        entities_summary = "\n".join(
            f"- {e.get('id','?')}: {e.get('name','?')} "
            f"(类型={e.get('type','?')}, 位置={e.get('location_in_image','?')})"
            for e in entities
        )
        pairs_list = "\n".join(
            f"{i+1}. {p['entity_a']} ↔ {p['entity_b']}"
            for i, p in enumerate(all_pairs)
        )
        logger.info(f"  [{img_id}] LLM 为 {len(all_pairs)} 对实体生成搜索计划...")
        query_data = call_vlm_json(
            CROSS_ENTITY_PROMPT.format(
                image_description=image_desc,
                entities_summary=entities_summary,
                pairs_list=pairs_list,
            ),
            "请为每对实体生成有针对性的搜索查询计划。",
            max_tokens=2048,
            temperature=0.3,
        )

        # LLM 失败时回退到简单拼接
        if query_data is None:
            logger.warning(f"  [{img_id}] LLM 生成搜索计划失败，回退到实体名直接拼接")
            search_tasks = [
                {"entity_a": p["entity_a"], "entity_b": p["entity_b"],
                 "query": f"{p['entity_a']} {p['entity_b']}", "purpose": "回退拼接搜索"}
                for p in all_pairs
            ]
        else:
            search_tasks = []
            llm_plans = {}
            for plan in query_data.get("search_plans", []):
                key = (plan.get("entity_a", "").lower(), plan.get("entity_b", "").lower())
                llm_plans[key] = plan
                key_rev = (plan.get("entity_b", "").lower(), plan.get("entity_a", "").lower())
                llm_plans[key_rev] = plan

            skipped_pairs = 0
            for p in all_pairs:
                key = (p["entity_a"].lower(), p["entity_b"].lower())
                plan = llm_plans.get(key)
                if plan and plan.get("skip", False):
                    logger.info(f"    跳过 {p['entity_a']} ↔ {p['entity_b']}: "
                                f"{plan.get('skip_reason', '')}")
                    skipped_pairs += 1
                    continue
                if plan and plan.get("queries"):
                    for q_item in plan["queries"]:
                        query = q_item.get("query", "")
                        if query:
                            search_tasks.append({
                                "entity_a": p["entity_a"],
                                "entity_b": p["entity_b"],
                                "query": query,
                                "purpose": q_item.get("purpose", ""),
                            })
                else:
                    search_tasks.append({
                        "entity_a": p["entity_a"],
                        "entity_b": p["entity_b"],
                        "query": f"{p['entity_a']} {p['entity_b']}",
                        "purpose": "回退拼接搜索",
                    })

            logger.info(f"  [{img_id}] LLM 搜索计划完成: {len(search_tasks)} 条查询, "
                        f"{skipped_pairs} 对跳过")

        logger.info(f"  [{img_id}] 跨实体搜索: {len(search_tasks)} 条查询，并行执行...")

        # 并行搜索所有任务
        pair_search_results = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(web_search, t["query"], 5): t for t in search_tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    sr = fut.result()
                except Exception as exc:
                    logger.warning(f"  [{img_id}] 跨实体搜索失败 "
                                   f"({task['entity_a']}-{task['entity_b']}): {exc}")
                    continue
                # web-first, visit-later: 只有 snippet 同时提到两个实体才 visit
                a_lower = task["entity_a"].lower()
                b_lower = task["entity_b"].lower()
                snippets_text = " ".join(
                    r.get("snippet", "") + " " + r.get("title", "")
                    for r in sr.get("results", []) if isinstance(r, dict)
                ).lower()
                both_mentioned = a_lower in snippets_text and b_lower in snippets_text
                if both_mentioned:
                    deep_reads = _deep_read_top_urls(sr, max_urls=2, max_chars=2000, source_stage="cross_entity")
                else:
                    deep_reads = []  # web snippet 没同时提到 → 跳过 visit，省 Jina 调用
                pair_search_results.append({
                    "entity_a": task["entity_a"],
                    "entity_b": task["entity_b"],
                    "query": task["query"],
                    "purpose": task.get("purpose", ""),
                    "search_result": sr,
                    "deep_reads": deep_reads,
                })
                n_results = len(sr.get("results", []))
                logger.info(f"  [{img_id}] '{task['entity_a']}' ↔ '{task['entity_b']}'"
                            f" → \"{task['query'][:40]}\" → {n_results} 条结果, "
                            f"{len(deep_reads)} 篇深读")

        if not pair_search_results:
            return []

        # 拼装搜索结果文本
        cross_search_text = ""
        for psr in pair_search_results:
            purpose_str = f" (目的: {psr['purpose']})" if psr.get("purpose") else ""
            cross_search_text += (
                f"\n### {psr['entity_a']} ↔ {psr['entity_b']}\n"
                f"搜索词: \"{psr['query']}\"{purpose_str}\n"
            )
            for r in psr["search_result"].get("results", []):
                if r.get("type") == "knowledge_graph":
                    cross_search_text += f"  [知识面板] {r['content']}\n"
                elif r.get("type") == "answer":
                    cross_search_text += f"  [精选摘要] {r['content']}\n"
                else:
                    cross_search_text += f"  [{r.get('title', '')}] {r.get('snippet', '')[:300]}\n"
            for d in psr.get("deep_reads", []):
                cross_search_text += f"\n  [深度读取: {d['title']}]\n  {d['content'][:1500]}\n"

        entities_summary = "\n".join(
            f"- {e.get('id', '?')}: {e.get('name', '?')} "
            f"(类型={e.get('type', '?')}, 位置={e.get('location_in_image', '?')})"
            for e in entities
        )

        logger.info(f"  [{img_id}] 从跨实体搜索结果提取桥接三元组...")
        triple_data = call_vlm_json(
            CROSS_ENTITY_TRIPLE_PROMPT.format(
                entities_summary=entities_summary,
                cross_search_results=cross_search_text,
            ),
            "请严格基于搜索结果提取跨实体桥接三元组。不要编造搜索结果中没有的信息。",
            max_tokens=4096,
            temperature=0.3,
        )

        if triple_data is None:
            logger.warning(f"  [{img_id}] 跨实体三元组提取失败")
            return []

        raw_triples = _sanitize_triples(triple_data.get("triples", []))

        # 去重：(head, relation, tail) 小写精确去重
        seen: set[tuple] = set()
        cross_triples = []
        for t in raw_triples:
            key = (
                t.get("head", "").strip().lower(),
                t.get("relation", "").strip().lower(),
                t.get("tail", "").strip().lower(),
            )
            if key[0] and key[2] and key[0] != key[2] and key not in seen:
                seen.add(key)
                t["source"] = "cross_entity"
                t["provenance"] = "cross_entity"
                cross_triples.append(t)

        logger.info(
            f"  [{img_id}] 跨实体桥接三元组提取完成: "
            f"原始 {len(raw_triples)} 个，去重后 {len(cross_triples)} 个"
        )
        return cross_triples, pair_search_results

    except Exception as e:
        logger.error(f"  [{img_id}] 跨实体关联发现异常: {e}")
        return [], []


# ============================================================
# 单张图片处理
# ============================================================

def enrich_image(img_path: str) -> dict | None:
    img_id = os.path.splitext(os.path.basename(img_path))[0]

    if is_done(2, img_id):
        logger.info(f"  [{img_id}] 已有检查点，跳过")
        return load_checkpoint(2, img_id)

    logger.info(f"  [{img_id}] 实体提取（纯 VLM）...")
    try:
        entity_data = extract_entities_vlm(img_path, img_id=img_id)
    except Exception as e:
        logger.error(f"  [{img_id}] 实体提取失败: {e}")
        return None

    entities = entity_data.get("entities", [])
    logger.info(f"  [{img_id}] 提取到 {len(entities)} 个实体")

    if len(entities) < 3:
        logger.warning(f"  [{img_id}] 实体数太少 ({len(entities)})，跳过")
        return None

    # ---- 2b-0. 实体池分层：core_anchors（Step3 出题用）+ expansion_seeds（Step2 扩图用）----
    _GENERIC_NAMES = {"car", "person", "building", "tree", "sign", "door", "window", "wall", "floor", "sky"}
    seen_entity_names = set()

    # core_anchors: confidence >= 0.8，给 Step3 出题优先级用
    core_anchors = []
    for e in sorted(entities, key=lambda x: float(x.get("confidence", 0.0)) if isinstance(x.get("confidence"), (int, float)) else 0.0, reverse=True):
        name_key = str(e.get("name", "")).strip().lower()
        if not name_key or name_key in seen_entity_names:
            continue
        conf = e.get("confidence")
        if (isinstance(conf, (int, float)) and conf >= 0.8) or str(e.get("confidence_level", "")).lower() == "high":
            core_anchors.append(e)
            seen_entity_names.add(name_key)

    # expansion_seeds: 所有 searchable 实体，不限数量（排除太泛的 object）
    expansion_seeds = []
    seen_exp = set()
    for e in entities:
        name_key = str(e.get("name", "")).strip().lower()
        if not name_key or name_key in seen_exp:
            continue
        if e.get("type") == "object" and name_key in _GENERIC_NAMES:
            continue
        expansion_seeds.append(e)
        seen_exp.add(name_key)

    if not core_anchors:
        core_anchors = entities[:5]
    if not expansion_seeds:
        expansion_seeds = entities[:5]

    # high_conf 保持向后兼容（Step3 的 HeteroSolveGraph 读这个字段）
    high_conf = core_anchors

    image_desc = entity_data.get("image_description", "")
    logger.info(f"  [{img_id}] 实体池: core_anchors={len(core_anchors)}, expansion_seeds={len(expansion_seeds)} (resolution 前)")

    # ---- 2a-2. 实体 resolution：image_search 验证 + visit 增加信息 ----
    logger.info(f"  [{img_id}] 实体 resolution（image_search + visit）...")
    entities, canonical_discoveries = _resolve_entities(entities, img_id)

    # ---- 2a-3. 把 reverse search 发现的新实体并入 expansion_seeds ----
    if canonical_discoveries:
        for disc in canonical_discoveries:
            name_key = disc.get("name", "").strip().lower()
            if not name_key or name_key in seen_exp:
                continue
            stub = {
                "id": f"DE{len([e for e in expansion_seeds if e.get('source') == 'image_reverse_discovery'])+1}",
                "name": disc["name"],
                "type": "other",
                "location_in_image": "",
                "confidence": 0.6,
                "confidence_level": "medium",
                "source": "image_reverse_discovery",
                "source_entity_id": disc.get("source_entity_id", ""),
                "source_url": disc.get("source_url", ""),
            }
            expansion_seeds.append(stub)
            seen_exp.add(name_key)
        logger.info(f"  [{img_id}] 并入 reverse 发现实体后: expansion_seeds={len(expansion_seeds)}")

    # ---- 2a-4. Promotion gate: 只有高价值实体才进入昂贵的跨实体搜索 ----
    # discovered entities 和 second_pass_proposal 不自动拿到与主实体同级的搜索预算。
    # promotion 条件：(1) 是原始 VLM 检测的 in-image 实体，或
    #                 (2) 有 Lens 证据（lens_visual_matches），或
    #                 (3) 是 core_anchor
    # 不满足的保留在 expansion_seeds（Step3 仍能用），但不进跨实体搜索。
    # promotion 分两层：
    #   Tier 1 (full budget): 原始 VLM 检测 + high confidence → 跨实体 + 主搜索 + 扩展
    #   Tier 2 (limited budget): Lens 证实的 proposal / discovered → 主搜索 + Lens，不参与 pairwise
    # 未 promote 的：保留在图中供 Step3 用，但 0 搜索预算
    MAX_PROPOSAL_PROMOTED = 3  # proposal 实体最多升格 3 个（recall 6 个全保留在图里）

    tier1 = []  # full budget
    tier2 = []  # limited budget
    for e in expansion_seeds:
        src = e.get("source", "")
        if src in ("vlm_only", "") or e.get("confidence_level") == "high":
            tier1.append(e)
        elif e.get("resolve_mode_evidence") == "lens_visual_matches":
            # proposal with Lens evidence → tier2 (有限预算)
            tier2.append(e)
        # else: demoted, 不搜索

    # proposal 里超过 MAX_PROPOSAL_PROMOTED 的部分降回 tier2 → demoted
    proposal_in_tier2 = [e for e in tier2 if "proposal" in (e.get("source") or "")]
    other_tier2 = [e for e in tier2 if "proposal" not in (e.get("source") or "")]
    if len(proposal_in_tier2) > MAX_PROPOSAL_PROMOTED:
        tier2 = other_tier2 + proposal_in_tier2[:MAX_PROPOSAL_PROMOTED]

    promoted_entities = tier1 + tier2  # 跨实体只用 tier1，主搜索用 tier1 + tier2
    tier1_set = set(id(e) for e in tier1)

    # ---- Brand-dense large-image mode ----
    # 触发条件：tier1 > 12，或 brand/text 占比 > 0.7 且实体 >= 12
    # 此模式下只让 top-12 tier1 进 full-budget，其余降级
    _brand_text_types = {"brand", "text"}
    n_bt = sum(1 for e in tier1 if e.get("type", "") in _brand_text_types)
    brand_dense = (len(tier1) > 12) or (len(tier1) >= 12 and n_bt / max(1, len(tier1)) > 0.7)
    if brand_dense:
        # 按综合分选 top-12：bbox 面积 × 0.3 + 空间分散度代理（location 字符串 hash 多样性）
        def _tier1_score(e):
            bbox = e.get("bbox", [0, 0, 1, 1])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) == 4 else 0
            # person/landmark/product 加分（更可能产出 hard closure）
            type_bonus = 1.5 if e.get("type", "") in ("person", "landmark", "product") else 1.0
            conf = float(e.get("confidence", 0.9))
            return area * type_bonus * conf

        tier1_sorted = sorted(tier1, key=_tier1_score, reverse=True)
        tier1_kept = tier1_sorted[:12]
        tier1_demoted = tier1_sorted[12:]
        logger.info(f"  [{img_id}] Brand-dense mode: tier1 {len(tier1)} → top-12 kept, "
                    f"{len(tier1_demoted)} demoted (brand/text={n_bt}/{len(tier1)})")
        tier1 = tier1_kept
        promoted_entities = tier1 + tier2
        tier1_set = set(id(e) for e in tier1)

    n_demoted = len(expansion_seeds) - len(promoted_entities)
    logger.info(f"  [{img_id}] Promotion gate: {len(expansion_seeds)} seeds → "
                f"tier1={len(tier1)} (full) + tier2={len(tier2)} (limited) + "
                f"{n_demoted} demoted"
                + (f" [brand-dense mode]" if brand_dense else ""))

    # ---- 2b-0. 跨实体关联发现（只用 tier1 full-budget 实体）----
    logger.info(f"  [{img_id}] 跨实体关联发现（{len(tier1)} 个 tier1 实体）...")
    cross_triples, cross_pair_search_results = _find_cross_entity_relations(img_path, tier1, img_id, image_desc)
    logger.info(f"  [{img_id}] 跨实体桥接三元组: {len(cross_triples)} 个")

    # ---- 2b-1. 确定性搜索计划（只对 promoted_entities）----
    # brand-dense 模式下：brand/text 只跑 exact-name query（不跑 fallback），
    # 只有前 6 个高分实体跑 fallback query
    searchable = []
    _top_tier1_ids = set(id(e) for e in tier1[:6]) if brand_dense else set()
    for e in promoted_entities:
        name = str(e.get("name", "")).strip()
        etype = e.get("type", "")
        if not name:
            continue
        if etype == "object" and len(name) <= 3:
            continue
        queries = [{"query": name, "purpose": "精确名搜索", "origin": "text_exact"}]
        # fallback query: brand-dense 模式只对 top-6 或 person/landmark/product 开放
        allow_fallback = (not brand_dense) or (id(e) in _top_tier1_ids) or (etype in ("person", "landmark", "product"))
        if allow_fallback:
            alias_map = {"brand": "brand company", "product": "product", "landmark": "landmark building", "person": "person biography"}
            alias = alias_map.get(etype, "")
            if alias:
                queries.append({"query": f"{name} {alias}", "purpose": "类型限定搜索", "origin": "text_exact"})
        searchable.append({"entity_id": e.get("id", ""), "entity_name": name, "skip": False, "queries": queries})
    skipped = [e for e in promoted_entities if str(e.get("name", "")).strip() == ""]
    logger.info(f"  [{img_id}] 确定性搜索计划: {len(searchable)} 个实体需搜索 (promoted), "
                f"{len(expansion_seeds) - len(promoted_entities)} 个 demoted 不搜")

    search_plan_data = {"search_plans": searchable}
    plans = searchable

    if not plans:
        logger.warning(f"  [{img_id}] 无可搜索实体，跳过知识扩展")
        result = {
            "img_id": img_id, "img_path": img_path,
            "image_description": image_desc,
            "domain": entity_data.get("domain", "other"),
            "entities": entities,
            "local_artifacts": entity_data.get("local_artifacts", {}),
            "search_results": [], "triples": [],
        }
        save_checkpoint(2, img_id, result)
        return result

    # ---- 2b-2. 并行执行搜索 + Jina深读 ----
    logger.info(f"  [{img_id}] 并行执行搜索...")

    search_tasks = []
    entity_lookup = {e.get("id", ""): e for e in expansion_seeds}
    for plan in searchable:
        eid = plan.get("entity_id", "")
        ename = plan.get("entity_name", "")
        seen_queries = set()
        for q_item in _exact_name_queries_for_entity(entity_lookup.get(eid, {"name": ename})):
            query = q_item.get("query", "")
            if query and query.lower() not in seen_queries:
                search_tasks.append((eid, ename, query, q_item.get("purpose", ""), q_item.get("origin", "text_exact")))
                seen_queries.add(query.lower())
        for q_item in plan.get("queries", []):
            query = q_item.get("query", "")
            purpose = q_item.get("purpose", "")
            if query and query.lower() not in seen_queries:
                search_tasks.append((eid, ename, query, purpose, "text_rewrite"))
                seen_queries.add(query.lower())

    search_results_map = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for eid, ename, query, purpose, origin in search_tasks:
            fut = pool.submit(web_search, query, 5)
            futures[fut] = (eid, ename, query, purpose, origin)
        for fut in as_completed(futures):
            eid, ename, query, purpose, origin = futures[fut]
            sr = fut.result()
            sr["purpose"] = purpose
            sr["query_origin"] = origin
            search_results_map[(eid, query)] = sr
            logger.info(f"    [{eid}] \"{query}\" [{origin}] → {len(sr.get('results', []))} 条结果")

    all_search_results = []
    for plan in searchable:
        eid = plan.get("entity_id", "")
        ename = plan.get("entity_name", "")
        entity_results = {"entity_id": eid, "entity_name": ename, "searches": []}
        for q_item in plan.get("queries", []):
            query = q_item.get("query", "")
            if query and (eid, query) in search_results_map:
                entity_results["searches"].append(search_results_map[(eid, query)])
        all_search_results.append(entity_results)

    logger.info(f"  [{img_id}] 并行 Jina Reader 深度读取...")

    jina_tasks = []
    for i, entity_result in enumerate(all_search_results):
        for j, s in enumerate(entity_result.get("searches", [])):
            jina_tasks.append((i, j, s))

    # brand-dense 模式下降低 visit 上限：brand/text → top-1，其他 → top-2
    # 普通模式保持 top-5（兼容旧行为）
    _main_visit_max = 2 if brand_dense else 5

    def _jina_read_task(task):
        idx_i, idx_j, s = task
        return idx_i, idx_j, _deep_read_top_urls(s, max_urls=_main_visit_max, max_chars=3000, source_stage="main_search")

    with ThreadPoolExecutor(max_workers=6) as pool:
        for idx_i, idx_j, deep in pool.map(_jina_read_task, jina_tasks):
            all_search_results[idx_i]["searches"][idx_j]["deep_reads"] = deep

    # ---- 2b-3. 从搜索结果中提取事实三元组 ----
    entities_summary_lines = []
    for e in expansion_seeds:
        line = f"- {e['id']}: {e['name']} (类型={e.get('type', '?')}, 位置={e.get('location_in_image', '?')})"
        res = e.get("resolution", {})
        if res.get("image_search_titles"):
            line += f" [图搜验证: {', '.join(res['image_search_titles'][:2])}]"
        if res.get("visited_pages"):
            page_info = "; ".join(p["content_preview"][:100] for p in res["visited_pages"][:1])
            line += f" [来源页: {page_info}]"
        entities_summary_lines.append(line)
    entities_summary = "\n".join(entities_summary_lines)

    search_results_text = ""
    for sr in all_search_results:
        search_results_text += f"\n### {sr['entity_name']} ({sr['entity_id']})\n"
        for s in sr.get("searches", []):
            search_results_text += f"\n搜索词: \"{s['query']}\" (目的: {s.get('purpose', '')}, provenance: {s.get('query_origin', 'text_rewrite')})\n"
            for r in s.get("results", []):
                if r.get("type") == "knowledge_graph":
                    search_results_text += f"  [知识面板] {r['content']}\n"
                elif r.get("type") == "answer":
                    search_results_text += f"  [精选摘要] {r['content']}\n"
                else:
                    search_results_text += f"  [{r.get('title', '')}] {r.get('snippet', '')[:300]}\n"
            for d in s.get("deep_reads", []):
                search_results_text += f"\n  [深度读取: {d['title']}]\n  {d['content'][:1500]}\n"

    logger.info(f"  [{img_id}] 从搜索结果提取三元组...")
    triple_data = call_vlm_json(
        TRIPLE_EXTRACTION_PROMPT.format(entities_summary=entities_summary, search_results=search_results_text),
        "请严格基于搜索结果提取事实三元组。不要编造搜索结果中没有的信息。",
        max_tokens=4096,
        temperature=0.3,
    )

    if triple_data is None:
        logger.warning(f"  [{img_id}] 三元组提取失败，仅保留实体和搜索结果")
        triple_data = {"triples": []}

    triples = _sanitize_triples(triple_data.get("triples", []))

    # ---- retrieval_mode 标记：snippet_only vs page_only ----
    # 收集 snippet 和 deep_read 语料（主搜索 + 跨实体都收集）
    deep_read_corpus = ""
    snippet_corpus = ""
    for sr in all_search_results:
        for s in sr.get("searches", []):
            for r in s.get("results", []):
                snippet_corpus += (r.get("snippet", "") + " " + r.get("content", "")) + " "
            for d in s.get("deep_reads", []):
                deep_read_corpus += d.get("content", "") + " "
    # 包含跨实体搜索的 snippet 和 deep_reads
    for psr in cross_pair_search_results:
        sr2 = psr.get("search_result", {})
        for r in sr2.get("results", []):
            snippet_corpus += (r.get("snippet", "") + " " + r.get("content", "")) + " "
        for d in psr.get("deep_reads", []):
            deep_read_corpus += d.get("content", "") + " "

    def _mark_retrieval_mode(triples_list):
        """对 triples 打 retrieval_mode 标记。

        现在产出 4 种状态：
        - spatial              ：空间关系（bbox 兜底），不参与 page/snippet 分类
        - page_only_evidenced  ：有真证据在 deep_read_corpus 里，且 snippet 不足（Signal 1/2）
        - page_only_semantic   ：relation override 或 tail_type fallback 推的"原则上需要 visit"，本次可能没实证（Signal 0/3）
        - snippet_only         ：snippet 已足够回答

        下游语义：
        - `visit_heavy` bucket 只吃 `page_only_evidenced`（严格不可约）
        - `ultra_long` / Step3 walker 可以混用 semantic + evidenced
        - `total_page_only` = evidenced + semantic（向后兼容旧统计）
        """
        # 空间关系（来自 bbox 兜底，不是从搜索拿的，不参与 retrieval_mode）
        spatial_relations = {
            "located_above", "located_below", "located_left_of", "located_right_of",
            "located_near", "located_at_top", "located_at_bottom", "adjacent_to",
        }

        # 深读倾向的 relation 关键词（在 snippet 里只有概述，详细数据在页面里）
        # 扩展版：包含 company ownership / biography / location / capacity / 业务关系 类
        page_leaning_relations = {
            # 财务数值类
            "revenue", "employees", "assets", "market_cap", "population",
            "contract_value", "salary", "net_worth", "capacity", "budget",
            "brand_value", "box_office", "attendance",
            # 组织/所有权类
            "founder", "founded_by", "ceo", "president", "chairman",
            "head_coach", "general_manager", "owner", "owned_by", "owned by",
            "parent_company", "parent of", "parent_of",
            "subsidiary_of", "subsidiaries_of", "subsidiary of",
            "acquired_by", "acquired by", "acquisition_by", "sold_brand_to",
            "manufacturer", "manufactured_by", "made_by",
            "formerly_division_of", "spun_off_from", "merged_with",
            "competitor_of", "competitor of",
            "market_leader_in", "market leader in",
            "specializes_in", "specializes in",
            # 位置/总部类
            "headquarters", "headquartered_in", "headquartered in",
            "headquarter_address", "based_in", "based in",
            "arena_name", "arena_location", "home_city", "located_at", "located at",
            # 时间/历史类
            "founded_in", "founded", "founded in", "established",
            "established_in", "established in", "introduced_in", "introduced in",
            "launched_in", "launched in", "founding_history", "founding_year",
            # 人物履历类
            "biography", "career_history", "draft_position", "draft_year",
            "born_in", "born in", "birthplace", "hometown_of", "alma_mater",
            "exact_population",
            # 产品线/品类
            "product_line", "product line", "brand_name",
        }

        def _tail_strongly_in_snippet(tail_str: str) -> bool:
            """tail 值在 snippet_corpus 里以 ≥15 字符完整出现 → 视为 snippet 就有。"""
            if not tail_str or len(tail_str) < 15:
                return False
            return tail_str in snippet_corpus

        for t in triples_list:
            if t.get("retrieval_mode"):
                continue  # 已标记
            src = (t.get("source_snippet") or "").strip()
            fact_text = (t.get("fact") or "").strip()
            tail = (t.get("tail") or "").strip()
            relation = (t.get("relation") or "").lower()

            # 空间关系跳过：它们来自 bbox 兜底，不是搜索结果，不参与 retrieval_mode
            if any(sr == relation or sr in relation for sr in spatial_relations):
                t["retrieval_mode"] = "spatial"
                continue

            # ---- Signal 1: chunk 匹配（级联 30/20/15/10 字符）----
            # 先计算所有信号再决定，因为我们要区分 evidenced vs semantic
            found_in_deep = False
            found_in_snippet = False
            for text in (src, fact_text):
                if not text or len(text) < 10:
                    continue
                for chunk_len in (30, 20, 15, 10):
                    if len(text) < chunk_len:
                        continue
                    chunk = text[:chunk_len]
                    if chunk in deep_read_corpus:
                        found_in_deep = True
                    if chunk in snippet_corpus:
                        found_in_snippet = True
                    if found_in_deep or found_in_snippet:
                        break
                if found_in_deep or found_in_snippet:
                    break

            # ---- Signal 2: tail 值在 deep_read 但不在 snippet ----
            tail_only_in_deep = False
            if tail and len(tail) >= 3:
                if tail in deep_read_corpus and tail not in snippet_corpus:
                    tail_only_in_deep = True

            # ---- Signal 0: relation 硬 override（语义级）----
            relation_leans_page = any(kw in relation for kw in page_leaning_relations)

            # ---- Signal 3: tail_type 通用 fallback（语义级）----
            tail_type = (t.get("tail_type") or "OTHER").upper()
            tail_is_specific = tail_type in ("PERSON", "LOCATION", "ORG", "TIME", "QUANTITY")
            tail_not_in_snippet = bool(tail) and not _tail_strongly_in_snippet(tail)

            # ---- 融合判断，区分 evidenced vs semantic ----
            # 优先级：evidenced > semantic > snippet_only
            #
            # evidenced: 有真证据（Signal 1 chunk 命中 deep_read 或 Signal 2 tail 命中 deep_read）
            # semantic:  没真证据但语义上应该 visit（Signal 0 relation override 或 Signal 3 tail_type fallback）
            evidenced = (found_in_deep and not found_in_snippet) or tail_only_in_deep
            semantic = (relation_leans_page or (tail_is_specific and tail_not_in_snippet)) and not _tail_strongly_in_snippet(tail)

            if evidenced:
                t["retrieval_mode"] = "page_only_evidenced"
            elif semantic:
                t["retrieval_mode"] = "page_only_semantic"
            else:
                t["retrieval_mode"] = "snippet_only"

    _mark_retrieval_mode(triples)
    n_page = sum(1 for t in triples if t.get("retrieval_mode") == "page_only")
    logger.info(f"  [{img_id}] 第一轮提取到 {len(triples)} 个三元组 (page_only={n_page})")

    # ---- 2b-3.5. 合并跨实体桥接三元组 ----
    # 先更新 corpus（把跨实体 deep_reads 也加进去）
    for psr in pair_search_results if 'pair_search_results' in dir() else []:
        pass  # pair_search_results 是局部变量，这里拿不到

    if cross_triples:
        existing_keys = {
            (t.get("head", "").lower(), t.get("relation", "").lower(), t.get("tail", "").lower())
            for t in triples
        }
        cross_added = 0
        for ct in cross_triples:
            key = (ct.get("head", "").lower(), ct.get("relation", "").lower(), ct.get("tail", "").lower())
            if key not in existing_keys:
                triples.append(ct)
                existing_keys.add(key)
                cross_added += 1
        # 给新并入的 cross_triples 也标记 retrieval_mode
        _mark_retrieval_mode(triples)
        logger.info(f"  [{img_id}] 合并跨实体三元组: 新增 {cross_added} 个, 共 {len(triples)} 个")

    # ---- 2b-4. 多轮扩展搜索（graph richness 停止条件）----
    MAX_EXTEND_ROUNDS = 3  # 默认 3 轮，但允许 1 轮 deficit-driven reopen（见下方 DEFICIT_REOPEN）
    DEFICIT_REOPEN = True  # 如果 3 轮后 closure richness 只差 1 项就达标，再开 1 轮定向扩展
    for round_idx in range(1, MAX_EXTEND_ROUNDS + 1):
        ok, richness_checks = _closure_richness_ok(triples, entities)
        logger.info(f"  [{img_id}] 第 {round_idx} 轮 closure richness: "
                    f"image_resolved={richness_checks['image_resolved_anchors']}/3 "
                    f"page_only={richness_checks['page_only_facts']}/6 "
                    f"compare_pairs={richness_checks['compare_ready_pairs']}/4 "
                    f"rank={richness_checks['rank_ready_triplets']}/2 "
                    f"shared={richness_checks['cross_anchor_shared_nodes']}/3 "
                    f"triples={richness_checks['total_triples']}/40 "
                    f"→ {richness_checks['passed_count']}/6 passed")
        if ok:
            logger.info(f"  [{img_id}] 第 {round_idx} 轮前 closure richness 达标，停止扩展")
            break
        logger.info(f"  [{img_id}] 第 {round_idx} 轮扩展搜索（图谱尚未达标）...")
        triples, extra_search_results = _extend_search(
            triples, entities, img_id, all_search_results, round_idx
        )
        all_search_results.extend(extra_search_results)
        # 更新 corpus 并重新标记 retrieval_mode
        for sr2 in extra_search_results:
            for s in sr2.get("searches", []):
                for r in s.get("results", []):
                    snippet_corpus += (r.get("snippet", "") + " " + r.get("content", "")) + " "
                for d in s.get("deep_reads", []):
                    deep_read_corpus += d.get("content", "") + " "
        _mark_retrieval_mode(triples)
        if not extra_search_results:
            logger.info(f"  [{img_id}] 第 {round_idx} 轮无可扩展实体，停止")
            break

    # ---- 2b-4b. Deficit-driven reopen（只差 1 项就达标时再开 1 轮）----
    if DEFICIT_REOPEN:
        ok_final, rc_final = _closure_richness_ok(triples, entities)
        # 只对小图（≤10 实体）reopen，大图额外一轮代价太高
        if not ok_final and rc_final.get("passed_count", 0) >= 2 and len(entities) <= 10:
            # 差 1 项达标（passed=2，需要 3）→ 再做 1 轮定向扩展
            logger.info(f"  [{img_id}] Deficit reopen: passed={rc_final['passed_count']}/3, 再开 1 轮")
            triples, extra = _extend_search(triples, entities, img_id, all_search_results, MAX_EXTEND_ROUNDS + 1)
            if extra:
                all_search_results.extend(extra)
                for sr2 in extra:
                    for s in sr2.get("searches", []):
                        for r in s.get("results", []):
                            snippet_corpus += (r.get("snippet", "") + " " + r.get("content", "")) + " "
                        for d in s.get("deep_reads", []):
                            deep_read_corpus += d.get("content", "") + " "
                _mark_retrieval_mode(triples)

    # ---- 2b-5. 实体名规范化（启发式合并漂移变体） ----
    before = len(triples)
    triples = _normalize_triple_entities(triples, entities)
    triples = _sanitize_triples(triples)
    logger.info(
        f"  [{img_id}] 启发式规范化: {before} → {len(triples)} 个三元组"
        f"（去除 {before - len(triples)} 条重复/自环）"
    )

    # ---- 2b-5b. LLM 别名归一（处理 substring match 管不住的变体） ----
    # 例如 "Nuggets" / "Denver Nuggets" / "the Denver Nuggets" 合并
    # 必须在 spatial fallback 之前跑，因为 spatial 会用干净的实体名建连接
    before_llm = len(triples)
    triples, entities, _alias_map = _llm_canonicalize_aliases(triples, entities, img_id)
    triples = _sanitize_triples(triples)
    logger.info(
        f"  [{img_id}] LLM 别名归一: {before_llm} → {len(triples)} 个三元组"
    )

    # ---- 2b-6. 空间关系兜底：为无知识连接的实体对补充空间三元组 ----
    spatial_added = _add_spatial_fallback(triples, high_conf)
    if spatial_added:
        logger.info(
            f"  [{img_id}] 空间关系兜底: 补充 {len(spatial_added)} 条三元组，"
            f"覆盖对: {[(t['head'], t['tail']) for t in spatial_added]}"
        )
    triples = _sanitize_triples(triples)
    # ★ 空间兜底后再跑一次 _mark_retrieval_mode，让新加的空间三元组也被标 "spatial"
    #   （否则它们 retrieval_mode 为空，既不算 page_only 也不算 snippet_only）
    _mark_retrieval_mode(triples)

    # ==================================================================
    # 持久化一等证据层（P1 改进）
    # 让 Step5/6 能按 provenance 归因，而不是只看 triples
    # ==================================================================

    # 1. resolution_edges: region → candidate_entity → canonical_entity 的证据链
    resolution_edges: list[dict] = []
    for e in entities:
        eid = e.get("id", "")
        res = e.get("resolution", {}) or {}
        # lens 路径
        if res.get("lens_provenance") == "lens_reverse":
            for cand in res.get("lens_candidate_titles", [])[:10]:
                resolution_edges.append({
                    "region_id": eid,
                    "candidate": cand,
                    "canonical": e.get("name", ""),
                    "provenance": "lens_reverse",
                    "lens_url": res.get("lens_url", ""),
                })
            # knowledge graph 单独标
            kg = res.get("lens_knowledge_graph") or {}
            if kg.get("title"):
                resolution_edges.append({
                    "region_id": eid,
                    "candidate": kg.get("title", ""),
                    "canonical": e.get("name", ""),
                    "provenance": "lens_kg",
                    "lens_url": res.get("lens_url", ""),
                })
        # image_search(text) 路径
        for title in (res.get("image_search_titles") or [])[:5]:
            resolution_edges.append({
                "region_id": eid,
                "candidate": title,
                "canonical": e.get("name", ""),
                "provenance": "image_search_text",
            })
        # VLM workaround 路径
        if res.get("reverse_visual_desc") and res.get("lens_provenance") != "lens_reverse":
            resolution_edges.append({
                "region_id": eid,
                "candidate": res.get("reverse_visual_desc", "")[:100],
                "canonical": e.get("name", ""),
                "provenance": "vlm_describe_workaround",
            })

    # 2. image_pages: 来自 lens / image_search 的 source pages
    # 3. web_pages: 来自 web_search 主流程 + 扩展搜索的 visited pages
    # 两个结构相同，区别是 source 字段
    image_pages: list[dict] = []
    web_pages: list[dict] = []
    seen_page_urls: set[str] = set()

    for e in entities:
        eid = e.get("id", "")
        res = e.get("resolution", {}) or {}
        # visited_pages 存的是所有 phase 3 visit 的结果
        for vp in res.get("visited_pages", []):
            url = vp.get("url", "")
            if not url or url in seen_page_urls:
                continue
            seen_page_urls.add(url)
            # 判断是 image 来源还是 web 来源（从 image_search_sources 对比）
            img_sources = {s.get("url", "") for s in res.get("image_search_sources", [])}
            is_image_source = url in img_sources or res.get("lens_provenance") == "lens_reverse"
            page_record = {
                "url": url,
                "title": vp.get("title", ""),
                "content_preview": vp.get("content_preview", "")[:300],
                "anchor_entity_id": eid,
                "visited": True,
                "provenance": "lens_source" if is_image_source else "image_text_source",
            }
            if is_image_source:
                image_pages.append(page_record)
            else:
                web_pages.append(page_record)

    # 从 all_search_results 抽 web_search 主流程的 visited pages
    for sr in all_search_results:
        ent_name = sr.get("entity_name", "")
        for s in sr.get("searches", []):
            for dr in s.get("deep_reads", []):
                url = dr.get("url", "")
                if not url or url in seen_page_urls:
                    continue
                seen_page_urls.add(url)
                web_pages.append({
                    "url": url,
                    "title": dr.get("title", ""),
                    "content_preview": (dr.get("content", "") or "")[:300],
                    "anchor_entity": ent_name,
                    "visited": True,
                    "provenance": "web_search",
                })

    # 4. local_artifacts（已存在，但可能为空）— 保持现状，Step2 未来补 OCR 等
    local_artifacts = entity_data.get("local_artifacts", {})

    # ==================================================================

    result = {
        "img_id": img_id,
        "img_path": img_path,
        "image_description": image_desc,
        "domain": entity_data.get("domain", "other"),
        "entities": entities,
        "high_conf": core_anchors,  # 向后兼容
        "core_anchors": core_anchors,
        "expansion_seeds": expansion_seeds,
        "local_artifacts": local_artifacts,
        "search_plans": plans,
        "search_results": all_search_results,
        "triples": triples,
        # ===== 一等证据层（新增，供 Step5/6 溯源）=====
        "resolution_edges": resolution_edges,
        "image_pages": image_pages,
        "web_pages": web_pages,
        "evidence_stats": {
            "n_resolution_edges": len(resolution_edges),
            "n_image_pages": len(image_pages),
            "n_web_pages": len(web_pages),
            "n_triples_total": len(triples),
            "n_triples_spatial": sum(1 for t in triples if t.get("retrieval_mode") == "spatial"),
            "n_triples_page_evidenced": sum(1 for t in triples if t.get("retrieval_mode") == "page_only_evidenced"),
            "n_triples_page_semantic": sum(1 for t in triples if t.get("retrieval_mode") == "page_only_semantic"),
            "n_triples_snippet": sum(1 for t in triples if t.get("retrieval_mode") == "snippet_only"),
        },
    }

    save_checkpoint(2, img_id, result)
    os.makedirs(ENTITY_DIR, exist_ok=True)
    entity_file = os.path.join(ENTITY_DIR, f"{img_id}.json")
    with open(entity_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"  [{img_id}] 实体文件写入: {entity_file}")

    return result


# ============================================================
# 主流程
# ============================================================

def main(workers: int = MAX_WORKERS, source_dir: str = ""):
    """Step2 主入口。

    source_dir: 图片来源目录。留空则先查 FILTERED_IMAGE_DIR（Step1 输出），
                再 fallback 到 images/（直接跳过 Step1）。
                也可以显式传一个目录（如 stress_suite/sports/）。
    """
    logger.info("=" * 60)
    logger.info("第二步：实体提取与知识扩展")
    logger.info(f"Pillow: {'可用' if Image is not None else '未安装⚠️'}")
    logger.info(f"Serper: {'已配置' if SERPER_KEY else '未配置⚠️'}  Jina: {'有key' if JINA_API_KEY else '无key(rate limited)'}")
    logger.info("=" * 60)

    if Image is None:
        logger.error("缺少 Pillow 依赖。请安装: pip install pillow")
        return []

    # 图片来源：显式传入 > FILTERED_IMAGE_DIR (Step1 输出) > images/ (跳过 Step1)
    search_dirs = [source_dir] if source_dir else [FILTERED_IMAGE_DIR, "images"]
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    img_paths = []
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(d, p)))
        if img_paths:
            logger.info(f"图片来源: {d} ({len(img_paths)} 张)")
            break
    img_paths.sort()

    if not img_paths:
        logger.error("没有找到图片。请把图片放到 images/ 目录，或用 --source 指定目录")
        return []

    actual_workers = min(workers, 3) if SERPER_KEY else workers
    results = []
    with ThreadPoolExecutor(max_workers=actual_workers) as pool:
        futures = {pool.submit(enrich_image, p): p for p in img_paths}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    logger.info("=" * 60)
    logger.info(f"第二步完成！处理 {len(results)}/{len(img_paths)} 张图片")
    total_entities = sum(len(r.get("entities", [])) for r in results)
    total_triples = sum(len(r.get("triples", [])) for r in results)
    total_searches = sum(
        sum(len(sr.get("searches", [])) for sr in r.get("search_results", []))
        for r in results
    )
    logger.info(f"总计实体: {total_entities}  三元组: {total_triples}  搜索请求: {total_searches}")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第二步：实体提取与知识扩展")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    parser.add_argument("--source", default="", help="图片来源目录（默认先查 output/images，再 fallback images/）")
    args = parser.parse_args()
    main(workers=args.workers, source_dir=args.source)
