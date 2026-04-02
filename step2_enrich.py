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

# 搜索/Jina 请求需要绕过代理，与 VLM 客户端保持一致
_http = httpx.Client(trust_env=False, timeout=30)

from core.config import (
    FILTERED_IMAGE_DIR, ENTITY_DIR, MAX_WORKERS,
    SERPER_KEY, JINA_API_KEY, JINA_READER_URL, JINA_TIMEOUT,
)
from core.vlm import call_vlm_json
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step2", "step2_enrich.log")

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

ENTITY_CROP_QUALITY = int(os.environ.get("ENTITY_CROP_QUALITY", "90"))


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
2. 如果实体本身太泛（如纯年份"1960"、通用概念"jazz musicians"），要加上方向性的词使搜索有意义，或者标记 skip
3. 查询词要自然、适合搜索引擎
4. 如果某个实体确实不值得搜索（太泛、太通用、搜不出有价值的信息），标记 skip=true

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

请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{{{
    "triples": [
        {{{{
            "head": "头实体名",
            "relation": "关系类型",
            "tail": "尾实体名",
            "tail_type": "TIME/QUANTITY/LOCATION/PERSON/ORG/OTHER 之一",
            "fact": "一句话描述这个事实",
            "source_snippet": "搜索结果中的佐证原文片段"
        }}}}
    ]
}}}}"""


# ============================================================
# 三元组字段规范化
# ============================================================

_TAIL_TYPE_ENUM = {"TIME", "QUANTITY", "LOCATION", "PERSON", "ORG", "OTHER"}


def _normalize_tail_type(value) -> str:
    if not isinstance(value, str):
        return "OTHER"
    normalized = value.strip().upper()
    return normalized if normalized in _TAIL_TYPE_ENUM else "OTHER"


def _sanitize_triples(triples: list[dict]) -> list[dict]:
    """规范化三元组字段，确保 tail_type 始终存在且合法。"""
    normalized = []
    for t in triples or []:
        if not isinstance(t, dict):
            continue
        item = dict(t)
        item["tail_type"] = _normalize_tail_type(item.get("tail_type"))
        normalized.append(item)
    return normalized


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
        crop_path = os.path.join(crop_dir, f"{eid}.jpg")
        rgb_image.crop((ix1, iy1, ix2, iy2)).save(
            crop_path, format="JPEG", quality=ENTITY_CROP_QUALITY
        )

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
            "crop_path": crop_path,
            "source": "vlm_only",
        })
        idx += 1
    domain = _guess_domain_from_labels([e["name"] for e in entities])

    logger.info(
        f"    [{img_id}] 最终: {len(entities)} 个实体 "
        f"(VLM 识别 {len(vlm_entities)}, 幻觉移除 {hallucination_count})"
    )

    return {"entities": entities, "image_description": image_desc, "domain": domain}


# ============================================================
# 搜索：SerpAPI (Google)
# ============================================================

def web_search(query: str, max_results: int = 5) -> dict:
    """调用 Serper 搜索 Google，返回结构化结果。"""
    if not SERPER_KEY:
        return {"query": query, "error": "未配置 SERPER_KEY", "results": []}
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

        return {"query": query, "results": results}
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


# ============================================================
# 网页深度读取：Jina Reader
# ============================================================

def visit_url(url: str, max_chars: int = 3000) -> dict:
    try:
        headers = {"Accept": "text/plain"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        resp = _http.get(f"{JINA_READER_URL}{url}", headers=headers)
        if resp.status_code != 200:
            return {"url": url, "error": f"HTTP {resp.status_code}", "content": ""}
        return {"url": url, "content": resp.text[:max_chars]}
    except Exception as e:
        return {"url": url, "error": str(e), "content": ""}


def _deep_read_top_urls(search_result: dict, max_urls: int = 2, max_chars: int = 3000) -> list:
    _SKIP_DOMAINS = ("youtube.com", "facebook.com", "instagram.com", "twitter.com", "tiktok.com")
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
        content = visit_url(url, max_chars=max_chars)
        if content.get("content"):
            return {"url": url, "title": title, "content": content["content"]}
        return None

    read_results = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        for result in pool.map(_fetch, candidates):
            if result:
                read_results.append(result)
    return read_results


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

    tail_counts: dict[str, int] = {}
    for t in triples:
        tail = t.get("tail", "").strip()
        tail_lower = tail.lower()
        if not tail or tail_lower in in_image_names or tail_lower in already_searched or len(tail) < 3:
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
        query = item.get("query", "")
        if query:
            ext_tasks.append((entity_name, query))

    round2_search_results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(web_search, q, 5): (name, q) for name, q in ext_tasks}
        for fut in as_completed(futures):
            entity_name, query = futures[fut]
            sr = fut.result()
            sr["purpose"] = f"第{round_idx}轮扩展搜索: {entity_name}"
            round2_search_results.append({
                "entity_id": f"R{round_idx+1}_{entity_name[:20]}",
                "entity_name": entity_name,
                "searches": [sr],
                "round": round_idx + 1,
            })
            logger.info(f"    [R{round_idx+1}] \"{entity_name}\" → \"{query}\" → {len(sr.get('results', []))} 条结果")

    def _jina_ext(er):
        for s in er.get("searches", []):
            s["deep_reads"] = _deep_read_top_urls(s, max_urls=5, max_chars=2000)
        return er

    with ThreadPoolExecutor(max_workers=6) as pool:
        round2_search_results = list(pool.map(_jina_ext, round2_search_results))

    round2_text = ""
    for sr in round2_search_results:
        round2_text += f"\n### {sr['entity_name']} (第 {round_idx} 轮扩展)\n"
        for s in sr.get("searches", []):
            round2_text += f"\n搜索词: \"{s['query']}\"\n"
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
) -> list[dict]:
    """发现图中不同实体之间的真实世界关联，返回带 source='cross_entity' 标记的三元组列表。

    对 high_conf 实体两两枚举（C(n,2) 对），由 LLM 为每对生成有针对性的搜索计划，再并行搜索。
    """
    if len(entities) < 2:
        return []

    try:
        from itertools import combinations

        # ── 枚举所有实体对 ──
        all_pairs = []
        for a, b in combinations(entities, 2):
            a_name = a.get("name", "")
            b_name = b.get("name", "")
            if a_name and b_name:
                all_pairs.append({"entity_a": a_name, "entity_b": b_name})

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
                deep_reads = _deep_read_top_urls(sr, max_urls=5, max_chars=2000)
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
                cross_triples.append(t)

        logger.info(
            f"  [{img_id}] 跨实体桥接三元组提取完成: "
            f"原始 {len(raw_triples)} 个，去重后 {len(cross_triples)} 个"
        )
        return cross_triples

    except Exception as e:
        logger.error(f"  [{img_id}] 跨实体关联发现异常: {e}")
        return []


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

    # ---- 2b-1 前. 先选出高置信度实体（最多5个），后续所有步骤均基于此子集 ----
    high_conf = []
    seen_entity_names = set()
    entities_for_plan = sorted(
        entities,
        key=lambda x: float(x.get("confidence", 0.0)) if isinstance(x.get("confidence"), (int, float)) else 0.0,
        reverse=True,
    )
    for e in entities_for_plan:
        name_key = str(e.get("name", "")).strip().lower()
        if not name_key or name_key in seen_entity_names:
            continue
        conf = e.get("confidence")
        if isinstance(conf, (int, float)) and conf >= 0.45:
            high_conf.append(e)
        elif str(e.get("confidence_level", "")).lower() in ("high", "medium"):
            high_conf.append(e)
        if high_conf and high_conf[-1] is e:
            seen_entity_names.add(name_key)
        if len(high_conf) >= 5:
            break
    if not high_conf:
        high_conf = entities[:5]

    entities_json_str = json.dumps(high_conf, ensure_ascii=False, indent=2)
    image_desc = entity_data.get("image_description", "")

    # ---- 2b-0. 跨实体关联发现（只在被搜索的 high_conf 实体之间查找）----
    logger.info(f"  [{img_id}] 跨实体关联发现（{len(high_conf)} 个实体）...")
    cross_triples = _find_cross_entity_relations(img_path, high_conf, img_id, image_desc)
    logger.info(f"  [{img_id}] 跨实体桥接三元组: {len(cross_triples)} 个")

    logger.info(f"  [{img_id}] 为 {len(high_conf)} 个实体生成搜索计划...")
    search_plan_data = call_vlm_json(
        SEARCH_PLAN_PROMPT.format(image_description=image_desc, entities_json=entities_json_str),
        "请为每个实体生成搜索计划。",
        max_tokens=2048,
        temperature=0.3,
    )

    if search_plan_data is None:
        logger.warning(f"  [{img_id}] 搜索计划生成失败，跳过知识扩展")
        result = {
            "img_id": img_id, "img_path": img_path,
            "image_description": image_desc,
            "domain": entity_data.get("domain", "other"),
            "entities": entities,
            "search_results": [], "triples": [],
        }
        save_checkpoint(2, img_id, result)
        return result

    plans = search_plan_data.get("search_plans", [])
    searchable = [p for p in plans if not p.get("skip", False)]
    skipped = [p for p in plans if p.get("skip", False)]
    logger.info(f"  [{img_id}] 搜索计划: {len(searchable)} 个实体需搜索, {len(skipped)} 个跳过")

    # ---- 2b-2. 并行执行搜索 + Jina深读 ----
    logger.info(f"  [{img_id}] 并行执行搜索...")

    search_tasks = []
    for plan in searchable:
        eid = plan.get("entity_id", "")
        ename = plan.get("entity_name", "")
        for q_item in plan.get("queries", []):
            query = q_item.get("query", "")
            purpose = q_item.get("purpose", "")
            if query:
                search_tasks.append((eid, ename, query, purpose))

    search_results_map = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {}
        for eid, ename, query, purpose in search_tasks:
            fut = pool.submit(web_search, query, 5)
            futures[fut] = (eid, ename, query, purpose)
        for fut in as_completed(futures):
            eid, ename, query, purpose = futures[fut]
            sr = fut.result()
            sr["purpose"] = purpose
            search_results_map[(eid, query)] = sr
            logger.info(f"    [{eid}] \"{query}\" → {len(sr.get('results', []))} 条结果")

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

    def _jina_read_task(task):
        idx_i, idx_j, s = task
        return idx_i, idx_j, _deep_read_top_urls(s, max_urls=5, max_chars=3000)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for idx_i, idx_j, deep in pool.map(_jina_read_task, jina_tasks):
            all_search_results[idx_i]["searches"][idx_j]["deep_reads"] = deep

    # ---- 2b-3. 从搜索结果中提取事实三元组 ----
    entities_summary = "\n".join(
        f"- {e['id']}: {e['name']} (类型={e.get('type', '?')}, 位置={e.get('location_in_image', '?')})"
        for e in high_conf
    )

    search_results_text = ""
    for sr in all_search_results:
        search_results_text += f"\n### {sr['entity_name']} ({sr['entity_id']})\n"
        for s in sr.get("searches", []):
            search_results_text += f"\n搜索词: \"{s['query']}\" (目的: {s.get('purpose', '')})\n"
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
    logger.info(f"  [{img_id}] 第一轮提取到 {len(triples)} 个三元组")

    # ---- 2b-3.5. 合并跨实体桥接三元组 ----
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
        logger.info(f"  [{img_id}] 合并跨实体三元组: 新增 {cross_added} 个, 共 {len(triples)} 个")

    # ---- 2b-4. 多轮扩展搜索 ----
    MAX_EXTEND_ROUNDS = 3
    for round_idx in range(1, MAX_EXTEND_ROUNDS + 1):
        if _has_l3_chains(triples, entities):
            logger.info(f"  [{img_id}] 第 {round_idx} 轮前检测到 L3 链，停止扩展")
            break
        logger.info(f"  [{img_id}] 第 {round_idx} 轮扩展搜索（尚无 L3 链）...")
        triples, extra_search_results = _extend_search(
            triples, entities, img_id, all_search_results, round_idx
        )
        all_search_results.extend(extra_search_results)
        if not extra_search_results:
            logger.info(f"  [{img_id}] 第 {round_idx} 轮无可扩展实体，停止")
            break

    # ---- 2b-5. 实体名规范化（启发式合并漂移变体） ----
    before = len(triples)
    triples = _normalize_triple_entities(triples, entities)
    triples = _sanitize_triples(triples)
    logger.info(
        f"  [{img_id}] 实体名规范化: {before} → {len(triples)} 个三元组"
        f"（去除 {before - len(triples)} 条重复/自环）"
    )

    # ---- 2b-6. 空间关系兜底：为无知识连接的实体对补充空间三元组 ----
    spatial_added = _add_spatial_fallback(triples, high_conf)
    if spatial_added:
        logger.info(
            f"  [{img_id}] 空间关系兜底: 补充 {len(spatial_added)} 条三元组，"
            f"覆盖对: {[(t['head'], t['tail']) for t in spatial_added]}"
        )
    triples = _sanitize_triples(triples)

    result = {
        "img_id": img_id,
        "img_path": img_path,
        "image_description": image_desc,
        "domain": entity_data.get("domain", "other"),
        "entities": entities,
        "search_plans": plans,
        "search_results": all_search_results,
        "triples": triples,
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

def main(workers: int = MAX_WORKERS):
    logger.info("=" * 60)
    logger.info("第二步：实体提取与知识扩展（纯 VLM + SerpAPI + Jina + 三元组提取）")
    logger.info(f"Pillow: {'可用' if Image is not None else '未安装⚠️'}")
    logger.info(f"Serper: {'已配置' if SERPER_KEY else '未配置⚠️'}  Jina: {'有key' if JINA_API_KEY else '无key(rate limited)'}")
    logger.info("=" * 60)

    if Image is None:
        logger.error("缺少 Pillow 依赖。请安装: pip install pillow")
        return []

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(os.path.join(FILTERED_IMAGE_DIR, p)))
    img_paths.sort()
    logger.info(f"找到 {len(img_paths)} 张筛选后图片")

    if not img_paths:
        logger.error("没有找到筛选后的图片，请先运行第一步")
        return []

    actual_workers = min(workers, 3) if SERPAPI_KEY else workers
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
    args = parser.parse_args()
    main(workers=args.workers)
