"""
从 Hugging Face 流式读取图像，筛选真实照片，调用 VLM 合成高阶 Agentic Q&A 训练数据。

输出格式：每行一道题（平铺），字段包含：
  image_path, visual_analysis, query, constraints_injected,
  tool_plan, target_answer_format, key_evidence_chain

流程：
  1. 流式加载 HF 数据集，逐条提取图片
  2. 调用 VLM 判断图片是否为真实物理世界照片
  3. 通过筛选的图片保存到本地 images/ 目录
  4. 再次调用 VLM，基于工具约束生成复杂 Agentic 任务轨迹
  5. 将每个 candidate 作为独立一行写入 JSONL
  6. 收集满 TARGET_COUNT 道题后退出

依赖安装：
  pip3 install openai datasets pillow python-dotenv
"""

import base64
import io
import json
import logging
import os
import sys
import time
import traceback

from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from PIL import Image

load_dotenv()

# ============================================================
# 全局配置
# ============================================================
API_KEY    = "sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH"
MODEL_NAME = "gemini-3-flash-preview"
BASE_URL   = "http://35.220.164.252:3888/v1"

TARGET_COUNT     = 50   # 目标题目数（每个 candidate 算一道题）
MAX_STREAM_ITEMS = 300   # 最多遍历的原始图片数
OUTPUT_FILE      = "synthetic_qa.jsonl"
IMAGE_DIR        = "images_new"
LOG_FILE         = "synthesize_qa.log"

VALID_TOOLS = {"web_search", "image_search", "visit", "code_interpreter"}

# ============================================================
# 日志
# ============================================================
logger = logging.getLogger("synthesize_qa")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)


def log(msg: str = ""):
    logger.info(msg)


# ============================================================
# VLM 客户端
# ============================================================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# Prompt 模板
# ============================================================
FILTER_SYSTEM_PROMPT = (
    "你是一个图片分类助手。你的任务是判断用户提供的图片是否为"
    "**真实的物理世界照片**（例如实拍的风景、人物、动物、建筑、食物、物品等）。\n\n"
    "如果图片属于以下任何一类，请回答 NO：\n"
    "- 屏幕截图、软件界面截图\n"
    "- 二次元 / 动漫 / 插画 / 卡通\n"
    "- 图表、流程图、数据可视化\n"
    "- AI 生成的明显不真实的图片\n"
    "- 纯文字图片、表情包\n\n"
    "只输出一个单词：YES 或 NO，不要输出任何其他内容。"
)

AGENTIC_SYSTEM_PROMPT = """\
你是一个高阶 Agentic 视觉任务数据合成专家。

## 你的工具集（也是唯一可用的工具）
用户的 Agent 只能使用以下四种工具来完成任务：
1. `web_search`：文本网络搜索，获取外部知识和事实。
2. `image_search`：图像搜索（文本搜图或以图搜图），用于识别物体、品牌、地标等。
3. `visit`：访问特定 URL 网页并提取其文本内容。
4. `code_interpreter`：执行 Python 代码（算术计算，或使用 PIL/OpenCV 进行图像裁剪、缩放、测量等视觉处理）。

## 你的任务
根据用户提供的一张真实照片，按以下步骤思考并输出：

### 步骤 1：视觉线索分析
仔细观察图片，识别其中的"可操作视觉线索"（如品牌 logo、文字、地标、商品、植物种类、设备型号等），\
推导出依赖这些线索才能回答的现实需求。

### 步骤 2：注入真实约束
将问题写成真实用户的请求形式。为每个候选任务随机注入 1-3 个真实世界约束，\
例如：时间限制、预算范围、地理范围、兼容性要求、安全规范等。

### 步骤 3：生成 3~5 个候选复杂任务
每个候选任务必须包含一个多步 tool_plan，其中：
- 至少使用上述四种工具中的两种不同工具
- 每一步的 tool 字段必须严格是 web_search / image_search / visit / code_interpreter 之一
- 步骤之间要有逻辑依赖（后续步骤依赖前序步骤的输出）

## 强制输出格式
严格输出以下 JSON，不要输出任何 JSON 之外的内容（不要加 markdown 代码块标记）：
{
  "visual_analysis": "对图中可操作视觉线索的简短分析",
  "candidates": [
    {
      "query": "结合了视觉线索和真实约束的复杂用户问题",
      "constraints_injected": ["约束1", "约束2"],
      "tool_plan": [
        {"step": 1, "tool": "code_interpreter", "intent": "裁剪出图片中的关键商品细节"},
        {"step": 2, "tool": "image_search", "intent": "使用裁剪出的局部图像进行反向搜索获取型号"},
        {"step": 3, "tool": "web_search", "intent": "搜索该型号的官网价格"}
      ],
      "target_answer_format": "目标答案的格式要求",
      "key_evidence_chain": "得出答案必须依赖的关键证据链"
    }
  ]
}
"""

# ============================================================
# 辅助函数
# ============================================================

def pil_image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def save_pil_image(img: Image.Image, index: int) -> str:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    rel_path = os.path.join(IMAGE_DIR, f"img_{index:04d}.jpg")
    img.save(rel_path, format="JPEG", quality=90)
    return rel_path


def call_vlm(system_prompt: str, image_b64: str, max_tokens: int = 4096,
             temperature: float = 0.5, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_b64}},
                        {"type": "text", "text": "请根据这张图片完成你的任务。"},
                    ]},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"    [VLM 调用失败] 第 {attempt}/{max_retries} 次: {e}")
            if attempt < max_retries:
                time.sleep(3 * attempt)
    return ""


# ============================================================
# 阶段逻辑
# ============================================================

def is_real_photo(image_b64: str) -> bool:
    answer = call_vlm(FILTER_SYSTEM_PROMPT, image_b64, max_tokens=64, temperature=0.3)
    log(f"    [筛选原始返回] {answer!r}")
    return answer.upper().startswith("YES")


def extract_json(raw: str) -> Optional[dict]:
    try:
        start = raw.index("{")
        depth, i = 0, start
        while i < len(raw):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start: i + 1])
            i += 1
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def validate_agentic_data(data: dict) -> bool:
    if "visual_analysis" not in data or "candidates" not in data:
        return False
    candidates = data["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 1:
        return False
    for c in candidates:
        if not all(k in c for k in ("query", "tool_plan")):
            return False
        plan = c["tool_plan"]
        if not isinstance(plan, list) or len(plan) < 2:
            return False
        tools_used = {step.get("tool", "") for step in plan}
        if not tools_used.issubset(VALID_TOOLS) or len(tools_used) < 2:
            return False
    return True


def generate_agentic_data(image_b64: str, max_attempts: int = 3) -> Optional[dict]:
    for attempt in range(1, max_attempts + 1):
        raw = call_vlm(AGENTIC_SYSTEM_PROMPT, image_b64, max_tokens=4096, temperature=0.6)
        if not raw:
            log(f"    [生成] 第 {attempt} 次：模型返回空")
            continue
        data = extract_json(raw)
        if data is None:
            log(f"    [生成] 第 {attempt} 次：JSON 提取失败，原文前 300 字: {raw[:300]}")
            continue
        if not validate_agentic_data(data):
            log(f"    [生成] 第 {attempt} 次：结构校验不通过")
            continue
        return data
    return None


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 60)
    log(f"开始生成问题，目标 {TARGET_COUNT} 道题，输出到 {OUTPUT_FILE}")
    log("=" * 60)

    ds = load_dataset(
        "WildVision/wildvision-internal-data",
        "battle",
        split="test",
        streaming=True,
    )

    collected    = 0   # 已写入的题目数
    processed    = 0   # 已遍历的原始图片数
    filtered_in  = 0
    saved_img_idx = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for item in ds:
            if collected >= TARGET_COUNT:
                break
            if processed >= MAX_STREAM_ITEMS:
                log(f"已遍历 {MAX_STREAM_ITEMS} 条，未凑满 {TARGET_COUNT} 道题，停止。")
                break

            processed += 1
            log(f"[图片 {processed}/{MAX_STREAM_ITEMS}] 已收集题目 {collected}/{TARGET_COUNT}")

            # ---- 提取图片 ----
            try:
                img = item.get("image")
                if img is None or not isinstance(img, Image.Image):
                    log("    跳过：无图片或类型异常")
                    continue
                image_b64 = pil_image_to_base64(img)
            except Exception:
                log(f"    跳过：图片编码异常\n{traceback.format_exc()}")
                continue

            # ---- 第一阶段：真实照片筛选 ----
            try:
                if not is_real_photo(image_b64):
                    log("    第一阶段：非真实照片，跳过")
                    continue
                filtered_in += 1
                log(f"    第一阶段：通过 ✓（累计通过 {filtered_in}）")
            except Exception:
                log(f"    跳过：筛选阶段异常\n{traceback.format_exc()}")
                continue

            # ---- 保存图片 ----
            try:
                saved_img_idx += 1
                local_path = save_pil_image(img, saved_img_idx)
                log(f"    图片已保存: {local_path}")
            except Exception:
                log(f"    跳过：图片保存异常\n{traceback.format_exc()}")
                continue

            # ---- 第二阶段：生成候选题目 ----
            try:
                agentic_data = generate_agentic_data(image_b64)
                if agentic_data is None:
                    log("    第二阶段：生成失败，跳过")
                    continue
            except Exception:
                log(f"    跳过：生成阶段异常\n{traceback.format_exc()}")
                continue

            # ---- 展平写入：每个 candidate 单独一行 ----
            visual_analysis = agentic_data.get("visual_analysis", "")
            candidates = agentic_data.get("candidates", [])
            written = 0
            for candidate in candidates:
                if collected >= TARGET_COUNT:
                    break
                record = {
                    "image_path":          local_path,
                    "visual_analysis":     visual_analysis,
                    "query":               candidate.get("query", ""),
                    "constraints_injected": candidate.get("constraints_injected", []),
                    "tool_plan":           candidate.get("tool_plan", []),
                    "target_answer_format": candidate.get("target_answer_format", ""),
                    "key_evidence_chain":  candidate.get("key_evidence_chain", ""),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                collected += 1
                written += 1

            log(f"    ✅ 写入 {written} 道题（累计 {collected} 道）")

    log("=" * 60)
    log(f"完成！遍历图片 {processed} 张 → 筛选通过 {filtered_in} 张 → 生成题目 {collected} 道")
    log(f"输出文件: {OUTPUT_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()
