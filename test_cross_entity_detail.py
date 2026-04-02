"""
单独调用 CROSS_ENTITY_PROMPT，打印 VLM 的完整输出（reasoning + pairs）。
复用 test_full_img_0010.json 中的实体列表，无需重新提取。
"""
import base64
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from core.vlm import call_vlm
from step2_enrich import CROSS_ENTITY_PROMPT

with open("test_full_img_0010.json", "r", encoding="utf-8") as f:
    data = json.load(f)

entities = data["entities"]
img_path = data["img_path"]

entities_list_text = "\n".join(
    f"- {e.get('id', '?')}: {e.get('name', '?')} "
    f"(类型={e.get('type', '?')}, 位置={e.get('location_in_image', '?')})"
    for e in entities
)

with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
image_data_url = f"data:image/jpeg;base64,{b64}"

prompt_text = CROSS_ENTITY_PROMPT.format(entities_list=entities_list_text)

print("=" * 70)
print("  输入给 VLM 的实体列表")
print("=" * 70)
print(entities_list_text)
print()
print("=" * 70)
print("  CROSS_ENTITY_PROMPT（格式化后）")
print("=" * 70)
print(prompt_text)
print()
print("=" * 70)
print("  调用 VLM 中...")
print("=" * 70)

raw_output = call_vlm(
    "你是一个关系分析专家。请结合图片内容和实体列表，分析这些实体之间可能存在的真实世界关联。",
    [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": prompt_text},
    ],
    max_tokens=2048,
    temperature=0.3,
)

print()
print("=" * 70)
print("  VLM 原始输出")
print("=" * 70)
print(raw_output)

print()
print("=" * 70)
print("  解析后的结构化结果")
print("=" * 70)
from core.vlm import extract_json
parsed = extract_json(raw_output)
if parsed:
    print(f"\nreasoning:")
    print(f"  {parsed.get('reasoning', '(无)')}")
    print(f"\npairs ({len(parsed.get('pairs', []))} 个):")
    for i, p in enumerate(parsed.get("pairs", []), 1):
        skip = "SKIP" if p.get("skip") else "搜索"
        print(f"\n  [{i}] {p.get('entity_a', '?')} ↔ {p.get('entity_b', '?')}  [{skip}]")
        print(f"      hypothesis: {p.get('hypothesis', '?')}")
        print(f"      query:      {p.get('query', '?')}")
else:
    print("  JSON 解析失败")
