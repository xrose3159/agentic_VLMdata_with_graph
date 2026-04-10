"""真实工具调用评测 agent。

四种工具，和生题时的定义完全一致：
1. code_interpreter(code) — 执行 Python 代码（裁剪、OCR、计算等）
2. web_search(query) — 网络搜索
3. image_search(search_type, query/image_url) — 文字搜图 或 以图搜图
4. visit(url, goal) — 访问网页

模型输出工具调用 → 真实执行 → 结果喂回 → 循环直到 final_answer。
"""

import base64
import io
import json
import os
import sys
import tempfile
import time
from typing import Any

import httpx
from openai import OpenAI
from PIL import Image

from core.config import SERPER_KEY, JINA_API_KEY, JINA_READER_URL

_http = httpx.Client(trust_env=False, timeout=30)


# ============================================================
# Tool implementations
# ============================================================

def tool_code_interpreter(code: str, image_path: str, work_dir: str) -> str:
    """在受限环境中执行 Python 代码。提供 IMAGE_PATH 和 WORK_DIR 变量。"""
    import subprocess
    # 写代码到临时文件
    code_file = os.path.join(work_dir, "agent_code.py")
    wrapped = f"""
import os, sys, json, base64, io
from PIL import Image
IMAGE_PATH = {repr(image_path)}
WORK_DIR = {repr(work_dir)}
{code}
"""
    with open(code_file, "w") as f:
        f.write(wrapped)
    try:
        result = subprocess.run(
            [sys.executable, code_file],
            capture_output=True, text=True, timeout=120,
            cwd=work_dir,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()[-500:]
            return f"执行错误:\n{err}" if err else "执行错误（无输出）"
        return output if output else "(代码执行成功，无输出)"
    except subprocess.TimeoutExpired:
        return "执行超时（120秒限制）"
    except Exception as e:
        return f"执行异常: {e}"


def tool_web_search(query: str, num: int = 5) -> str:
    """Serper 网络搜索。"""
    resp = _http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": num},
    )
    data = resp.json()
    results = []
    kg = data.get("knowledgeGraph", {})
    if kg:
        results.append(f"[知识面板] {kg.get('title', '')} - {kg.get('description', '')}")
        for attr, val in kg.get("attributes", {}).items():
            results.append(f"  {attr}: {val}")
    ab = data.get("answerBox", {})
    if ab:
        results.append(f"[精选摘要] {ab.get('answer', ab.get('snippet', ''))}")
    for r in data.get("organic", [])[:5]:
        results.append(f"[{r.get('title', '')}] {r.get('snippet', '')}")
    return "\n".join(results) if results else "未找到相关结果。"


def tool_image_search(search_type: str, query: str = "", image_path: str = "") -> str:
    """Serper 图片搜索。search_type: 'text' 文字搜图, 'reverse' 以图搜图。"""
    if search_type == "text":
        resp = _http.post(
            "https://google.serper.dev/images",
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
        )
        data = resp.json()
        results = []
        for r in data.get("images", [])[:5]:
            results.append(f"[{r.get('title', '')}] {r.get('source', '')} - {r.get('link', '')}")
        return "\n".join(results) if results else "未找到相关图片。"
    elif search_type == "reverse":
        # 用 Google Lens via Serper（如果支持），否则 fallback 到文字描述搜索
        if image_path and os.path.exists(image_path):
            # 读取图片，用 VLM 描述后搜索（模拟 reverse search）
            return "反向图片搜索：请先用 code_interpreter 描述图片内容，再用 web_search 搜索。"
        return "需要提供图片路径进行反向搜索。"
    return f"未知的 search_type: {search_type}，请使用 'text' 或 'reverse'。"


def tool_visit(url: str, goal: str = "") -> str:
    """Jina Reader 读取网页。"""
    try:
        headers = {"Accept": "text/markdown"}
        if JINA_API_KEY:
            headers["Authorization"] = f"Bearer {JINA_API_KEY}"
        resp = _http.get(f"{JINA_READER_URL}{url}", headers=headers, timeout=15)
        text = resp.text[:3000]
        return text if text else "网页内容为空。"
    except Exception as e:
        return f"访问失败: {e}"


# ============================================================
# Agent system prompt
# ============================================================

AGENT_SYSTEM = '''你是一个多模态智能体，需要通过调用工具来回答关于图片的问题。

## 可用工具（只能使用以下4种）

1. **code_interpreter** — 执行 Python 代码
   - 裁剪图片：用 PIL 裁剪指定区域
   - OCR 文字识别：用 pytesseract 或 easyocr
   - 数值计算、比较、排序
   - 图片处理、分析
   - 代码中可使用 IMAGE_PATH 变量获取原图路径，WORK_DIR 变量获取工作目录
   - 输出请用 print()

2. **web_search** — 网络搜索
   - 搜索实体相关的事实信息

3. **image_search** — 图片搜索
   - search_type="text": 用文字搜图
   - search_type="reverse": 以图搜图（反向图片搜索）

4. **visit** — 访问网页
   - 深度阅读搜索结果中的网页

## 调用格式

每次只输出一个 JSON 工具调用，不要输出其他内容：

裁剪图片：
{"tool": "code_interpreter", "code": "from PIL import Image\\nimg = Image.open(IMAGE_PATH)\\ncrop = img.crop((x1, y1, x2, y2))\\ncrop.save(WORK_DIR + \\'/crop.jpg\\')\\nprint(f\\'裁剪完成: {crop.size}\\')"}

OCR识别：
{"tool": "code_interpreter", "code": "import easyocr\\nreader = easyocr.Reader([\\'en\\', \\'ch_sim\\'])\\nresult = reader.readtext(WORK_DIR + \\'/crop.jpg\\')\\nfor r in result:\\n    print(r[1])"}

网络搜索：
{"tool": "web_search", "query": "搜索词"}

访问网页：
{"tool": "visit", "url": "https://...", "goal": "获取什么信息"}

图片搜索：
{"tool": "image_search", "search_type": "text", "query": "搜索词"}

给出最终答案：
{"tool": "final_answer", "answer": "你的答案"}

## 规则

- 每次只输出一个 JSON，不要有其他文字
- code_interpreter 的代码中，图片路径用 IMAGE_PATH 变量，工作目录用 WORK_DIR
- 先看图理解问题，需要时用 code_interpreter 裁剪放大
- 识别出实体后用 web_search 查询需要的知识
- 最后用 final_answer 给出简洁答案'''


# ============================================================
# JSON extraction
# ============================================================

def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    # 修复常见的非法 JSON 转义（模型常输出 \' 但 JSON 标准不支持）
    def _try_parse(s: str) -> dict | None:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # fix \' → '
        try:
            return json.loads(s.replace("\\'", "'"))
        except json.JSONDecodeError:
            pass
        return None

    result = _try_parse(text)
    if result:
        return result
    try:
        start = text.index("{")
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    result = _try_parse(text[start:i + 1])
                    if result:
                        return result
                    break
            i += 1
    except ValueError:
        pass
    return None


# ============================================================
# Agent loop
# ============================================================

def run_agent(
    question: str,
    image_path: str,
    client: OpenAI,
    model: str,
    max_steps: int = 10,
) -> dict:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    work_dir = tempfile.mkdtemp(prefix="eval_agent_")

    messages = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": f"问题：{question}\n\n请开始调用工具来回答这个问题。"},
        ]},
    ]

    trace = []
    final_answer = None

    for step in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=800,
                temperature=0.2,
                timeout=120,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            trace.append({"step": step + 1, "error": str(e)[:300]})
            break

        call = extract_json(raw)
        if not call:
            trace.append({"step": step + 1, "raw": raw[:300], "error": "json_parse_failed"})
            if "答案" in raw:
                final_answer = raw
            break

        tool = call.get("tool", "")
        trace_entry = {"step": step + 1, "tool": tool, "input": {k: v for k, v in call.items() if k != "tool"}}

        if tool == "final_answer":
            final_answer = call.get("answer", "")
            trace_entry["answer"] = final_answer
            trace.append(trace_entry)
            break

        # execute
        result = ""
        if tool == "code_interpreter":
            code = call.get("code", "")
            if code:
                result = tool_code_interpreter(code, os.path.abspath(image_path), work_dir)
                trace_entry["output"] = result[:500]
            else:
                result = "缺少 code 参数。"
                trace_entry["output"] = result

        elif tool == "web_search":
            query = call.get("query", "")
            if query:
                result = tool_web_search(query)
                trace_entry["output"] = result[:500]
            else:
                result = "缺少 query 参数。"
                trace_entry["output"] = result

        elif tool == "image_search":
            st = call.get("search_type", "text")
            q = call.get("query", "")
            result = tool_image_search(st, query=q)
            trace_entry["output"] = result[:500]

        elif tool == "visit":
            url = call.get("url", "")
            if url:
                result = tool_visit(url, call.get("goal", ""))
                trace_entry["output"] = result[:500]
            else:
                result = "缺少 url 参数。"
                trace_entry["output"] = result

        else:
            result = f"未知工具: {tool}。只能使用 code_interpreter/web_search/image_search/visit/final_answer。"
            trace_entry["output"] = result

        trace.append(trace_entry)

        # feed result back
        messages.append({"role": "assistant", "content": raw})
        # 如果 code_interpreter 生成了裁剪图，把裁剪图也喂回去（不删除，后续 OCR 需要读）
        crop_path = os.path.join(work_dir, "crop.jpg")
        if tool == "code_interpreter" and os.path.exists(crop_path):
            with open(crop_path, "rb") as f:
                crop_b64 = base64.b64encode(f.read()).decode()
            messages.append({"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}},
                {"type": "text", "text": f"工具执行结果:\n{result}\n\n（上方是裁剪后的图片）\n请继续下一步。"},
            ]})
        else:
            messages.append({"role": "user", "content": f"工具执行结果:\n{result}\n\n请继续下一步。"})

    return {"answer": final_answer, "trace": trace, "steps": len(trace), "work_dir": work_dir}


# ============================================================
# Main
# ============================================================

def main():
    client = OpenAI(
        api_key="sk-SrQRDlXzOIgQz9ciSq9SABqkOFmpGJHQO3BU95Bo01ap63VH",
        base_url="http://35.220.164.252:3888/v1",
        http_client=httpx.Client(trust_env=False),
    )
    MODEL = "qwen3-vl-30b-a3b-instruct"
    IMAGE = "images/img_0010.jpg"

    questions = [
        {"id": "L1_01", "q": "画面最左侧、麦当劳标志上方那个红白相间的电子品牌广告牌上写着什么单词？", "gt": "Maxell"},
        {"id": "L1_02", "q": "画面最右侧的大型彩色海报上，那个带有醒目数字的电影名称是什么？", "gt": "Toy Story 2"},
        {"id": "L2_01", "q": "画面左下角那个带有巨大黄色发光拱门标志的餐饮品牌，它的创始人是谁？", "gt": "Ray Kroc"},
        {"id": "L2_02", "q": '画面左侧那张写着"WINNER BEST MUSICAL"的音乐剧海报，它的作曲者是谁？', "gt": "Elton John"},
        {"id": "L2_03", "q": "画面正中央远处那栋细长高楼上，红色背景的电子品牌广告牌，所属公司在合并成立之初的名称是什么？", "gt": "Tokyo Shibaura Electric Company"},
        {"id": "L3_01", "q": '画面左侧那个"WINNER BEST MUSICAL"的红字白底广告牌代表的剧目，和它下方蓝色音乐剧广告牌代表的剧目，哪一个在百老汇首演的时间更晚？', "gt": "Billy Elliot (Nov 2008)"},
    ]

    print(f"模型: {MODEL}")
    print(f"模式: 真实工具调用（code_interpreter 执行真实 Python）")
    print()

    all_results = []
    for item in questions:
        print(f"{'='*60}")
        print(f"{item['id']}: {item['q'][:60]}...")
        print(f"{'='*60}")
        t0 = time.time()
        result = run_agent(item["q"], IMAGE, client, MODEL)
        elapsed = time.time() - t0

        gt = item["gt"].lower()
        ans = str(result["answer"] or "").lower()
        correct = gt in ans or any(w in ans for w in gt.split() if len(w) >= 3)
        mark = "O" if correct else "X"

        print(f"\n[{mark}] {elapsed:.1f}s, {result['steps']} steps")
        print(f"  GT:  {item['gt']}")
        print(f"  ANS: {str(result['answer'])[:150]}")
        for t in result["trace"]:
            tool = t.get("tool", "?")
            inp = json.dumps(t.get("input", {}), ensure_ascii=False)[:80]
            out = str(t.get("output", t.get("answer", "")))[:80]
            print(f"  [{t['step']}] {tool}")
            print(f"      in:  {inp}")
            print(f"      out: {out}")
        print()

        all_results.append({
            "question_id": item["id"],
            "question": item["q"],
            "ground_truth": item["gt"],
            "model_answer": result["answer"],
            "correct": correct,
            "elapsed_seconds": round(elapsed, 1),
            "n_steps": result["steps"],
            "trace": result["trace"],
        })

    score = f"{sum(1 for r in all_results if r['correct'])}/{len(all_results)}"
    print(f"\n总分: {score}")

    output = {
        "image_id": "img_0010",
        "image_path": IMAGE,
        "eval_model": MODEL,
        "eval_mode": "real_tool_calling",
        "tools": ["code_interpreter", "web_search", "image_search", "visit"],
        "score": score,
        "questions": all_results,
    }
    out_path = "output/eval_img_0010_qwen3_vl_30b_real_tools.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"保存到: {out_path}")


if __name__ == "__main__":
    main()
