"""
针对已生成的 synthetic_agent_data.jsonl，为每条记录的每个 candidate 执行
Agent 工具链循环，得出 gold_answer，并将结果写入新文件。

用法：
    python generate_answers.py
    python generate_answers.py --input synthetic_agent_data.jsonl --output answered_data.jsonl
    python generate_answers.py --resume   # 跳过已有答案的条目
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
import traceback

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# 配置
# ============================================================
API_KEY    = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-vl-max")
BASE_URL   = os.environ.get("BASE_URL", "")

SERPER_KEY  = os.environ.get("SERPER_KEY", "")
E2B_API_KEY = os.environ.get("E2B_API_KEY", "")

INPUT_FILE  = "synthetic_qa.jsonl"
OUTPUT_FILE = "answered_data.jsonl"
LOG_FILE    = "generate_answers.log"

AGENT_MAX_TURNS  = 15
TOOL_OUTPUT_LIMIT = 3000

# ============================================================
# 日志
# ============================================================
logger = logging.getLogger("generate_answers")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8").setFormatter(_fmt)
logger.addHandler(logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"))
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

def log(msg=""):
    logger.info(msg)

# ============================================================
# VLM 客户端
# ============================================================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

AGENT_EXECUTE_PROMPT = """\
你是一个多模态视觉推理 Agent。根据图片和用户问题，通过调用工具逐步推理，最终给出简洁的唯一答案。

## 可用工具
- web_search：文本网络搜索
- image_search：图像搜索（文本搜图或以图搜图）
- visit：访问指定 URL 并提取正文
- code_interpreter：在沙箱中执行 Python 代码（可用 PIL、NumPy、OpenCV）

## 重要：优先使用 code_interpreter
遇到以下情况，必须首先使用 code_interpreter，不要跳过：
- 图片中有文字、Logo、型号需要识别 → 先裁剪放大局部再搜索，效果远好于直接搜原图
- 需要对图片做任何处理（裁剪、旋转、增强对比度、测量尺寸）→ 用 PIL/OpenCV
- 涉及数值计算（价格换算、面积计算、日期推算）→ 用 Python 计算，不要心算
- 需要从多个工具结果中整合数据 → 用代码处理字符串和数值

## 调用格式（每次只能调一个）
<tool_call>
{"tool": "web_search", "args": {"query": "..."}}
</tool_call>

<tool_call>
{"tool": "image_search", "args": {"search_type": "text", "query": "...", "max_results": 5}}
</tool_call>

<tool_call>
{"tool": "image_search", "args": {"search_type": "reverse", "image_b64": "__CURRENT_IMAGE__", "max_results": 5}}
</tool_call>

<tool_call>
{"tool": "visit", "args": {"url": "https://..."}}
</tool_call>

<tool_call>
{"tool": "code_interpreter", "args": {"code": "print(1+1)"}}
</tool_call>

## code_interpreter 使用须知
- 沙箱中原始图片已预加载，路径固定为：/home/user/original_image.jpg
- 用 PIL 读取：original_image = Image.open('/home/user/original_image.jpg')
- 必须用 print() 输出结果，否则无法获取返回值
- 裁剪后的图片保存到 /home/user/ 目录下，可供后续步骤使用

## 输出答案
<answer>最终答案（数字/短语/日期/单句结论）</answer>
"""

# ============================================================
# 工具实现
# ============================================================

def tool_web_search(query: str, max_results: int = 5) -> str:
    if not SERPER_KEY:
        return "[web_search] 未配置 SERPER_KEY"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            timeout=20,
        )
        data = resp.json()
        parts = []
        if data.get("answerBox", {}).get("answer"):
            parts.append(f"[摘要] {data['answerBox']['answer']}")
        for r in data.get("organic", []):
            parts.append(f"[{r['title']}] {r['link']}\n{r.get('snippet', '')[:300]}")
        return "\n\n".join(parts)[:TOOL_OUTPUT_LIMIT]
    except Exception as e:
        return f"[web_search 异常] {e}"


def tool_image_search(
    search_type: str = "text",
    query: str = "",
    image_b64: str = "",
    max_results: int = 5,
) -> str:
    if not SERPER_KEY:
        return "[image_search] 未配置 SERPER_KEY"
    try:
        if search_type == "reverse":
            # Google Lens：先把图片上传到 tmpfiles 拿到公网 URL，再传给 Serper
            raw_b64 = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
            img_bytes = base64.b64decode(raw_b64)

            upload_resp = requests.post(
                "https://tmpfiles.org/api/v1/upload",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                timeout=30,
            )
            upload_data = upload_resp.json()
            page_url = upload_data["data"]["url"]
            direct_url = page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

            resp = requests.post(
                "https://google.serper.dev/lens",
                json={"url": direct_url},
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                timeout=30,
            )
            data = resp.json()
            items = data.get("visual_matches", [])[:max_results]
        else:
            resp = requests.post(
                "https://google.serper.dev/images",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                timeout=20,
            )
            data = resp.json()
            items = data.get("images", [])[:max_results]

        if not resp.text.strip():
            return "[image_search] 接口返回空响应"
        lines = [f"- {it.get('title', '')} | {it.get('link', it.get('imageUrl', ''))}" for it in items]
        return "\n".join(lines) if lines else "[image_search] 无结果"
    except Exception as e:
        return f"[image_search 异常] {e}"


def tool_visit(url: str) -> str:
    try:
        resp = requests.get(
            url, timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        return text[:TOOL_OUTPUT_LIMIT]
    except Exception as e:
        return f"[visit 异常] {e}"


def tool_code_interpreter(code: str, image_b64: str = "") -> str:
    if E2B_API_KEY:
        return _e2b_run(code, image_b64)
    return _local_exec_fallback(code)


def _e2b_run(code: str, image_b64: str = "") -> str:
    sbx = None
    try:
        from e2b_code_interpreter import Sandbox
        # v2.x：通过环境变量传 key，直接实例化，不用 context manager
        os.environ["E2B_API_KEY"] = E2B_API_KEY
        sbx = Sandbox.create()

        if image_b64 and ("original_image" in code or "img" in code.lower()):
            raw = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
            sbx.files.write("/home/user/original_image.jpg", base64.b64decode(raw))
            preamble = (
                "from PIL import Image\n"
                "original_image = Image.open('/home/user/original_image.jpg')\n\n"
            )
            code = preamble + code

        result = sbx.run_code(code)
        parts = []
        if result.logs.stdout:
            parts.append("".join(result.logs.stdout))
        if result.logs.stderr:
            parts.append("[stderr] " + "".join(result.logs.stderr))
        for i, artifact in enumerate(result.results):
            if hasattr(artifact, "png") and artifact.png:
                parts.append(f"[图片输出 {i}] data:image/png;base64,{artifact.png[:80]}...")
        return "\n".join(parts)[:TOOL_OUTPUT_LIMIT] or "[无输出]"
    except ImportError:
        return "[code_interpreter] 请安装: pip install e2b-code-interpreter"
    except Exception as e:
        return f"[code_interpreter E2B 异常] {e}"
    finally:
        if sbx:
            try:
                sbx.kill()
            except Exception:
                pass


def _local_exec_fallback(code: str) -> str:
    buf = io.StringIO()
    local_vars: dict = {}
    try:
        import contextlib
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, local_vars)  # noqa: S102
    except Exception as e:
        return f"[本地执行异常] {e}"
    output = buf.getvalue() or str(local_vars.get("result", ""))
    return output[:TOOL_OUTPUT_LIMIT] or "[无输出]"


# ============================================================
# 工具分发
# ============================================================

def dispatch_tool(tool_name: str, args: dict, image_b64: str) -> str:
    for k, v in args.items():
        if isinstance(v, str) and v == "__CURRENT_IMAGE__":
            args[k] = image_b64
    dispatch = {
        "web_search":       lambda: tool_web_search(**args),
        "image_search":     lambda: tool_image_search(**args),
        "visit":            lambda: tool_visit(**args),
        "code_interpreter": lambda: tool_code_interpreter(image_b64=image_b64, **args),
    }
    fn = dispatch.get(tool_name)
    return fn() if fn else f"[未知工具] {tool_name}"


# ============================================================
# Agent 执行循环
# ============================================================

def run_agent(image_b64: str, query: str) -> dict:
    """多轮 Agent 循环，返回 {answer, turns, trajectory}。"""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_b64}},
            {"type": "text", "text": query},
        ],
    }]
    trajectory = []

    for turn in range(1, AGENT_MAX_TURNS + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": AGENT_EXECUTE_PROMPT}] + messages,
                max_tokens=2048,
                temperature=0.2,
                timeout=120,
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            log(f"      [Turn {turn}] VLM 调用失败: {e}")
            break

        messages.append({"role": "assistant", "content": reply})
        trajectory.append({"role": "assistant", "content": reply, "turn": turn})
        log(f"      [Turn {turn}] {reply[:120].replace(chr(10), ' ')}")

        # 检查最终答案
        m = re.search(r"<answer>(.*?)</answer>", reply, re.DOTALL)
        if m:
            answer = m.group(1).strip()
            log(f"      → 答案: {answer!r}（{turn} 轮）")
            return {"answer": answer, "turns": turn, "trajectory": trajectory}

        # 提取工具调用
        m = re.search(r"<tool_call>(.*?)</tool_call>", reply, re.DOTALL)
        if not m:
            log(f"      [Turn {turn}] 无工具调用也无答案，终止")
            break

        try:
            call_obj = json.loads(m.group(1))
            tool_name = call_obj["tool"]
            tool_args = call_obj.get("args", {})
            log(f"      [工具] {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:80]})")
            result = dispatch_tool(tool_name, tool_args, image_b64)
        except Exception as e:
            result = f"[工具解析失败] {e}"

        log(f"      [结果] {result[:100].replace(chr(10), ' ')}")
        messages.append({"role": "user", "content": f"[{tool_name} 结果]\n{result}"})
        trajectory.append({"role": "tool", "tool": tool_name, "content": result, "turn": turn})

    return {"answer": "", "turns": turn, "trajectory": trajectory}


# ============================================================
# 主逻辑：读 JSONL → 补充 gold_answers → 写新文件
# ============================================================

def load_done_image_paths(output_file: str) -> set:
    """读取已完成的 image_path||query 联合 key，用于断点续跑。"""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(f"{rec['image_path']}||{rec['query']}")
            except Exception:
                pass
    return done


def image_path_to_b64(image_path: str) -> str:
    """读取本地图片文件，转为 data URI base64。"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=INPUT_FILE,  help="输入 JSONL 文件")
    parser.add_argument("--output", default=OUTPUT_FILE, help="输出 JSONL 文件")
    parser.add_argument("--resume", action="store_true",  help="跳过已有答案的条目")
    args = parser.parse_args()

    log("=" * 60)
    log(f"输入: {args.input}  输出: {args.output}  断点续跑: {args.resume}")
    log(f"Serper: {'✓' if SERPER_KEY else '✗'}  "
        f"E2B: {'✓' if E2B_API_KEY else '✗'}")
    log("=" * 60)

    done_paths = load_done_image_paths(args.output) if args.resume else set()
    if done_paths:
        log(f"断点续跑：已完成 {len(done_paths)} 条，跳过")

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "a", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log(f"[行 {line_no}] JSON 解析失败，跳过")
                continue

            image_path = record.get("image_path", "")
            query      = record.get("query", "")

            # 断点续跑：用 image_path + query 联合去重
            done_key = f"{image_path}||{query}"
            if done_key in done_paths:
                log(f"[行 {line_no}] 已完成，跳过")
                continue

            log(f"[行 {line_no}] {image_path} | {query[:60]}...")

            # 读取图片
            if not os.path.exists(image_path):
                log(f"  图片不存在，跳过: {image_path}")
                continue
            try:
                image_b64 = image_path_to_b64(image_path)
            except Exception as e:
                log(f"  图片读取失败: {e}，跳过")
                continue

            # 执行 Agent 循环
            try:
                result = run_agent(image_b64, query)
            except Exception:
                log(f"  Agent 异常:\n{traceback.format_exc()}")
                result = {"answer": "", "turns": 0, "trajectory": []}

            log(f"  答案: {result['answer']!r}（{result['turns']} 轮）")

            # 写出：保留原有所有字段，追加 answer / turns / trajectory
            out_record = {
                **record,
                "answer":     result["answer"],
                "turns":      result["turns"],
                "trajectory": result["trajectory"],
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            fout.flush()

    log("=" * 60)
    log(f"全部完成！结果写入: {args.output}")
    log("=" * 60)


if __name__ == "__main__":
    main()
