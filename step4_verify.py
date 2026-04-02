"""
第四步：实体模糊化验证。

对第三步生成的问题进行模糊化质量校验，修正不合格的题目。

用法：
    python step4_verify.py
    python step4_verify.py --workers 4
"""

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import QUESTION_DIR, FINAL_DIR, STATS_DIR, VALID_TOOLS, MAX_WORKERS
import re
from core.vlm import call_vlm_json
from core.checkpoint import is_done, save_checkpoint, load_checkpoint
from core.logging_setup import get_logger

logger = get_logger("step4", "step4_verify.log")


# ============================================================
# Prompt：模糊化验证
# ============================================================
OBFUSCATION_VERIFY_PROMPT = """\
请检查以下问题集的实体模糊化是否正确执行。

## 检查规则

### 层级1：不应做模糊化
- web_search类题目中，实体名应直接出现在问题文本中
- visit类题目中，URL应直接提供
- 如果发现层级1的题被不必要地模糊化了，请还原

### 层级2：部分模糊化
- 需要code_interpreter作为第一步的题，被识别的实体名应已从问题文本中移除
- 移除后问题仍然清晰可理解（不能太含糊）
- 正确示例："图中标记为U3的芯片"（明确指向位置，但不透露型号）
- 错误示例："图中的某个芯片"（太含糊）

### 层级3：完全模糊化
- 问题文本中不应存在任何可直接搜索获得有用结果的实体名
- 模糊化表述应使用图中的位置标记来替代直接实体名

## 输入问题集
{questions_json}

## 输出格式
请输出严格的JSON格式（不要加 markdown 代码块标记）：
{{
    "verifications": [
        {{
            "question_id": "L2_01",
            "level": 2,
            "obfuscation_correct": true,
            "issues": [],
            "fixed_question": null
        }},
        {{
            "question_id": "L3_01",
            "level": 3,
            "obfuscation_correct": false,
            "issues": ["问题文本中仍包含'某实体名'，应替换为位置描述"],
            "fixed_question": "修正后的问题文本"
        }}
    ],
    "summary": {{
        "total_checked": 0,
        "passed": 0,
        "fixed": 0,
        "dropped": 0
    }}
}}"""


# ============================================================
# 工具名修正映射：将LLM常见的非法工具名映射到合法工具
# ============================================================
_TOOL_FIX_MAP = {
    "crop": "code_interpreter",
    "ocr": "code_interpreter",
    "OCR": "code_interpreter",
    "image_processing": "code_interpreter",
    "count_objects": "code_interpreter",
    "count": "code_interpreter",
    "color_detection": "code_interpreter",
    "color_analysis": "code_interpreter",
    "Color Analysis": "code_interpreter",
    "reverse_image_search": "image_search",
    "Image Search": "image_search",
}


# ============================================================
# 结构性验证（不需要 VLM）
# ============================================================
def structural_check(question_data: dict) -> tuple[list[str], int, dict]:
    """对单个问题文件做结构性检查。

    返回: (问题描述列表, 自动修复数, 应丢弃的question_id→原因映射)
    """
    issues = []
    auto_fixes = 0
    drop_ids = {}  # question_id → 丢弃原因

    for level_key in ["level_1", "level_2", "level_3"]:
        for q in question_data.get(level_key, []):
            qid = q.get("question_id", "?")
            seq = q.get("tool_sequence", [])

            # 自动修正无效工具名
            for step in seq:
                t = step.get("tool", "")
                if t not in VALID_TOOLS and t in _TOOL_FIX_MAP:
                    step["tool"] = _TOOL_FIX_MAP[t]
                    auto_fixes += 1
                elif t not in VALID_TOOLS:
                    issues.append(f"{qid}: 无效工具 '{t}'")

            # 合并连续相同工具步骤（修正后可能出现连续 code_interpreter）
            if len(seq) >= 2:
                deduped = [seq[0]]
                for s in seq[1:]:
                    if s.get("tool") == deduped[-1].get("tool"):
                        prev_action = deduped[-1].get("action", "")
                        curr_action = s.get("action", "")
                        if curr_action and curr_action not in prev_action:
                            deduped[-1]["action"] = f"{prev_action}; {curr_action}"
                        if s.get("expected_output"):
                            deduped[-1]["expected_output"] = s["expected_output"]
                    else:
                        deduped.append(s)
                if len(deduped) < len(seq):
                    q["tool_sequence"] = deduped
                    for i, s in enumerate(deduped):
                        s["step"] = i + 1

            # 检查必要字段
            if not q.get("question"):
                issues.append(f"{qid}: 缺少 question 字段")
                drop_ids[qid] = "缺少 question 字段"
                continue
            if not q.get("answer"):
                issues.append(f"{qid}: 缺少 answer 字段")
                drop_ids[qid] = "缺少 answer 字段"
                continue

            # 答案泄露检测：答案文本出现在问题中
            question_text = q.get("question", "")
            answer_text = str(q.get("answer", ""))
            if answer_text and len(answer_text) >= 2:
                if answer_text.lower() in question_text.lower():
                    reason = f"答案泄露 — '{answer_text}' 出现在问题文本中"
                    issues.append(f"{qid}: {reason}")
                    drop_ids[qid] = reason

            # 循环答案检测：答案与问题的关键短语重复
            if qid not in drop_ids and answer_text and len(answer_text) >= 4:
                answer_clean = re.sub(r'[的之]?(组成部分|关键组件|意思|含义)$', '', answer_text).strip()
                if answer_clean and len(answer_clean) >= 4 and answer_clean in question_text:
                    reason = f"循环答案 — '{answer_text}' 是问题文本的重述"
                    issues.append(f"{qid}: {reason}")
                    drop_ids[qid] = reason

    return issues, auto_fixes, drop_ids


# ============================================================
# 单张图片验证
# ============================================================
def verify_image_questions(q_file: str) -> dict | None:
    img_id = os.path.splitext(os.path.basename(q_file))[0]

    if is_done(4, img_id):
        logger.info(f"  [{img_id}] 已验证，跳过")
        return load_checkpoint(4, img_id)

    with open(q_file, encoding="utf-8") as f:
        question_data = json.load(f)

    logger.info(f"  [{img_id}] 开始验证...")

    # 结构性检查（含工具名自动修正）
    struct_issues, auto_fixes, drop_ids = structural_check(question_data)
    if auto_fixes:
        logger.info(f"  [{img_id}] 自动修正 {auto_fixes} 个无效工具名")
    if struct_issues:
        logger.warning(f"  [{img_id}] 结构性问题: {struct_issues}")

    # 丢弃有问题的题目，存入 rejected 目录
    rejected_questions = []
    if drop_ids:
        for level_key in ["level_1", "level_2", "level_3"]:
            kept = []
            for q in question_data.get(level_key, []):
                qid = q.get("question_id", "?")
                if qid in drop_ids:
                    q["reject_reason"] = drop_ids[qid]
                    rejected_questions.append(q)
                    logger.info(f"  [{img_id}] 丢弃 {qid}: {drop_ids[qid]}")
                else:
                    kept.append(q)
            question_data[level_key] = kept

        # 重新编号
        for level_key, prefix in [("level_1", "L1"), ("level_2", "L2"), ("level_3", "L3")]:
            for i, q in enumerate(question_data.get(level_key, [])):
                q["question_id"] = f"{prefix}_{i+1:02d}"

        # 保存被丢弃的题目
        if rejected_questions:
            rejected_dir = os.path.join(os.path.dirname(QUESTION_DIR), "rejected")
            os.makedirs(rejected_dir, exist_ok=True)
            rejected_path = os.path.join(rejected_dir, f"{img_id}.json")
            with open(rejected_path, "w", encoding="utf-8") as f:
                json.dump({"image_id": img_id, "rejected": rejected_questions}, f, ensure_ascii=False, indent=2)
            logger.info(f"  [{img_id}] {len(rejected_questions)} 道题已存入 {rejected_path}")

    # 更新 metadata
    question_data["metadata"] = {
        "total_questions": sum(len(question_data.get(k, [])) for k in ["level_1", "level_2", "level_3"]),
        "level_1_count": len(question_data.get("level_1", [])),
        "level_2_count": len(question_data.get("level_2", [])),
        "level_3_count": len(question_data.get("level_3", [])),
        "rejected_count": len(rejected_questions),
    }

    # VLM 模糊化验证（只检查 L2 和 L3）
    questions_to_check = []
    for q in question_data.get("level_2", []):
        questions_to_check.append({"question_id": q.get("question_id"), "level": 2, "question": q.get("question"), "obfuscation_applied": q.get("obfuscation_applied"), "obfuscated_entities": q.get("obfuscated_entities", [])})
    for q in question_data.get("level_3", []):
        questions_to_check.append({"question_id": q.get("question_id"), "level": 3, "question": q.get("question"), "obfuscation_applied": q.get("obfuscation_applied"), "obfuscated_entities": q.get("obfuscated_entities", [])})

    vlm_result = None
    if questions_to_check:
        vlm_result = call_vlm_json(
            OBFUSCATION_VERIFY_PROMPT.format(
                questions_json=json.dumps(questions_to_check, ensure_ascii=False, indent=2)
            ),
            "请检查并修正上述问题的模糊化。",
            max_tokens=4096,
            temperature=0.3,
        )

    # 应用修正
    fixes_applied = 0
    if vlm_result and "verifications" in vlm_result:
        fix_map = {}
        for v in vlm_result["verifications"]:
            if not v.get("obfuscation_correct") and v.get("fixed_question"):
                fix_map[v["question_id"]] = v["fixed_question"]
                fixes_applied += 1

        # 回写修正到 question_data
        for level_key in ["level_2", "level_3"]:
            for q in question_data.get(level_key, []):
                qid = q.get("question_id")
                if qid in fix_map:
                    q["question_original"] = q["question"]
                    q["question"] = fix_map[qid]
                    logger.info(f"  [{img_id}] {qid} 已修正模糊化")

    # 标记已验证
    for level_key in ["level_1", "level_2", "level_3"]:
        for q in question_data.get(level_key, []):
            q["verified"] = True

    # 回写文件
    with open(q_file, "w", encoding="utf-8") as f:
        json.dump(question_data, f, ensure_ascii=False, indent=2)

    result = {
        "img_id": img_id,
        "structural_issues": struct_issues,
        "tool_name_fixes": auto_fixes,
        "obfuscation_fixes": fixes_applied,
        "rejected_count": len(rejected_questions),
        "vlm_verification": vlm_result,
    }
    save_checkpoint(4, img_id, result)
    logger.info(f"  [{img_id}] 验证完成: 结构问题={len(struct_issues)} 工具修正={auto_fixes} 模糊化修正={fixes_applied}")
    return result


# ============================================================
# 重新聚合（验证后）
# ============================================================
def reaggregate():
    """验证后重新聚合最终 JSONL。"""
    from step3_generate import aggregate_final
    return aggregate_final()


# ============================================================
# 主流程
# ============================================================
def main(workers: int = MAX_WORKERS):
    logger.info("=" * 60)
    logger.info("第四步：模糊化验证")
    logger.info("=" * 60)

    q_files = sorted(glob.glob(os.path.join(QUESTION_DIR, "*.json")))
    logger.info(f"找到 {len(q_files)} 个问题文件")

    if not q_files:
        logger.error("没有找到问题文件，请先运行第三步")
        return

    # 并发验证
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(verify_image_questions, qf): qf for qf in q_files}
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                results.append(r)

    total_struct_issues = sum(len(r.get("structural_issues", [])) for r in results)
    total_tool_fixes = sum(r.get("tool_name_fixes", 0) for r in results)
    total_obf_fixes = sum(r.get("obfuscation_fixes", 0) for r in results)
    total_rejected = sum(r.get("rejected_count", 0) for r in results)
    logger.info(f"验证完成：{len(results)} 个文件, 结构问题={total_struct_issues}, 工具名修正={total_tool_fixes}, 模糊化修正={total_obf_fixes}, 丢弃={total_rejected}")

    # 写验证报告
    os.makedirs(STATS_DIR, exist_ok=True)
    report = {
        "total_files": len(q_files),
        "verified": len(results),
        "total_structural_issues": total_struct_issues,
        "total_tool_name_fixes": total_tool_fixes,
        "total_obfuscation_fixes": total_obf_fixes,
        "total_rejected": total_rejected,
        "details": results,
    }
    report_path = os.path.join(STATS_DIR, "verification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"验证报告: {report_path}")

    # 重新聚合
    logger.info("重新聚合最终数据...")
    stats = reaggregate()

    logger.info("=" * 60)
    logger.info(f"第四步完成！最终数据集: {stats['total_questions']} 道题")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第四步：模糊化验证")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    args = parser.parse_args()
    main(workers=args.workers)
