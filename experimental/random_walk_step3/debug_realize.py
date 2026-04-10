from __future__ import annotations

import argparse
import json

from core.image_utils import file_to_b64
from core.vlm import call_vlm_json
from experimental.random_walk_step3.runtime import (
    RandomWalkQuestionGenerator,
    filter_walk_candidates,
    load_graph_from_entity_file,
)
from step3_graphenv_runtime import GraphEnv, _ONTOLOGYESE_BANLIST, _ontologyese_penalty


def postcheck_reason(question: str, hidden_names: list[str]) -> str:
    if not question or question.count("?") + question.count("？") != 1:
        return "bad_question_mark_count"
    if _ontologyese_penalty(question) > 0:
        return "ontologyese_penalty"
    for phrase in _ONTOLOGYESE_BANLIST:
        if phrase in question:
            return f"banlist:{phrase}"
    for name in hidden_names:
        if name and name in question:
            return f"hidden_leak:{name}"
    return "pass"


def build_prompt(candidate, env: GraphEnv, image_description: str, domain: str) -> str:
    return (
        "你会拿到一条从图中实体出发、沿知识图随机探索得到的知识链。"
        "你的任务不是机械复述链条，而是基于这条链生成一条自然、值得问的中文问题。\n"
        "要求：\n"
        "- 问题必须可由这条知识链支撑。\n"
        "- 不要泄露中间隐藏节点名。\n"
        "- 要像真人会问的问题，不要用“对应的/实体/对象/相关信息/组织或机构是什么”。\n"
        "- 可以压缩链条，但不能改变链条蕴含的逻辑。\n"
        "- 优先把问题写成具体意图，比如“作曲者是谁”“位于哪座城市”“出生于哪一天”，不要写成泛问句。\n"
        "- 只输出一个单问号问题。\n"
        "输出严格 JSON：\n"
        "{\n"
        "  \"question\": \"...\",\n"
        "  \"answer\": \"...\",\n"
        "  \"why_question_works\": \"...\",\n"
        "  \"tool_sequence\": [{\"step\":1,\"tool\":\"...\",\"action\":\"...\",\"input\":\"...\",\"expected_output\":\"...\"}]\n"
        "}\n"
        f"图片描述：{image_description}\n"
        f"领域：{domain}\n"
        f"锚点：{candidate.chain_summary['anchor_text']}\n"
        f"补充锚点：{json.dumps(candidate.chain_summary['support_anchor_texts'], ensure_ascii=False)}\n"
        f"知识链：{json.dumps(candidate.chain_summary['chain_text'], ensure_ascii=False)}\n"
        f"最终答案：{candidate.answer_value}\n"
        f"隐藏节点：{json.dumps([env.node_name(node) for node in candidate.hidden_nodes], ensure_ascii=False)}\n"
        f"链条价值分：{json.dumps(candidate.scores, ensure_ascii=False)}\n"
        f"提示：问题头可以围绕“{candidate.chain_summary['question_head_hint']}”或“{candidate.chain_summary['question_tail_hint']}”组织，但不要机械照抄。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_file")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--walks-per-anchor", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    G, nodes, data = load_graph_from_entity_file(args.entity_file)
    env = GraphEnv(G, nodes)
    generator = RandomWalkQuestionGenerator(
        env,
        seed=args.seed,
        max_steps=args.max_steps,
        walks_per_anchor=args.walks_per_anchor,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    candidates, _ = generator.generate_candidates()
    filtered, critic_meta = filter_walk_candidates(candidates, env)
    image_b64 = file_to_b64(data["img_path"]) if data.get("img_path") else ""
    image_description = data.get("image_description", "")
    domain = data.get("domain", "")

    report = {
        "critic_selected_count": len(filtered),
        "critic_meta": critic_meta,
        "debug": [],
    }
    selected = filtered[args.start : args.start + args.top_k]
    for candidate in selected:
        prompt = build_prompt(candidate, env, image_description, domain)
        obj = call_vlm_json(
            prompt,
            "请根据给定知识链生成自然中文问题。",
            image_b64=image_b64,
            max_tokens=1600,
            temperature=0.8,
            max_attempts=1,
        )
        hidden_names = [env.node_name(node) for node in candidate.hidden_nodes]
        question = obj.get("question", "") if isinstance(obj, dict) else ""
        report["debug"].append(
            {
                "candidate_id": candidate.candidate_id,
                "answer": candidate.answer_value,
                "chain_text": candidate.chain_summary["chain_text"],
                "hidden_names": hidden_names,
                "raw_obj": obj,
                "postcheck_reason": postcheck_reason(question, hidden_names),
            }
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
