from __future__ import annotations

import argparse
import json
import os

from core.image_utils import file_to_b64
from experimental.random_walk_step3.runtime import (
    build_randomwalk_shadow_result,
    load_graph_from_entity_file,
)
from step3_graphenv_runtime import (
    GraphEnv,
    build_question_programs,
    realize_programs,
    select_programs_for_levels,
)


def _parse_triplet(value: str, cast=float) -> tuple:
    parts = [cast(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(parts)


def _graphenv_questions(entity_file: str, *, max_l2: int, max_l3: int) -> dict:
    G, nodes, data = load_graph_from_entity_file(entity_file)
    env = GraphEnv(G, nodes)
    programs, program_meta = build_question_programs(G, nodes)
    l2_programs, l3_programs, level_meta = select_programs_for_levels(programs, G, nodes)
    image_b64 = file_to_b64(data["img_path"]) if data.get("img_path") else ""
    image_description = data.get("image_description", "")
    domain = data.get("domain", "")
    realized_l2 = realize_programs(l2_programs[:max_l2], env, image_b64, image_description, domain)
    realized_l3 = realize_programs(l3_programs[:max_l3], env, image_b64, image_description, domain)
    return {
        "program_meta": program_meta,
        "level_meta": level_meta,
        "questions": [
            {
                "level": "L2",
                "family": item.get("reasoning_path", {}).get("program", {}).get("family", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            for item in realized_l2
        ]
        + [
            {
                "level": "L3",
                "family": item.get("reasoning_path", {}).get("program", {}).get("family", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            for item in realized_l3
        ],
    }


def _random_walk_questions(
    entity_file: str,
    *,
    seed: int,
    walks_per_anchor: int,
    max_steps: int,
    temperature: float,
    top_p: float,
    epsilon: float,
    length_weights: tuple[float, float, float],
    length_quota: tuple[int, int, int],
    utility_threshold: float,
    long_boost_walks: int,
    long_boost_temperature: float,
    long_boost_top_p: float,
    long_boost_epsilon: float,
    max_realize: int,
    best_of_n: int,
) -> dict:
    G, nodes, data = load_graph_from_entity_file(entity_file)
    image_b64 = file_to_b64(data["img_path"]) if data.get("img_path") else ""
    shadow = build_randomwalk_shadow_result(
        G,
        nodes,
        image_b64,
        data.get("image_description", ""),
        data.get("domain", ""),
        seed=seed,
        max_steps=max_steps,
        walks_per_anchor=walks_per_anchor,
        temperature=temperature,
        top_p=top_p,
        epsilon=epsilon,
        length_weights=length_weights,
        length_quota=length_quota,
        utility_threshold=utility_threshold,
        long_boost_walks=long_boost_walks,
        long_boost_temperature=long_boost_temperature,
        long_boost_top_p=long_boost_top_p,
        long_boost_epsilon=long_boost_epsilon,
        max_questions=max_realize,
        best_of_n=best_of_n,
        do_realize=True,
    )
    realized = shadow.get("level_2", []) + shadow.get("level_3", [])
    return {
        "meta": shadow.get("metadata", {}).get("search_meta", {}),
        "critic_meta": shadow.get("metadata", {}).get("critic_meta", {}),
        "shadow_meta": shadow.get("metadata", {}),
        "questions": [
            {
                "level": "L3" if item.get("question_id", "").startswith("L3_") else "L2",
                "family": "random_walk_chain",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
            }
            for item in realized
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_file")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--walks-per-anchor", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--length-weights", type=lambda s: _parse_triplet(s, float), default=(0.45, 0.35, 0.20))
    parser.add_argument("--length-quota", type=lambda s: _parse_triplet(s, int), default=(2, 2, 1))
    parser.add_argument("--utility-threshold", type=float, default=4.8)
    parser.add_argument("--long-boost-walks", type=int, default=16)
    parser.add_argument("--long-boost-temperature", type=float, default=1.15)
    parser.add_argument("--long-boost-top-p", type=float, default=0.95)
    parser.add_argument("--long-boost-epsilon", type=float, default=0.2)
    parser.add_argument("--max-realize", type=int, default=5)
    parser.add_argument("--best-of-n", type=int, default=3)
    parser.add_argument("--max-graphenv-l2", type=int, default=3)
    parser.add_argument("--max-graphenv-l3", type=int, default=2)
    parser.add_argument("--shadow-file", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    graphenv = _graphenv_questions(
        args.entity_file,
        max_l2=args.max_graphenv_l2,
        max_l3=args.max_graphenv_l3,
    )
    if args.shadow_file and os.path.exists(args.shadow_file):
        with open(args.shadow_file, "r", encoding="utf-8") as f:
            shadow_data = json.load(f)
        random_walk = {
            "meta": shadow_data.get("metadata", {}).get("search_meta", {}),
            "critic_meta": shadow_data.get("metadata", {}).get("critic_meta", {}),
            "shadow_meta": shadow_data.get("metadata", {}),
            "questions": [
                {
                    "level": "L1",
                    "family": "random_walk_chain",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                }
                for item in shadow_data.get("level_1", [])
            ] + [
                {
                    "level": "L2",
                    "family": "random_walk_chain",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                }
                for item in shadow_data.get("level_2", [])
            ] + [
                {
                    "level": "L3",
                    "family": "random_walk_chain",
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                }
                for item in shadow_data.get("level_3", [])
            ],
        }
    else:
        random_walk = _random_walk_questions(
            args.entity_file,
            seed=args.seed,
            walks_per_anchor=args.walks_per_anchor,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            epsilon=args.epsilon,
            length_weights=args.length_weights,
            length_quota=args.length_quota,
            utility_threshold=args.utility_threshold,
            long_boost_walks=args.long_boost_walks,
            long_boost_temperature=args.long_boost_temperature,
            long_boost_top_p=args.long_boost_top_p,
            long_boost_epsilon=args.long_boost_epsilon,
            max_realize=args.max_realize,
            best_of_n=args.best_of_n,
        )

    report = {
        "entity_file": args.entity_file,
        "graphenv": graphenv,
        "random_walk": random_walk,
    }

    output = args.output
    if not output:
        img_id = os.path.splitext(os.path.basename(args.entity_file))[0]
        output = os.path.join("output", "questions", f"{img_id}_graphenv_vs_randomwalk.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
