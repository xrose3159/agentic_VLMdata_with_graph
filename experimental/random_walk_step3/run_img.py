from __future__ import annotations

import argparse
import json

from core.image_utils import file_to_b64
from experimental.random_walk_step3.runtime import (
    build_randomwalk_shadow_result,
    load_graph_from_entity_file,
)


def _parse_triplet(value: str, cast=float) -> tuple:
    parts = [cast(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, e.g. 0.45,0.35,0.20")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_file")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--walks-per-anchor", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--length-weights", type=lambda s: _parse_triplet(s, float), default=(0.45, 0.35, 0.20))
    parser.add_argument("--length-quota", type=lambda s: _parse_triplet(s, int), default=(2, 2, 1))
    parser.add_argument("--utility-threshold", type=float, default=4.8)
    parser.add_argument("--long-boost-walks", type=int, default=16)
    parser.add_argument("--long-boost-temperature", type=float, default=1.15)
    parser.add_argument("--long-boost-top-p", type=float, default=0.95)
    parser.add_argument("--long-boost-epsilon", type=float, default=0.2)
    parser.add_argument("--realize", action="store_true")
    parser.add_argument("--max-realize", type=int, default=5)
    parser.add_argument("--best-of-n", type=int, default=3)
    args = parser.parse_args()

    G, nodes, data = load_graph_from_entity_file(args.entity_file)
    shadow = build_randomwalk_shadow_result(
        G,
        nodes,
        file_to_b64(data.get("img_path")) if args.realize and data.get("img_path") else "",
        data.get("image_description", ""),
        data.get("domain", ""),
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
        max_questions=args.max_realize,
        best_of_n=args.best_of_n,
        do_realize=args.realize,
    )

    report = {
        "entity_file": args.entity_file,
        "candidate_count": shadow["metadata"].get("deduped_candidate_count", 0),
        "filtered_candidate_count": shadow["metadata"].get("critic_selected_count", 0),
        "meta": shadow["metadata"].get("search_meta", {}),
        "critic_meta": shadow["metadata"].get("critic_meta", {}),
        "shadow_metadata": shadow["metadata"],
        "top_candidates": [],
    }
    selected_questions = shadow["level_2"] + shadow["level_3"]
    report["realized_questions"] = [
        {
            "level": "L3" if item.get("question_id", "").startswith("L3_") else "L2",
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "scores": item.get("scores", {}),
            "chain_summary": item.get("chain_summary", {}),
        }
        for item in selected_questions
    ]

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
