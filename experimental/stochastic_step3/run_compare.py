from __future__ import annotations

import argparse
import json
from pathlib import Path

from experimental.stochastic_step3.runtime import build_question_programs_stochastic, summarize_programs
from step3_generate import _build_nx_graph, _sanitize_triples
from step3_graphenv_runtime import GraphEnv, build_question_programs, select_programs_for_levels


def load_graph(entity_file: Path):
    data = json.loads(entity_file.read_text(encoding="utf-8"))
    triples = _sanitize_triples(data.get("triples", []))
    entities = data.get("high_conf", []) or data.get("entities", [])
    G, nodes = _build_nx_graph(triples, entities)
    return G, nodes


def format_program(program, env: GraphEnv) -> dict:
    return {
        "program_id": program.program_id,
        "family": program.family,
        "difficulty": program.difficulty,
        "anchors": [env.obfuscated_anchor(a) for a in program.anchors],
        "question_hint": program.surface_plan.get("microplans", [{}])[0].get("template_hint") or program.surface_plan,
        "answer": program.answer_value,
        "utility": program.scores.get("utility", 0.0),
        "selection_score": program.scores.get("selection_score", 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entity_file")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--beam-size", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    entity_file = Path(args.entity_file)
    G, nodes = load_graph(entity_file)
    env = GraphEnv(G, nodes)

    deterministic_programs, deterministic_meta = build_question_programs(G, nodes)
    det_l2, det_l3, det_level_meta = select_programs_for_levels(deterministic_programs, G, nodes)

    stochastic_programs, stochastic_meta = build_question_programs_stochastic(
        G,
        nodes,
        seed=args.seed,
        beam_size=args.beam_size,
        max_depth=args.max_depth,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    sto_l2, sto_l3, sto_level_meta = select_programs_for_levels(stochastic_programs, G, nodes)

    report = {
        "entity_file": str(entity_file),
        "deterministic": {
            "program_summary": summarize_programs(deterministic_programs, env),
            "meta": deterministic_meta,
            "level_meta": det_level_meta,
            "l2": [format_program(p, env) for p in det_l2],
            "l3": [format_program(p, env) for p in det_l3],
        },
        "stochastic": {
            "program_summary": summarize_programs(stochastic_programs, env),
            "meta": stochastic_meta,
            "level_meta": sto_level_meta,
            "l2": [format_program(p, env) for p in sto_l2],
            "l3": [format_program(p, env) for p in sto_l3],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
