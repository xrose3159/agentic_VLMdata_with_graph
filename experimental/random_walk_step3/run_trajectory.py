"""CLI: 对单张图跑 trajectory-conditioned random walk 生题。

用法:
    python -m experimental.random_walk_step3.run_trajectory output/entities/img_0010.json
    python -m experimental.random_walk_step3.run_trajectory output/entities/1.json --realize --image try/1.jpg
"""

from __future__ import annotations

import argparse
import json
import sys

from .trajectory_runtime import (
    HeteroSolveGraph,
    available_schemas,
    generate_questions,
)


def main():
    parser = argparse.ArgumentParser(description="Trajectory-conditioned random walk question generator")
    parser.add_argument("entity_file", help="Step2 entity JSON file")
    parser.add_argument("--image", default=None, help="Image path for LLM realization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=0.7)
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--max-realize", type=int, default=6)
    parser.add_argument("--realize", action="store_true", help="Enable LLM realization (requires --image)")
    args = parser.parse_args()

    # 图统计
    with open(args.entity_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = HeteroSolveGraph(data)
    stats = graph.stats()
    print(f"图谱: {stats['nodes']} | 边: {stats['edges']}")
    print(f"Regions: {len(graph.regions())}")
    for rk in graph.regions():
        info = graph.region_info(rk)
        ek = graph.entity_for_region(rk)
        n_facts = len(graph.retrieve_edges(ek)) if ek else 0
        print(f"  {info.get('name', '?'):25s} ({info.get('entity_type', '?'):10s}) "
              f"score={graph.region_score(rk):.2f} facts={n_facts} "
              f"ref={graph.visual_descriptor(rk)}")

    schemas = available_schemas(graph)
    print(f"\n可行 Schema: {[s.name for s in schemas]}")

    # 生成
    image_path = args.image if args.realize else None
    result = generate_questions(
        args.entity_file,
        image_path=image_path,
        seed=args.seed,
        tau=args.tau,
        n_candidates=args.n_candidates,
        max_realize=args.max_realize,
    )

    meta = result.get("metadata", {})
    print(f"\n候选: {meta.get('candidate_count', 0)}, 通过检查: {meta.get('irreducibility_pass', 0)}, 最终: {meta.get('realized_count', 0)}")

    if meta.get("irreducibility_reject"):
        print(f"被拒 {len(meta['irreducibility_reject'])} 条:")
        for r in meta["irreducibility_reject"][:5]:
            print(f"  {r['schema']:30s} {r['reason']}")

    for level_key in ("level_1", "level_2", "level_3"):
        questions = result.get(level_key, [])
        if not questions:
            continue
        print(f"\n{'='*50}")
        print(f"  {level_key.upper()} ({len(questions)} 题)")
        print(f"{'='*50}")
        for q in questions:
            print(f"\n  [{q.get('question_id')}] schema={q.get('schema')} tool_depth={q.get('tool_depth', q.get('rationale', ''))}")
            if "question" in q:
                print(f"  Q: {q['question']}")
                print(f"  A: {q['answer']}")
            elif "frame" in q:
                f = q["frame"]
                print(f"  Frame: {f['operation']} | refs={f['visible_refs']} | criterion={f.get('criterion','')}")
                print(f"  Answer: {f['answer']} ({f['answer_type']})")

    if meta.get("realize_rejects"):
        print(f"\nRealize 被拒 {len(meta['realize_rejects'])} 条:")
        for r in meta["realize_rejects"][:5]:
            print(f"  {r.get('reason')} | {r.get('question', '')[:60]}")


if __name__ == "__main__":
    main()
