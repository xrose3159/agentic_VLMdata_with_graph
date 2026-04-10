from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from step3_graphenv_runtime import (
    BeamProgramProposer,
    GraphEnv,
    ProgramVerifier,
    QuestionProgram,
    SearchState,
    _apply_editor_enrichment,
    _pairwise_rerank,
    _program_anchor_signature,
    _program_answer_signature,
)


@dataclass
class PolicyAction:
    action_id: str
    state_id: str
    action_type: str
    rationale: str
    score: float
    probability: float = 0.0
    terminal_program: QuestionProgram | None = None
    next_state: SearchState | None = None


class StochasticPolicyProposer(BeamProgramProposer):
    """
    实验版 proposer：
    - 保留 typed action / GraphEnv / 现有 program 构造逻辑
    - 把 deterministic beam 扩展替换成 think -> action -> new_state 的概率搜索
    """

    def __init__(
        self,
        env: GraphEnv,
        *,
        beam_size: int = 24,
        max_depth: int = 3,
        seed: int = 7,
        temperature: float = 0.8,
        top_p: float = 0.88,
        deterministic_keep: int = 1,
        sampled_keep: int = 2,
    ):
        super().__init__(env, beam_size=beam_size, max_depth=max_depth)
        self.rng = random.Random(seed)
        self.temperature = max(0.2, temperature)
        self.top_p = min(max(top_p, 0.5), 1.0)
        self.deterministic_keep = max(1, deterministic_keep)
        self.sampled_keep = max(1, sampled_keep)
        self.action_counter = 0

    def next_action_id(self) -> str:
        self.action_counter += 1
        return f"act_{self.action_counter:04d}"

    def _state_signature(self, state: SearchState) -> tuple:
        proof_sig = tuple(
            sorted((edge.get("src"), edge.get("dst"), edge.get("relation")) for edge in state.proof_edges)
        )
        return (
            state.kind,
            tuple(sorted(state.anchors)),
            tuple(op.get("op") for op in state.operations),
            proof_sig,
            tuple(sorted(state.frontier)),
        )

    def _weighted_sample_without_replacement(self, items: list[tuple[float, Any]], k: int) -> list[Any]:
        pool = list(items)
        chosen: list[Any] = []
        for _ in range(min(k, len(pool))):
            total = sum(max(weight, 0.0) for weight, _ in pool)
            if total <= 0:
                idx = self.rng.randrange(len(pool))
                chosen.append(pool.pop(idx)[1])
                continue
            threshold = self.rng.random() * total
            running = 0.0
            picked_idx = 0
            for idx, (weight, _) in enumerate(pool):
                running += max(weight, 0.0)
                if running >= threshold:
                    picked_idx = idx
                    break
            chosen.append(pool.pop(picked_idx)[1])
        return chosen

    def _softmax_probabilities(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        pivot = max(scores)
        exps = [math.exp((score - pivot) / self.temperature) for score in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0 / len(scores)] * len(scores)
        return [value / total for value in exps]

    def _sample_seed_states(self, states: list[SearchState]) -> list[SearchState]:
        grouped: dict[str, list[SearchState]] = defaultdict(list)
        for state in states:
            grouped[state.kind].append(state)

        selected: list[SearchState] = []
        seen = set()
        for kind, quota in self.seed_kind_quota.items():
            bucket = sorted(grouped.get(kind, []), key=lambda s: s.score, reverse=True)
            if not bucket:
                continue
            keep_top = bucket[: min(2, len(bucket))]
            for state in keep_top:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)

            remainder = bucket[len(keep_top):]
            weighted = [(max(state.score, 0.1), state) for state in remainder]
            sampled = self._weighted_sample_without_replacement(weighted, max(0, quota - len(keep_top)))
            for state in sampled:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)

        if len(selected) < self.beam_size:
            remainder = sorted(states, key=lambda s: s.score, reverse=True)
            for state in remainder:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)
                if len(selected) >= self.beam_size:
                    break
        return selected[: self.beam_size]

    def _state_priority(self, state: SearchState) -> float:
        op_bonus = {
            "followed": 0.3,
            "compared": 0.9,
            "joined": 0.8,
            "triple": 0.6,
            "pair": 0.5,
            "single": 0.2,
        }.get(state.kind, 0.0)
        return state.score + op_bonus

    def _trim_next_beam(self, states: list[SearchState]) -> list[SearchState]:
        if not states:
            return []
        grouped: dict[str, list[SearchState]] = defaultdict(list)
        for state in states:
            grouped[state.kind].append(state)

        selected: list[SearchState] = []
        seen = set()
        for kind, quota in self.next_kind_quota.items():
            bucket = sorted(grouped.get(kind, []), key=self._state_priority, reverse=True)
            if not bucket:
                continue
            keep_top = bucket[:1]
            for state in keep_top:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)

            remainder = bucket[1:]
            weighted = [(max(self._state_priority(state), 0.1), state) for state in remainder]
            sampled = self._weighted_sample_without_replacement(weighted, max(0, quota - len(keep_top)))
            for state in sampled:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)

        if len(selected) < self.beam_size:
            remainder = sorted(states, key=self._state_priority, reverse=True)
            for state in remainder:
                signature = self._state_signature(state)
                if signature in seen:
                    continue
                selected.append(state)
                seen.add(signature)
                if len(selected) >= self.beam_size:
                    break
        return selected[: self.beam_size]

    def think(self, state: SearchState) -> list[PolicyAction]:
        terminal, expanded = self.expand_state(state)
        actions: list[PolicyAction] = []

        for program in terminal:
            emit_penalty = {
                "lookup": -1.6,
                "compare": -0.3,
                "delta": -0.2,
                "same_target": -0.1,
                "same_group": -0.1,
                "extremum": 0.1,
                "threshold": 0.2,
                "multihop": 0.5,
                "select_then_follow": 0.8,
                "join_then_follow": 0.8,
            }.get(program.family, 0.0)
            score = (
                program.scores.get("overall", 0.0)
                + program.scores.get("relation_value", 0.0)
                + emit_penalty
            )
            actions.append(
                PolicyAction(
                    action_id=self.next_action_id(),
                    state_id=state.state_id,
                    action_type=f"emit:{program.family}",
                    rationale=f"emit {program.family} program with target {program.target_type}",
                    score=score,
                    terminal_program=program,
                )
            )

        for next_state in expanded:
            follow_bonus = {
                "followed": 0.35,
                "compared": 2.0,
                "joined": 1.8,
                "triple": 0.5,
            }.get(next_state.kind, 0.2)
            actions.append(
                PolicyAction(
                    action_id=self.next_action_id(),
                    state_id=state.state_id,
                    action_type=f"transition:{next_state.kind}",
                    rationale=f"transition {state.kind} -> {next_state.kind}",
                    score=next_state.score + follow_bonus,
                    next_state=next_state,
                )
            )

        if not actions:
            return []

        scores = [action.score for action in actions]
        probs = self._softmax_probabilities(scores)
        for action, prob in zip(actions, probs):
            action.probability = prob
        return sorted(actions, key=lambda item: item.score, reverse=True)

    def _top_p_candidates(self, actions: list[PolicyAction]) -> list[PolicyAction]:
        cumulative = 0.0
        selected: list[PolicyAction] = []
        for action in actions:
            selected.append(action)
            cumulative += action.probability
            if cumulative >= self.top_p:
                break
        return selected or actions

    def _sample_actions(self, actions: list[PolicyAction]) -> list[PolicyAction]:
        if not actions:
            return []
        kept: list[PolicyAction] = []
        seen = set()

        emit_actions = [action for action in actions if action.terminal_program is not None]
        transition_actions = [action for action in actions if action.next_state is not None]

        deterministic = []
        deterministic.extend(actions[: self.deterministic_keep])
        if transition_actions:
            deterministic.append(transition_actions[0])
        if emit_actions:
            deterministic.append(emit_actions[0])

        for action in deterministic:
            if action.action_id in seen:
                continue
            kept.append(action)
            seen.add(action.action_id)

        candidate_pool = self._top_p_candidates(actions)
        weighted = [(action.probability, action) for action in candidate_pool if action.action_id not in seen]
        sampled = self._weighted_sample_without_replacement(weighted, self.sampled_keep)
        for action in sampled:
            if action.action_id in seen:
                continue
            kept.append(action)
            seen.add(action.action_id)

        return kept

    def step(self, action: PolicyAction) -> tuple[QuestionProgram | None, SearchState | None]:
        return action.terminal_program, action.next_state

    def propose(self) -> tuple[list[QuestionProgram], dict]:
        programs: list[QuestionProgram] = []
        traces: list[dict] = []
        beam = self._sample_seed_states(self.seed_states())

        for depth in range(self.max_depth):
            next_beam: list[SearchState] = []
            for state in beam:
                actions = self.think(state)
                chosen = self._sample_actions(actions)
                traces.append(
                    {
                        "depth": depth,
                        "state_id": state.state_id,
                        "state_kind": state.kind,
                        "anchors": list(state.anchors),
                        "candidate_actions": [
                            {
                                "action_id": action.action_id,
                                "action_type": action.action_type,
                                "probability": round(action.probability, 4),
                                "score": round(action.score, 4),
                                "rationale": action.rationale,
                            }
                            for action in actions[:8]
                        ],
                        "chosen_actions": [action.action_id for action in chosen],
                    }
                )
                for action in chosen:
                    program, next_state = self.step(action)
                    if program is not None:
                        programs.append(program)
                    if next_state is not None:
                        next_beam.append(next_state)
            if not next_beam:
                break
            beam = self._trim_next_beam(next_beam)

        deduped: dict[tuple, QuestionProgram] = {}
        for program in programs:
            signature = (
                program.family,
                tuple(program.anchors),
                program.answer_value,
                tuple((edge.get("src"), edge.get("dst"), edge.get("relation")) for edge in program.proof_graph),
            )
            existing = deduped.get(signature)
            if existing is None or existing.scores.get("overall", 0.0) < program.scores.get("overall", 0.0):
                deduped[signature] = program
        return list(deduped.values()), {"search_traces": traces}


def build_question_programs_stochastic(
    G: nx.DiGraph,
    nodes: dict,
    *,
    seed: int = 7,
    beam_size: int = 24,
    max_depth: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.88,
) -> tuple[list[QuestionProgram], dict]:
    env = GraphEnv(G, nodes)
    proposer = StochasticPolicyProposer(
        env,
        beam_size=beam_size,
        max_depth=max_depth,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )
    programs, search_meta = proposer.propose()

    verifier = ProgramVerifier(env)
    accepted: list[QuestionProgram] = []
    rejected: list[dict] = []
    for program in programs:
        ok, reason = verifier.verify(program)
        if ok:
            accepted.append(_apply_editor_enrichment(program, env))
        else:
            rejected.append({"program_id": program.program_id, "family": program.family, "reason": reason})

    accepted = _pairwise_rerank(accepted, env)
    meta = {
        "raw_program_count": len(programs),
        "accepted_program_count": len(accepted),
        "rejected_program_count": len(rejected),
        "family_counts": dict(_count_by_family(accepted)),
        "top_programs": [
            {
                "program_id": p.program_id,
                "family": p.family,
                "utility": p.scores.get("utility", 0.0),
                "selection_score": p.scores.get("selection_score", 0.0),
                "anchors": list(p.anchors),
                "answer": p.answer_value,
            }
            for p in accepted[:12]
        ],
        "rejected_samples": rejected[:20],
        **search_meta,
    }
    return accepted, meta


def summarize_programs(programs: list[QuestionProgram], env: GraphEnv) -> dict:
    family_counts = defaultdict(int)
    anchor_counts = defaultdict(int)
    for program in programs:
        family_counts[program.family] += 1
        anchor_counts[_program_anchor_signature(program, env)] += 1
    return {
        "count": len(programs),
        "family_counts": dict(family_counts),
        "distinct_anchor_groups": len(anchor_counts),
        "answers": [_program_answer_signature(p) for p in programs[:10]],
    }


def _count_by_family(programs: list[QuestionProgram]) -> defaultdict[str, int]:
    counts = defaultdict(int)
    for program in programs:
        counts[program.family] += 1
    return counts
