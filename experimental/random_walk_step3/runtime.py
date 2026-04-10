from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from core.vlm import call_vlm_json, call_vlm_json_with_meta
from step3_generate import _build_nx_graph, _sanitize_triples
from step3_graphenv_runtime import (
    GraphEnv,
    _ONTOLOGYESE_BANLIST,
    _answer_specificity_score,
    _generic_answer_penalty,
    _ontologyese_penalty,
    _path_follow_clause,
    _relation_natural_text,
)


TARGET_TYPE_BONUS = {
    "PERSON": 1.0,
    "ORG": 0.9,
    "LOCATION": 0.7,
    "TIME": 0.7,
    "QUANTITY": 0.7,
    "OTHER": 0.1,
}

LENGTH_MODE_SPECS = {
    "short": {"min_depth": 2, "max_depth": 3, "stop_bias": 0.9, "continue_bias": -0.2},
    "medium": {"min_depth": 3, "max_depth": 5, "stop_bias": 0.2, "continue_bias": 0.2},
    "long": {"min_depth": 5, "max_depth": 9, "stop_bias": -0.8, "continue_bias": 1.0},
}

BRIDGE_RELATION = "bridge_to_in_image_anchor"


@dataclass
class WalkState:
    state_id: str
    anchor: str
    current_node: str
    steps: list[dict]
    visited_nodes: tuple[str, ...]
    length_mode: str
    target_min_depth: int
    target_max_depth: int
    score_trace: list[dict] = field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.steps)


@dataclass
class WalkCandidate:
    candidate_id: str
    anchor: str
    answer_node: str
    answer_value: str
    chain_edges: list[dict]
    hidden_nodes: list[str]
    support_anchors: list[str]
    length_mode: str
    chain_depth: int
    scores: dict
    chain_summary: dict


@dataclass
class CritiquedWalkCandidate:
    candidate: WalkCandidate
    accepted: bool
    critic_scores: dict
    reject_reason: str = ""


def load_graph_from_entity_file(entity_file: str) -> tuple[nx.DiGraph, dict, dict]:
    with open(entity_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    triples = _sanitize_triples(data.get("triples", []))
    entities = data.get("high_conf", []) or data.get("entities", [])
    G, nodes = _build_nx_graph(triples, entities)
    return G, nodes, data


class RandomWalkQuestionGenerator:
    def __init__(
        self,
        env: GraphEnv,
        *,
        seed: int = 7,
        max_steps: int = 10,
        walks_per_anchor: int = 24,
        temperature: float = 0.9,
        top_p: float = 0.9,
        epsilon: float = 0.1,
        length_weights: tuple[float, float, float] = (0.45, 0.35, 0.20),
    ):
        self.env = env
        self.rng = random.Random(seed)
        self.max_steps = max(2, max_steps)
        self.walks_per_anchor = max(4, walks_per_anchor)
        self.temperature = max(0.2, temperature)
        self.top_p = min(max(top_p, 0.5), 1.0)
        self.epsilon = min(max(epsilon, 0.0), 0.5)
        self.length_weights = self._normalize_length_weights(length_weights)
        self.counter = 0
        self._incoming_anchor_edges = self._build_incoming_anchor_index()

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def _normalize_length_weights(self, weights: tuple[float, float, float]) -> dict[str, float]:
        short_w, medium_w, long_w = weights
        fixed = [max(0.0, short_w), max(0.0, medium_w), max(0.0, long_w)]
        total = sum(fixed)
        if total <= 0:
            fixed = [0.45, 0.35, 0.20]
            total = 1.0
        return {
            "short": fixed[0] / total,
            "medium": fixed[1] / total,
            "long": fixed[2] / total,
        }

    def _build_incoming_anchor_index(self) -> dict[str, list[dict]]:
        index: dict[str, list[dict]] = {}
        for anchor in self.env.in_image_nodes:
            for edge in self.env.unique_follows(anchor):
                index.setdefault(edge["dst"], []).append({
                    "anchor": anchor,
                    "edge": edge,
                })
        return index

    def _sample_length_mode(self, mode_override: str | None = None) -> str:
        if mode_override in LENGTH_MODE_SPECS:
            return mode_override
        keys = ["short", "medium", "long"]
        probs = [self.length_weights[k] for k in keys]
        threshold = self.rng.random()
        running = 0.0
        for key, prob in zip(keys, probs):
            running += prob
            if threshold <= running:
                return key
        return "long"

    def initial_state(self, anchor: str, *, mode: str) -> WalkState:
        spec = LENGTH_MODE_SPECS.get(mode, LENGTH_MODE_SPECS["medium"])
        return WalkState(
            state_id=self.next_id("state"),
            anchor=anchor,
            current_node=anchor,
            steps=[],
            visited_nodes=(anchor,),
            length_mode=mode,
            target_min_depth=int(spec["min_depth"]),
            target_max_depth=int(spec["max_depth"]),
            score_trace=[],
        )

    def _support_anchor_bonus(self, node: str, anchor: str) -> tuple[float, list[str]]:
        support = []
        for other in self.env.in_image_nodes:
            if other == anchor:
                continue
            if any(edge["dst"] == node for edge in self.env.unique_follows(other)):
                support.append(other)
        bonus = 0.35 * min(len(support), 2)
        return bonus, support

    def _natural_anchor_ref(self, node: str) -> str:
        return self.env.obfuscated_anchor(node)

    def _in_image_ratio_excluding_anchor(self, state: WalkState) -> float:
        if len(state.visited_nodes) <= 1:
            return 0.0
        trailing_nodes = state.visited_nodes[1:]
        in_image_count = sum(1 for node in trailing_nodes if node in self.env.in_image_nodes)
        return in_image_count / max(len(trailing_nodes), 1)

    def _state_support_count(self, state: WalkState) -> int:
        support = set()
        for step in state.steps:
            for anchor in step.get("support_anchors", []):
                support.add(anchor)
        return len(support)

    def _semantic_edges(self, state: WalkState) -> list[dict]:
        edges = [step["edge"] for step in state.steps]
        semantic = [edge for edge in edges if edge.get("relation") != BRIDGE_RELATION]
        return semantic or edges

    def _bridge_candidates(self, state: WalkState) -> list[dict]:
        # A bridge lets the walk pivot to another in-image anchor that also points to current_node.
        incoming = self._incoming_anchor_edges.get(state.current_node, [])
        if not incoming:
            return []
        last_node = state.steps[-1]["edge"]["src"] if state.steps else None
        options = []
        for item in incoming:
            anchor = item["anchor"]
            if anchor == state.anchor:
                continue
            if anchor == state.current_node:
                continue
            if anchor == last_node:
                continue
            visit_count = state.visited_nodes.count(anchor)
            if state.length_mode != "long" and visit_count > 0:
                continue
            if state.length_mode == "long" and visit_count >= 2:
                continue
            if not self.env.unique_follows(anchor):
                continue
            options.append(item)
        return options

    def _bridge_step_edge(self, state: WalkState, bridge_anchor: str, via_edge: dict) -> dict:
        return {
            "src": state.current_node,
            "dst": bridge_anchor,
            "relation": BRIDGE_RELATION,
            "tail_type": self.env.node_type(bridge_anchor),
            "normalized_value": "",
            "unit": "",
            "fact": via_edge.get("fact", ""),
            "source_snippet": via_edge.get("source_snippet", ""),
            "source": via_edge.get("source", ""),
            "evidence": 0.35 + 0.25 * min(via_edge.get("evidence", 0.0), 1.0),
            "relation_profile": {
                "askability": 0.7,
                "lexicalizability": 0.9,
            },
        }

    def _state_utility(self, state: WalkState) -> float:
        if not state.steps:
            return 0.0
        chain = self._semantic_edges(state)
        final_edge = chain[-1]
        target_type = final_edge["tail_type"]
        answer_text = self.env.node_name(state.current_node)
        evidence = sum(edge.get("evidence", 0.0) for edge in chain) / max(len(chain), 1)
        lexicalizability = sum(
            edge.get("relation_profile", {}).get("lexicalizability", 0.0)
            for edge in chain
        ) / max(len(chain), 1)
        askability = sum(
            edge.get("relation_profile", {}).get("askability", 0.0)
            for edge in chain
        ) / max(len(chain), 1)
        answer_specificity = max(0.0, _answer_specificity_score(answer_text, target_type))
        generic_penalty = _generic_answer_penalty(answer_text, target_type)
        bridge_bonus = 0.25 * min(self._state_support_count(state), 2)
        clue_minimality = 1.0 if state.depth >= 3 else 0.65 if state.depth == 2 else 0.2
        in_image_ratio = self._in_image_ratio_excluding_anchor(state)
        recoverability = 0.9
        if any(
            edge.get("relation") != BRIDGE_RELATION
            and _relation_natural_text(edge.get("relation", ""), edge.get("tail_type", "OTHER")).startswith("相关")
            for edge in chain
        ):
            recoverability -= 0.4
        return (
            1.1 * evidence
            + 1.0 * lexicalizability
            + 0.9 * askability
            + 0.8 * answer_specificity
            + bridge_bonus
            + 0.5 * clue_minimality
            + 0.25 * in_image_ratio
            + recoverability
            - generic_penalty
        )

    def _projected_utility(self, state: WalkState, edge: dict, support_anchors: list[str]) -> float:
        pseudo_step = {
            "edge": edge,
            "support_anchors": support_anchors,
            "support_bonus": 0.35 * min(len(support_anchors), 2),
        }
        projected = WalkState(
            state_id=state.state_id,
            anchor=state.anchor,
            current_node=edge["dst"],
            steps=state.steps + [pseudo_step],
            visited_nodes=state.visited_nodes + (edge["dst"],),
            length_mode=state.length_mode,
            target_min_depth=state.target_min_depth,
            target_max_depth=state.target_max_depth,
            score_trace=state.score_trace,
        )
        return self._state_utility(projected)

    def _can_follow_edge(self, state: WalkState, edge: dict) -> bool:
        dst = edge["dst"]
        if dst not in state.visited_nodes:
            return True
        if state.length_mode != "long":
            return False
        if dst == state.anchor:
            return False
        previous_node = state.steps[-1]["edge"]["src"] if state.steps else None
        if previous_node and dst == previous_node:
            return False
        return state.visited_nodes.count(dst) < 2

    def _future_branch_bonus(self, state: WalkState, edge: dict) -> float:
        next_node = edge["dst"]
        previous_node = state.current_node
        usable = 0
        for out_edge in self.env.unique_follows(next_node):
            dst2 = out_edge["dst"]
            if dst2 == previous_node:
                continue
            if dst2 in state.visited_nodes and state.length_mode != "long":
                continue
            usable += 1
        return min(1.0, 0.25 * usable)

    def think(self, state: WalkState) -> dict:
        outgoing = []
        current_utility = self._state_utility(state)
        for edge in self.env.unique_follows(state.current_node):
            if not self._can_follow_edge(state, edge):
                continue
            relation_profile = edge.get("relation_profile", {})
            askability = relation_profile.get("askability", 0.0)
            lexicalizability = relation_profile.get("lexicalizability", 0.0)
            target_type = edge["tail_type"]
            answer_text = self.env.node_name(edge["dst"])
            support_bonus, support_anchors = self._support_anchor_bonus(edge["dst"], state.anchor)
            revisit_penalty = 0.22 if edge["dst"] in state.visited_nodes else 0.0
            future_branch_bonus = self._future_branch_bonus(state, edge)
            projected_utility = self._projected_utility(state, edge, support_anchors)
            delta_utility = projected_utility - current_utility
            step_score = (
                1.5 * delta_utility
                + 0.7 * askability
                + 0.6 * lexicalizability
                + 0.5 * edge.get("evidence", 0.0)
                + 0.35 * TARGET_TYPE_BONUS.get(target_type, 0.0)
                + 0.25 * support_bonus
                + (0.15 if state.length_mode == "long" and support_anchors else 0.0)
                + (0.12 if state.length_mode == "long" and target_type in {"PERSON", "ORG", "LOCATION"} else 0.0)
                + (0.45 if state.length_mode == "long" else 0.18) * future_branch_bonus
                - 0.4 * _generic_answer_penalty(answer_text, target_type)
                - revisit_penalty
            )
            outgoing.append(
                {
                    "type": "walk",
                    "edge": edge,
                    "score": step_score,
                    "delta_utility": delta_utility,
                    "projected_utility": projected_utility,
                    "support_anchors": support_anchors,
                    "support_bonus": support_bonus,
                    "future_branch_bonus": future_branch_bonus,
                    "revisit_penalty": revisit_penalty,
                }
            )

        for bridge in self._bridge_candidates(state):
            bridge_anchor = bridge["anchor"]
            via_edge = bridge["edge"]
            edge = self._bridge_step_edge(state, bridge_anchor, via_edge)
            support_anchors = [bridge_anchor]
            projected_utility = self._projected_utility(state, edge, support_anchors)
            delta_utility = projected_utility - current_utility
            anchor_out = len(self.env.unique_follows(bridge_anchor))
            bridge_score = (
                1.25 * delta_utility
                + 0.55
                + 0.12 * min(anchor_out, 6)
                + (0.22 if state.length_mode == "long" else 0.08)
                + (0.18 if bridge_anchor not in state.visited_nodes else -0.18)
            )
            outgoing.append(
                {
                    "type": "bridge",
                    "edge": edge,
                    "score": bridge_score,
                    "delta_utility": delta_utility,
                    "projected_utility": projected_utility,
                    "support_anchors": support_anchors,
                    "support_bonus": 0.25,
                    "future_branch_bonus": 0.0,
                    "revisit_penalty": 0.0,
                    "bridge_anchor": bridge_anchor,
                    "via_relation": via_edge.get("relation", ""),
                }
            )

        outgoing.sort(key=lambda item: item["score"], reverse=True)

        if not outgoing:
            return {
                "state": state,
                "gate": {"continue_probability": 0.0, "stop_probability": 1.0},
                "actions": [{"type": "stop", "probability": 1.0, "score": 0.0}],
                "current_utility": current_utility,
            }

        in_image_ratio = self._in_image_ratio_excluding_anchor(state)
        best_delta = max(item["delta_utility"] for item in outgoing)
        mean_delta = sum(item["delta_utility"] for item in outgoing) / max(len(outgoing), 1)
        mode_spec = LENGTH_MODE_SPECS.get(state.length_mode, LENGTH_MODE_SPECS["medium"])
        target_min = state.target_min_depth
        target_max = state.target_max_depth

        if state.length_mode == "long" and state.depth < 5:
            continue_prob, stop_prob = 1.0, 0.0
            continue_logit, stop_logit = 10.0, -10.0
        elif state.steps and state.steps[-1]["edge"].get("relation") == BRIDGE_RELATION:
            continue_prob, stop_prob = 0.95, 0.05
            continue_logit, stop_logit = 4.0, -1.0
        elif state.depth < target_min:
            continue_prob, stop_prob = 1.0, 0.0
            continue_logit, stop_logit = 8.0, -8.0
        elif state.depth >= target_max:
            continue_prob, stop_prob = 0.0, 1.0
            continue_logit, stop_logit = -8.0, 8.0
        else:
            stop_logit = (
                -1.4
                + float(mode_spec["stop_bias"])
                + 0.40 * max(current_utility, 0.0)
                + 0.45 * in_image_ratio
                + 0.15 * max(state.depth - target_min, 0)
            )
            continue_logit = (
                -0.2
                + float(mode_spec["continue_bias"])
                + 0.85 * max(best_delta, 0.0)
                + 0.35 * max(mean_delta, 0.0)
                + 0.20 * (1.0 - in_image_ratio)
            )
            gate_probs = self._softmax([continue_logit, stop_logit])
            continue_prob, stop_prob = gate_probs[0], gate_probs[1]

        walk_scores = [item["score"] for item in outgoing]
        walk_probs = self._softmax(walk_scores)
        actions = []
        for idx, item in enumerate(outgoing):
            actions.append(
                {
                    "type": item.get("type", "walk"),
                    "probability": continue_prob * walk_probs[idx],
                    "conditional_probability": walk_probs[idx],
                    "score": item["score"],
                    "delta_utility": item["delta_utility"],
                    "projected_utility": item["projected_utility"],
                    "edge": item["edge"],
                    "support_anchors": item["support_anchors"],
                    "support_bonus": item["support_bonus"],
                    "via_relation": item.get("via_relation", ""),
                    "bridge_anchor": item.get("bridge_anchor", ""),
                }
            )
        actions.append(
            {
                "type": "stop",
                "probability": stop_prob,
                "score": stop_logit,
            }
        )
        actions.sort(key=lambda item: item["probability"], reverse=True)
        return {
            "state": state,
            "gate": {
                "continue_probability": continue_prob,
                "stop_probability": stop_prob,
                "continue_logit": continue_logit,
                "stop_logit": stop_logit,
                "length_mode": state.length_mode,
                "target_min_depth": target_min,
                "target_max_depth": target_max,
            },
            "actions": actions,
            "current_utility": current_utility,
        }

    def _softmax(self, scores: list[float]) -> list[float]:
        pivot = max(scores) if scores else 0.0
        exps = [math.exp((score - pivot) / self.temperature) for score in scores]
        total = sum(exps) or 1.0
        return [value / total for value in exps]

    def _sample_action(self, action_bundle: dict) -> dict | None:
        actions = action_bundle["actions"]
        gate = action_bundle.get("gate", {})
        walk_actions = [action for action in actions if action["type"] in {"walk", "bridge"}]
        stop_action = next((action for action in actions if action["type"] == "stop"), None)
        if stop_action is None and not walk_actions:
            return None
        continue_prob = gate.get("continue_probability", 0.0)
        stop_prob = gate.get("stop_probability", 1.0)
        if not walk_actions:
            return stop_action
        if self.rng.random() < stop_prob / max(stop_prob + continue_prob, 1e-9):
            return stop_action

        ranked_walks = sorted(
            walk_actions,
            key=lambda item: item.get("conditional_probability", item["probability"]),
            reverse=True,
        )
        top_p_actions = []
        cumulative = 0.0
        for action in ranked_walks:
            top_p_actions.append(action)
            cumulative += action.get("conditional_probability", action["probability"])
            if cumulative >= self.top_p:
                break
        if not top_p_actions:
            top_p_actions = ranked_walks
        if top_p_actions and self.rng.random() < self.epsilon:
            return self.rng.choice(top_p_actions)
        total = sum(item.get("conditional_probability", item["probability"]) for item in top_p_actions) or 1.0
        threshold = self.rng.random() * total
        running = 0.0
        for action in top_p_actions:
            running += action.get("conditional_probability", action["probability"])
            if running >= threshold:
                return action
        return top_p_actions[-1]

    def act(self, state: WalkState, action: dict) -> WalkState | None:
        if action["type"] == "stop":
            return None
        edge = action["edge"]
        next_node = edge["dst"]
        return WalkState(
            state_id=self.next_id("state"),
            anchor=state.anchor,
            current_node=next_node,
            steps=state.steps + [action],
            visited_nodes=state.visited_nodes + (next_node,),
            length_mode=state.length_mode,
            target_min_depth=state.target_min_depth,
            target_max_depth=state.target_max_depth,
            score_trace=state.score_trace
            + [
                {
                    "at": state.current_node,
                    "relation": edge["relation"],
                    "to": next_node,
                    "score": round(action["score"], 4),
                    "probability": round(action["probability"], 4),
                }
            ],
        )

    def _candidate_from_state(self, state: WalkState) -> WalkCandidate | None:
        if state.depth < 2:
            return None
        if state.length_mode == "long" and state.depth < 5:
            return None
        if state.steps and state.steps[-1]["edge"].get("relation") == BRIDGE_RELATION:
            return None
        answer_node = state.current_node
        answer_value = self.env.node_name(answer_node)
        chain_for_scoring = self._semantic_edges(state)
        final_edge = chain_for_scoring[-1]
        if not answer_value or _generic_answer_penalty(answer_value, final_edge["tail_type"]) >= 1.2:
            return None

        support_anchors = []
        for step in state.steps:
            for support in step.get("support_anchors", []):
                if support not in support_anchors:
                    support_anchors.append(support)

        evidence = sum(edge.get("evidence", 0.0) for edge in chain_for_scoring) / max(len(chain_for_scoring), 1)
        lexicalizability = sum(
            edge.get("relation_profile", {}).get("lexicalizability", 0.0)
            for edge in chain_for_scoring
        ) / max(len(chain_for_scoring), 1)
        bridge_bonus = 0.25 * len(support_anchors)
        depth_bonus = 0.45 * min(state.depth, 4)
        overall = evidence + lexicalizability + bridge_bonus + depth_bonus + _answer_specificity_score(answer_value, final_edge["tail_type"])

        hidden_nodes = [step["edge"]["dst"] for step in state.steps[:-1]]
        chain_summary = {
            "anchor_text": self.env.obfuscated_anchor(state.anchor),
            "anchor_ref_text": self._natural_anchor_ref(state.anchor),
            "support_anchor_texts": [self.env.obfuscated_anchor(anchor) for anchor in support_anchors],
            "support_anchor_refs": [self._natural_anchor_ref(anchor) for anchor in support_anchors],
            "chain_text": [],
            "question_head_hint": _relation_natural_text(final_edge["relation"], final_edge["tail_type"]),
            "question_tail_hint": _path_follow_clause(final_edge["relation"], final_edge["tail_type"]),
        }
        for step in state.steps:
            edge = step["edge"]
            rel = edge.get("relation", "")
            if rel == BRIDGE_RELATION:
                relation_zh = "转到图中另一处相关实体"
            else:
                relation_zh = _relation_natural_text(rel, edge.get("tail_type", "OTHER"))
            chain_summary["chain_text"].append({
                "src": self.env.node_name(edge["src"]),
                "relation": rel,
                "relation_zh": relation_zh,
                "dst": self.env.node_name(edge["dst"]),
                "tail_type": edge["tail_type"],
            })
        return WalkCandidate(
            candidate_id=self.next_id("cand"),
            anchor=state.anchor,
            answer_node=answer_node,
            answer_value=answer_value,
            chain_edges=[step["edge"] for step in state.steps],
            hidden_nodes=hidden_nodes,
            support_anchors=support_anchors,
            length_mode=state.length_mode,
            chain_depth=state.depth,
            scores={
                "evidence": round(evidence, 3),
                "lexicalizability": round(lexicalizability, 3),
                "bridge_bonus": round(bridge_bonus, 3),
                "depth_bonus": round(depth_bonus, 3),
                "overall": round(overall, 3),
            },
            chain_summary=chain_summary,
        )

    def generate_candidates(
        self,
        *,
        mode_override: str | None = None,
        walks_override: int | None = None,
    ) -> tuple[list[WalkCandidate], dict]:
        candidates: list[WalkCandidate] = []
        traces: list[dict] = []
        walks_per_anchor = self.walks_per_anchor if walks_override is None else max(1, int(walks_override))
        sampled_mode_counts = {"short": 0, "medium": 0, "long": 0}
        for anchor in self.env.in_image_nodes:
            for _ in range(walks_per_anchor):
                mode = self._sample_length_mode(mode_override)
                sampled_mode_counts[mode] = sampled_mode_counts.get(mode, 0) + 1
                state = self.initial_state(anchor, mode=mode)
                steps_taken = 0
                while state is not None and steps_taken < self.max_steps:
                    action_bundle = self.think(state)
                    action = self._sample_action(action_bundle)
                    traces.append(
                        {
                            "state_id": state.state_id,
                            "anchor": state.anchor,
                            "length_mode": state.length_mode,
                            "depth": state.depth,
                            "current_node": state.current_node,
                            "gate": {
                                "continue_probability": round(action_bundle.get("gate", {}).get("continue_probability", 0.0), 4),
                                "stop_probability": round(action_bundle.get("gate", {}).get("stop_probability", 1.0), 4),
                            },
                            "actions": [
                                {
                                    "type": item["type"],
                                    "relation": item.get("edge", {}).get("relation", ""),
                                    "dst": item.get("edge", {}).get("dst", ""),
                                    "probability": round(item["probability"], 4),
                                    "score": round(item["score"], 4),
                                }
                                for item in action_bundle["actions"][:8]
                            ],
                            "chosen": {
                                "type": action["type"],
                                "relation": action.get("edge", {}).get("relation", ""),
                                "dst": action.get("edge", {}).get("dst", ""),
                            }
                            if action
                            else None,
                        }
                    )
                    if action is None or action["type"] == "stop":
                        candidate = self._candidate_from_state(state)
                        if candidate is not None:
                            candidates.append(candidate)
                        break
                    state = self.act(state, action)
                    steps_taken += 1
                else:
                    if state is not None:
                        candidate = self._candidate_from_state(state)
                        if candidate is not None:
                            candidates.append(candidate)

        deduped: dict[tuple, WalkCandidate] = {}
        for candidate in candidates:
            signature = (
                candidate.anchor,
                candidate.answer_value,
                candidate.length_mode,
                tuple((edge["src"], edge["dst"], edge["relation"]) for edge in candidate.chain_edges),
            )
            existing = deduped.get(signature)
            if existing is None or existing.scores["overall"] < candidate.scores["overall"]:
                deduped[signature] = candidate

        ranked = sorted(deduped.values(), key=lambda item: item.scores["overall"], reverse=True)
        depth_dist: dict[int, int] = {}
        mode_dist: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
        for candidate in ranked:
            depth_dist[candidate.chain_depth] = depth_dist.get(candidate.chain_depth, 0) + 1
            mode_dist[candidate.length_mode] = mode_dist.get(candidate.length_mode, 0) + 1
        return ranked, {
            "raw_candidate_count": len(candidates),
            "deduped_candidate_count": len(ranked),
            "sampled_mode_counts": sampled_mode_counts,
            "candidate_mode_counts": mode_dist,
            "candidate_depth_dist": depth_dist,
            "search_traces": traces[:120],
        }


def _relation_chain_penalty(candidate: WalkCandidate) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    chain = candidate.chain_summary.get("chain_text", [])
    if not chain:
        return 2.0, ["empty_chain"]

    generic_count = 0
    structural_count = 0
    for item in chain:
        rel = (item.get("relation") or "").strip().lower()
        rel_zh = (item.get("relation_zh") or "").strip()
        if rel == BRIDGE_RELATION:
            continue
        if rel_zh.startswith("相关"):
            generic_count += 1
        if any(token in rel for token in ("formed_by_intersection", "located_in", "has_location", "has_location_at")):
            structural_count += 1

    if generic_count:
        penalty += 0.8 * generic_count
        reasons.append(f"generic_relation_x{generic_count}")
    if structural_count >= max(2, len(chain)):
        penalty += 0.8
        reasons.append("over_structural_chain")
    if len(chain) >= 3 and structural_count >= 2:
        penalty += 0.4
        reasons.append("structural_middle_hop")
    return penalty, reasons


def _length_bucket(depth: int) -> str:
    if depth <= 3:
        return "short"
    if depth <= 4:
        return "medium"
    return "long"


def _step_necessity_score(candidate: WalkCandidate) -> float:
    chain = [edge for edge in candidate.chain_edges if edge.get("relation") != BRIDGE_RELATION]
    if not chain:
        return 0.0
    if len(chain) <= 2:
        return 0.8
    unique_targets = len({edge.get("dst", "") for edge in chain})
    lexical_edges = sum(
        1 for edge in chain
        if edge.get("relation_profile", {}).get("lexicalizability", 0.0) >= 0.55
    )
    structural_edges = sum(
        1
        for edge in chain
        if any(token in (edge.get("relation", "") or "").lower() for token in ("located_in", "has_location", "formed_by_intersection"))
    )
    diversity = min(1.0, unique_targets / max(len(chain), 1))
    lexical_ratio = lexical_edges / max(len(chain), 1)
    structural_penalty = 0.35 if structural_edges >= max(2, len(chain) - 1) else 0.0
    score = 0.35 + 0.45 * diversity + 0.35 * lexical_ratio - structural_penalty
    return max(0.0, min(score, 1.0))


def _min_hops_from_in_image_to_answer(env: GraphEnv, answer_node: str, max_hops: int = 2) -> int | None:
    if not answer_node:
        return None
    frontier = list(env.in_image_nodes)
    if answer_node in frontier:
        return 0
    visited = set(frontier)
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        nxt = []
        for node in frontier:
            for edge in env.unique_follows(node):
                dst = edge.get("dst")
                if not dst:
                    continue
                if dst == answer_node:
                    return hop
                if dst in visited:
                    continue
                visited.add(dst)
                nxt.append(dst)
        frontier = nxt
    return None


def critique_walk_candidate(candidate: WalkCandidate, env: GraphEnv) -> CritiquedWalkCandidate:
    chain = candidate.chain_edges
    semantic_chain = [edge for edge in chain if edge.get("relation") != BRIDGE_RELATION]
    final_edge = semantic_chain[-1] if semantic_chain else chain[-1]
    target_type = final_edge["tail_type"]
    answer = candidate.answer_value
    support_count = len(candidate.support_anchors)
    depth = len(chain)

    answer_specificity = max(0.0, _answer_specificity_score(answer, target_type))
    answer_uniqueness = 1.0 if _generic_answer_penalty(answer, target_type) < 0.8 else 0.2
    lexicalizability = sum(
        edge.get("relation_profile", {}).get("lexicalizability", 0.0) for edge in semantic_chain
    ) / max(len(semantic_chain), 1)
    askability = sum(
        edge.get("relation_profile", {}).get("askability", 0.0) for edge in semantic_chain
    ) / max(len(semantic_chain), 1)
    final_relation_strength = (
        final_edge.get("relation_profile", {}).get("askability", 0.0)
        + final_edge.get("relation_profile", {}).get("lexicalizability", 0.0)
    )
    bridge_value = min(1.0, 0.45 * support_count)
    clue_minimality = 1.0 if depth >= 3 else 0.65 if depth == 2 else 0.2
    if depth in {2, 3}:
        reasoning_compactness = 1.0
    elif depth in {4, 5}:
        reasoning_compactness = 0.78
    elif depth <= 7:
        reasoning_compactness = 0.62
    else:
        reasoning_compactness = 0.45
    hidden_recoverability = 0.9
    chain_penalty, penalty_reasons = _relation_chain_penalty(candidate)
    step_necessity = _step_necessity_score(candidate)
    length_bucket = _length_bucket(depth)
    bridge_steps = sum(1 for edge in chain if edge.get("relation") == BRIDGE_RELATION)
    min_hops_visible = _min_hops_from_in_image_to_answer(env, candidate.answer_node, max_hops=2)
    shortcut_risk = 0.0
    if min_hops_visible == 1:
        shortcut_risk += 1.0
    elif min_hops_visible == 2:
        shortcut_risk += 0.45
    if bridge_steps >= 1 and min_hops_visible == 1:
        shortcut_risk += 0.7
    if semantic_chain and semantic_chain[-1].get("src") in env.in_image_nodes:
        shortcut_risk += 0.35

    if any((item.get("relation_zh") or "").startswith("相关") for item in candidate.chain_summary.get("chain_text", [])):
        hidden_recoverability -= 0.5

    if length_bucket == "short":
        reasoning_compactness += 0.15
    elif length_bucket == "long":
        reasoning_compactness -= 0.1

    utility = (
        answer_uniqueness
        + answer_specificity
        + lexicalizability
        + askability
        + final_relation_strength
        + bridge_value
        + clue_minimality
        + reasoning_compactness
        + hidden_recoverability
        + 0.8 * step_necessity
        - 1.1 * shortcut_risk
        - chain_penalty
    )

    # 宽松 critic：只拦截硬伤
    reject_reason = ""
    accepted = True
    if answer_uniqueness < 0.4:
        accepted = False
        reject_reason = "generic_answer"
    elif final_relation_strength < 0.6:
        accepted = False
        reject_reason = "weak_final_relation"
    elif utility < 3.5:
        accepted = False
        reject_reason = "low_chain_utility"
    elif candidate.answer_node in env.in_image_nodes:
        # 答案本身就是图中可见实体，看图就能得到，不需要推理
        accepted = False
        reject_reason = "answer_visible_in_image"
    elif min_hops_visible is not None and min_hops_visible <= 1 and depth >= 4:
        # 链条走了 4+ 步但答案从图中 1 跳可达，多步推理是绕路
        accepted = False
        reject_reason = "answer_shortcut"

    critic_scores = {
        "answer_uniqueness": round(answer_uniqueness, 3),
        "answer_specificity": round(answer_specificity, 3),
        "lexicalizability": round(lexicalizability, 3),
        "askability": round(askability, 3),
        "final_relation_strength": round(final_relation_strength, 3),
        "bridge_value": round(bridge_value, 3),
        "clue_minimality": round(clue_minimality, 3),
        "reasoning_compactness": round(reasoning_compactness, 3),
        "hidden_recoverability": round(hidden_recoverability, 3),
        "step_necessity": round(step_necessity, 3),
        "shortcut_risk": round(shortcut_risk, 3),
        "min_hops_visible": min_hops_visible if min_hops_visible is not None else -1,
        "bridge_steps": bridge_steps,
        "length_bucket": length_bucket,
        "chain_penalty": round(chain_penalty, 3),
        "chain_utility": round(utility, 3),
        "penalty_reasons": penalty_reasons,
    }
    return CritiquedWalkCandidate(
        candidate=candidate,
        accepted=accepted,
        critic_scores=critic_scores,
        reject_reason=reject_reason,
    )


def filter_walk_candidates(
    candidates: list[WalkCandidate],
    env: GraphEnv,
    *,
    utility_threshold: float = 3.5,
    max_selected: int = 6,
    length_quota: tuple[int, int, int] = (3, 2, 1),
) -> tuple[list[WalkCandidate], dict]:
    critiqued = [critique_walk_candidate(candidate, env) for candidate in candidates]
    accepted = [
        item
        for item in critiqued
        if item.accepted and item.critic_scores.get("chain_utility", 0.0) >= utility_threshold
    ]
    accepted.sort(key=lambda item: item.critic_scores["chain_utility"], reverse=True)
    short_q, medium_q, long_q = [max(0, int(x)) for x in length_quota]
    quotas = {"short": short_q, "medium": medium_q, "long": long_q}

    # First dedupe by answer while preserving utility order.
    deduped_ranked: list[CritiquedWalkCandidate] = []
    seen_answers = set()
    for item in accepted:
        answer_sig = item.candidate.answer_value.strip().lower()
        if answer_sig in seen_answers:
            continue
        seen_answers.add(answer_sig)
        deduped_ranked.append(item)

    by_bucket: dict[str, list[CritiquedWalkCandidate]] = {"short": [], "medium": [], "long": []}
    for item in deduped_ranked:
        bucket = item.critic_scores.get("length_bucket", _length_bucket(item.candidate.chain_depth))
        by_bucket.setdefault(bucket, []).append(item)

    selected_items: list[CritiquedWalkCandidate] = []
    used_sigs = set()
    for bucket in ("short", "medium", "long"):
        need = quotas.get(bucket, 0)
        if need <= 0:
            continue
        for item in by_bucket.get(bucket, []):
            if len(selected_items) >= max_selected:
                break
            sig = (
                item.candidate.anchor,
                item.candidate.answer_value.strip().lower(),
                item.candidate.chain_depth,
                tuple((edge.get("src", ""), edge.get("relation", ""), edge.get("dst", "")) for edge in item.candidate.chain_edges),
            )
            if sig in used_sigs:
                continue
            selected_items.append(item)
            used_sigs.add(sig)
            if sum(1 for x in selected_items if x.critic_scores.get("length_bucket") == bucket) >= need:
                break

    # Backfill only with stronger candidates to keep high precision.
    fallback_threshold = utility_threshold + 0.35
    if len(selected_items) < max_selected:
        for item in deduped_ranked:
            if len(selected_items) >= max_selected:
                break
            sig = (
                item.candidate.anchor,
                item.candidate.answer_value.strip().lower(),
                item.candidate.chain_depth,
                tuple((edge.get("src", ""), edge.get("relation", ""), edge.get("dst", "")) for edge in item.candidate.chain_edges),
            )
            if sig in used_sigs:
                continue
            if item.critic_scores.get("chain_utility", 0.0) < fallback_threshold:
                continue
            selected_items.append(item)
            used_sigs.add(sig)

    deduped: list[WalkCandidate] = []
    for item in selected_items:
        item.candidate.scores.update(item.critic_scores)
        deduped.append(item.candidate)

    meta = {
        "critic_input_count": len(candidates),
        "critic_pass_count": len(accepted),
        "critic_selected_count": len(deduped),
        "utility_threshold": utility_threshold,
        "fallback_threshold": fallback_threshold,
        "max_selected": max_selected,
        "length_quota": quotas,
        "selected_length_dist": {
            "short": sum(1 for c in deduped if _length_bucket(c.chain_depth) == "short"),
            "medium": sum(1 for c in deduped if _length_bucket(c.chain_depth) == "medium"),
            "long": sum(1 for c in deduped if _length_bucket(c.chain_depth) == "long"),
        },
        "accepted_length_dist": {
            "short": sum(1 for item in deduped_ranked if item.critic_scores.get("length_bucket") == "short"),
            "medium": sum(1 for item in deduped_ranked if item.critic_scores.get("length_bucket") == "medium"),
            "long": sum(1 for item in deduped_ranked if item.critic_scores.get("length_bucket") == "long"),
        },
        "reject_samples": [
            {
                "candidate_id": item.candidate.candidate_id,
                "answer": item.candidate.answer_value,
                "reject_reason": item.reject_reason,
                "critic_scores": item.critic_scores,
            }
            for item in critiqued
            if not item.accepted
        ][:20],
        "top_chain_utility": [
            {
                "candidate_id": item.candidate.candidate_id,
                "answer": item.candidate.answer_value,
                "chain_utility": item.critic_scores["chain_utility"],
                "anchor": item.candidate.anchor,
                "length_bucket": item.critic_scores.get("length_bucket", ""),
            }
            for item in deduped_ranked[:12]
        ],
    }
    return deduped, meta


def _postcheck_walk_question_with_reason(
    question: str,
    candidate: WalkCandidate,
    env: GraphEnv,
) -> tuple[bool, str]:
    """轻量 postcheck：只拦截硬伤（无问号、名称泄露）。"""
    if not question:
        return False, "empty_question"
    if question.count("?") + question.count("？") < 1:
        return False, "no_question_mark"
    # 隐藏节点名称泄露
    hidden_names = {env.node_name(node) for node in candidate.hidden_nodes}
    for name in hidden_names:
        if name and name in question:
            return False, "hidden_node_leak"
    # 图中实体名称泄露
    q_lower = question.lower()
    for node in env.in_image_nodes:
        name = (env.node_name(node) or "").strip()
        if not name:
            continue
        if bool(re.search(r"[A-Za-z]", name)):
            if len(name) >= 4 and name.lower() in q_lower:
                return False, f"in_image_name_leak:{name}"
        else:
            if len(name) >= 2 and name in question:
                return False, f"in_image_name_leak:{name}"
    return True, "pass"


def _postcheck_walk_question(question: str, candidate: WalkCandidate, env: GraphEnv) -> bool:
    ok, _ = _postcheck_walk_question_with_reason(question, candidate, env)
    return ok


def _roundtrip_walk_semantic_check(question: str, candidate: WalkCandidate) -> tuple[bool, str]:
    final_edge = candidate.chain_edges[-1]
    tail_type = final_edge.get("tail_type", "OTHER")
    q = question.lower()
    if tail_type == "PERSON":
        if not any(token in question for token in ("谁", "哪位", "哪个人")):
            return False, "missing_person_wh"
    elif tail_type == "LOCATION":
        if not any(token in question for token in ("哪里", "哪座", "哪个城市", "哪座城市", "哪条")):
            return False, "missing_location_wh"
    elif tail_type == "TIME":
        if not any(token in question for token in ("何时", "什么时候", "哪一年", "哪天", "哪一天", "日期")):
            return False, "missing_time_wh"
    elif tail_type == "QUANTITY":
        if not any(token in question for token in ("多少", "几")):
            return False, "missing_quantity_wh"
    elif tail_type == "ORG":
        if not any(token in question for token in ("哪家", "哪个机构", "哪个组织", "哪个公司", "哪一个", "哪个团体", "哪支乐队", "哪一组合")):
            return False, "missing_org_wh"
    if "组织或机构是什么" in question:
        return False, "generic_org_phrase"
    if "entity" in q or "object" in q:
        return False, "english_placeholder_phrase"
    return True, "pass"


def _walk_question_quality_score(question: str, candidate: WalkCandidate) -> float:
    final_edge = candidate.chain_edges[-1]
    tail_type = final_edge.get("tail_type", "OTHER")
    wh_bonus = 0.0
    if tail_type == "PERSON" and any(token in question for token in ("谁", "哪位")):
        wh_bonus = 0.6
    elif tail_type == "LOCATION" and any(token in question for token in ("哪里", "哪座城市", "哪个城市")):
        wh_bonus = 0.6
    elif tail_type == "TIME" and any(token in question for token in ("哪一年", "什么时候", "哪一天")):
        wh_bonus = 0.6
    elif tail_type == "QUANTITY" and any(token in question for token in ("多少", "几")):
        wh_bonus = 0.6
    elif tail_type == "ORG" and any(token in question for token in ("哪家", "哪个组织", "哪个公司", "哪个团体", "哪支乐队", "哪一组合")):
        wh_bonus = 0.6
    ontology_penalty = _ontologyese_penalty(question)
    length_penalty = 0.15 if len(question) < 12 else 0.0
    return 1.2 + wh_bonus - ontology_penalty - length_penalty


def _latest_vlm_failure(meta: dict) -> tuple[str, str]:
    attempts = meta.get("attempts") or []
    if not attempts:
        err = str(meta.get("last_error") or "").strip()
        return ("vlm_failed", err[:240])
    last = attempts[-1]
    status = str(last.get("status") or "")
    if status == "call_error":
        err = str(last.get("error") or meta.get("last_error") or "").strip()
        return ("vlm_call_error", err[:240])
    if status == "json_parse_failed":
        return ("vlm_json_parse_failed", str(last.get("raw_preview") or "")[:240])
    if status == "empty_response":
        return ("vlm_empty_response", "")
    return ("vlm_failed", str(meta.get("last_error") or "")[:240])


def _fallback_walk_questions(candidate: WalkCandidate) -> list[str]:
    """当 LLM 生成失败时，用规则模板兜底。

    模板遵循"视觉指代 + 具体提问意图"的自然句式，
    避免 "根据...线索" "最终关联到" 等机械表达。
    """
    final_edge = candidate.chain_edges[-1]
    tail_type = final_edge.get("tail_type", "OTHER")
    anchor_ref = candidate.chain_summary.get("anchor_ref_text") or "图中的标识"
    follow_clause = (candidate.chain_summary.get("question_tail_hint") or "").strip().rstrip("？?")
    head_hint = (candidate.chain_summary.get("question_head_hint") or "").strip()
    if head_hint.startswith("相关"):
        head_hint = ""

    # 用视觉指代开头，直接接具体提问
    variants = []
    if tail_type == "PERSON":
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")
        variants.append(f"{anchor_ref}上展示的内容，其创作者是谁？")
    elif tail_type == "LOCATION":
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")
        variants.append(f"{anchor_ref}上展示的内容，发源于哪座城市？")
    elif tail_type == "TIME":
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")
        variants.append(f"{anchor_ref}上展示的内容，最早出现在哪一年？")
    elif tail_type == "QUANTITY":
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")
        variants.append(f"{anchor_ref}上展示的内容，涉及的数值是多少？")
    elif tail_type == "ORG":
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")
        variants.append(f"{anchor_ref}上展示的内容，隶属于哪家公司？")
    else:
        if follow_clause and not follow_clause.startswith("相关"):
            variants.append(f"{anchor_ref}，{follow_clause}？")

    if head_hint and head_hint != follow_clause:
        variants.append(f"{anchor_ref}，它的{head_hint}是什么？")

    if not variants:
        variants.append(f"{anchor_ref}上展示的内容，具体是什么？")
    return variants


def _build_realize_prompt(
    candidate: WalkCandidate,
    env: GraphEnv,
    image_description: str,
    domain: str,
) -> str:
    anchor_ref = candidate.chain_summary.get("anchor_ref_text", "")
    hidden_names = [env.node_name(node) for node in candidate.hidden_nodes]
    chain_text = json.dumps(candidate.chain_summary["chain_text"], ensure_ascii=False)
    hidden_text = json.dumps(hidden_names, ensure_ascii=False)

    lines = [
        "你看到一张图片，想基于图中某样东西问一个问题。",
        "下面有一条知识链。如果这条链能变成一个自然的问题就输出它；如果这条链不适合出题就输出 null。",
        "",
        "要求：",
        "- 用视觉特征指代图中实体（如 \"左侧那张音乐剧海报\"），不要写出实体名字",
        "- 问题要具体，像真人好奇心（\"作曲者是谁\" \"第一家店开在哪\"），不要泛问（\"是什么\" \"隶属于哪个机构\"）",
        f"- 隐藏节点名不可出现在题面中：{hidden_text}",
        "",
        f"图片描述：{image_description}",
        f"图中起点：{anchor_ref}",
        f"知识链：{chain_text}",
        f"答案：{candidate.answer_value}",
        "",
        '输出 JSON（不适合出题则 question 填 ""）：',
        '{"question": "...", "answer": "..."}',
    ]
    return "\n".join(lines)


def _realize_single_candidate(
    candidate: WalkCandidate,
    env: GraphEnv,
    image_b64: str,
    image_description: str,
    domain: str,
    best_of_n: int,
    json_retries: int,
) -> tuple[dict | None, list[dict], bool]:
    """单条候选链 → 1 次 LLM 调用 → 出题或跳过。"""
    reject_traces: list[dict] = []
    prompt = _build_realize_prompt(candidate, env, image_description, domain)

    obj = call_vlm_json(
        prompt,
        "请根据知识链生成自然中文问题，不适合出题则 question 填空字符串。",
        image_b64=image_b64,
        max_tokens=800,
        temperature=0.7,
        max_attempts=1,
    )
    if not isinstance(obj, dict):
        reject_traces.append({"candidate_id": candidate.candidate_id, "reason": "vlm_failed"})
        return None, reject_traces, False

    question = str(obj.get("question") or "").strip()
    if not question:
        reject_traces.append({"candidate_id": candidate.candidate_id, "reason": "llm_skipped"})
        return None, reject_traces, False

    post_ok, post_reason = _postcheck_walk_question_with_reason(question, candidate, env)
    if not post_ok:
        reject_traces.append({
            "candidate_id": candidate.candidate_id,
            "reason": post_reason,
            "question": question,
        })
        return None, reject_traces, False

    payload = {
        "question": question,
        "answer": obj.get("answer") or candidate.answer_value,
        "tool_sequence": obj.get("tool_sequence") or [],
        "rationale": obj.get("why_question_works") or "",
        "chain_summary": candidate.chain_summary,
        "scores": candidate.scores,
        "candidate_id": candidate.candidate_id,
        "length_mode": candidate.length_mode,
        "chain_depth": candidate.chain_depth,
    }
    return payload, reject_traces, False


def realize_walk_candidates(
    candidates: list[WalkCandidate],
    env: GraphEnv,
    image_b64: str,
    image_description: str,
    domain: str,
    *,
    max_candidates: int = 6,
    return_meta: bool = False,
    max_workers: int = 4,
    **_kwargs,
) -> list[dict] | tuple[list[dict], dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    to_realize = candidates[:max_candidates]
    results: list[tuple[int, dict]] = []
    all_reject_traces: list[dict] = []

    def _worker(idx: int, cand: WalkCandidate):
        return idx, _realize_single_candidate(
            cand, env, image_b64, image_description, domain, 0, 0,
        )

    with ThreadPoolExecutor(max_workers=min(max_workers, len(to_realize))) as pool:
        futures = {pool.submit(_worker, i, c): i for i, c in enumerate(to_realize)}
        for future in as_completed(futures):
            idx, (payload, traces, _) = future.result()
            all_reject_traces.extend(traces)
            if payload is not None:
                results.append((idx, payload))

    results.sort(key=lambda x: x[0])
    ordered_results = [r[1] for r in results]

    meta = {
        "realize_input_count": len(to_realize),
        "realize_output_count": len(ordered_results),
        "reject_traces": all_reject_traces[:120],
    }
    if return_meta:
        return ordered_results, meta
    return ordered_results


def _candidate_signature(candidate: WalkCandidate) -> tuple:
    return (
        candidate.anchor,
        candidate.answer_value.strip().lower(),
        candidate.length_mode,
        tuple((edge.get("src", ""), edge.get("relation", ""), edge.get("dst", "")) for edge in candidate.chain_edges),
    )


def _merge_candidates(primary: list[WalkCandidate], extra: list[WalkCandidate]) -> list[WalkCandidate]:
    merged = { _candidate_signature(c): c for c in primary }
    for candidate in extra:
        sig = _candidate_signature(candidate)
        existing = merged.get(sig)
        if existing is None or candidate.scores.get("overall", 0.0) > existing.scores.get("overall", 0.0):
            merged[sig] = candidate
    return sorted(merged.values(), key=lambda c: c.scores.get("overall", 0.0), reverse=True)


def build_randomwalk_shadow_result(
    G: nx.DiGraph,
    nodes: dict,
    image_b64: str,
    image_description: str,
    domain: str,
    *,
    seed: int = 13,
    walks_per_anchor: int = 32,
    max_steps: int = 10,
    temperature: float = 0.95,
    top_p: float = 0.9,
    epsilon: float = 0.1,
    length_weights: tuple[float, float, float] = (0.45, 0.35, 0.2),
    length_quota: tuple[int, int, int] = (2, 2, 1),
    utility_threshold: float = 4.8,
    long_boost_walks: int = 16,
    long_boost_temperature: float = 1.15,
    long_boost_top_p: float = 0.95,
    long_boost_epsilon: float = 0.2,
    max_questions: int = 5,
    best_of_n: int = 3,
    do_realize: bool = True,
) -> dict:
    env = GraphEnv(G, nodes)
    generator = RandomWalkQuestionGenerator(
        env,
        seed=seed,
        max_steps=max_steps,
        walks_per_anchor=walks_per_anchor,
        temperature=temperature,
        top_p=top_p,
        epsilon=epsilon,
        length_weights=length_weights,
    )
    candidates, search_meta = generator.generate_candidates()
    selected, critic_meta = filter_walk_candidates(
        candidates,
        env,
        utility_threshold=utility_threshold,
        max_selected=max_questions,
        length_quota=length_quota,
    )

    long_boost_triggered = False
    long_boost_added = 0
    if not any(c.chain_depth >= 5 for c in selected) and long_boost_walks > 0:
        long_boost_triggered = True
        long_generator = RandomWalkQuestionGenerator(
            env,
            seed=seed + 101,
            max_steps=max_steps,
            walks_per_anchor=walks_per_anchor,
            temperature=long_boost_temperature,
            top_p=long_boost_top_p,
            epsilon=long_boost_epsilon,
            length_weights=length_weights,
        )
        extra_candidates, extra_meta = long_generator.generate_candidates(mode_override="long", walks_override=long_boost_walks)
        long_boost_added = extra_meta.get("deduped_candidate_count", 0)
        merged = _merge_candidates(candidates, extra_candidates)
        selected, critic_meta = filter_walk_candidates(
            merged,
            env,
            utility_threshold=utility_threshold,
            max_selected=max_questions,
            length_quota=length_quota,
        )
        search_meta = {
            **search_meta,
            "long_boost_extra": {
                "raw_candidate_count": extra_meta.get("raw_candidate_count", 0),
                "deduped_candidate_count": extra_meta.get("deduped_candidate_count", 0),
                "candidate_mode_counts": extra_meta.get("candidate_mode_counts", {}),
                "candidate_depth_dist": extra_meta.get("candidate_depth_dist", {}),
            },
            "post_merge_deduped_count": len(merged),
        }

    if do_realize:
        realized, realize_meta = realize_walk_candidates(
            selected,
            env,
            image_b64,
            image_description,
            domain,
            max_candidates=max_questions,
            best_of_n=best_of_n,
            json_retries=2,
            return_meta=True,
        )
    else:
        realized, realize_meta = [], {"realize_input_count": 0, "realize_output_count": 0, "reject_traces": []}

    level_2: list[dict] = []
    level_3: list[dict] = []
    for idx, item in enumerate(realized, start=1):
        chain_depth = int(item.get("chain_depth") or 0)
        record = {
            "question_id": "",
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "tool_sequence": item.get("tool_sequence", []),
            "rationale": item.get("rationale", ""),
            "entities_involved": [],
            "obfuscation_applied": True,
            "obfuscated_entities": item.get("chain_summary", {}).get("support_anchor_texts", []),
            "reasoning_path": {
                "program": {
                    "program_id": item.get("candidate_id", ""),
                    "program_type": "random_walk_chain",
                    "family": "random_walk_chain",
                    "reasoning_family": "random_walk_chain",
                    "difficulty": "L3" if chain_depth >= 4 else "L2",
                    "length_mode": item.get("length_mode", ""),
                    "chain_depth": chain_depth,
                },
                "proof_graph": item.get("chain_summary", {}).get("chain_text", []),
                "visibility_plan": {
                    "visible_anchors": [item.get("chain_summary", {}).get("anchor_text", "")],
                    "support_anchors": item.get("chain_summary", {}).get("support_anchor_texts", []),
                    "hard_hidden_nodes": [],
                    "soft_hidden_nodes": [],
                    "answer_node": item.get("answer", ""),
                },
                "surface_plan": {
                    "question_head_hint": item.get("chain_summary", {}).get("question_head_hint", ""),
                    "question_tail_hint": item.get("chain_summary", {}).get("question_tail_hint", ""),
                },
                "scores": item.get("scores", {}),
            },
            "chain_summary": item.get("chain_summary", {}),
            "scores": item.get("scores", {}),
        }
        if chain_depth >= 4:
            record["question_id"] = f"L3_{len(level_3) + 1:02d}"
            level_3.append(record)
        else:
            record["question_id"] = f"L2_{len(level_2) + 1:02d}"
            level_2.append(record)

    meta = {
        "raw_candidate_count": search_meta.get("raw_candidate_count", 0),
        "deduped_candidate_count": search_meta.get("deduped_candidate_count", 0),
        "critic_pass_count": critic_meta.get("critic_pass_count", 0),
        "critic_selected_count": critic_meta.get("critic_selected_count", 0),
        "final_selected_count": len(realized),
        "length_dist": {
            "short": sum(1 for item in realized if int(item.get("chain_depth") or 0) <= 3),
            "medium": sum(1 for item in realized if int(item.get("chain_depth") or 0) == 4),
            "long": sum(1 for item in realized if int(item.get("chain_depth") or 0) >= 5),
        },
        "long_boost_triggered": long_boost_triggered,
        "long_boost_added": long_boost_added,
        "long_bucket_present": any(int(item.get("chain_depth") or 0) >= 5 for item in realized),
        "long_boost_status": (
            "not_needed"
            if not long_boost_triggered
            else ("success" if any(int(item.get("chain_depth") or 0) >= 5 for item in realized) else "no_long_after_boost")
        ),
        "utility_threshold": utility_threshold,
        "long_boost_temperature": long_boost_temperature,
        "long_boost_top_p": long_boost_top_p,
        "long_boost_epsilon": long_boost_epsilon,
        "search_meta": search_meta,
        "critic_meta": critic_meta,
        "realize_meta": realize_meta,
        "reject_reasons": critic_meta.get("reject_samples", []),
    }
    return {
        "level_2": level_2,
        "level_3": level_3,
        "metadata": meta,
    }
