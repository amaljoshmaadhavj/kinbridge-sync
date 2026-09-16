"""
Positive and negative evaluation pairs for the pilot study.

Positive pairs: two actions from the SAME ground-truth intent.
Negative pairs: actions from DIFFERENT ground-truth intents.

Ground-truth labels come exclusively from intent_id, never from
similarity scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


@dataclass
class EvalPair:
    """A single evaluation pair with ground-truth label."""
    pair_id: str
    action_a_id: str          # prompt_id of first action
    action_b_id: str          # prompt_id of second action
    intent_id_a: str
    intent_id_b: str
    same_intent: bool         # ground truth, NOT derived from similarity
    tool_class: str
    similarity: float = 0.0   # filled in later by the similarity module

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "action_a": self.action_a_id,
            "action_b": self.action_b_id,
            "intent_id_a": self.intent_id_a,
            "intent_id_b": self.intent_id_b,
            "same_intent": self.same_intent,
            "tool_class": self.tool_class,
            "similarity": self.similarity,
        }


def build_pairs_from_actions(
    actions: list[dict[str, Any]],
) -> list[EvalPair]:
    """Build all unique pairwise evaluation sets from a list of actions.

    Each action dict must contain:
        prompt_id, intent_id, tool_class, tool, args

    Positive pairs: same intent_id, distinct prompt_ids.
    Negative pairs: different intent_ids.
    """
    pairs: list[EvalPair] = []
    pair_counter = 0

    for i, j in combinations(range(len(actions)), 2):
        a = actions[i]
        b = actions[j]
        same_intent = a["intent_id"] == b["intent_id"]

        # Skip self-pairs (same prompt_id)
        if a["prompt_id"] == b["prompt_id"]:
            continue

        pair_counter += 1
        pair_id = f"pair_{pair_counter:04d}"

        pairs.append(EvalPair(
            pair_id=pair_id,
            action_a_id=a["prompt_id"],
            action_b_id=b["prompt_id"],
            intent_id_a=a["intent_id"],
            intent_id_b=b["intent_id"],
            same_intent=same_intent,
            tool_class=a["tool_class"],
        ))

    return pairs


def split_pairs(pairs: list[EvalPair]) -> tuple[list[EvalPair], list[EvalPair]]:
    """Split into (positive_pairs, negative_pairs)."""
    pos = [p for p in pairs if p.same_intent]
    neg = [p for p in pairs if not p.same_intent]
    return pos, neg
