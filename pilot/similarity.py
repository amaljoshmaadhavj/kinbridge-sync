"""
Semantic similarity for extracted tool calls.

Uses sentence-transformers (all-MiniLM-L6-v2) to embed a normalised
textual representation of each action, then computes cosine similarity.

The model is loaded once and cached for the process lifetime.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

# Lazy-loaded singleton — only loaded when first needed.
_model = None
_model_name = "all-MiniLM-L6-v2"


def _get_model():
    """Load (or return cached) sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
    return _model


def normalise_action(tool: str, args: dict[str, Any]) -> str:
    """Produce a deterministic textual representation of an extracted action.

    Format:  TOOL_NAME key1=value1 key2=value2 ...
    Keys are sorted; values are JSON-serialised for determinism.
    """
    parts = [tool.upper()]
    for k in sorted(args.keys()):
        v = args[k]
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, sort_keys=True, separators=(",", ":"))
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return " ".join(parts)


def embed_action(tool: str, args: dict[str, Any]) -> np.ndarray:
    """Return the embedding vector for a single action."""
    text = normalise_action(tool, args)
    model = _get_model()
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors.

    Since the vectors are already normalised by sentence-transformers,
    this is simply the dot product.  Clamps to [-1, 1] for safety.
    """
    dot = float(np.dot(vec_a, vec_b))
    return max(-1.0, min(1.0, dot))


def action_similarity(
    tool_a: str, args_a: dict[str, Any],
    tool_b: str, args_b: dict[str, Any],
    _cache: dict[str, np.ndarray] | None = None,
) -> float:
    """Compute cosine similarity between two actions.

    An optional _cache dict avoids re-embedding the same action text.
    """
    text_a = normalise_action(tool_a, args_a)
    text_b = normalise_action(tool_b, args_b)

    if _cache is not None:
        if text_a not in _cache:
            _cache[text_a] = embed_action(tool_a, args_a)
        if text_b not in _cache:
            _cache[text_b] = embed_action(tool_b, args_b)
        vec_a, vec_b = _cache[text_a], _cache[text_b]
    else:
        vec_a = embed_action(tool_a, args_a)
        vec_b = embed_action(tool_b, args_b)

    return cosine_similarity(vec_a, vec_b)
