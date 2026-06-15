"""Vector store utilities for exercise RAG search."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from app.config import EMBEDDINGS_FILE, EXERCISES_FILE, INDEX_FILE

logger = logging.getLogger("trainer.rag")

_store_cache: tuple[np.ndarray, list[str], list[dict[str, Any]]] | None = None


def exercise_to_text(exercise: dict[str, Any]) -> str:
    primary = ", ".join(exercise.get("primaryMuscles") or [])
    secondary = ", ".join(exercise.get("secondaryMuscles") or [])
    instructions = " ".join(exercise.get("instructions") or [])
    parts = [
        f"Name: {exercise.get('name', '')}",
        f"Category: {exercise.get('category', '')}",
        f"Level: {exercise.get('level', '')}",
        f"Equipment: {exercise.get('equipment', 'none')}",
        f"Primary muscles: {primary}",
    ]
    if secondary:
        parts.append(f"Secondary muscles: {secondary}")
    if exercise.get("mechanic"):
        parts.append(f"Mechanic: {exercise['mechanic']}")
    if exercise.get("force"):
        parts.append(f"Force: {exercise['force']}")
    if instructions:
        parts.append(f"Instructions: {instructions}")
    return ". ".join(parts)


def is_store_ready() -> bool:
    return EMBEDDINGS_FILE.exists() and INDEX_FILE.exists() and EXERCISES_FILE.exists()


def load_store() -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    if not is_store_ready():
        raise FileNotFoundError(
            "RAG database not found. Run `python -m scripts.build_embeddings` first."
        )

    started = time.perf_counter()
    embeddings = np.load(EMBEDDINGS_FILE)
    with INDEX_FILE.open("r", encoding="utf-8") as handle:
        ids: list[str] = json.load(handle)
    with EXERCISES_FILE.open("r", encoding="utf-8") as handle:
        exercises: list[dict[str, Any]] = json.load(handle)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "RAG store loaded: %d exercises, embedding dim=%d (%.0f ms)",
        len(exercises),
        embeddings.shape[1] if embeddings.ndim == 2 else 0,
        elapsed_ms,
    )
    _store_cache = (embeddings, ids, exercises)
    return _store_cache


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = query_vec.astype(np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    scores = matrix @ query / (safe_norms * query_norm)
    return scores.astype(np.float32)


_LEVEL_SYNONYMS: dict[str, str] = {
    "beginner": "beginner",
    "basic": "beginner",
    "easy": "beginner",
    "novice": "beginner",
    "newbie": "beginner",
    "starter": "beginner",
    "intermediate": "intermediate",
    "medium": "intermediate",
    "moderate": "intermediate",
    "expert": "expert",
    "advanced": "expert",
    "hard": "expert",
    "pro": "expert",
    "difficult": "expert",
}


def normalize_level(text: Any) -> str | None:
    """Map a free-form level word to one of beginner/intermediate/expert, or None."""
    if not text:
        return None
    key = str(text).strip().lower()
    return _LEVEL_SYNONYMS.get(key)


def keyword_boost(query: str, exercise: dict[str, Any], base_score: float) -> float:
    query_lower = query.lower()
    boost = 0.0

    for muscle in exercise.get("primaryMuscles") or []:
        if muscle.lower() in query_lower:
            boost += 0.18

    for muscle in exercise.get("secondaryMuscles") or []:
        if muscle.lower() in query_lower:
            boost += 0.08

    name = (exercise.get("name") or "").lower()
    if any(token in name for token in query_lower.split() if len(token) > 3):
        boost += 0.05

    for token in query_lower.split():
        mapped = normalize_level(token)
        if mapped and exercise.get("level") == mapped:
            boost += 0.04
            break

    return float(base_score + boost)


def search_exercises(
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int = 8,
    level: str | None = None,
    exclude_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank exercises by similarity (+ keyword boost), optionally hard-filtering by
    difficulty ``level`` and skipping any names in ``exclude_names``."""
    embeddings, _, exercises = load_store()
    scores = cosine_similarity(query_embedding, embeddings)

    ranked_indices = np.argsort(scores)[::-1]
    wanted_level = normalize_level(level) if level else None
    excluded = {n.lower() for n in exclude_names} if exclude_names else set()

    # With a hard filter we may need to scan well past the usual window to find
    # enough matches, so when filtering we consider the whole ranked list.
    window = ranked_indices if (wanted_level or excluded) else ranked_indices[: top_k * 3]

    results: list[dict[str, Any]] = []
    for idx in window:
        exercise = exercises[int(idx)]
        if wanted_level and (exercise.get("level") or "").lower() != wanted_level:
            continue
        if excluded and (exercise.get("name") or "").lower() in excluded:
            continue
        final_score = keyword_boost(query_text, exercise, float(scores[int(idx)]))
        results.append({"score": final_score, "exercise": exercise})

    results.sort(key=lambda item: item["score"], reverse=True)
    picked = results[:top_k]
    logger.debug(
        "search(%r, top_k=%d, level=%s, exclude=%d) -> %d results",
        query_text[:60],
        top_k,
        wanted_level,
        len(excluded),
        len(picked),
    )
    return picked


def find_by_name(name: str) -> dict[str, Any] | None:
    """Look up a single exercise by (fuzzy) name: exact match first, else substring."""
    target = " ".join((name or "").strip().lower().split())
    if not target:
        return None
    _, _, exercises = load_store()

    for exercise in exercises:
        if (exercise.get("name") or "").strip().lower() == target:
            logger.debug("find_by_name exact match %r", exercise.get("name"))
            return exercise

    # Substring / token overlap fallback.
    best: dict[str, Any] | None = None
    best_overlap = 0
    target_tokens = set(target.split())
    for exercise in exercises:
        ex_name = (exercise.get("name") or "").strip().lower()
        if not ex_name:
            continue
        if target in ex_name or ex_name in target:
            return exercise
        overlap = len(target_tokens & set(ex_name.split()))
        if overlap > best_overlap:
            best_overlap = overlap
            best = exercise
    return best if best_overlap > 0 else None


def invalidate_store_cache() -> None:
    """Drop the in-memory store cache (used after rebuilding embeddings)."""
    global _store_cache
    _store_cache = None
    logger.debug("RAG store cache invalidated")
