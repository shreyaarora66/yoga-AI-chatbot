"""The agent's RAG skills: build an adaptive workout plan, and look up a single
exercise by name.

This module is provider-agnostic. Both the Gemini and Ollama providers call
``run_plan_workout`` / ``run_find_exercise`` and read results from the shared
``collector`` list.

When a ``user_id`` is supplied and the user did not request a specific level, the
difficulty is chosen by ``app.progression`` from the user's last 7 days, exercises
already served in that window are skipped (uniqueness), and the result is logged so
future sessions can progress. Without a ``user_id`` (e.g. unit tests) the behaviour
is the original plain similarity search.
"""

import logging
from datetime import date
from typing import Any

import numpy as np

from app import progression, rag_store, storage
from app.config import get_daily_total, get_image_base_url, get_max_per_group
from app.embeddings import embed_text
from app.muscles import normalize_muscles
from app.rag_store import search_exercises as rag_search

logger = logging.getLogger("trainer.agent")


def build_image_urls(images: Any) -> list[str]:
    """Turn the extended image paths in data.json into absolute URLs."""
    if not isinstance(images, list):
        return []
    base = get_image_base_url()
    urls: list[str] = []
    for path in images:
        path = str(path or "").strip()
        if not path:
            continue
        if path.startswith(("http://", "https://")):
            urls.append(path)
        else:
            urls.append(f"{base}/{path.lstrip('/')}")
    return urls


def distribute(num_groups: int, total: int) -> list[int]:
    """Split ``total`` exercises across ``num_groups`` as evenly as possible.

    Any remainder is added to the LAST groups, so 10 over 3 groups -> [3, 3, 4].
    Every group gets at least 1.
    """
    if num_groups <= 0:
        return []
    base = max(0, total) // num_groups
    remainder = max(0, total) % num_groups
    counts = [base] * num_groups
    for i in range(num_groups - remainder, num_groups):
        counts[i] += 1
    return [max(1, c) for c in counts]


def _summary(exercise: dict[str, Any]) -> dict[str, Any]:
    """Lightweight view of an exercise (no image URLs, to keep LLM prompts small)."""
    return {
        "name": exercise.get("name"),
        "level": exercise.get("level"),
        "equipment": exercise.get("equipment") or "none",
        "primaryMuscles": exercise.get("primaryMuscles") or [],
        "instructions": exercise.get("instructions") or [],
    }


def _embed_query(muscle: str) -> tuple[np.ndarray, str]:
    query = f"{muscle} exercises"
    vector = np.array(embed_text(query, text_type="query"), dtype=np.float32)
    return vector, query


def _canonical_key(muscle: str) -> str:
    """A stable canonical muscle name used to key history/progression."""
    canon = normalize_muscles([muscle])
    return canon[0] if canon else (muscle or "").strip().lower()


def _do_search(
    vector: np.ndarray,
    query: str,
    count: int,
    level: str | None,
    exclude: set[str] | None,
) -> list[dict[str, Any]]:
    """Call the RAG store. The no-level/no-exclude path keeps the original 3-arg
    signature so existing stubbed tests keep working."""
    count = max(1, int(count))
    if level is None and not exclude:
        return rag_search(vector, query, top_k=count)
    return rag_search(vector, query, top_k=count, level=level, exclude_names=exclude)


def _collect_unique(
    vector: np.ndarray,
    query: str,
    total: int,
    level_plan: list[tuple[str, int]] | None,
    window_served: set[str],
    enforce_unique: bool,
) -> list[dict[str, Any]]:
    """Gather up to ``total`` exercises following the level plan, keeping them unique.

    ``level_plan`` is None for the anonymous/plain path. ``window_served`` are names
    already given to the user in the 7-day window (only enforced when
    ``enforce_unique``). Falls back across levels, and finally allows repeats, so the
    requested count is met whenever the muscle has enough exercises at all.
    """
    used = {n.lower() for n in window_served} if enforce_unique else set()
    chosen: list[dict[str, Any]] = []
    chosen_names: set[str] = set()

    def take(matches: list[dict[str, Any]], cap: int) -> None:
        for match in matches:
            if len(chosen) >= total or cap <= 0:
                return
            exercise = match["exercise"]
            low = (exercise.get("name") or "").lower()
            if not low or low in chosen_names:
                continue
            if enforce_unique and low in used:
                continue
            chosen.append(exercise)
            chosen_names.add(low)
            used.add(low)
            cap -= 1

    if level_plan is None:
        take(_do_search(vector, query, total, None, None), total)
        return chosen[:total]

    # Phase 1: fulfil each planned (level, count).
    for level, n in level_plan:
        if len(chosen) >= total:
            break
        matches = _do_search(vector, query, n + 5, level, used if enforce_unique else None)
        take(matches, n)

    # Phase 2: top up any deficit from fallback levels (unique).
    if len(chosen) < total:
        planned_levels = [lvl for lvl, _ in level_plan]
        for level in progression.fallback_order(planned_levels):
            if len(chosen) >= total:
                break
            need = total - len(chosen)
            matches = _do_search(vector, query, need + 5, level, used if enforce_unique else None)
            take(matches, need)

    # Phase 3: still short only because everything was already served -> allow repeats.
    if len(chosen) < total:
        need = total - len(chosen)
        matches = _do_search(vector, query, need + len(used) + 5, None, None)
        for match in matches:
            if len(chosen) >= total:
                break
            exercise = match["exercise"]
            low = (exercise.get("name") or "").lower()
            if not low or low in chosen_names:
                continue
            chosen.append(exercise)
            chosen_names.add(low)

    return chosen[:total]


def _fetch_for_group(
    muscle: str,
    count: int,
    collector: list[dict[str, Any]],
    *,
    user_id: int | None,
    today: date,
    level: str | None,
) -> dict[str, Any]:
    """Resolve difficulty, fetch unique exercises for one muscle, and log them."""
    muscle = str(muscle or "").strip()
    safe_count = max(1, min(int(count), get_max_per_group()))
    if not muscle:
        return {"muscle": muscle, "count": 0, "exercises": [], "levels": []}

    vector, query = _embed_query(muscle)
    canonical = _canonical_key(muscle)
    explicit_level = rag_store.normalize_level(level) if level else None

    if explicit_level:
        level_plan: list[tuple[str, int]] | None = [(explicit_level, safe_count)]
    elif user_id is not None:
        level_plan = progression.next_level_plan(user_id, canonical, today, safe_count)
    else:
        level_plan = None

    window_served = (
        progression.served_names(user_id, canonical, today)
        if user_id is not None
        else set()
    )

    chosen = _collect_unique(
        vector,
        query,
        safe_count,
        level_plan,
        window_served,
        enforce_unique=user_id is not None,
    )

    exercises: list[dict[str, Any]] = []
    log_items: list[tuple[str | None, str]] = []
    for exercise in chosen:
        summary = _summary(exercise)
        exercises.append(summary)

        tagged = dict(summary)
        tagged["group"] = muscle
        tagged["images"] = build_image_urls(exercise.get("images"))
        collector.append(tagged)

        log_items.append((exercise.get("level"), exercise.get("name")))

    if user_id is not None and log_items:
        storage.log_workout(user_id, canonical, log_items, today.isoformat())

    levels_used = sorted({lvl for lvl, _ in log_items if lvl})
    logger.info(
        "  group(%r, count=%d) -> %d exercises, levels=%s",
        muscle,
        safe_count,
        len(exercises),
        levels_used or ("auto" if level_plan else "any"),
    )
    return {
        "muscle": muscle,
        "count": len(exercises),
        "exercises": exercises,
        "levels": levels_used,
    }


# Kept for backward compatibility / direct single-muscle searches.
def search_muscle(
    muscle: str,
    count: int,
    collector: list[dict[str, Any]],
    level: str | None = None,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    """Search the RAG DB for ``count`` exercises for one ``muscle``; record in ``collector``."""
    muscle = str(muscle or "").strip()
    safe_count = max(1, min(int(count), get_max_per_group()))
    if not muscle:
        return {"muscle": muscle, "count": 0, "exercises": []}

    vector, query = _embed_query(muscle)
    wanted = rag_store.normalize_level(level) if level else None
    exclude = {n.lower() for n in exclude_names} if exclude_names else set()
    matches = _do_search(vector, query, safe_count, wanted, exclude)

    exercises: list[dict[str, Any]] = []
    for match in matches[:safe_count]:
        exercise = match["exercise"]
        summary = _summary(exercise)
        exercises.append(summary)
        tagged = dict(summary)
        tagged["group"] = muscle
        tagged["images"] = build_image_urls(exercise.get("images"))
        collector.append(tagged)

    logger.info("  search(%r, count=%d) -> %d exercises", muscle, safe_count, len(exercises))
    return {"muscle": muscle, "count": len(exercises), "exercises": exercises}


def run_plan_workout(
    muscles: Any,
    counts: Any,
    collector: list[dict[str, Any]],
    *,
    user_id: int | None = None,
    today: date | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    """Build a plan over one or more muscle groups, recording results in ``collector``.

    ``counts`` may be empty/partial; any unspecified groups share the remaining
    portion of the daily total, split as evenly as possible. When ``level`` is not
    given and ``user_id`` is known, difficulty auto-progresses per muscle.
    """
    today = today or date.today()

    # Accept a single string for robustness (some models pass a string).
    if isinstance(muscles, str):
        muscles = [muscles]
    groups = [str(m).strip() for m in (muscles or []) if str(m).strip()]

    logger.info(
        "TOOL plan_workout(muscles=%s, counts=%s, level=%s, user=%s)",
        groups,
        counts,
        level,
        user_id,
    )
    if not groups:
        return {"groups": [], "total": 0}

    daily_total = get_daily_total()
    max_per_group = get_max_per_group()
    raw_counts = list(counts) if isinstance(counts, (list, tuple)) else []
    explicit: dict[int, int] = {}
    for i in range(len(groups)):
        if i < len(raw_counts):
            try:
                value = int(raw_counts[i])
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                explicit[i] = min(value, max_per_group)

    if len(explicit) == len(groups):
        final_counts = [explicit[i] for i in range(len(groups))]
    else:
        remaining = max(0, daily_total - sum(explicit.values()))
        unspecified = [i for i in range(len(groups)) if i not in explicit]
        shares = distribute(len(unspecified), remaining)
        final_counts = []
        share_idx = 0
        for i in range(len(groups)):
            if i in explicit:
                final_counts.append(explicit[i])
            else:
                final_counts.append(shares[share_idx])
                share_idx += 1

    logger.info("  plan: %s", list(zip(groups, final_counts)))
    plan = [
        _fetch_for_group(
            muscle, count, collector, user_id=user_id, today=today, level=level
        )
        for muscle, count in zip(groups, final_counts)
    ]
    total = sum(item["count"] for item in plan)
    logger.info("TOOL plan_workout returned %d exercises across %d groups", total, len(groups))
    return {"groups": plan, "total": total}


def run_find_exercise(
    name: Any,
    collector: list[dict[str, Any]],
    *,
    user_id: int | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Look up a single exercise by name and add it to ``collector``."""
    today = today or date.today()
    name = str(name or "").strip()
    logger.info("TOOL find_exercise(%r, user=%s)", name, user_id)
    if not name:
        return {"found": False, "name": name}

    exercise = rag_store.find_by_name(name)
    if not exercise:
        logger.info("  find_exercise: no match for %r", name)
        return {"found": False, "name": name}

    summary = _summary(exercise)
    primary = (exercise.get("primaryMuscles") or ["exercise"])[0]
    tagged = dict(summary)
    tagged["group"] = primary
    tagged["images"] = build_image_urls(exercise.get("images"))
    collector.append(tagged)

    if user_id is not None:
        canonical = _canonical_key(primary)
        storage.log_workout(
            user_id, canonical, [(exercise.get("level"), exercise.get("name"))], today.isoformat()
        )

    logger.info("  find_exercise -> %r", summary["name"])
    return {"found": True, "exercise": summary}
