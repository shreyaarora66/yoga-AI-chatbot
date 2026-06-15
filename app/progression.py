"""Adaptive difficulty engine.

When the user does not specify a level, difficulty auto-progresses per muscle based
on the level-set used on the most recent training day within a 7-day window:

    no history / gap > 7 days  -> [beginner]
    {beginner}                 -> [beginner, intermediate]
    {beginner, intermediate}   -> [intermediate]
    {intermediate}             -> [intermediate, expert]
    {intermediate, expert}     -> [intermediate, expert]   (cap)

Training the same muscle again on the same day does not advance the stage (state is
read from the most recent *day*). History rows are never deleted here - this module
only reads a windowed slice.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from app import storage

logger = logging.getLogger("trainer.progress")

WINDOW_DAYS = 7

# Difficulty ordering for fallback when a level lacks enough exercises.
LEVEL_ORDER = ("beginner", "intermediate", "expert")


def _since_day(today: date) -> str:
    return (today - timedelta(days=WINDOW_DAYS)).isoformat()


def split_counts(num_levels: int, total: int) -> list[int]:
    """Split ``total`` across ``num_levels`` as evenly as possible (10/2 -> [5, 5]).

    Remainder goes to the later levels (the harder ones in a two-level stage).
    """
    if num_levels <= 0:
        return []
    total = max(0, total)
    base = total // num_levels
    remainder = total % num_levels
    counts = [base] * num_levels
    for i in range(num_levels - remainder, num_levels):
        counts[i] += 1
    return counts


def last_level_set(user_id: int, muscle: str, today: date) -> frozenset[str]:
    """Levels used on the most recent PRIOR training day for ``muscle``.

    Only days strictly before ``today`` count, so training the same muscle again on
    the same day does not advance the stage (it repeats today's stage with fresh,
    unique exercises). Days older than the 7-day window are ignored (a reset).
    """
    rows = storage.recent_logs(user_id, muscle, _since_day(today))
    today_str = today.isoformat()
    prior = [row for row in rows if row["day"] < today_str]
    if not prior:
        return frozenset()
    latest_day = prior[0]["day"]  # rows are ordered day DESC -> first prior is newest
    levels = {
        row["level"]
        for row in prior
        if row["day"] == latest_day and row["level"]
    }
    return frozenset(levels)


def next_levels(last: frozenset[str]) -> list[str]:
    """The state machine: given last session's level-set, pick the next levels."""
    if not last:
        return ["beginner"]
    if last == frozenset({"beginner"}):
        return ["beginner", "intermediate"]
    if last == frozenset({"beginner", "intermediate"}):
        return ["intermediate"]
    if last == frozenset({"intermediate"}):
        return ["intermediate", "expert"]
    if last == frozenset({"intermediate", "expert"}):
        return ["intermediate", "expert"]
    # Defensive fallbacks for any unexpected combination.
    if "expert" in last:
        return ["intermediate", "expert"]
    if "intermediate" in last:
        return ["intermediate"]
    return ["beginner", "intermediate"]


def next_level_plan(
    user_id: int, muscle: str, today: date, count: int
) -> list[tuple[str, int]]:
    """Return [(level, count), ...] for an auto-difficulty request of ``count`` items."""
    last = last_level_set(user_id, muscle, today)
    levels = next_levels(last)
    counts = split_counts(len(levels), count)
    plan = [(level, n) for level, n in zip(levels, counts)]
    logger.info(
        "Progression user=%s muscle=%s last=%s -> plan=%s (count=%d)",
        user_id,
        muscle,
        sorted(last) if last else "none",
        plan,
        count,
    )
    return plan


def served_names(user_id: int, muscle: str, today: date) -> set[str]:
    """Exercise names already given for this muscle within the 7-day window."""
    rows = storage.recent_logs(user_id, muscle, _since_day(today))
    return {row["exercise_name"] for row in rows if row["exercise_name"]}


def fallback_order(planned_levels: list[str]) -> list[str]:
    """Levels to try when topping up a shortage: planned first, then the rest."""
    order = list(dict.fromkeys(planned_levels))  # de-dupe, keep order
    for level in LEVEL_ORDER:
        if level not in order:
            order.append(level)
    return order
