"""Tests for the adaptive difficulty engine."""

from __future__ import annotations

from datetime import date

from app import progression, storage


def _user():
    return storage.create_user("lifter", "h", "s")


# --------------------------------------------------------------------------- #
# split_counts + state machine (pure)
# --------------------------------------------------------------------------- #
def test_split_counts_even_and_remainder():
    assert progression.split_counts(2, 10) == [5, 5]
    assert progression.split_counts(2, 11) == [5, 6]
    assert progression.split_counts(1, 7) == [7]
    assert progression.split_counts(3, 10) == [3, 3, 4]


def test_next_levels_state_machine():
    assert progression.next_levels(frozenset()) == ["beginner"]
    assert progression.next_levels(frozenset({"beginner"})) == ["beginner", "intermediate"]
    assert progression.next_levels(frozenset({"beginner", "intermediate"})) == ["intermediate"]
    assert progression.next_levels(frozenset({"intermediate"})) == ["intermediate", "expert"]
    assert progression.next_levels(frozenset({"intermediate", "expert"})) == ["intermediate", "expert"]


def test_fallback_order_prepends_planned():
    assert progression.fallback_order(["intermediate", "expert"]) == [
        "intermediate",
        "expert",
        "beginner",
    ]
    assert progression.fallback_order(["beginner"]) == ["beginner", "intermediate", "expert"]


# --------------------------------------------------------------------------- #
# next_level_plan against seeded history
# --------------------------------------------------------------------------- #
def test_no_history_starts_at_beginner():
    uid = _user()
    plan = progression.next_level_plan(uid, "chest", date(2026, 6, 15), 10)
    assert plan == [("beginner", 10)]


def test_after_beginner_serves_beginner_plus_intermediate():
    uid = _user()
    storage.log_workout(uid, "chest", [("beginner", "A"), ("beginner", "B")], "2026-06-14")
    plan = progression.next_level_plan(uid, "chest", date(2026, 6, 15), 10)
    assert plan == [("beginner", 5), ("intermediate", 5)]


def test_after_beginner_intermediate_serves_intermediate():
    uid = _user()
    storage.log_workout(
        uid, "chest", [("beginner", "A"), ("intermediate", "B")], "2026-06-14"
    )
    plan = progression.next_level_plan(uid, "chest", date(2026, 6, 15), 8)
    assert plan == [("intermediate", 8)]


def test_after_intermediate_serves_intermediate_plus_expert():
    uid = _user()
    storage.log_workout(uid, "chest", [("intermediate", "A")], "2026-06-14")
    plan = progression.next_level_plan(uid, "chest", date(2026, 6, 15), 10)
    assert plan == [("intermediate", 5), ("expert", 5)]


def test_gap_over_seven_days_resets_to_beginner():
    uid = _user()
    # Trained intermediate, but 10 days ago -> outside the 7-day window -> reset.
    storage.log_workout(uid, "chest", [("intermediate", "A")], "2026-06-05")
    plan = progression.next_level_plan(uid, "chest", date(2026, 6, 15), 6)
    assert plan == [("beginner", 6)]


def test_same_day_repeat_does_not_advance():
    uid = _user()
    today = date(2026, 6, 15)
    # Already trained beginner today; asking again the same day must NOT advance
    # (stage is computed from prior days only, of which there are none -> beginner).
    storage.log_workout(uid, "chest", [("beginner", "A")], today.isoformat())
    plan = progression.next_level_plan(uid, "chest", today, 4)
    assert plan == [("beginner", 4)]


def test_served_names_within_window():
    uid = _user()
    storage.log_workout(uid, "chest", [("beginner", "Push Up"), ("beginner", "Dip")], "2026-06-14")
    storage.log_workout(uid, "chest", [("beginner", "Ancient")], "2026-05-01")
    names = progression.served_names(uid, "chest", date(2026, 6, 15))
    assert names == {"Push Up", "Dip"}
