"""Tests for the SQLite persistence layer (uses the isolated temp DB fixture)."""

from __future__ import annotations

import sqlite3

import pytest

from app import storage


def _make_user(username="alice"):
    return storage.create_user(username, "hash", "salt")


def test_create_user_and_lookup():
    uid = _make_user()
    by_name = storage.get_user_by_username("alice")
    assert by_name["id"] == uid
    assert storage.get_user(uid)["username"] == "alice"


def test_duplicate_username_raises():
    _make_user("bob")
    with pytest.raises(sqlite3.IntegrityError):
        _make_user("bob")


def test_token_round_trip():
    uid = _make_user()
    token = storage.create_token(uid)
    assert storage.user_id_for_token(token) == uid
    storage.delete_token(token)
    assert storage.user_id_for_token(token) is None
    assert storage.user_id_for_token("nope") is None


def test_profile_upsert_merges_fields():
    uid = _make_user()
    storage.upsert_profile(uid, height_cm=180.0)
    storage.upsert_profile(uid, weight_kg=75.0)
    profile = storage.get_profile(uid)
    # Setting weight must not wipe the previously saved height.
    assert profile["height_cm"] == 180.0
    assert profile["weight_kg"] == 75.0

    storage.upsert_profile(uid, avatar_path="/tmp/a.png")
    assert storage.get_profile(uid)["avatar_path"] == "/tmp/a.png"
    assert storage.get_profile(uid)["height_cm"] == 180.0


def test_log_and_recent_logs_window():
    uid = _make_user()
    storage.log_workout(uid, "chest", [("beginner", "Push Up"), ("beginner", "Dip")], "2026-06-10")
    storage.log_workout(uid, "chest", [("beginner", "Old Move")], "2026-05-01")

    recent = storage.recent_logs(uid, "chest", "2026-06-08")
    names = {row["exercise_name"] for row in recent}
    assert names == {"Push Up", "Dip"}  # the May row is outside the window


def test_history_retained_beyond_seven_days():
    uid = _make_user()
    storage.log_workout(uid, "chest", [("beginner", "Push Up")], "2026-01-01")
    # Full history (no purge) still has the old day.
    days = storage.history_by_day(uid)
    assert any(d["day"] == "2026-01-01" for d in days)


def test_history_by_day_grouping():
    uid = _make_user()
    storage.log_workout(uid, "chest", [("beginner", "A"), ("intermediate", "B")], "2026-06-10")
    storage.log_workout(uid, "biceps", [("beginner", "C")], "2026-06-10")
    storage.log_workout(uid, "chest", [("beginner", "D")], "2026-06-11")

    days = storage.history_by_day(uid)
    # Newest day first.
    assert days[0]["day"] == "2026-06-11"
    day10 = next(d for d in days if d["day"] == "2026-06-10")
    chest = next(g for g in day10["groups"] if g["muscle"] == "chest")
    assert chest["count"] == 2
    assert set(chest["exercises"]) == {"A", "B"}
    assert set(chest["levels"]) == {"beginner", "intermediate"}


def test_muscle_totals_all_time():
    uid = _make_user()
    storage.log_workout(uid, "chest", [("beginner", "A"), ("beginner", "B")], "2026-06-10")
    storage.log_workout(uid, "chest", [("beginner", "C")], "2026-06-11")
    storage.log_workout(uid, "biceps", [("beginner", "D")], "2026-06-11")
    totals = storage.muscle_totals(uid)
    assert totals["chest"] == 3
    assert totals["biceps"] == 1
