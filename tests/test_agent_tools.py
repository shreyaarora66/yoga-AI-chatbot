"""Unit tests for the agent's planning logic (RAG calls are stubbed)."""

from __future__ import annotations

from datetime import date

import numpy as np

from app import rag_store, storage
from app.agent import tools


# --------------------------------------------------------------------------- #
# distribute()
# --------------------------------------------------------------------------- #
def test_distribute_single_group_gets_all():
    assert tools.distribute(1, 10) == [10]


def test_distribute_two_groups_even():
    assert tools.distribute(2, 10) == [5, 5]


def test_distribute_three_groups_remainder_to_last():
    assert tools.distribute(3, 10) == [3, 3, 4]


def test_distribute_two_groups_fifteen():
    # New default daily total of 15.
    assert tools.distribute(2, 15) == [7, 8]


def test_distribute_three_groups_fifteen():
    assert tools.distribute(3, 15) == [5, 5, 5]


def test_distribute_four_groups():
    assert tools.distribute(4, 10) == [2, 2, 3, 3]


def test_distribute_minimum_one_each():
    # 2 exercises across 3 groups -> nobody gets zero.
    assert tools.distribute(3, 2) == [1, 1, 1]


def test_distribute_zero_groups():
    assert tools.distribute(0, 10) == []


# --------------------------------------------------------------------------- #
# build_image_urls()
# --------------------------------------------------------------------------- #
def test_build_image_urls_prefixes_base(monkeypatch):
    monkeypatch.setattr(tools, "get_image_base_url", lambda: "https://cdn.example.com")
    urls = tools.build_image_urls(["Bench_Press/0.jpg", "Bench_Press/1.jpg"])
    assert urls == [
        "https://cdn.example.com/Bench_Press/0.jpg",
        "https://cdn.example.com/Bench_Press/1.jpg",
    ]


def test_build_image_urls_passthrough_absolute(monkeypatch):
    monkeypatch.setattr(tools, "get_image_base_url", lambda: "https://cdn.example.com")
    urls = tools.build_image_urls(["https://other.com/x.jpg"])
    assert urls == ["https://other.com/x.jpg"]


def test_build_image_urls_handles_non_list():
    assert tools.build_image_urls(None) == []
    assert tools.build_image_urls("nope") == []


# --------------------------------------------------------------------------- #
# run_plan_workout() with stubbed embedding + RAG search
# --------------------------------------------------------------------------- #
def _stub_rag(monkeypatch):
    """Make embed_text + rag_search return deterministic fake exercises."""
    monkeypatch.setattr(tools, "embed_text", lambda text, text_type="query": [0.0, 0.0, 0.0])

    def fake_search(vector, query, top_k):
        muscle = query.split()[0]
        return [
            {
                "score": 1.0,
                "exercise": {
                    "name": f"{muscle}-ex-{i}",
                    "level": "beginner",
                    "equipment": "none",
                    "primaryMuscles": [muscle],
                    "instructions": ["step one", "step two"],
                    "images": [f"{muscle}_{i}/0.jpg"],
                },
            }
            for i in range(top_k)
        ]

    monkeypatch.setattr(tools, "rag_search", fake_search)


def test_run_plan_workout_no_counts_splits_daily_total(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 10)
    collector: list[dict] = []
    result = tools.run_plan_workout(["chest", "back", "legs"], [], collector)

    assert result["total"] == 10
    counts = [g["count"] for g in result["groups"]]
    assert counts == [3, 3, 4]
    assert len(collector) == 10
    # Each collected exercise carries group + image URLs for the UI.
    assert all("group" in ex and "images" in ex for ex in collector)


def test_run_plan_workout_explicit_counts_are_honored(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 10)
    collector: list[dict] = []
    result = tools.run_plan_workout(["chest", "back"], [3, 2], collector)

    counts = [g["count"] for g in result["groups"]]
    assert counts == [3, 2]
    assert result["total"] == 5


def test_run_plan_workout_partial_counts_fill_remaining(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 10)
    collector: list[dict] = []
    # chest explicitly 4; back unspecified -> gets remaining 6.
    result = tools.run_plan_workout(["chest", "back"], [4], collector)

    counts = [g["count"] for g in result["groups"]]
    assert counts == [4, 6]


def test_run_plan_workout_single_string_muscle(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 10)
    collector: list[dict] = []
    result = tools.run_plan_workout("chest", [], collector)
    assert result["total"] == 10
    assert result["groups"][0]["muscle"] == "chest"


def test_run_plan_workout_empty_returns_nothing(monkeypatch):
    _stub_rag(monkeypatch)
    collector: list[dict] = []
    result = tools.run_plan_workout([], [], collector)
    assert result["total"] == 0
    assert collector == []


# --- New behaviour: honor exact requested numbers, default total of 15 -------- #
def test_run_plan_workout_large_explicit_count(monkeypatch):
    """'20 shoulder exercises' must fetch 20, not clamp down to 10."""
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 15)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    collector: list[dict] = []
    result = tools.run_plan_workout(["shoulder"], [20], collector)

    assert result["total"] == 20
    assert result["groups"][0]["count"] == 20
    assert len(collector) == 20


def test_run_plan_workout_three_explicit_counts_in_order(monkeypatch):
    """'10 shoulder, 10 chest and 10 leg' -> 10 each, groups kept in order."""
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 15)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    collector: list[dict] = []
    result = tools.run_plan_workout(["shoulder", "chest", "legs"], [10, 10, 10], collector)

    assert [g["count"] for g in result["groups"]] == [10, 10, 10]
    assert [g["muscle"] for g in result["groups"]] == ["shoulder", "chest", "legs"]
    assert result["total"] == 30
    assert len(collector) == 30


def test_run_plan_workout_default_total_is_fifteen(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 15)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    collector: list[dict] = []
    result = tools.run_plan_workout(["chest", "back"], [], collector)
    assert [g["count"] for g in result["groups"]] == [7, 8]
    assert result["total"] == 15


def test_run_plan_workout_clamps_above_max(monkeypatch):
    _stub_rag(monkeypatch)
    monkeypatch.setattr(tools, "get_daily_total", lambda: 15)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    collector: list[dict] = []
    result = tools.run_plan_workout(["shoulder"], [100], collector)
    assert result["groups"][0]["count"] == 30


# --------------------------------------------------------------------------- #
# Adaptive (user_id) path: auto difficulty, uniqueness, logging, find_exercise
# --------------------------------------------------------------------------- #
def _stub_rag_leveled(monkeypatch):
    """A stub that honors level + exclude_names, generating deterministic names."""
    monkeypatch.setattr(tools, "embed_text", lambda text, text_type="query": [0.0, 0.0, 0.0])

    def fake_search(vector, query, top_k, level=None, exclude_names=None):
        muscle = query.split()[0]
        excl = {e.lower() for e in (exclude_names or set())}
        out = []
        i = 0
        while len(out) < top_k and i < top_k + 100:
            name = f"{muscle}-{level or 'any'}-{i}"
            i += 1
            if name.lower() in excl:
                continue
            out.append(
                {
                    "score": 1.0,
                    "exercise": {
                        "name": name,
                        "level": level or "beginner",
                        "equipment": "none",
                        "primaryMuscles": [muscle],
                        "instructions": ["step"],
                        "images": [],
                    },
                }
            )
        return out

    monkeypatch.setattr(tools, "rag_search", fake_search)


def test_user_path_auto_level_beginner_and_logs(monkeypatch):
    _stub_rag_leveled(monkeypatch)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    uid = storage.create_user("u1", "h", "s")
    collector: list[dict] = []

    result = tools.run_plan_workout(
        ["chest"], [5], collector, user_id=uid, today=date(2026, 6, 15)
    )
    assert result["total"] == 5
    assert result["groups"][0]["levels"] == ["beginner"]  # no history -> beginner
    # The session was logged so future sessions can progress.
    assert storage.muscle_totals(uid)["chest"] == 5


def test_user_path_progresses_next_day(monkeypatch):
    _stub_rag_leveled(monkeypatch)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    uid = storage.create_user("u2", "h", "s")

    tools.run_plan_workout(["chest"], [4], [], user_id=uid, today=date(2026, 6, 14))
    collector: list[dict] = []
    result = tools.run_plan_workout(
        ["chest"], [4], collector, user_id=uid, today=date(2026, 6, 15)
    )
    # Day 2 after a beginner day -> beginner + intermediate.
    assert result["groups"][0]["levels"] == ["beginner", "intermediate"]


def test_user_path_unique_across_same_day(monkeypatch):
    _stub_rag_leveled(monkeypatch)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    uid = storage.create_user("u3", "h", "s")
    today = date(2026, 6, 15)

    first: list[dict] = []
    tools.run_plan_workout(["chest"], [3], first, user_id=uid, today=today)
    second: list[dict] = []
    tools.run_plan_workout(["chest"], [3], second, user_id=uid, today=today)

    names1 = {e["name"] for e in first}
    names2 = {e["name"] for e in second}
    assert names1.isdisjoint(names2)  # no repeats within the window


def test_explicit_level_overrides_progression(monkeypatch):
    _stub_rag_leveled(monkeypatch)
    monkeypatch.setattr(tools, "get_max_per_group", lambda: 30)
    uid = storage.create_user("u4", "h", "s")
    collector: list[dict] = []
    result = tools.run_plan_workout(
        ["chest"], [3], collector, user_id=uid, today=date(2026, 6, 15), level="expert"
    )
    assert result["groups"][0]["levels"] == ["expert"]


def test_run_find_exercise_found_and_logged(monkeypatch):
    monkeypatch.setattr(
        rag_store,
        "find_by_name",
        lambda name: {
            "name": "Barbell Curl",
            "level": "beginner",
            "equipment": "barbell",
            "primaryMuscles": ["biceps"],
            "instructions": ["curl it"],
            "images": [],
        },
    )
    uid = storage.create_user("u5", "h", "s")
    collector: list[dict] = []
    result = tools.run_find_exercise("barbell curl", collector, user_id=uid, today=date(2026, 6, 15))
    assert result["found"] is True
    assert result["exercise"]["name"] == "Barbell Curl"
    assert collector[0]["group"] == "biceps"
    assert storage.muscle_totals(uid)["biceps"] == 1


def test_run_find_exercise_not_found(monkeypatch):
    monkeypatch.setattr(rag_store, "find_by_name", lambda name: None)
    collector: list[dict] = []
    result = tools.run_find_exercise("does not exist", collector)
    assert result["found"] is False
    assert collector == []
