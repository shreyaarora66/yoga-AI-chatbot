"""Shared pytest fixtures.

Tests are split into two kinds:
- Fast unit tests (default): pure logic, no model or network. They stub the
  embedding model and RAG store so they run in milliseconds.
- Slow integration tests (``@pytest.mark.slow``): load the real Sentence
  Transformers model to verify embeddings are actually correct. Run them with
  ``pytest -m slow``; they are skipped automatically if the model cannot load.
"""

from __future__ import annotations

import pytest

from app import config, storage


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Point persistence at a throwaway SQLite file for every test."""
    storage.configure(tmp_path / "test.db")
    storage.init_db()
    yield
    storage.configure(config.DB_FILE)


# A tiny fake exercise database used by the fast unit tests.
FAKE_EXERCISES = [
    {
        "name": "Bench Press",
        "level": "intermediate",
        "equipment": "barbell",
        "primaryMuscles": ["chest"],
        "secondaryMuscles": ["triceps"],
        "instructions": ["Lie on the bench.", "Press the bar up."],
        "images": ["Bench_Press/0.jpg", "Bench_Press/1.jpg"],
        "id": "Bench_Press",
    },
    {
        "name": "Pull Up",
        "level": "intermediate",
        "equipment": "body only",
        "primaryMuscles": ["lats"],
        "secondaryMuscles": ["biceps"],
        "instructions": ["Hang from the bar.", "Pull yourself up."],
        "images": ["Pull_Up/0.jpg"],
        "id": "Pull_Up",
    },
    {
        "name": "Squat",
        "level": "beginner",
        "equipment": "barbell",
        "primaryMuscles": ["quadriceps"],
        "secondaryMuscles": ["glutes"],
        "instructions": ["Stand with the bar.", "Squat down and up."],
        "images": [],
        "id": "Squat",
    },
]


@pytest.fixture
def fake_exercises():
    # Return copies so tests can mutate freely without cross-contamination.
    return [dict(item) for item in FAKE_EXERCISES]
