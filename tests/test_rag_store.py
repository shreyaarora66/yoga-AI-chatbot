"""Unit tests for the vector store math and ranking (no model needed)."""

from __future__ import annotations

import numpy as np

from app import rag_store


def test_exercise_to_text_includes_key_fields(fake_exercises):
    text = rag_store.exercise_to_text(fake_exercises[0])
    assert "Name: Bench Press" in text
    assert "Primary muscles: chest" in text
    assert "Secondary muscles: triceps" in text
    assert "Instructions:" in text


def test_cosine_similarity_identical_vectors_is_one():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    scores = rag_store.cosine_similarity(query, matrix)
    assert scores[0] == np.float32(1.0)
    assert scores[1] == np.float32(0.0)


def test_cosine_similarity_zero_query_returns_zeros():
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    query = np.array([0.0, 0.0], dtype=np.float32)
    scores = rag_store.cosine_similarity(query, matrix)
    assert np.all(scores == 0)


def test_cosine_similarity_orthogonal_and_opposite():
    matrix = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([1.0, 0.0], dtype=np.float32)
    scores = rag_store.cosine_similarity(query, matrix)
    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], -1.0)
    assert np.isclose(scores[2], 0.0)


def test_keyword_boost_rewards_matching_primary_muscle(fake_exercises):
    bench = fake_exercises[0]
    boosted = rag_store.keyword_boost("i want chest exercises", bench, 0.5)
    assert boosted > 0.5  # "chest" is a primary muscle -> boosted


def test_keyword_boost_no_match_keeps_base(fake_exercises):
    bench = fake_exercises[0]
    same = rag_store.keyword_boost("calf raises", bench, 0.5)
    assert same == 0.5


def test_normalize_level_maps_synonyms():
    assert rag_store.normalize_level("beginner") == "beginner"
    assert rag_store.normalize_level("basic") == "beginner"
    assert rag_store.normalize_level("advanced") == "expert"
    assert rag_store.normalize_level("EXPERT") == "expert"
    assert rag_store.normalize_level("moderate") == "intermediate"
    assert rag_store.normalize_level("nonsense") is None
    assert rag_store.normalize_level(None) is None


def _stub_store(monkeypatch, fake_exercises):
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    ids = [ex["id"] for ex in fake_exercises]
    monkeypatch.setattr(rag_store, "load_store", lambda: (embeddings, ids, fake_exercises))


def test_search_exercises_level_filter(monkeypatch, fake_exercises):
    _stub_store(monkeypatch, fake_exercises)
    query_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    results = rag_store.search_exercises(query_vec, "workout", top_k=5, level="beginner")
    # Only the Squat is beginner in the fake set.
    assert [r["exercise"]["name"] for r in results] == ["Squat"]


def test_search_exercises_excludes_names(monkeypatch, fake_exercises):
    _stub_store(monkeypatch, fake_exercises)
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = rag_store.search_exercises(
        query_vec, "chest", top_k=5, exclude_names={"bench press"}
    )
    names = {r["exercise"]["name"] for r in results}
    assert "Bench Press" not in names


def test_find_by_name_exact_and_fuzzy(monkeypatch, fake_exercises):
    _stub_store(monkeypatch, fake_exercises)
    assert rag_store.find_by_name("Bench Press")["name"] == "Bench Press"
    assert rag_store.find_by_name("bench")["name"] == "Bench Press"
    assert rag_store.find_by_name("totally unknown move") is None


def test_search_exercises_ranks_relevant_first(monkeypatch, fake_exercises):
    # Build a fake store: each exercise gets a 3-dim one-hot vector.
    # chest -> [1,0,0], lats -> [0,1,0], quads -> [0,0,1]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    ids = [ex["id"] for ex in fake_exercises]

    monkeypatch.setattr(
        rag_store, "load_store", lambda: (embeddings, ids, fake_exercises)
    )

    # Query vector aligned with the chest exercise.
    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = rag_store.search_exercises(query_vec, "chest workout", top_k=2)

    assert len(results) == 2
    assert results[0]["exercise"]["name"] == "Bench Press"
    # Scores are sorted descending.
    assert results[0]["score"] >= results[1]["score"]
