"""Embedding tests.

Fast tests cover the pure helpers (vector normalization, prompt prefixing).
The ``@pytest.mark.slow`` tests load the REAL Sentence Transformers model and
verify it produces correct, useful output:
  - the right dimensionality (384 for all-MiniLM-L6-v2),
  - unit-normalized vectors,
  - semantically related text scores higher than unrelated text.

Run them with:  pytest -m slow
They auto-skip if the model cannot be loaded (e.g. not downloaded / offline).
"""

from __future__ import annotations

import numpy as np
import pytest

from app import embeddings


# --------------------------------------------------------------------------- #
# Fast unit tests (no model)
# --------------------------------------------------------------------------- #
def test_to_vector_normalizes():
    vec = embeddings.to_vector([3.0, 4.0])  # norm 5 -> normalized to 0.6, 0.8
    assert pytest.approx(vec, abs=1e-6) == [0.6, 0.8]


def test_to_vector_averages_token_matrix():
    # A 2-token x 2-dim matrix should be mean-pooled to a single 2-dim vector.
    vec = embeddings.to_vector([[1.0, 0.0], [0.0, 1.0]])
    assert len(vec) == 2
    assert pytest.approx(np.linalg.norm(vec), abs=1e-6) == 1.0


def test_prepare_text_adds_bge_prefix(monkeypatch):
    monkeypatch.setattr(embeddings, "get_embed_model", lambda: "BAAI/bge-small-en")
    assert embeddings.prepare_text("hello", "query").startswith("query: ")
    assert embeddings.prepare_text("hello", "document").startswith("passage: ")


def test_prepare_text_no_prefix_for_minilm(monkeypatch):
    monkeypatch.setattr(
        embeddings, "get_embed_model", lambda: "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert embeddings.prepare_text("hello", "query") == "hello"


# --------------------------------------------------------------------------- #
# Slow integration tests (real model) - verify the embeddings are correct
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model_available():
    try:
        embeddings.get_local_model()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"Embedding model unavailable: {error}")
    return True


@pytest.mark.slow
def test_embedding_dimension_and_norm(model_available):
    vec = embeddings.embed_text("barbell bench press for the chest", text_type="query")
    assert len(vec) == 384, "all-MiniLM-L6-v2 produces 384-dim vectors"
    norm = float(np.linalg.norm(vec))
    assert pytest.approx(norm, abs=1e-2) == 1.0, "vectors should be unit-normalized"


@pytest.mark.slow
def test_embedding_is_deterministic(model_available):
    a = embeddings.embed_text("push ups", text_type="query")
    b = embeddings.embed_text("push ups", text_type="query")
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.slow
def test_embedding_semantic_similarity(model_available):
    """A chest query must be closer to a chest exercise than to an unrelated one."""
    query = np.array(embeddings.embed_text("chest workout", text_type="query"))
    chest = np.array(
        embeddings.embed_text(
            "Bench Press. Primary muscles: chest. Press the barbell from your chest.",
            text_type="document",
        )
    )
    unrelated = np.array(
        embeddings.embed_text(
            "Calf Raise. Primary muscles: calves. Rise onto your toes.",
            text_type="document",
        )
    )

    sim_chest = float(query @ chest)
    sim_unrelated = float(query @ unrelated)
    assert sim_chest > sim_unrelated, (
        f"chest query should rank chest exercise higher "
        f"({sim_chest:.3f} vs {sim_unrelated:.3f})"
    )
