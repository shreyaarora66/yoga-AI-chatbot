"""Build the Hugging Face embedding index from data.json.

Run from the project root with:  python -m scripts.build_embeddings
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

from app import config
from app.logging_config import setup_logging
from app.ssl_fix import configure_ssl

setup_logging()
configure_ssl()

from app.embeddings import embed_texts
from app.rag_store import exercise_to_text, invalidate_store_cache

logger = logging.getLogger("trainer.rag")

BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))
BATCH_DELAY = float(os.getenv("HF_BATCH_DELAY", "0.5"))


def main() -> None:
    if not config.DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {config.DATA_FILE}")

    with config.DATA_FILE.open("r", encoding="utf-8") as handle:
        exercises = json.load(handle)

    documents = [exercise_to_text(exercise) for exercise in exercises]
    ids = [exercise["id"] for exercise in exercises]
    model = config.get_embed_model()

    logger.info(
        "Building embeddings: %d exercises, mode=%s, model=%s, batch_size=%d",
        len(exercises),
        config.get_embed_mode(),
        model,
        BATCH_SIZE,
    )

    all_embeddings: list[list[float]] = []
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    started = time.perf_counter()

    for batch_index in range(total_batches):
        start = batch_index * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(documents))
        batch = documents[start:end]

        logger.info(
            "Embedding batch %d/%d (%d items)...",
            batch_index + 1,
            total_batches,
            len(batch),
        )
        batch_embeddings = embed_texts(batch, text_type="document")
        all_embeddings.extend(batch_embeddings)

        if config.get_embed_mode() == "api" and batch_index + 1 < total_batches and BATCH_DELAY > 0:
            time.sleep(BATCH_DELAY)

    matrix = np.array(all_embeddings, dtype=np.float32)

    config.RAG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, matrix)

    with config.INDEX_FILE.open("w", encoding="utf-8") as handle:
        json.dump(ids, handle, ensure_ascii=False, indent=2)

    with config.EXERCISES_FILE.open("w", encoding="utf-8") as handle:
        json.dump(exercises, handle, ensure_ascii=False)

    invalidate_store_cache()
    elapsed_s = time.perf_counter() - started
    logger.info(
        "RAG index saved to %s (shape=%s, %.1f s total)",
        config.RAG_DIR,
        matrix.shape,
        elapsed_s,
    )


if __name__ == "__main__":
    main()
