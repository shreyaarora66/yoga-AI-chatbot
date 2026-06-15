"""Embedding provider with local Sentence Transformers (default) or HF Inference API.

Settings (model name, mode, token, offline flag) come from ``app.config``.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

import numpy as np

from app.config import (
    embed_offline_enabled,
    get_embed_mode,
    get_embed_model,
    get_hf_token,
)

logger = logging.getLogger("trainer.rag")

_inference_client = None


def prepare_text(text: str, text_type: str) -> str:
    model = get_embed_model().lower()
    if "bge" in model:
        prefix = "query: " if text_type == "query" else "passage: "
        return prefix + text
    return text


def to_vector(raw: Any) -> list[float]:
    array = np.array(raw, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=0)
    elif array.ndim > 2:
        array = array.reshape(-1, array.shape[-1]).mean(axis=0)

    norm = float(np.linalg.norm(array))
    if norm > 0:
        array = array / norm
    return array.tolist()


@lru_cache(maxsize=1)
def get_local_model():
    # The model is downloaded once and cached. Default to offline mode so we skip
    # the slow per-file online cache-verification requests on every load (~2 min).
    # Set EMBED_OFFLINE=0 to allow downloads if the cache is missing.
    if embed_offline_enabled():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    model_name = get_embed_model()
    logger.info("Loading embedding model %s (offline=%s)...", model_name, embed_offline_enabled())
    started = time.perf_counter()
    token = get_hf_token()
    model = SentenceTransformer(model_name, token=token)
    elapsed_s = time.perf_counter() - started
    logger.info("Embedding model ready in %.1f s", elapsed_s)
    return model


def get_inference_client():
    global _inference_client
    if _inference_client is not None:
        return _inference_client

    token = get_hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required for EMBED_MODE=api. "
            "Create a token with Inference permissions at "
            "https://huggingface.co/settings/tokens, "
            "or set EMBED_MODE=local in .env."
        )

    from huggingface_hub import InferenceClient

    _inference_client = InferenceClient(model=get_embed_model(), token=token)
    return _inference_client


def embed_text_api(text: str, text_type: str = "document") -> list[float]:
    client = get_inference_client()
    prepared = prepare_text(text, text_type)
    max_retries = int(os.getenv("HF_MAX_RETRIES", "6"))
    delay = float(os.getenv("HF_RETRY_DELAY", "2.0"))

    for attempt in range(max_retries):
        try:
            result = client.feature_extraction(prepared)
            return to_vector(result)
        except Exception as error:
            message = str(error).lower()
            retryable = any(
                marker in message
                for marker in (
                    "503",
                    "429",
                    "loading",
                    "timeout",
                    "rate",
                    "temporarily",
                )
            )
            if not retryable or attempt == max_retries - 1:
                raise RuntimeError(
                    f"Hugging Face API embedding failed for {get_embed_model()}: {error}"
                ) from error
            wait = delay * (attempt + 1)
            logger.warning(
                "HF embed retry in %.1fs (%d/%d): %s",
                wait,
                attempt + 1,
                max_retries,
                error,
            )
            time.sleep(wait)

    raise RuntimeError("Hugging Face API embedding failed after retries.")


def embed_texts_local(texts: list[str], text_type: str = "document") -> list[list[float]]:
    model = get_local_model()
    prepared = [prepare_text(text, text_type) for text in texts]
    batch_size = int(os.getenv("LOCAL_EMBED_BATCH_SIZE", "32"))
    vectors = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=len(prepared) > 32,
        normalize_embeddings=True,
    )
    return [vector.tolist() for vector in vectors]


def embed_text(text: str, text_type: str = "document") -> list[float]:
    if get_embed_mode() == "api":
        return embed_text_api(text, text_type=text_type)
    return embed_texts_local([text], text_type=text_type)[0]


def embed_texts(texts: list[str], text_type: str = "document") -> list[list[float]]:
    if get_embed_mode() == "api":
        request_delay = float(os.getenv("HF_REQUEST_DELAY", "0.12"))
        vectors: list[list[float]] = []
        for index, text in enumerate(texts, start=1):
            vectors.append(embed_text_api(text, text_type=text_type))
            if request_delay > 0 and index < len(texts):
                time.sleep(request_delay)
        return vectors

    return embed_texts_local(texts, text_type=text_type)
