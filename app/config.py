"""Central configuration: filesystem paths and environment-driven settings.

Every module reads its settings from here so configuration lives in one place.
Paths are anchored to the project root (the parent of this ``app`` package), so
the app behaves the same regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
WEB_DIR = PROJECT_ROOT / "web"
DATA_FILE = PROJECT_ROOT / "data.json"

RAG_DIR = PROJECT_ROOT / "rag_db"
EMBEDDINGS_FILE = RAG_DIR / "embeddings.npy"
INDEX_FILE = RAG_DIR / "index.json"
EXERCISES_FILE = RAG_DIR / "exercises.json"

# Per-user persistence (accounts, profiles, workout history) + uploaded avatars.
USER_DATA_DIR = PROJECT_ROOT / "user_data"
DB_FILE = USER_DATA_DIR / "trainer.db"
AVATAR_DIR = USER_DATA_DIR / "avatars"

# Application logs (plain-text files, created on server start).
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "trainer.txt"

# Load .env from the project root once, on import. override=True makes the .env file
# authoritative even when a variable (e.g. OLLAMA_MODEL) is already set in the OS
# environment, so editing .env reliably changes the app's behaviour after a restart.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _is_falsey(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no", "off"}


# --------------------------------------------------------------------------- #
# Agent / LLM provider settings
# --------------------------------------------------------------------------- #
def get_provider() -> str:
    return os.getenv("AGENT_PROVIDER", "gemini").strip().lower()


def get_api_key() -> str | None:
    key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key or key.startswith("your_"):
        return None
    return key


def get_model_name() -> str:
    if get_provider() == "ollama":
        return os.getenv("OLLAMA_MODEL", "mistral").strip()
    return os.getenv("AGENT_MODEL", "gemini-2.5-flash").strip()


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_ollama_tool_rounds() -> int:
    return int(os.getenv("OLLAMA_TOOL_ROUNDS", "1"))


def is_agent_enabled() -> bool:
    """True when the agent is toggled on and the selected provider is usable."""
    if _is_falsey(os.getenv("USE_LLM_AGENT", "true")):
        return False
    if get_provider() == "ollama":
        return True  # we trust the local Ollama server is running
    return bool(get_api_key())


# --------------------------------------------------------------------------- #
# Workout / RAG behaviour
# --------------------------------------------------------------------------- #
def get_daily_total() -> int:
    """Exercises to split across groups when the user gives no explicit counts."""
    return int(os.getenv("DAILY_EXERCISE_TOTAL", "15"))


def get_max_per_group() -> int:
    """Upper bound on exercises fetched for a single muscle group when the user
    asks for an explicit number (e.g. "20 shoulder exercises")."""
    return int(os.getenv("MAX_EXERCISES_PER_GROUP", "30"))


def get_image_base_url() -> str:
    """Base URL prefixed to the extended image paths stored in data.json."""
    return os.getenv("EXERCISE_IMAGE_BASE", "https://ik.imagekit.io/yuhonas/").rstrip("/")


# --------------------------------------------------------------------------- #
# Embedding settings
# --------------------------------------------------------------------------- #
def get_embed_model() -> str:
    return os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embed_mode() -> str:
    return os.getenv("EMBED_MODE", "local").strip().lower()


def get_hf_token() -> str | None:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    if not token or token.startswith("your_"):
        return None
    return token


def embed_offline_enabled() -> bool:
    return not _is_falsey(os.getenv("EMBED_OFFLINE", "1"))
