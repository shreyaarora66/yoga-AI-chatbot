"""Central logging setup for the trainer application.

Call ``setup_logging()`` once at process start (from ``app.main``). Logs go to:

    1. The terminal (stdout)
    2. A plain-text file: ``logs/trainer.txt`` (append mode, UTF-8)

All modules use named loggers under the ``trainer.*`` namespace:

    trainer.server   — HTTP endpoints, auth, startup
    trainer.agent    — LLM providers and tool calls
    trainer.rag      — embedding model load and vector search
    trainer.storage  — SQLite persistence
    trainer.auth     — login/register outcomes
    trainer.progress — adaptive difficulty decisions

Set ``LOG_LEVEL`` in ``.env`` (DEBUG, INFO, WARNING, ERROR). Default: INFO.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from app.config import LOG_DIR, LOG_FILE

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    formatter = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    root.setLevel(level)

    startup = logging.getLogger("trainer.server")
    startup.info("Logging to %s", _display_path(LOG_FILE))

    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)
