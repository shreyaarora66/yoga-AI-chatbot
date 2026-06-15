"""Trainer agent package.

Public surface used by the web layer and tests:
- ``start_conversation`` builds a per-session conversation for the active provider.
- ``run_plan_workout`` is the RAG-backed planning skill the LLM calls as a tool.
"""

from app.agent.providers import (
    GeminiConversation,
    OllamaConversation,
    start_conversation,
)
from app.agent.tools import (
    build_image_urls,
    distribute,
    run_find_exercise,
    run_plan_workout,
    search_muscle,
)

__all__ = [
    "GeminiConversation",
    "OllamaConversation",
    "start_conversation",
    "run_plan_workout",
    "run_find_exercise",
    "search_muscle",
    "distribute",
    "build_image_urls",
]
