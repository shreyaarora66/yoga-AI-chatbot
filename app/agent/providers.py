"""LLM providers for the trainer agent: Google Gemini (cloud) and Ollama (local).

Both providers expose the same skills, ``plan_workout`` and ``find_exercise`` (see
``app.agent.tools``), and the same conversational behaviour (see
``app.agent.prompts``). The provider is selected with ``AGENT_PROVIDER``.

Each conversation is bound to a ``user_id`` so the planning tool can read/write that
user's workout history for adaptive difficulty.

NOTE: do NOT add ``from __future__ import annotations`` here. google-genai's
automatic function calling reads the tool function's raw __annotations__ and runs
isinstance() against them; stringized annotations break it.
"""

import json
import logging
import time
from datetime import date
from typing import Any

from app.config import (
    get_api_key,
    get_model_name,
    get_ollama_base_url,
    get_ollama_tool_rounds,
    get_provider,
)
from app.agent.prompts import (
    FIND_EXERCISE_DESCRIPTION,
    PLAN_WEEKLY_WORKOUT_DESCRIPTION,
    PLAN_WORKOUT_DESCRIPTION,
    SYSTEM_INSTRUCTION,
)
from app.agent.tools import run_find_exercise, run_plan_weekly_workout, run_plan_workout

logger = logging.getLogger("trainer.agent")

_gemini_client = None


# --------------------------------------------------------------------------- #
# Gemini provider
# --------------------------------------------------------------------------- #
def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    from google import genai

    logger.info("Creating Gemini client (model=%s)", get_model_name())
    _gemini_client = genai.Client(api_key=get_api_key())
    return _gemini_client


def _make_gemini_tools(
    collector: list[dict[str, Any]],
    user_id: int | None,
    summary_holder: dict[str, str | None],
):
    # NOTE: keep the return annotations as bare `dict`. With automatic function
    # calling, google-genai runs isinstance() against them; `dict[str, Any]` is not
    # a valid isinstance type and breaks tool calling.
    def plan_workout(muscles: list[str], counts: list[int] = [], level: str = "") -> dict:
        """Build a workout plan by searching the exercise database.

        Call this exactly ONCE per request, listing ALL requested muscle groups
        together in `muscles`.

        Args:
            muscles: The muscle groups to target, in the order the user said
                them, for example ["shoulder", "chest", "legs"]. Use simple names
                like "chest", "back", "abdominals", "shoulders", "legs", "biceps",
                "triceps".
            counts: Optional. The EXACT number of exercises for each muscle,
                aligned by position with `muscles` (e.g. "20 shoulder exercises"
                -> muscles=["shoulder"], counts=[20]). Never reduce the requested
                number. If empty, a daily total of 15 exercises is split evenly
                across the groups.
            level: Optional difficulty: "beginner", "intermediate" or "expert"
                (map "advanced" -> expert). Leave empty when the user does not say
                a level; the system then auto-picks difficulty from their history.

        Returns:
            A dictionary describing the plan: each group with its muscle name and
            a list of exercises (name, level, equipment, primary muscles, and an
            ordered list of instruction steps).
        """
        result = run_plan_workout(
            muscles, counts, collector, user_id=user_id, today=date.today(), level=level or None
        )
        if result.get("summary"):
            summary_holder["text"] = result["summary"]
        return result

    def find_exercise(name: str) -> dict:
        """Look up ONE specific exercise the user named, by its name.

        Use this instead of plan_workout when the user asks to do a particular
        exercise (for example "I want to do barbell curls").

        Args:
            name: The exercise name the user mentioned.

        Returns:
            A dictionary with `found` and, when found, the `exercise` (name, level,
            equipment, primary muscles, instructions).
        """
        return run_find_exercise(name, collector, user_id=user_id, today=date.today())

    def plan_weekly_workout(muscles: list[str], level: str = "") -> dict:
        """Build a 7-day weekly workout plan by searching the exercise database.

        Call this exactly ONCE when the user asks for a weekly plan. Muscles are spread
        evenly across 7 days; each day gets 15 exercises split across that day's muscles.

        Args:
            muscles: All muscle groups to cover this week, in the order the user said
                them, for example ["shoulders", "chest", "biceps", "hamstrings"].
            level: Optional difficulty: "beginner", "intermediate" or "expert"
                (map "advanced" -> expert). Leave empty when the user does not say
                a level; the system then auto-picks difficulty from their history.

        Returns:
            A dictionary with a "days" list (each day has date, muscles, groups, total)
            and overall "total" exercise count.
        """
        result = run_plan_weekly_workout(
            muscles, collector, user_id=user_id, today=date.today(), level=level or None
        )
        if result.get("summary"):
            summary_holder["text"] = result["summary"]
        return result

    return [plan_workout, plan_weekly_workout, find_exercise]


def _create_gemini_chat(
    collector: list[dict[str, Any]],
    user_id: int | None,
    summary_holder: dict[str, str | None],
):
    from google.genai import types

    client = _get_gemini_client()
    return client.chats.create(
        model=get_model_name(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=_make_gemini_tools(collector, user_id, summary_holder),
            temperature=0.5,
        ),
    )


class GeminiConversation:
    def __init__(self, user_id: int | None = None) -> None:
        self.user_id = user_id
        self._collector: list[dict[str, Any]] = []
        self._summary_holder: dict[str, str | None] = {"text": None}
        self._chat = _create_gemini_chat(self._collector, user_id, self._summary_holder)

    def send(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        self._collector.clear()
        self._summary_holder["text"] = None
        logger.info("USER -> %r", text)
        started = time.perf_counter()
        response = self._chat.send_message(text)
        reply = (getattr(response, "text", None) or "").strip()
        tool_summary = self._summary_holder.get("text")
        if tool_summary:
            reply = tool_summary
        elapsed_s = time.perf_counter() - started
        logger.info(
            "GEMINI -> %r (tool exercises=%d, %.1f s)",
            reply[:200] + ("..." if len(reply) > 200 else ""),
            len(self._collector),
            elapsed_s,
        )
        return reply, list(self._collector)


# --------------------------------------------------------------------------- #
# Ollama provider (local, OpenAI-style tool calling via /api/chat)
# --------------------------------------------------------------------------- #
OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_workout",
            "description": PLAN_WORKOUT_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "muscles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All muscle groups to target, in the order the user said them, "
                            "e.g. [\"shoulder\", \"chest\", \"legs\"]. Use simple names like "
                            "chest, back, abdominals, shoulders, legs, biceps, triceps."
                        ),
                    },
                    "counts": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": (
                            "Optional. The EXACT number of exercises per muscle, aligned by "
                            "position with muscles (e.g. '20 shoulder exercises' -> "
                            "counts=[20]); never reduce it. Leave empty to auto-split a "
                            "total of 15 exercises across the groups."
                        ),
                    },
                    "level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "expert"],
                        "description": (
                            "Optional difficulty. Map 'advanced' to 'expert'. Leave empty "
                            "when the user does not state a level; difficulty is then "
                            "auto-selected from their training history."
                        ),
                    },
                },
                "required": ["muscles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_weekly_workout",
            "description": PLAN_WEEKLY_WORKOUT_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "muscles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "All muscle groups to cover this week, in the order the user "
                            "said them, e.g. [\"shoulders\", \"chest\", \"biceps\"]."
                        ),
                    },
                    "level": {
                        "type": "string",
                        "enum": ["beginner", "intermediate", "expert"],
                        "description": (
                            "Optional difficulty. Map 'advanced' to 'expert'. Leave empty "
                            "when the user does not state a level; difficulty is then "
                            "auto-selected from their training history."
                        ),
                    },
                },
                "required": ["muscles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_exercise",
            "description": FIND_EXERCISE_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The specific exercise name the user asked to do.",
                    }
                },
                "required": ["name"],
            },
        },
    },
]


def _tool_plan_workout(
    args: dict[str, Any], collector: list[dict[str, Any]], user_id: int | None, today: date
) -> dict[str, Any]:
    return run_plan_workout(
        args.get("muscles"),
        args.get("counts"),
        collector,
        user_id=user_id,
        today=today,
        level=args.get("level") or None,
    )


def _tool_plan_weekly_workout(
    args: dict[str, Any], collector: list[dict[str, Any]], user_id: int | None, today: date
) -> dict[str, Any]:
    return run_plan_weekly_workout(
        args.get("muscles"),
        collector,
        user_id=user_id,
        today=today,
        level=args.get("level") or None,
    )


def _tool_find_exercise(
    args: dict[str, Any], collector: list[dict[str, Any]], user_id: int | None, today: date
) -> dict[str, Any]:
    return run_find_exercise(args.get("name"), collector, user_id=user_id, today=today)


OLLAMA_TOOL_REGISTRY = {
    "plan_workout": _tool_plan_workout,
    "plan_weekly_workout": _tool_plan_weekly_workout,
    "find_exercise": _tool_find_exercise,
}


def _ollama_chat(messages: list[dict[str, Any]]) -> dict[str, Any]:
    import requests

    # Tools are always advertised. Ollama's chat template errors (HTTP 500) when
    # the history contains tool_calls/tool messages but the request omits "tools".
    url = f"{get_ollama_base_url()}/api/chat"
    payload = {
        "model": get_model_name(),
        "messages": messages,
        "tools": OLLAMA_TOOLS,
        "stream": False,
        "options": {"temperature": 0.4},
    }

    response = requests.post(url, json=payload, timeout=300)
    if response.status_code >= 400:
        logger.error("Ollama %s error: %s", response.status_code, response.text[:500])
        response.raise_for_status()
    data = response.json()
    message = data.get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []
    logger.debug(
        "Ollama round model=%s tool_calls=%d content_len=%d",
        get_model_name(),
        len(tool_calls),
        len(message.get("content") or ""),
    )
    return data


def _execute_ollama_tool_call(
    call: dict[str, Any],
    collector: list[dict[str, Any]],
    user_id: int | None,
    today: date,
) -> dict[str, Any]:
    """Dispatch a single model tool call to the registered Python function."""
    function = call.get("function", {}) or {}
    name = function.get("name")
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    logger.info("TOOL CALL (ollama) %s(%s)", name, args)
    executor = OLLAMA_TOOL_REGISTRY.get(name)
    if executor is None:
        return {"error": f"unknown tool {name}"}

    try:
        return executor(args, collector, user_id, today)
    except Exception as error:  # noqa: BLE001 - report failure back to the model
        logger.exception("Tool %s failed", name)
        return {"error": str(error)}


class OllamaConversation:
    def __init__(self, user_id: int | None = None) -> None:
        self.user_id = user_id
        # How many rounds the model may call tools before we force a text answer.
        # Small local models tend to loop on tool calls, so keep this tight.
        self.tool_round_budget = get_ollama_tool_rounds()
        self._collector: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]

    def send(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        self._collector.clear()
        today = date.today()
        self._messages.append({"role": "user", "content": text})
        logger.info("USER -> %r", text)
        chat_started = time.perf_counter()
        tool_summary: str | None = None

        message: dict[str, Any] = {}
        for round_index in range(self.tool_round_budget + 1):
            is_final = round_index == self.tool_round_budget
            if is_final and not tool_summary:
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You now have all the exercise data you need. Do NOT call "
                            "any more tools. Read the summary field from the tool result "
                            "and present every exercise name grouped by muscle or day. "
                            "Ask if I want to start the first exercise."
                        ),
                    }
                )

            data = _ollama_chat(self._messages)
            message = data.get("message", {}) or {}
            self._messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    name = (call.get("function", {}) or {}).get("name")
                    result = _execute_ollama_tool_call(
                        call, self._collector, self.user_id, today
                    )
                    summary = result.get("summary")
                    if summary and name in ("plan_workout", "plan_weekly_workout"):
                        tool_summary = summary
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_name": name,
                            "name": name,
                            "content": json.dumps(result),
                        }
                    )
                if tool_summary:
                    reply = tool_summary.strip()
                    elapsed_s = time.perf_counter() - chat_started
                    logger.info(
                        "OLLAMA (tool summary) -> %r (exercises=%d, rounds=%d, %.1f s)",
                        reply[:200] + ("..." if len(reply) > 200 else ""),
                        len(self._collector),
                        round_index + 1,
                        elapsed_s,
                    )
                    return reply, list(self._collector)
                continue

            reply = (message.get("content") or "").strip()
            elapsed_s = time.perf_counter() - chat_started
            logger.info(
                "OLLAMA%s -> %r (exercises=%d, rounds=%d, %.1f s)",
                " (forced final)" if is_final else "",
                reply[:200] + ("..." if len(reply) > 200 else ""),
                len(self._collector),
                round_index + 1,
                elapsed_s,
            )
            if not reply:
                reply = tool_summary or (
                    "Here is your plan. Would you like to begin the first exercise?"
                )
            elif tool_summary:
                reply = tool_summary
            return reply, list(self._collector)

        return (
            tool_summary
            or "Here is your plan. Would you like to begin the first exercise?",
            list(self._collector),
        )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def start_conversation(user_id: int | None = None):
    provider = get_provider()
    logger.info(
        "Starting conversation (provider=%s, model=%s, user=%s)",
        provider,
        get_model_name(),
        user_id,
    )
    if provider == "ollama":
        return OllamaConversation(user_id=user_id)
    return GeminiConversation(user_id=user_id)
