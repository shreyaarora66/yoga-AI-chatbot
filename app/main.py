"""FastAPI backend. The LLM agent owns the whole conversation and calls the RAG
database as a tool; this server routes messages to per-session, per-user agents and
exposes account, profile, history and stats endpoints backed by SQLite.

Run with:  uvicorn app.main:app --host 127.0.0.1 --port 8000

Note: uvicorn --reload only watches .py files, so changing .env requires a
server restart to reload environment variables.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

# Load .env before reading LOG_LEVEL / paths.
from app import config
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("trainer.server")

from app.ssl_fix import configure_ssl

configure_ssl()

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import auth, storage
from app.agent import start_conversation
from app.rag_store import is_store_ready

INDEX_FILE = config.WEB_DIR / "index.html"
STATIC_DIR = config.WEB_DIR / "static"
IMAGES_DIR = config.PROJECT_ROOT / "images"

# Ensure persistence is ready before serving any request.
config.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
storage.init_db()

logger.info(
    "Trainer starting | ragReady=%s agentReady=%s provider=%s model=%s",
    is_store_ready(),
    config.is_agent_enabled(),
    config.get_provider(),
    config.get_model_name(),
)

app = FastAPI(title="AI Gym Trainer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each API request with status and duration (skip static asset noise)."""
    path = request.url.path
    if path.startswith(("/static/", "/avatars/", "/images/")):
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s -> %s (%.0f ms)",
        request.method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response


# session_id -> {"conversation": obj, "user_id": int}
conversations: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AuthRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TrainerChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ProfileUpdate(BaseModel):
    height_cm: float | None = Field(default=None, ge=0, le=300)
    weight_kg: float | None = Field(default=None, ge=0, le=600)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ragReady": is_store_ready(),
        "agentReady": config.is_agent_enabled(),
        "agentProvider": config.get_provider(),
        "agentModel": config.get_model_name(),
        "embedProvider": "huggingface",
        "embedMode": config.get_embed_mode(),
        "embedModel": config.get_embed_model(),
    }


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
@app.post("/api/auth/register")
def register(req: AuthRequest) -> dict[str, Any]:
    return auth.register(req.username, req.password)


@app.post("/api/auth/login")
def login(req: AuthRequest) -> dict[str, Any]:
    return auth.login(req.username, req.password)


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if authorization:
        token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else authorization
        user_id = storage.user_id_for_token(token)
        storage.delete_token(token)
        if user_id is not None:
            logger.info("Logout user id=%s", user_id)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
@app.post("/api/trainer/chat")
def trainer_chat(
    request: TrainerChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if not is_store_ready():
        raise HTTPException(
            status_code=503,
            detail="RAG database is not ready. Run `python -m scripts.build_embeddings` first.",
        )

    if not config.is_agent_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "The trainer LLM is not configured. Add GEMINI_API_KEY to .env "
                "and set USE_LLM_AGENT=true, then restart the server."
            ),
        )

    user_id = auth.resolve_token(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")

    session_id = request.session_id
    entry = conversations.get(session_id) if session_id else None
    # Sessions belong to one user; ignore a session id that is unknown or not ours.
    if not entry or entry.get("user_id") != user_id:
        session_id = str(uuid.uuid4())
        logger.info("Starting new conversation %s for user %s", session_id, user_id)
        try:
            conversation = start_conversation(user_id=user_id)
        except Exception as error:  # noqa: BLE001
            logger.exception("Failed to start conversation")
            raise HTTPException(
                status_code=502, detail=f"Could not start the trainer: {error}"
            ) from error
        entry = {"conversation": conversation, "user_id": user_id}
        conversations[session_id] = entry

    conversation = entry["conversation"]

    logger.info("Chat [%s] message=%r", session_id, request.message.strip())
    try:
        reply, exercises = conversation.send(request.message.strip())
    except Exception as error:  # noqa: BLE001
        logger.exception("Agent call failed")
        raise HTTPException(
            status_code=502, detail=f"Trainer could not respond: {error}"
        ) from error

    logger.info("Reply [%s] len=%d exercises=%d", session_id, len(reply), len(exercises))

    return {
        "type": "reply",
        "sessionId": session_id,
        "reply": reply,
        "spokenText": reply,
        "exercises": exercises,
    }


# --------------------------------------------------------------------------- #
# Profile / history / stats
# --------------------------------------------------------------------------- #
def _avatar_url(avatar_path: str | None) -> str | None:
    if not avatar_path:
        return None
    return f"/avatars/{Path(avatar_path).name}"


@app.get("/api/profile")
def get_profile(user_id: int = Depends(auth.current_user)) -> dict[str, Any]:
    user = storage.get_user(user_id) or {}
    profile = storage.get_profile(user_id)
    return {
        "userId": user_id,
        "username": user.get("username"),
        "heightCm": profile.get("height_cm"),
        "weightKg": profile.get("weight_kg"),
        "avatarUrl": _avatar_url(profile.get("avatar_path")),
    }


@app.post("/api/profile")
def update_profile(
    update: ProfileUpdate, user_id: int = Depends(auth.current_user)
) -> dict[str, Any]:
    storage.upsert_profile(
        user_id, height_cm=update.height_cm, weight_kg=update.weight_kg
    )
    logger.info(
        "Profile updated user=%s height=%s weight=%s",
        user_id,
        update.height_cm,
        update.weight_kg,
    )
    return get_profile(user_id)


_ALLOWED_AVATAR_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@app.post("/api/profile/avatar")
async def upload_avatar(
    file: UploadFile = File(...), user_id: int = Depends(auth.current_user)
) -> dict[str, Any]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_AVATAR_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type. Use one of: {', '.join(sorted(_ALLOWED_AVATAR_EXT))}",
        )
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 5 MB).")

    config.AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.AVATAR_DIR / f"{user_id}{ext}"
    # Remove any previous avatar with a different extension.
    for old in config.AVATAR_DIR.glob(f"{user_id}.*"):
        if old != dest:
            old.unlink(missing_ok=True)
    dest.write_bytes(contents)

    storage.upsert_profile(user_id, avatar_path=str(dest))
    logger.info("Avatar uploaded user=%s size=%d ext=%s", user_id, len(contents), ext)
    return get_profile(user_id)


@app.get("/api/history")
def get_history(user_id: int = Depends(auth.current_user)) -> dict[str, Any]:
    days = storage.history_by_day(user_id)
    logger.debug("History fetched user=%s days=%d", user_id, len(days))
    return {"days": days}


@app.delete("/api/history")
def clear_history(user_id: int = Depends(auth.current_user)) -> dict[str, Any]:
    removed = storage.clear_history(user_id)
    return {"ok": True, "removed": removed}


@app.get("/api/stats/muscles")
def get_muscle_stats(user_id: int = Depends(auth.current_user)) -> dict[str, Any]:
    totals = storage.muscle_totals(user_id)
    return {
        "totals": totals,
        "totalExercises": sum(totals.values()),
    }


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #
@app.get("/")
def serve_index() -> FileResponse:
    # Disable caching so UI changes are always picked up on refresh.
    return FileResponse(
        INDEX_FILE,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/trainer.png")
def serve_trainer_image() -> FileResponse:
    image_path = STATIC_DIR / "trainer.png"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Trainer image not found.")
    return FileResponse(image_path, media_type="image/png")


if config.AVATAR_DIR.is_dir():
    app.mount("/avatars", StaticFiles(directory=config.AVATAR_DIR), name="avatars")

if IMAGES_DIR.is_dir():
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
