"""Lightweight account auth: password hashing, register/login, and a FastAPI
dependency that resolves the current user from a bearer token.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib ``hashlib``) using a unique
per-user salt, so no third-party crypto dependency is needed.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3

from fastapi import Header, HTTPException

from app import storage

logger = logging.getLogger("trainer.auth")

_PBKDF2_ROUNDS = 200_000
_USERNAME_MIN = 3
_PASSWORD_MIN = 4


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex). Generates a salt when one is not supplied."""
    if salt is None:
        salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    )
    return derived.hex(), salt


def verify_password(password: str, salt_hex: str, expected_hash: str) -> bool:
    candidate, _ = hash_password(password, salt_hex)
    return secrets.compare_digest(candidate, expected_hash)


def register(username: str, password: str) -> dict[str, object]:
    """Create a user and return {token, username, user_id}. Raises HTTPException."""
    username = (username or "").strip()
    if len(username) < _USERNAME_MIN:
        raise HTTPException(
            status_code=400,
            detail=f"Username must be at least {_USERNAME_MIN} characters.",
        )
    if len(password or "") < _PASSWORD_MIN:
        raise HTTPException(
            status_code=400,
            detail=f"Password must be at least {_PASSWORD_MIN} characters.",
        )

    password_hash, salt = hash_password(password)
    try:
        user_id = storage.create_user(username, password_hash, salt)
    except sqlite3.IntegrityError as error:
        logger.warning("Register rejected: username %r already taken", username)
        raise HTTPException(status_code=409, detail="Username is already taken.") from error

    token = storage.create_token(user_id)
    logger.info("Registered user %r (id=%s)", username, user_id)
    return {"token": token, "username": username, "user_id": user_id}


def login(username: str, password: str) -> dict[str, object]:
    username = (username or "").strip()
    user = storage.get_user_by_username(username)
    if not user or not verify_password(password or "", user["salt"], user["password_hash"]):
        logger.warning("Login failed for username %r", username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = storage.create_token(int(user["id"]))
    logger.info("Login user %r (id=%s)", username, user["id"])
    return {"token": token, "username": username, "user_id": int(user["id"])}


def resolve_token(authorization: str | None) -> int | None:
    """Extract the bearer token from an Authorization header and resolve a user id."""
    if not authorization:
        return None
    token = authorization
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return storage.user_id_for_token(token)


def current_user(authorization: str | None = Header(default=None)) -> int:
    """FastAPI dependency: returns the authenticated user id or raises 401."""
    user_id = resolve_token(authorization)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated. Please log in.")
    return user_id
