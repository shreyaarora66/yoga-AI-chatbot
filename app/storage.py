"""SQLite-backed persistence: users, profiles, auth tokens, and a permanent workout log.

The workout log is never deleted - the profile page needs the full history and an
all-time muscle pie chart. The adaptive difficulty engine only ever *queries* a
7-day window (see ``app.progression``); it does not purge old rows.

Connections are opened per call (SQLite handles its own file locking), which keeps
the module safe under the threaded TestClient and uvicorn worker alike.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app import config

logger = logging.getLogger("trainer.storage")

# The active DB path. Tests can point this at a temp file via ``configure``.
_db_path: Path = config.DB_FILE


def configure(db_path: str | Path) -> None:
    """Override the database file location (used by tests)."""
    global _db_path
    _db_path = Path(db_path)


def get_db_path() -> Path:
    return _db_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    _db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profiles (
    user_id     INTEGER PRIMARY KEY,
    height_cm   REAL,
    weight_kg   REAL,
    avatar_path TEXT,
    updated_at  TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS auth_tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE IF NOT EXISTS workout_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    muscle        TEXT NOT NULL,
    level         TEXT,
    exercise_name TEXT NOT NULL,
    day           TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE INDEX IF NOT EXISTS idx_log_user_muscle_day
    ON workout_log (user_id, muscle, day);
CREATE INDEX IF NOT EXISTS idx_log_user_day
    ON workout_log (user_id, day);
"""


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database ready at %s", _db_path)


# --------------------------------------------------------------------------- #
# Users + auth
# --------------------------------------------------------------------------- #
def create_user(username: str, password_hash: str, salt: str) -> int:
    """Insert a new user and an empty profile. Raises sqlite3.IntegrityError if taken."""
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) "
            "VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, _now()),
        )
        user_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO profiles (user_id, updated_at) VALUES (?, ?)",
            (user_id, _now()),
        )
        return user_id


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user(user_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _now()),
        )
    return token


def user_id_for_token(token: str) -> int | None:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id FROM auth_tokens WHERE token = ?", (token,)
        ).fetchone()
        return int(row["user_id"]) if row else None


def delete_token(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #
def get_profile(user_id: int) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id, height_cm, weight_kg, avatar_path, updated_at "
            "FROM profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return {
                "user_id": user_id,
                "height_cm": None,
                "weight_kg": None,
                "avatar_path": None,
                "updated_at": None,
            }
        return dict(row)


def upsert_profile(
    user_id: int,
    *,
    height_cm: float | None = None,
    weight_kg: float | None = None,
    avatar_path: str | None = None,
) -> dict[str, Any]:
    """Update only the provided fields, leaving the others untouched."""
    current = get_profile(user_id)
    new_height = current["height_cm"] if height_cm is None else height_cm
    new_weight = current["weight_kg"] if weight_kg is None else weight_kg
    new_avatar = current["avatar_path"] if avatar_path is None else avatar_path
    with _connect() as conn:
        conn.execute(
            "INSERT INTO profiles (user_id, height_cm, weight_kg, avatar_path, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "height_cm = excluded.height_cm, "
            "weight_kg = excluded.weight_kg, "
            "avatar_path = excluded.avatar_path, "
            "updated_at = excluded.updated_at",
            (user_id, new_height, new_weight, new_avatar, _now()),
        )
    return get_profile(user_id)


# --------------------------------------------------------------------------- #
# Workout log
# --------------------------------------------------------------------------- #
def log_workout(
    user_id: int, muscle: str, items: Iterable[tuple[str | None, str]], day: str
) -> None:
    """Record one row per served exercise. ``items`` is an iterable of (level, name)."""
    rows = [
        (user_id, muscle, level, name, day, _now())
        for level, name in items
        if name
    ]
    if not rows:
        return
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO workout_log "
            "(user_id, muscle, level, exercise_name, day, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
    logger.info(
        "Logged workout user=%s muscle=%s day=%s exercises=%d",
        user_id,
        muscle,
        day,
        len(rows),
    )


def recent_logs(user_id: int, muscle: str, since_day: str) -> list[dict[str, Any]]:
    """Windowed query for the progression engine: rows on or after ``since_day``."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT muscle, level, exercise_name, day FROM workout_log "
            "WHERE user_id = ? AND muscle = ? AND day >= ? "
            "ORDER BY day DESC, id DESC",
            (user_id, muscle, since_day),
        ).fetchall()
        return [dict(row) for row in rows]


def history_by_day(user_id: int) -> list[dict[str, Any]]:
    """All-time history grouped by day then muscle, newest day first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT day, muscle, level, exercise_name FROM workout_log "
            "WHERE user_id = ? ORDER BY day DESC, id ASC",
            (user_id,),
        ).fetchall()

    days: dict[str, dict[str, dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        day = row["day"]
        if day not in days:
            days[day] = {}
            order.append(day)
        groups = days[day]
        muscle = row["muscle"]
        if muscle not in groups:
            groups[muscle] = {"muscle": muscle, "exercises": [], "levels": set()}
        groups[muscle]["exercises"].append(row["exercise_name"])
        if row["level"]:
            groups[muscle]["levels"].add(row["level"])

    result: list[dict[str, Any]] = []
    for day in order:
        group_list = []
        for muscle, info in days[day].items():
            group_list.append(
                {
                    "muscle": muscle,
                    "count": len(info["exercises"]),
                    "exercises": info["exercises"],
                    "levels": sorted(info["levels"]),
                }
            )
        result.append({"day": day, "groups": group_list})
    return result


def clear_history(user_id: int) -> int:
    """Delete all workout-log rows for a user. Returns the number of rows removed.

    This resets the profile history, the all-time pie chart, and the adaptive
    difficulty engine back to a clean slate for the user.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "DELETE FROM workout_log WHERE user_id = ?", (user_id,)
        )
        removed = cursor.rowcount
    logger.info("Cleared workout history user=%s removed=%d", user_id, removed)
    return removed


def muscle_totals(user_id: int) -> dict[str, int]:
    """All-time count of exercises performed per muscle (for the pie chart)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT muscle, COUNT(*) AS n FROM workout_log "
            "WHERE user_id = ? GROUP BY muscle ORDER BY n DESC",
            (user_id,),
        ).fetchall()
        return {row["muscle"]: int(row["n"]) for row in rows}
