"""API tests for the FastAPI endpoints using Starlette's TestClient.

The LLM/RAG layer is stubbed so these tests are fast and deterministic - they
verify the HTTP contract (status codes, JSON shape, auth, session handling), not
the model itself. Persistence runs against the per-test temp DB (see conftest).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import config, main


class FakeConversation:
    """Stand-in for a real provider conversation; echoes a canned plan."""

    def __init__(self, user_id=None) -> None:
        self.user_id = user_id
        self.messages: list[str] = []

    def send(self, text: str):
        self.messages.append(text)
        reply = f"Got it: {text}"
        exercises = [
            {
                "name": "Bench Press",
                "level": "intermediate",
                "equipment": "barbell",
                "primaryMuscles": ["chest"],
                "instructions": ["Lie down.", "Press up."],
                "group": "chest",
                "images": ["https://cdn.example.com/Bench_Press/0.jpg"],
            }
        ]
        return reply, exercises


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "is_store_ready", lambda: True)
    monkeypatch.setattr(config, "is_agent_enabled", lambda: True)
    monkeypatch.setattr(main, "start_conversation", lambda user_id=None: FakeConversation(user_id))
    main.conversations.clear()
    return TestClient(main.app)


def _auth_headers(client, username="tester", password="secret"):
    resp = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


# --------------------------------------------------------------------------- #
# Health + auth
# --------------------------------------------------------------------------- #
def test_health_returns_expected_shape(client):
    body = client.get("/api/health").json()
    for key in ("ok", "ragReady", "agentReady", "agentProvider", "agentModel", "embedMode", "embedModel"):
        assert key in body
    assert body["ok"] is True


def test_register_then_login(client):
    reg = client.post("/api/auth/register", json={"username": "newbie", "password": "pw12"})
    assert reg.status_code == 200
    assert reg.json()["token"]

    dupe = client.post("/api/auth/register", json={"username": "newbie", "password": "pw12"})
    assert dupe.status_code == 409

    good = client.post("/api/auth/login", json={"username": "newbie", "password": "pw12"})
    assert good.status_code == 200
    bad = client.post("/api/auth/login", json={"username": "newbie", "password": "wrong"})
    assert bad.status_code == 401


# --------------------------------------------------------------------------- #
# Chat (now requires auth)
# --------------------------------------------------------------------------- #
def test_chat_requires_auth(client):
    resp = client.post("/api/trainer/chat", json={"message": "hi"})
    assert resp.status_code == 401


def test_chat_returns_reply_and_exercises(client):
    headers = _auth_headers(client)
    resp = client.post("/api/trainer/chat", json={"message": "give me chest exercises"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "reply"
    assert body["reply"].startswith("Got it:")
    assert body["sessionId"]
    assert body["exercises"][0]["images"][0].startswith("https://")


def test_chat_reuses_session(client):
    headers = _auth_headers(client)
    first = client.post("/api/trainer/chat", json={"message": "hi"}, headers=headers).json()
    sid = first["sessionId"]
    second = client.post(
        "/api/trainer/chat", json={"message": "chest", "session_id": sid}, headers=headers
    ).json()
    assert second["sessionId"] == sid
    assert main.conversations[sid]["conversation"].messages == ["hi", "chest"]


def test_chat_rejects_empty_message(client):
    headers = _auth_headers(client)
    resp = client.post("/api/trainer/chat", json={"message": ""}, headers=headers)
    assert resp.status_code == 422


def test_chat_503_when_store_not_ready(monkeypatch):
    monkeypatch.setattr(main, "is_store_ready", lambda: False)
    c = TestClient(main.app)
    resp = c.post("/api/trainer/chat", json={"message": "hi"})
    assert resp.status_code == 503


def test_chat_503_when_agent_disabled(monkeypatch):
    monkeypatch.setattr(main, "is_store_ready", lambda: True)
    monkeypatch.setattr(config, "is_agent_enabled", lambda: False)
    c = TestClient(main.app)
    resp = c.post("/api/trainer/chat", json={"message": "hi"})
    assert resp.status_code == 503


# --------------------------------------------------------------------------- #
# Profile / history / stats
# --------------------------------------------------------------------------- #
def test_profile_get_and_update(client):
    headers = _auth_headers(client, "profileuser")
    prof = client.get("/api/profile", headers=headers).json()
    assert prof["username"] == "profileuser"
    assert prof["heightCm"] is None

    updated = client.post(
        "/api/profile", json={"height_cm": 178, "weight_kg": 72}, headers=headers
    ).json()
    assert updated["heightCm"] == 178
    assert updated["weightKg"] == 72


def test_profile_requires_auth(client):
    assert client.get("/api/profile").status_code == 401
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/stats/muscles").status_code == 401


def test_history_and_stats_reflect_workouts(client):
    from app import storage

    headers = _auth_headers(client, "logger")
    user_id = storage.user_id_for_token(headers["Authorization"].split()[1])
    storage.log_workout(user_id, "chest", [("beginner", "A"), ("beginner", "B")], "2026-06-10")
    storage.log_workout(user_id, "biceps", [("intermediate", "C")], "2026-06-11")

    stats = client.get("/api/stats/muscles", headers=headers).json()
    assert stats["totals"]["chest"] == 2
    assert stats["totalExercises"] == 3

    history = client.get("/api/history", headers=headers).json()
    assert history["days"][0]["day"] == "2026-06-11"


def test_avatar_upload(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AVATAR_DIR", tmp_path / "avatars")
    headers = _auth_headers(client, "pixel")
    files = {"file": ("me.png", b"\x89PNG\r\n\x1a\n fake", "image/png")}
    resp = client.post("/api/profile/avatar", files=files, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["avatarUrl"].startswith("/avatars/")


def test_avatar_rejects_bad_extension(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "AVATAR_DIR", tmp_path / "avatars")
    headers = _auth_headers(client, "pixel2")
    files = {"file": ("evil.txt", b"hello", "text/plain")}
    resp = client.post("/api/profile/avatar", files=files, headers=headers)
    assert resp.status_code == 400
