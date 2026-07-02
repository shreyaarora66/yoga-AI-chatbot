# AI Gym Trainer

**Live demo:** https://web-production-a6883.up.railway.app/

A voice-enabled chat app where an LLM agent (Google Gemini in the cloud, or a local
Ollama model) acts as a personal trainer. It calls a RAG database of exercises as a
tool, builds a workout plan, and returns each exercise with images and step-by-step
instructions.

Each user has an account and a persistent profile. When no difficulty is requested,
the trainer **auto-progresses difficulty per muscle** from the user's last 7 days
(beginner → beginner+intermediate → intermediate → intermediate+expert), avoids
repeating exercises within that window, and resets to beginner after a 7-day gap. The
profile page shows the user's avatar/height/weight, a muscle pie chart, a date-wise
workout history, and a **Clear history** button to erase past exercise data. Users can
also ask for a specific difficulty or a specific named exercise.

## Prerequisites

- **Python 3.10+** (3.11 or 3.14 tested)
- **Git**
- **data.json** is included in the repo (exercise dataset)
- For **Ollama** (local LLM): [Ollama](https://ollama.com) installed and a tool-capable
  model pulled (e.g. `ministral-3:8b`, `mistral`, `llama3.1`)
- For **Gemini** (cloud LLM): a free API key from
  [Google AI Studio](https://aistudio.google.com/apikey)

## Project structure

```
.
├── app/                      # Python application package
│   ├── config.py             # All paths + environment-driven settings (one place)
│   ├── logging_config.py     # Central logging setup (LOG_LEVEL in .env)
│   ├── main.py               # FastAPI app (endpoints) -> run as app.main:app
│   ├── ssl_fix.py            # SSL workaround for restricted networks
│   ├── embeddings.py         # Embedding provider (local Sentence Transformers / HF API)
│   ├── rag_store.py          # Vector store: search, level filter, name exclusion, find_by_name
│   ├── storage.py            # SQLite: users, profiles, tokens, permanent workout log
│   ├── auth.py               # Account register/login, password hashing, current_user
│   ├── muscles.py            # Canonical muscle names + alias normalization
│   ├── progression.py        # Adaptive difficulty engine (7-day window state machine)
│   └── agent/                # The trainer agent
│       ├── prompts.py        # System prompt + tool descriptions
│       ├── tools.py          # plan_workout + find_exercise skills (RAG, progression, logging)
│       └── providers.py      # Gemini + Ollama conversation classes + factory
├── user_data/                # Generated at runtime: SQLite DB + uploaded avatars
├── logs/                       # Plain-text app logs (trainer.txt, created on start)
├── scripts/
│   └── build_embeddings.py   # Build the RAG index from data.json
├── web/                      # Frontend (served by FastAPI)
│   ├── index.html
│   └── static/trainer.png
├── tests/                    # Pytest suite (see "Testing")
├── data.json                 # Source exercise dataset
├── rag_db/                   # Generated embeddings + index (created by build script)
├── requirements.txt          # Runtime dependencies
└── requirements-dev.txt      # Test dependencies (pytest, httpx)
```

## Clone and setup

```bash
git clone <your-repo-url>
cd "Yoga AI chatbot"

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
# If pip hits SSL errors, add:
#   --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

Copy the example environment file and edit it:

```bash
# Windows
copy .env.example .env

# macOS / Linux
# cp .env.example .env
```

Build the RAG index (required once, or after changing `data.json`):

```bash
python -m scripts.build_embeddings
```

This creates `rag_db/embeddings.npy`, `rag_db/index.json`, and
`rag_db/exercises.json`. The first run downloads the embedding model (~90 MB) and may
take a few minutes.

### Choose an LLM provider

Edit `.env` and pick **one** of these paths:

**Option A — Ollama (local, no API key)**

```env
AGENT_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=ministral-3:8b
USE_LLM_AGENT=true
```

Start Ollama and pull the model before running the app:

```bash
ollama pull ministral-3:8b
ollama serve
```

**Option B — Gemini (cloud)**

```env
AGENT_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
AGENT_MODEL=gemini-2.5-flash
USE_LLM_AGENT=true
```

### Key `.env` settings

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `AGENT_PROVIDER` | `gemini` | `ollama` (local) or `gemini` (cloud) |
| `DAILY_EXERCISE_TOTAL` | `15` | Exercises split across groups when user gives no counts |
| `MAX_EXERCISES_PER_GROUP` | `30` | Cap when user asks for an explicit number |
| `EMBED_MODE` | `local` | `local` (Sentence Transformers) or `api` (Hugging Face) |
| `LOG_LEVEL` | `INFO` | Set to `DEBUG` for detailed request/RAG/progression traces |

> **Note:** After editing `.env`, restart the server. Uvicorn `--reload` only watches
> `.py` files, not `.env`.

## Run the app

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000.

1. **Register** or **log in** (chat requires authentication).
2. Ask for exercises, e.g. `give me 5 chest exercises`.
3. Open the **Profile** tab for avatar, stats, history, and **Clear history**.

Interactive API docs: http://127.0.0.1:8000/docs

## Logging and debugging

All logs are written to **two places**:

1. **Terminal** — where uvicorn runs
2. **`logs/trainer.txt`** — plain UTF-8 text file (appended on each server start)

The `logs/` folder is created automatically when the server starts. Open `trainer.txt`
in any text editor to review past requests, tool calls, and errors.

Logger namespaces:

| Logger | What it traces |
| ------ | -------------- |
| `trainer.server` | HTTP requests (method, path, status, duration), startup, profile changes |
| `trainer.auth` | Register, login, failed login |
| `trainer.agent` | User messages, LLM replies, tool calls (`plan_workout`, `find_exercise`) |
| `trainer.rag` | Embedding model load, RAG store load, search (DEBUG) |
| `trainer.storage` | Database init, workout logging, history clears |
| `trainer.progress` | Adaptive difficulty decisions per muscle |

For deep traces during development, set in `.env`:

```env
LOG_LEVEL=DEBUG
```

Typical flow in logs for one chat message:

```
trainer.server: POST /api/trainer/chat -> 200 (45000 ms)
trainer.agent: USER -> 'give me 5 chest exercises'
trainer.agent: TOOL CALL (ollama) plan_workout({'muscles': ['chest'], 'counts': [5]})
trainer.progress: Progression user=1 muscle=chest last=none -> plan=[('beginner', 5)]
trainer.rag: Loading embedding model ...
trainer.rag: RAG store loaded: 873 exercises ...
trainer.agent: OLLAMA -> '...' (exercises=5, rounds=2, 42.3 s)
```

## API endpoints

All endpoints except health and auth require an `Authorization: Bearer <token>` header
(the token comes from register/login).

| Method | Path                    | Description                                            |
| ------ | ----------------------- | ------------------------------------------------------ |
| GET    | `/api/health`           | Readiness + active provider/model/embedding info       |
| POST   | `/api/auth/register`    | `{ "username", "password" }` → `{ token, username }`   |
| POST   | `/api/auth/login`       | `{ "username", "password" }` → `{ token, username }`   |
| POST   | `/api/auth/logout`      | Invalidate the current token                           |
| POST   | `/api/trainer/chat`     | `{ "message", "session_id?" }` → reply + exercises (auth) |
| GET    | `/api/profile`          | Username, height, weight, avatar URL (auth)            |
| POST   | `/api/profile`          | Update `{ height_cm?, weight_kg? }` (auth)             |
| POST   | `/api/profile/avatar`   | Multipart image upload (auth)                          |
| GET    | `/api/history`          | Date-wise workout history (auth)                       |
| DELETE | `/api/history`          | Erase all workout history for the current user (auth)  |
| GET    | `/api/stats/muscles`    | All-time exercise counts per muscle for the pie (auth) |
| GET    | `/`                     | Serves the UI                                          |

## Testing

### Manual / exploratory

FastAPI auto-generates interactive docs — open **http://127.0.0.1:8000/docs**.

Chat requires auth. Example flow (PowerShell):

```powershell
# 1. Register
$reg = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/auth/register `
  -Method Post -ContentType 'application/json' `
  -Body '{"username":"demo","password":"demo1234"}'

# 2. Chat with token
$h = @{ Authorization = "Bearer $($reg.token)" }
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/trainer/chat `
  -Method Post -Headers $h -ContentType 'application/json' `
  -Body '{"message":"give me 3 chest exercises"}'
```

bash / curl equivalent:

```bash
# Register
curl -X POST http://127.0.0.1:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo1234"}'

# Chat (replace TOKEN with the token from register/login)
curl -X POST http://127.0.0.1:8000/api/trainer/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"message":"give me 3 chest exercises"}'
```

### Automated (pytest)

```bash
pip install -r requirements-dev.txt

pytest -m "not slow"   # fast unit + API tests (default, ~10s)
pytest -m slow         # loads the real embedding model (~2 min)
pytest                 # everything
```

What the suite covers:

- **`tests/test_rag_store.py`** — cosine similarity, keyword boosting, level filtering,
  exclude_names, find_by_name.
- **`tests/test_agent_tools.py`** — the 15/day distribution (`1→15`, `2→8,7`, etc.),
  explicit counts, image-URL building, and `plan_workout` / `find_exercise` with
  stubbed RAG (no model needed).
- **`tests/test_api.py`** — `/api/health`, register/login, auth-gated chat (401 without
  token), session reuse, profile/history/stats/avatar/clear-history endpoints.
- **`tests/test_storage.py`** — SQLite round-trips, workout logging, 7-day window,
  permanent history.
- **`tests/test_progression.py`** — adaptive difficulty state machine, 7-day reset,
  same-day-no-advance rule.
- **`tests/test_embeddings.py`** — vector normalization; slow tests load the real model.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `503 RAG database is not ready` | Run `python -m scripts.build_embeddings` |
| `503 LLM not configured` | Set `USE_LLM_AGENT=true` and configure Gemini or Ollama in `.env`, then restart |
| `401 Not authenticated` | Register/login first; send `Authorization: Bearer <token>` |
| Ollama `502` / connection errors | Ensure `ollama serve` is running and `OLLAMA_MODEL` is pulled |
| First chat very slow | Normal — embedding model + RAG store load once (~1–2 min on first request) |
| `.env` changes ignored | Restart uvicorn (reload does not watch `.env`) |

## Files not committed

Do not commit these (they contain secrets or runtime data):

- `.env` — API keys and local settings
- `user_data/` — SQLite database and uploaded avatars
- `logs/` — runtime log files (optional to commit)
- `rag_db/` — optional to commit (large); can be rebuilt with `build_embeddings`
