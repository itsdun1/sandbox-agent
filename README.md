# sandbox-agent

An AI data analyst agent with container-per-thread workspace isolation.

Each conversation thread gets its own sandboxed Python environment (Docker or e2b). The agent runs LLM-generated pandas/numpy code against uploaded CSVs and streams results back via Server-Sent Events.

## Architecture

```
User → FastAPI router → OpenAI (tool calls) → Sandbox (Docker / e2b)
                ↑                                      ↓
            Redis (thread state)              CSV + Python namespace
                ↑
            SQLite (durable message history)
```

## Features

- **Container-per-thread isolation** — each thread has its own Python kernel
- **HOT / WARM / COLD** sandbox lifecycle (pause/resume, evict/restore)
- **Streaming responses** via Server-Sent Events (TTFT optimised)
- **Pluggable runtime** — swap Docker ↔ e2b via `SANDBOX_RUNTIME` env var
- **Redis thread store** — stateless server, horizontally scalable
- **Chat UI** at `/ui`

---

## Prerequisites

| Dependency | Install |
|------------|---------|
| Python 3.11+ | [python.org](https://python.org) |
| Docker Desktop | [docker.com](https://docker.com) — required for `docker` runtime |
| Redis | `brew install redis` (macOS) / `apt install redis` (Linux) |
| OpenAI API key | [platform.openai.com](https://platform.openai.com) |
| e2b API key | [e2b.dev](https://e2b.dev) — only needed for `e2b` runtime |

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/itsdun1/sandbox-agent.git
cd sandbox-agent

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file (or export directly):

```bash
# .env
OPENAI_API_KEY=your_openai_api_key

# Choose runtime: docker (default) or e2b
SANDBOX_RUNTIME=docker

# Only required for e2b runtime
E2B_API_KEY=your_e2b_api_key

# Redis (defaults shown)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Start Redis

```bash
# macOS
brew services start redis

# Linux
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:alpine

# Verify
redis-cli ping   # should return PONG
```

### 4a. Docker runtime — build the worker image

```bash
docker build -t sandbox-worker ./worker
```

Verify the image exists:

```bash
docker images sandbox-worker
```

### 4b. e2b runtime — no build step needed

e2b spins up cloud sandboxes automatically. Just set `SANDBOX_RUNTIME=e2b` and `E2B_API_KEY`.

### 5. Run the server

```bash
# Load .env and start
export $(cat .env | xargs)
python server.py
```

Server starts on [http://localhost:8000](http://localhost:8000).

---

## Usage

Open the chat UI:

```
http://localhost:8000/ui
```

Or use the API directly:

```bash
# Create a thread (uses bundled transactions.csv by default)
curl -X POST http://localhost:8000/thread/new

# Send a message
curl -X POST http://localhost:8000/thread/{thread_id}/message \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top 5 categories by spend?"}'

# Stream a message (SSE)
curl -N -X POST http://localhost:8000/thread/{thread_id}/message/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me monthly trends"}'

# Upload your own CSV
curl -X POST http://localhost:8000/upload \
  -F "files=@your_data.csv"
# returns file_id — pass it to /thread/new:
curl -X POST http://localhost:8000/thread/new \
  -H "Content-Type: application/json" \
  -d '{"file_ids": ["<file_id>"]}'
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/ui` | Chat frontend |
| `POST` | `/upload` | Upload CSV or other files — returns `file_id` |
| `POST` | `/thread/new` | Create thread and spin up sandbox |
| `POST` | `/thread/{id}/message` | Send message — blocking JSON response |
| `POST` | `/thread/{id}/message/stream` | Send message — SSE stream |
| `GET`  | `/thread/{id}/status` | Sandbox state: `hot` / `warm` / `cold` |
| `GET`  | `/thread/{id}/history` | Conversation history |
| `GET`  | `/threads` | List all threads |

### SSE event types (`/message/stream`)

| Event | Payload | Description |
|-------|---------|-------------|
| `status` | `{text}` | Progress update (e.g. "Executing Python…") |
| `token` | `{text}` | One streamed token of the final answer |
| `code_run` | `{code, output, exec_ms}` | Code executed + output |
| `done` | `{timings_ms, container_state, …}` | Completion metadata |
| `error` | `{message}` | Fatal error |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_RUNTIME` | `docker` | `docker` or `e2b` |
| `OPENAI_API_KEY` | — | Required |
| `E2B_API_KEY` | — | Required for `e2b` runtime |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |

---

## Benchmarking

```bash
# Compare non-streaming vs streaming latency
python benchmark_stream.py <thread_id> "your question here"
```
