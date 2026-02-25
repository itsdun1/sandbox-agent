# sandbox-agent

An AI data analyst agent with container-per-thread workspace isolation.

Each conversation thread gets its own sandboxed Python environment (Docker or e2b). The agent runs LLM-generated pandas/numpy code against uploaded CSVs and streams results back via SSE.

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

## Quickstart

```bash
pip install -r requirements.txt

# Docker runtime (default)
SANDBOX_RUNTIME=docker python server.py

# e2b runtime
SANDBOX_RUNTIME=e2b E2B_API_KEY=... OPENAI_API_KEY=... python server.py
```

Open [http://localhost:8000/ui](http://localhost:8000/ui) to chat.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/thread/new` | Create thread, spin up sandbox |
| `POST` | `/thread/{id}/message` | Send message (blocking) |
| `POST` | `/thread/{id}/message/stream` | Send message (SSE streaming) |
| `GET`  | `/thread/{id}/status` | Container state (hot/warm/cold) |
| `GET`  | `/thread/{id}/history` | Conversation history |
| `POST` | `/upload` | Upload CSV or other files |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_RUNTIME` | `docker` | `docker` or `e2b` |
| `OPENAI_API_KEY` | — | Required |
| `E2B_API_KEY` | — | Required for e2b runtime |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
