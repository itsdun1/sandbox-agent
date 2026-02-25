#!/usr/bin/env python3
"""
Agent router — pluggable sandbox backend.

Each thread gets its own sandbox (Docker or e2b) controlled by SANDBOX_RUNTIME env var:
  SANDBOX_RUNTIME=docker  (default) — Docker container per thread
  SANDBOX_RUNTIME=e2b                — e2b cloud sandbox per thread

States (Docker):
  HOT  → container running, namespace in memory    → instant
  WARM → container paused (docker pause)           → ~10ms unpause
  COLD → container destroyed, pickle on disk only  → ~2-3s new container

States (e2b):
  HOT  → sandbox running, Jupyter kernel live in e2b cloud
  WARM → beta_pause() — full kernel state saved to e2b cloud

The router holds conversation history and calls OpenAI.
The sandbox holds the Python namespace and executes code.
"""

import asyncio
import io
import json
import os
import logging
import shutil
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import redis as _redis_lib
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import Annotated
from openai import AsyncOpenAI
from pydantic import BaseModel

from runtime import create_runtime

# ---------- Logging ----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------- Config ----------

TRANSACTIONS_PATH = Path(__file__).parent / "transactions.csv"
WORKSPACES_DIR    = Path(__file__).parent / "workspaces"
UPLOADS_DIR       = Path(__file__).parent / "uploads"
DB_PATH           = Path(__file__).parent / "aqqrue.db"
BACKUP_DIR        = Path(__file__).parent / "backup"


# ---------- Database ----------

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    """Create tables and enable WAL mode for concurrent reads."""
    with _get_db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id           TEXT PRIMARY KEY,
                runtime             TEXT NOT NULL,
                system_prompt       TEXT NOT NULL,
                workspace_backup_path TEXT,
                created_at          TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id    TEXT NOT NULL,
                role         TEXT NOT NULL,
                content      TEXT,
                tool_calls   TEXT,
                tool_call_id TEXT,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)")
        # Migration: add column if upgrading from older schema
        try:
            conn.execute("ALTER TABLE threads ADD COLUMN workspace_backup_path TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
    log.info("Database ready (%s)", DB_PATH)


def _backup_db() -> None:
    """Copy DB to backup dir — overwrites previous backup (latest only)."""
    try:
        BACKUP_DIR.mkdir(exist_ok=True)
        shutil.copy2(DB_PATH, BACKUP_DIR / "aqqrue.db")
    except Exception as e:
        log.warning("DB backup failed: %s", e)


def _backup_workspace(thread_id: str) -> None:
    """Copy workspace dir to backup/workspaces/{thread_id}/ and record path in DB."""
    src = WORKSPACES_DIR / thread_id
    if not src.exists():
        return
    dest = BACKUP_DIR / "workspaces" / thread_id
    try:
        BACKUP_DIR.mkdir(exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        with _get_db() as conn:
            conn.execute(
                "UPDATE threads SET workspace_backup_path = ? WHERE thread_id = ?",
                (str(dest), thread_id),
            )
    except Exception as e:
        log.warning("[%s] Workspace backup failed: %s", thread_id, e)


def _save_thread(thread_id: str) -> None:
    """Append only new messages (since last save) to the DB, then backup."""
    thread = _get_thread(thread_id)
    if not thread:
        return
    messages = thread["messages"]
    saved_count = thread.get("_saved_count", 0)
    new_messages = messages[saved_count:]
    if not new_messages:
        return
    with _get_db() as conn:
        conn.executemany(
            "INSERT INTO messages (thread_id, role, content, tool_calls, tool_call_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    thread_id,
                    msg["role"],
                    msg.get("content"),
                    json.dumps(msg["tool_calls"]) if msg.get("tool_calls") else None,
                    msg.get("tool_call_id"),
                )
                for msg in new_messages
            ],
        )
    thread["_saved_count"] = len(messages)
    _set_thread(thread_id, thread)  # persist updated saved_count back to Redis
    _backup_db()
    _backup_workspace(thread_id)


def _ensure_thread_loaded(thread_id: str) -> bool:
    """Ensure thread is in Redis. Load from DB if not cached. Returns False if not found."""
    if _get_thread(thread_id) is not None:
        return True
    with _get_db() as conn:
        thread_row = conn.execute(
            "SELECT runtime, system_prompt FROM threads WHERE thread_id = ?", (thread_id,)
        ).fetchone()
        if not thread_row:
            return False
        msg_rows = conn.execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages "
            "WHERE thread_id = ? ORDER BY id", (thread_id,)
        ).fetchall()

    messages = []
    for m in msg_rows:
        msg: dict = {"role": m["role"]}
        if m["content"] is not None:
            msg["content"] = m["content"]
        if m["tool_calls"]:
            msg["tool_calls"] = json.loads(m["tool_calls"])
        if m["tool_call_id"]:
            msg["tool_call_id"] = m["tool_call_id"]
        messages.append(msg)

    _set_thread(thread_id, {
        "messages":      messages,
        "system_prompt": thread_row["system_prompt"],
        "runtime":       thread_row["runtime"],
        "_saved_count":  len(messages),  # all loaded messages already in DB
    })
    log.info("[%s] Lazy-loaded %d message(s) from DB → Redis", thread_id, len(messages))
    return True


@asynccontextmanager
async def lifespan(_: FastAPI):
    _init_db()
    runtime.on_startup()
    yield
    log.info("Server shutting down — cleaning up sandboxes...")
    runtime.cleanup_all()


app = FastAPI(title="Transaction Agent", description="Container-per-thread workspace demo", lifespan=lifespan)

# ---------- Redis thread store ----------
# Replaces the in-memory threads dict so multiple server instances share state.
# Key schema:
#   thread:{thread_id}  → JSON blob {messages, system_prompt, runtime, _saved_count}
# Configure via REDIS_HOST / REDIS_PORT env vars (defaults: localhost:6379).

_redis_client: _redis_lib.Redis | None = None


def _get_redis() -> _redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = _redis_lib.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=0,
            decode_responses=True,
        )
    return _redis_client


_THREAD_KEY_PREFIX = "thread:"
_THREAD_TTL_SECS   = 86400 * 30  # 30 days — auto-expire stale keys


def _rkey(thread_id: str) -> str:
    return f"{_THREAD_KEY_PREFIX}{thread_id}"


def _get_thread(thread_id: str) -> dict | None:
    """Read thread state from Redis. Returns None if not cached."""
    raw = _get_redis().get(_rkey(thread_id))
    return json.loads(raw) if raw else None


def _set_thread(thread_id: str, thread: dict) -> None:
    """Write thread state to Redis with a rolling TTL."""
    _get_redis().setex(_rkey(thread_id), _THREAD_TTL_SECS, json.dumps(thread))


# ---------- System prompt ----------

SYSTEM_PROMPT_BASE = """You are a data analyst agent.

A DataFrame `df` is pre-loaded in your Python environment from a CSV file.
{schema}
Use the execute_python tool to analyze data with pandas/numpy.
IMPORTANT: Always use print() to show results — bare expressions produce no output.
Variables you define persist across messages in the same thread — reuse them freely.
After running code, always explain the result clearly in plain English."""


def build_system_prompt(df) -> str:
    """Generate a system prompt by introspecting the actual DataFrame."""
    lines = [f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n", "Columns:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        if df[col].dtype == "object":
            unique = df[col].nunique()
            sample = df[col].dropna().unique()[:4].tolist()
            detail = f"string, {unique} unique values, sample: {sample}"
        elif "datetime" in dtype:
            detail = f"datetime, range: {df[col].min().date()} → {df[col].max().date()}"
        elif df[col].dtype in ("float64", "int64"):
            detail = f"numeric, min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
        else:
            detail = dtype
        null_note = f", {nulls} nulls" if nulls else ""
        lines.append(f"  - {col}: {detail}{null_note}")

    return SYSTEM_PROMPT_BASE.format(schema="\n".join(lines))


# ---------- OpenAI tools ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Execute Python code in the workspace container. "
                "`df` is pre-loaded. Variables persist across calls in the same thread."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to show results.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

# ---------- Runtime (pluggable sandbox backend) ----------

runtime = create_runtime()

# ---------- OpenAI client (module-level, reuses connection pool) ----------

client = AsyncOpenAI()


def _restore_sandbox(thread_id: str, workspace_dir: Path) -> tuple[str, float]:
    """Sync helper: restore sandbox to HOT. Returns (state_label, restore_ms)."""
    t = time.perf_counter()
    state = runtime.get_or_restore(thread_id)
    if state != "cold":
        restore_ms = round((time.perf_counter() - t) * 1000, 2)
        return state, restore_ms
    if not workspace_dir.exists():
        raise RuntimeError("Workspace not found. Create a new thread.")
    current_backend  = os.getenv("SANDBOX_RUNTIME", "docker")
    thread = _get_thread(thread_id)
    original_backend = thread.get("runtime", current_backend) if thread else current_backend
    if original_backend != current_backend:
        log.warning(
            "[%s] Backend mismatch: thread was created with '%s' but server is running '%s'. "
            "Conversation history is preserved but computed variables (non-CSV data) will be lost.",
            thread_id, original_backend, current_backend,
        )
    log.info("[%s] Restoring from COLD state...", thread_id)
    runtime.create(thread_id, workspace_dir)
    restore_ms = round((time.perf_counter() - t) * 1000, 2)
    log.info("[%s] Restored in %.0f ms", thread_id, restore_ms)
    return f"cold→hot ({restore_ms}ms)", restore_ms


# ---------- Request models ----------

class MessageRequest(BaseModel):
    message: str


# ---------- Routes ----------

@app.get("/")
def root():
    return {
        "status": "ok",
        "ui":   "/ui",
        "docs": "/docs",
        "endpoints": {
            "POST /upload":                         "Upload any file (CSV, image, PDF…); returns file_id",
            "POST /thread/new":                     "Create a new thread; pass file_ids[] to attach uploads",
            "POST /thread/{id}/message":            "Send a message — returns JSON when fully done",
            "POST /thread/{id}/message/stream":     "Send a message — SSE stream: token/status/code_run/done/error events",
            "GET  /thread/{id}/status":             "Show container state (hot/warm/cold)",
            "GET  /thread/{id}/history":            "Get conversation history",
            "GET  /threads":                        "List all threads",
        },
    }


@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    """Serve the chat frontend."""
    return (Path(__file__).parent / "frontend" / "index.html").read_text()


async def _process_upload(file: UploadFile) -> dict:
    """Save one uploaded file, validate CSVs. Returns the per-file result dict."""
    import pandas as pd  # noqa: PLC0415

    contents = await file.read()
    if not contents:
        return {"filename": file.filename, "error": "File is empty"}

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower() or ".bin"

    file_id = str(uuid.uuid4())[:8]
    dest = UPLOADS_DIR / f"{file_id}{ext}"
    dest.write_bytes(contents)

    result: dict = {
        "file_id": file_id,
        "filename": filename,
        "size_bytes": len(contents),
    }

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            dest.unlink(missing_ok=True)
            return {"filename": filename, "error": f"Could not parse CSV: {e}"}
        if len(df) == 0:
            dest.unlink(missing_ok=True)
            return {"filename": filename, "error": "CSV has no data rows"}
        log.info("CSV upload '%s': %d rows | columns: %s", filename, len(df), list(df.columns))
        result["rows"] = len(df)
        result["columns"] = list(df.columns)
    else:
        log.info("File upload '%s': %d bytes", filename, len(contents))

    return result


@app.post("/upload")
async def upload_files(files: Annotated[list[UploadFile], File(...)]):
    """Upload one or more files (CSV, image, PDF, …).

    Returns a list of results — each with a file_id to use in POST /thread/new.
    CSVs are validated (must have at least one data row); other types just need to be non-empty.
    """
    UPLOADS_DIR.mkdir(exist_ok=True)
    results = []
    errors = []
    for f in files:
        r = await _process_upload(f)
        if "error" in r:
            errors.append(r)
        else:
            results.append(r)

    if errors and not results:
        # All files failed
        raise HTTPException(status_code=400, detail={"errors": errors})

    return {
        "message": f"{len(results)} file(s) uploaded successfully",
        "files": results,
        **({"errors": errors} if errors else {}),
    }


class NewThreadRequest(BaseModel):
    file_ids: list[str] = []  # from POST /upload; empty = use default transactions.csv


def _resolve_upload(file_id: str) -> Path:
    """Return the path for a previously uploaded file_id (any extension)."""
    matches = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"file_id '{file_id}' not found. Upload a file first via POST /upload",
        )
    return matches[0]


@app.post("/thread/new")
def new_thread(body: NewThreadRequest = NewThreadRequest()):
    """Create a new thread. Pass file_ids (from POST /upload) to attach files to the workspace."""
    thread_id = str(uuid.uuid4())[:8]
    total_start = time.perf_counter()
    log.info("── Creating thread %s ──────────────────────", thread_id)

    # 1. Set up workspace directory
    workspace_dir = WORKSPACES_DIR / thread_id
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy files into workspace; track which CSV to use for schema
    csv_for_schema: Path | None = None
    attached_files: list[str] = []

    if not body.file_ids:
        # Default: use the bundled transactions.csv
        dest = workspace_dir / "data.csv"
        shutil.copy(TRANSACTIONS_PATH, dest)
        csv_for_schema = dest
        attached_files = ["data.csv"]
    else:
        for fid in body.file_ids:
            src = _resolve_upload(fid)
            dest = workspace_dir / src.name   # keeps original name e.g. abc12345.csv / abc12345.png
            shutil.copy(src, dest)
            attached_files.append(src.name)
            if src.suffix.lower() == ".csv" and csv_for_schema is None:
                csv_for_schema = dest

    log.info("  Attached files: %s", attached_files)

    # 3. Build system prompt
    t = time.perf_counter()
    import pandas as pd  # noqa: PLC0415

    if csv_for_schema:
        df_schema = pd.read_csv(csv_for_schema)
        for col in df_schema.select_dtypes(include="object").columns:
            try:
                df_schema[col] = pd.to_datetime(df_schema[col], infer_datetime_format=True)
            except (ValueError, TypeError):
                pass
        non_csv = [f for f in attached_files if not f.endswith(".csv")]
        extra = f"\nOther files in workspace: {non_csv}" if non_csv else ""
        system_prompt = build_system_prompt(df_schema) + extra
        schema_ms = round((time.perf_counter() - t) * 1000, 2)
        log.info("  Schema built in %.0f ms (%d rows, %d cols)", schema_ms, *df_schema.shape)
        del df_schema
    else:
        # No CSV — generic prompt listing whatever files are present
        file_list = ", ".join(attached_files) or "(none)"
        system_prompt = SYSTEM_PROMPT_BASE.format(
            schema=f"No CSV data is pre-loaded. Files available in workspace: {file_list}"
        )
        schema_ms = round((time.perf_counter() - t) * 1000, 2)

    # 4. Spin up the sandbox
    t = time.perf_counter()
    runtime.create(thread_id, workspace_dir)
    container_ms = round((time.perf_counter() - t) * 1000, 2)

    # 5. Store thread state in DB and memory (no namespace — it lives in the sandbox)
    backend = os.getenv("SANDBOX_RUNTIME", "docker")
    with _get_db() as conn:
        conn.execute(
            "INSERT INTO threads (thread_id, runtime, system_prompt) VALUES (?, ?, ?)",
            (thread_id, backend, system_prompt),
        )
    _backup_db()
    _backup_workspace(thread_id)
    _set_thread(thread_id, {
        "messages":      [],
        "system_prompt": system_prompt,
        "runtime":       backend,
        "_saved_count":  0,
    })

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    log.info("── Thread %s ready in %.0f ms total ──", thread_id, total_ms)

    return {
        "thread_id": thread_id,
        "container_status": "hot",
        "workspace": str(workspace_dir),
        "attached_files": attached_files,
        "timings_ms": {
            "schema_build":      schema_ms,
            "container_startup": container_ms,
            "total":             total_ms,
        },
    }


@app.post("/thread/{thread_id}/message")
async def send_message(thread_id: str, body: MessageRequest, background_tasks: BackgroundTasks):
    """Send a message. Sandbox restore and first OpenAI call run in parallel."""
    if not _ensure_thread_loaded(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found. Create one via POST /thread/new")

    thread = _get_thread(thread_id)
    workspace_dir = WORKSPACES_DIR / thread_id
    thread["messages"].append({"role": "user", "content": body.message})

    code_runs: list[dict] = []
    openai_calls = 0
    openai_timings_ms: list[float] = []
    restore_ms = 0.0
    msg_start = time.perf_counter()

    # Parallel: sandbox restore (sync→thread) + first OpenAI call (async)
    t0 = time.perf_counter()
    try:
        (state, restore_ms), response = await asyncio.gather(
            asyncio.to_thread(_restore_sandbox, thread_id, workspace_dir),
            client.chat.completions.create(
                model="gpt-5.2-chat-latest",
                messages=[{"role": "system", "content": thread["system_prompt"]}] + thread["messages"],
                tools=TOOLS,
                tool_choice="auto",
            ),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    openai_calls = 1
    first_gather_ms = round((time.perf_counter() - t0) * 1000, 2)
    openai_timings_ms.append(first_gather_ms)
    log.info("[%s] First gather (OpenAI #1 ∥ sandbox restore): %.0f ms | restore: %.0f ms | tokens: %d",
             thread_id, first_gather_ms, restore_ms,
             response.usage.prompt_tokens if response.usage else -1)
    log.info("[%s] Sandbox state: %s", thread_id, state)

    # Agent loop — process responses, exec code, repeat
    while True:
        msg = response.choices[0].message

        if msg.tool_calls:
            thread["messages"].append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                if tc.function.name == "execute_python":
                    code = json.loads(tc.function.arguments)["code"]
                    log.info("[%s] LLM generated code:\n%s", thread_id, code)
                    result, exec_ms = await asyncio.to_thread(runtime.exec_code, thread_id, code)
                    log.info("[%s] Code executed in sandbox: %.0f ms | output:\n%s",
                             thread_id, exec_ms, result if result else "(none)")
                    code_runs.append({"code": code, "output": result, "exec_ms": exec_ms})
                    thread["messages"].append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
        else:
            final = msg.content or ""
            thread["messages"].append({"role": "assistant", "content": final})
            total_ms = round((time.perf_counter() - msg_start) * 1000, 2)
            log.info("[%s] Done — %d OpenAI call(s), %.0f ms total", thread_id, openai_calls, total_ms)
            _set_thread(thread_id, thread)  # flush to Redis before background tasks read it
            # All post-response work runs after response is sent, in order:
            # 1. save namespace pkl (e2b only — Docker worker handles it internally)
            # 2. save messages to DB + backup
            # 3. suspend sandbox
            background_tasks.add_task(runtime.save_namespace, thread_id)
            background_tasks.add_task(_save_thread, thread_id)
            background_tasks.add_task(runtime.suspend, thread_id)
            exec_total_ms  = round(sum(r["exec_ms"] for r in code_runs), 2)
            openai_total_ms = round(sum(openai_timings_ms), 2)
            return {
                "response": final,
                "thread_id": thread_id,
                "container_state": state,
                "openai_calls": openai_calls,
                "total_ms": total_ms,
                "timings_ms": {
                    # openai[0] ran in parallel with sandbox_restore — wall time shown
                    "sandbox_restore":  restore_ms,
                    "openai_per_call":  openai_timings_ms,
                    "openai_total":     openai_total_ms,
                    "exec_per_call":    [r["exec_ms"] for r in code_runs],
                    "exec_total":       exec_total_ms,
                    # overhead = routing, DB, message append, Python glue
                    "overhead":         round(total_ms - first_gather_ms - exec_total_ms - sum(openai_timings_ms[1:]), 2),
                },
                "code_runs": code_runs,
            }

        if openai_calls >= 10:
            break

        t = time.perf_counter()
        # Parallel: next OpenAI call + namespace save (e2b: ~550ms saved; Docker: save_namespace is no-op)
        response, _ = await asyncio.gather(
            client.chat.completions.create(
                model="gpt-5.2-chat-latest",
                messages=[{"role": "system", "content": thread["system_prompt"]}] + thread["messages"],
                tools=TOOLS,
                tool_choice="auto",
            ),
            asyncio.to_thread(runtime.save_namespace, thread_id),
        )
        openai_calls += 1
        call_ms = round((time.perf_counter() - t) * 1000, 2)
        openai_timings_ms.append(call_ms)
        log.info("[%s] OpenAI call #%d + namespace save: %.0f ms (prompt tokens: %d)",
                 thread_id, openai_calls, call_ms,
                 response.usage.prompt_tokens if response.usage else -1)

    # Hit max iterations — flush to Redis then save and suspend in background
    _set_thread(thread_id, thread)
    background_tasks.add_task(runtime.save_namespace, thread_id)
    background_tasks.add_task(_save_thread, thread_id)
    background_tasks.add_task(runtime.suspend, thread_id)
    total_ms = round((time.perf_counter() - msg_start) * 1000, 2)
    return {
        "response": "[agent hit max iterations]",
        "thread_id": thread_id,
        "container_state": state,
        "openai_calls": openai_calls,
        "total_ms": total_ms,
        "timings_ms": {
            "sandbox_restore": restore_ms,
            "openai_per_call": openai_timings_ms,
            "openai_total":    round(sum(openai_timings_ms), 2),
            "exec_per_call":   [r["exec_ms"] for r in code_runs],
            "exec_total":      round(sum(r["exec_ms"] for r in code_runs), 2),
            "overhead":        round(total_ms - first_gather_ms - sum(r["exec_ms"] for r in code_runs) - sum(openai_timings_ms[1:]), 2),
        },
        "code_runs": code_runs,
    }


@app.post("/thread/{thread_id}/message/stream")
async def send_message_stream(thread_id: str, body: MessageRequest, background_tasks: BackgroundTasks):
    """Send a message with Server-Sent Events streaming for the final text response.

    SSE event types (each line: ``data: <json>\\n\\n``):
      {"type": "status",   "text": "..."}                         — progress update
      {"type": "token",    "text": "..."}                         — one token of the final answer
      {"type": "code_run", "code": "...", "output": "...", "exec_ms": N} — executed code + result
      {"type": "done",     "thread_id": "...", "timings_ms": {...}, ...} — completion metadata
      {"type": "error",    "message": "..."}                      — fatal error
    """
    if not _ensure_thread_loaded(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found. Create one via POST /thread/new")

    async def generate():
        thread = _get_thread(thread_id)
        workspace_dir = WORKSPACES_DIR / thread_id
        thread["messages"].append({"role": "user", "content": body.message})

        code_runs: list[dict] = []
        openai_calls = 0
        openai_timings_ms: list[float] = []
        restore_ms = 0.0
        ttft_ms: float | None = None
        msg_start = time.perf_counter()

        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        # Restore sandbox — runs in a thread so the event loop stays responsive
        try:
            state, restore_ms = await asyncio.to_thread(_restore_sandbox, thread_id, workspace_dir)
            log.info("[%s] Sandbox state: %s (%.0f ms)", thread_id, state, restore_ms)
        except RuntimeError as e:
            yield sse({"type": "error", "message": str(e)})
            return

        # Agent loop — stream every OpenAI call; accumulate tool-call chunks silently
        while True:
            if openai_calls >= 10:
                _set_thread(thread_id, thread)
                background_tasks.add_task(runtime.save_namespace, thread_id)
                background_tasks.add_task(_save_thread, thread_id)
                background_tasks.add_task(runtime.suspend, thread_id)
                yield sse({"type": "error", "message": "Agent hit max iterations"})
                return

            t = time.perf_counter()
            openai_calls += 1

            tool_calls_acc: dict[int, dict] = {}   # index → {id, name, arguments}
            text_buffer = ""

            stream = await client.chat.completions.create(
                model="gpt-5.2-chat-latest",
                messages=[{"role": "system", "content": thread["system_prompt"]}] + thread["messages"],
                tools=TOOLS,
                tool_choice="auto",
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    if ttft_ms is None:
                        ttft_ms = round((time.perf_counter() - msg_start) * 1000, 2)
                    text_buffer += delta.content
                    yield sse({"type": "token", "text": delta.content})

                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_chunk.id:
                            tool_calls_acc[idx]["id"] += tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_calls_acc[idx]["name"] += tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc_chunk.function.arguments

            call_ms = round((time.perf_counter() - t) * 1000, 2)
            openai_timings_ms.append(call_ms)
            log.info("[%s] OpenAI call #%d: %.0f ms (tool_calls: %d)",
                     thread_id, openai_calls, call_ms, len(tool_calls_acc))

            # No tool calls → final text answer; tokens already streamed
            if not tool_calls_acc:
                thread["messages"].append({"role": "assistant", "content": text_buffer})
                total_ms = round((time.perf_counter() - msg_start) * 1000, 2)
                log.info("[%s] Done — %d OpenAI call(s), %.0f ms total", thread_id, openai_calls, total_ms)
                _set_thread(thread_id, thread)  # flush to Redis before background tasks read it
                background_tasks.add_task(runtime.save_namespace, thread_id)
                background_tasks.add_task(_save_thread, thread_id)
                background_tasks.add_task(runtime.suspend, thread_id)
                exec_total_ms   = round(sum(r["exec_ms"] for r in code_runs), 2)
                openai_total_ms = round(sum(openai_timings_ms), 2)
                yield sse({
                    "type":            "done",
                    "thread_id":       thread_id,
                    "container_state": state,
                    "openai_calls":    openai_calls,
                    "total_ms":        total_ms,
                    "timings_ms": {
                        "ttft":            ttft_ms,
                        # How many ms earlier the user saw content vs waiting for the full response.
                        # = time saved by streaming on the final OpenAI call.
                        "ttft_saved":      round(total_ms - ttft_ms, 2) if ttft_ms else None,
                        "sandbox_restore": restore_ms,
                        "openai_per_call": openai_timings_ms,
                        "openai_total":    openai_total_ms,
                        "exec_per_call":   [r["exec_ms"] for r in code_runs],
                        "exec_total":      exec_total_ms,
                        "overhead":        round(total_ms - restore_ms - exec_total_ms - openai_total_ms, 2),
                    },
                    "code_runs": code_runs,
                })
                return

            # Has tool calls — build assistant message and execute
            tool_calls_list = [
                {
                    "id":   tool_calls_acc[idx]["id"],
                    "type": "function",
                    "function": {
                        "name":      tool_calls_acc[idx]["name"],
                        "arguments": tool_calls_acc[idx]["arguments"],
                    },
                }
                for idx in sorted(tool_calls_acc.keys())
            ]
            thread["messages"].append({
                "role":       "assistant",
                "content":    text_buffer or None,
                "tool_calls": tool_calls_list,
            })

            for tc in tool_calls_list:
                if tc["function"]["name"] == "execute_python":
                    code = json.loads(tc["function"]["arguments"])["code"]
                    log.info("[%s] LLM generated code:\n%s", thread_id, code)
                    yield sse({"type": "status", "text": "Executing Python code..."})
                    result, exec_ms = await asyncio.to_thread(runtime.exec_code, thread_id, code)
                    log.info("[%s] Code executed in %.0f ms | output:\n%s",
                             thread_id, exec_ms, result or "(none)")
                    code_runs.append({"code": code, "output": result, "exec_ms": exec_ms})
                    yield sse({"type": "code_run", "code": code, "output": result, "exec_ms": exec_ms})
                    thread["messages"].append({
                        "role":        "tool",
                        "tool_call_id": tc["id"],
                        "content":     result,
                    })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/thread/{thread_id}/status")
def get_status(thread_id: str):
    """Show the container lifecycle state for this thread."""
    if not _ensure_thread_loaded(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    thread = _get_thread(thread_id)
    return {
        "thread_id": thread_id,
        "sandbox_status": runtime.status(thread_id),
        "message_count": len(thread["messages"]),
    }


@app.get("/thread/{thread_id}/history")
def get_history(thread_id: str):
    """Return the conversation history (user and assistant messages only)."""
    if not _ensure_thread_loaded(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    thread = _get_thread(thread_id)
    readable = [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in thread["messages"]
        if m.get("role") in ("user", "assistant")
    ]
    return {"thread_id": thread_id, "messages": readable}


@app.get("/threads")
def list_threads():
    """List all threads (from DB) with their Redis-cached state and sandbox status."""
    with _get_db() as conn:
        rows = conn.execute("SELECT thread_id FROM threads").fetchall()
    result = {}
    for row in rows:
        tid = row["thread_id"]
        thread = _get_thread(tid)
        result[tid] = {
            "message_count": len(thread["messages"]) if thread else None,
            "sandbox_status": runtime.status(tid),
            "cached": thread is not None,
        }
    return {"threads": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
