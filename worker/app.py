#!/usr/bin/env python3
"""
Worker container — holds the Python namespace for one thread.
Loaded once on startup, exec'd on demand, paused between messages.
Namespace is persisted to /workspace/namespace.pkl after every exec so it
survives container eviction and server restarts.
"""

import io
import logging
import pickle
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("worker")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI()

WORKSPACE = Path("/workspace")
PICKLE_PATH = WORKSPACE / "namespace.pkl"

# Modules that can't be pickled — always re-injected on startup
_BUILTINS = {"pd", "np"}

# Shared namespace persists for the lifetime of this container
namespace: dict = {"pd": pd, "np": np}


def _save_namespace() -> None:
    """Pickle all user-created variables (not pd/np) to disk.
    Skips individual variables that are not picklable so one bad value
    (e.g. a matplotlib figure) doesn't block everything else from saving.
    """
    to_save = {}
    skipped = []
    for k, v in namespace.items():
        if k in _BUILTINS:
            continue
        try:
            pickle.dumps(v)  # test picklability without writing
            to_save[k] = v
        except Exception:
            skipped.append(k)
    if skipped:
        log.warning("Skipped non-picklable variables: %s", skipped)
    try:
        with PICKLE_PATH.open("wb") as f:
            pickle.dump(to_save, f)
        log.info("Namespace saved to pickle: %s", list(to_save.keys()))
    except Exception as e:
        log.warning("Could not save namespace pickle: %s", e)


def _load_namespace() -> bool:
    """Restore namespace from pickle. Returns True if pickle was loaded."""
    if not PICKLE_PATH.exists():
        return False
    try:
        with PICKLE_PATH.open("rb") as f:
            saved = pickle.load(f)
        namespace.update(saved)
        log.info("Namespace restored from pickle: %s", list(saved.keys()))
        return True
    except Exception as e:
        log.warning("Could not load namespace pickle (will fall back to CSV): %s", e)
        return False


@app.on_event("startup")
def load_workspace() -> None:
    """
    Startup priority:
      1. If namespace.pkl exists → restore full namespace (df + all variables)
      2. Otherwise → load df from CSV as fresh start
    """
    # Try pickle first
    if _load_namespace():
        return

    # Fall back to CSV
    csv_files = sorted(WORKSPACE.glob("*.csv"))
    if not csv_files:
        return  # no data yet — namespace has pd/np only

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # Auto-detect date-like object columns
    for col in df.select_dtypes(include="object").columns:
        try:
            df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
        except (ValueError, TypeError):
            pass

    namespace["df"] = df
    log.info("Loaded df from CSV: %s rows, %s cols", len(df), len(df.columns))


class ExecRequest(BaseModel):
    code: str


@app.post("/exec")
def exec_code(req: ExecRequest) -> dict:
    """Execute code in the shared namespace, return stdout + stderr."""
    log.info("Executing code:\n%s", req.code)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            exec(compile(req.code, "<agent>", "exec"), namespace)  # noqa: S102
        output = buf_out.getvalue().strip() or "(executed, no output)"
        error = buf_err.getvalue() or None
        log.info("Output:\n%s", output)
        if error:
            log.warning("Stderr:\n%s", error)
        # Persist namespace to disk after every exec
        _save_namespace()
        return {"output": output, "error": error}
    except Exception:
        tb = traceback.format_exc()
        log.error("Exception:\n%s", tb)
        return {"output": "", "error": tb}


@app.get("/ping")
def ping() -> dict:
    """Health check — also reports whether df is loaded."""
    df = namespace.get("df")
    return {
        "ok": True,
        "rows": len(df) if df is not None else 0,
        "has_df": df is not None,
    }
