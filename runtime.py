"""
runtime.py — Pluggable sandbox backend abstraction.

Select backend via env var:
  SANDBOX_RUNTIME=docker   (default) — Docker container per thread
  SANDBOX_RUNTIME=e2b      — e2b cloud sandbox per thread

Both backends expose the same interface so server.py never knows which
one is running. Key behavioural difference:

  Docker suspend → docker pause  (memory frozen in RAM, pickle on disk)
  e2b    suspend → beta_pause()  (full kernel state saved by e2b cloud)
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

WORKER_IMAGE      = "sandbox-worker"
IDLE_TIMEOUT_SECS = 900   # 15 min idle → eviction
MAX_WARM          = 50    # max warm Docker containers
WORKSPACES_DIR    = Path(__file__).parent / "workspaces"


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class RuntimeBackend(ABC):
    """Common interface for all sandbox backends."""

    @abstractmethod
    def create(self, thread_id: str, workspace_dir: Path) -> None:
        """Spin up a fresh sandbox for a new thread.

        Docker: docker run with workspace volume mount.
        e2b:    Sandbox.create() + upload CSV files from workspace_dir.
        """

    @abstractmethod
    def get_or_restore(self, thread_id: str) -> str:
        """Ensure sandbox is running and ready.

        Docker: unpause (WARM→HOT) or create new container + load pickle (COLD→HOT).
        e2b:    Sandbox.connect(sandbox_id) auto-resumes the paused kernel.

        Returns a state label string, e.g. 'hot', 'warm→hot (12ms)', 'cold→hot (843ms)'.
        """

    @abstractmethod
    def suspend(self, thread_id: str) -> None:
        """Suspend sandbox after a message response.

        Docker: docker pause — memory stays frozen in RAM (WARM).
        e2b:    beta_pause() — full kernel state saved by e2b, sandbox removed.
        """

    @abstractmethod
    def exec_code(self, thread_id: str, code: str) -> tuple[str, float]:
        """Run code in the sandbox. Returns (output, exec_ms)."""

    def save_namespace(self, thread_id: str) -> None:
        """Persist kernel namespace to local disk after exec.

        Called in the background — parallel with next OpenAI call (mid-loop)
        or as a background task on final response.

        Docker: no-op — worker's /exec handler saves pickle automatically.
        e2b:    serialize kernel globals → download to workspace/namespace.pkl.
        """

    @abstractmethod
    def status(self, thread_id: str) -> str:
        """Return 'hot', 'warm', or 'cold'."""

    @abstractmethod
    def on_startup(self) -> None:
        """Called once when the server starts (e.g. orphan cleanup)."""

    @abstractmethod
    def cleanup_all(self) -> None:
        """Called on server shutdown — stop/remove all active sandboxes."""


# ─────────────────────────────────────────────────────────────────────────────
# Docker backend
# ─────────────────────────────────────────────────────────────────────────────

class DockerRuntime(RuntimeBackend):
    """
    Docker container per thread.

    States:
      HOT  → container running, namespace live in RAM
      WARM → container paused (SIGSTOP), namespace frozen in RAM
             + pickle on disk (survives eviction)
      COLD → container destroyed, pickle on disk only
    """

    def __init__(self) -> None:
        import docker as docker_sdk  # noqa: PLC0415
        self._docker = docker_sdk.from_env()
        # thread_id → {container, url, status, timer}
        self._entries: dict = {}
        self._lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────

    def create(self, thread_id: str, workspace_dir: Path) -> None:
        log.info("[%s] Starting Docker container (workspace: %s)", thread_id, workspace_dir)
        total_start = time.perf_counter()

        t = time.perf_counter()
        container = self._docker.containers.run(
            WORKER_IMAGE,
            detach=True,
            ports={"8080/tcp": None},
            volumes={str(workspace_dir): {"bind": "/workspace", "mode": "rw"}},
            labels={"sandbox.thread_id": thread_id},
        )
        # macOS / Docker Desktop can take a moment to assign the host port
        for _ in range(20):
            container.reload()
            port_bindings = container.ports.get("8080/tcp")
            if port_bindings:
                break
            time.sleep(0.1)
        else:
            raise RuntimeError(f"[{thread_id}] Container port 8080 never assigned")
        boot_ms = round((time.perf_counter() - t) * 1000, 2)
        log.info("[%s] Container booted in %.0f ms (id: %s)", thread_id, boot_ms, container.short_id)

        port = port_bindings[0]["HostPort"]
        url  = f"http://localhost:{port}"
        t = time.perf_counter()
        self._wait_for_ready(url, thread_id)
        ready_ms = round((time.perf_counter() - t) * 1000, 2)

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        log.info("[%s] Container ready in %.0f ms total (boot: %.0f ms, health: %.0f ms) → %s",
                 thread_id, total_ms, boot_ms, ready_ms, url)

        with self._lock:
            self._entries[thread_id] = {
                "container": container,
                "url":       url,
                "status":    "hot",
                "timer":     None,
            }
            self._schedule_eviction(thread_id)

    def get_or_restore(self, thread_id: str) -> str:
        """Unpause WARM container or signal COLD (caller must call create())."""
        with self._lock:
            entry = self._entries.get(thread_id)
            if not entry:
                return "cold"

            if entry["status"] == "warm":
                t = time.perf_counter()
                entry["container"].unpause()
                entry["status"] = "hot"
                unpause_ms = round((time.perf_counter() - t) * 1000, 2)
                log.info("[%s] Container unpaused in %.0f ms (warm→hot)", thread_id, unpause_ms)
                self._schedule_eviction(thread_id)
                return f"warm→hot ({unpause_ms}ms)"

            self._schedule_eviction(thread_id)
            return "hot"

    def suspend(self, thread_id: str) -> None:
        with self._lock:
            entry = self._entries.get(thread_id)
            if entry and entry["status"] == "hot":
                t = time.perf_counter()
                entry["container"].pause()
                entry["status"] = "warm"
                pause_ms = round((time.perf_counter() - t) * 1000, 2)
                log.info("[%s] Container paused in %.0f ms (hot→warm)", thread_id, pause_ms)

    def exec_code(self, thread_id: str, code: str) -> tuple[str, float]:
        with self._lock:
            entry = self._entries.get(thread_id)
            if not entry:
                raise RuntimeError(f"[{thread_id}] No active container for exec")
            url = entry["url"]
        t = time.perf_counter()
        resp = httpx.post(f"{url}/exec", json={"code": code}, timeout=30.0)
        exec_ms = round((time.perf_counter() - t) * 1000, 2)
        data = resp.json()
        output = data.get("output", "(no output)")
        if data.get("error"):
            output += f"\n[error]: {data['error']}"
        return output.strip(), exec_ms

    def status(self, thread_id: str) -> str:
        entry = self._entries.get(thread_id)
        if not entry:
            return "cold"
        return entry["status"]

    def on_startup(self) -> None:
        """Remove containers left over from a previous server run."""
        try:
            orphans = self._docker.containers.list(filters={"label": "sandbox.thread_id"})
            for c in orphans:
                log.info("Removing orphaned container %s", c.short_id)
                try:
                    c.stop(timeout=2)
                    c.remove()
                except Exception as e:
                    log.warning("Could not remove orphan %s: %s", c.short_id, e)
            if orphans:
                log.info("Cleaned up %d orphaned container(s)", len(orphans))
        except Exception as e:
            log.warning("Orphan cleanup failed (Docker unavailable?): %s", e)

    def cleanup_all(self) -> None:
        with self._lock:
            tids = list(self._entries.keys())
        for tid in tids:
            self._evict(tid)

    # ── internals ─────────────────────────────────────────────────────────────

    def _evict(self, thread_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(thread_id, None)
        if not entry:
            return
        log.info("[%s] Evicting container (warm→cold) — pickle preserved on disk", thread_id)
        t = time.perf_counter()
        try:
            if entry["status"] == "warm":
                entry["container"].unpause()
            entry["container"].stop(timeout=2)
            entry["container"].remove()
            evict_ms = round((time.perf_counter() - t) * 1000, 2)
            log.info("[%s] Container evicted in %.0f ms", thread_id, evict_ms)
        except Exception as e:
            log.warning("[%s] Eviction error (ignored): %s", thread_id, e)

    def _wait_for_ready(self, url: str, thread_id: str, timeout: float = 20.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = httpx.get(f"{url}/ping", timeout=1.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"[{thread_id}] Worker at {url} not ready after {timeout}s")

    def _schedule_eviction(self, thread_id: str) -> None:
        """Reset idle eviction timer. Must be called with lock held."""
        entry = self._entries.get(thread_id)
        if not entry:
            return
        if entry.get("timer"):
            entry["timer"].cancel()
        timer = threading.Timer(IDLE_TIMEOUT_SECS, self._evict, args=[thread_id])
        timer.daemon = True
        timer.start()
        entry["timer"] = timer


# ─────────────────────────────────────────────────────────────────────────────
# e2b backend
# ─────────────────────────────────────────────────────────────────────────────

class E2BRuntime(RuntimeBackend):
    """
    e2b cloud sandbox per thread.

    States:
      HOT  → sandbox running, Jupyter kernel live in e2b cloud
      WARM → beta_pause() called — e2b saves full kernel state to cloud storage,
             sandbox is removed. Reconnect via Sandbox.connect(sandbox_id)
             to auto-resume with full state intact (no pickle needed).

    No COLD state: e2b snapshot is always available until explicitly deleted.
    """

    def __init__(self, api_key: str) -> None:
        os.environ["E2B_API_KEY"] = api_key
        # thread_id → {sandbox, sandbox_id, status}
        self._entries: dict = {}
        self._lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────

    def create(self, thread_id: str, workspace_dir: Path) -> None:
        from e2b_code_interpreter import Sandbox  # noqa: PLC0415
        log.info("[%s] Creating e2b sandbox", thread_id)
        total_start = time.perf_counter()

        t = time.perf_counter()
        sbx = Sandbox.create()
        boot_ms = round((time.perf_counter() - t) * 1000, 2)
        log.info("[%s] e2b sandbox booted in %.0f ms (id: %s)", thread_id, boot_ms, sbx.sandbox_id)

        # Upload all files from the workspace directory into the sandbox
        files = [f for f in workspace_dir.iterdir() if f.is_file()]
        upload_start = time.perf_counter()
        for file_path in files:
            t = time.perf_counter()
            sbx.files.write(file_path.name, file_path.read_bytes())
            file_ms = round((time.perf_counter() - t) * 1000, 2)
            log.info("[%s] Uploaded '%s' (%d bytes) in %.0f ms",
                     thread_id, file_path.name, file_path.stat().st_size, file_ms)
        upload_ms = round((time.perf_counter() - upload_start) * 1000, 2)
        log.info("[%s] Uploaded %d file(s) in %.0f ms total", thread_id, len(files), upload_ms)

        # Initialize the Jupyter kernel — pickle fallback first, then CSV
        csv_files = [f for f in files if f.suffix.lower() == ".csv"]
        has_pickle = (workspace_dir / "namespace.pkl").exists()
        t = time.perf_counter()
        if has_pickle:
            init_code = (
                "import pandas as pd\n"
                "import numpy as np\n"
                "import pickle\n"
                "_ns = pickle.load(open('namespace.pkl', 'rb'))\n"
                "globals().update(_ns)\n"
                "print(f'Restored {len(_ns)} variable(s) from namespace.pkl')\n"
            )
        elif csv_files:
            csv_name = csv_files[0].name
            init_code = (
                "import pandas as pd\n"
                "import numpy as np\n"
                f"df = pd.read_csv('{csv_name}')\n"
                "# Auto-detect date columns\n"
                "for col in df.select_dtypes(include='object').columns:\n"
                "    try:\n"
                "        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)\n"
                "    except Exception:\n"
                "        pass\n"
                "print(f'df loaded: {df.shape[0]} rows x {df.shape[1]} cols')"
            )
        else:
            init_code = "import pandas as pd\nimport numpy as np\nprint('kernel ready (no CSV)')"

        init_exec = sbx.run_code(init_code)
        init_ms = round((time.perf_counter() - t) * 1000, 2)
        init_out = "\n".join(str(o) for o in init_exec.logs.stdout).strip()
        log.info("[%s] Kernel initialized in %.0f ms — %s", thread_id, init_ms, init_out)

        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        log.info("[%s] e2b sandbox ready in %.0f ms (boot: %.0f ms, upload: %.0f ms, init: %.0f ms)",
                 thread_id, total_ms, boot_ms, upload_ms, init_ms)

        # Persist sandbox_id to disk so it survives server restarts
        self._save_sandbox_id(thread_id, workspace_dir, sbx.sandbox_id)

        with self._lock:
            self._entries[thread_id] = {
                "sandbox":      sbx,
                "sandbox_id":   sbx.sandbox_id,
                "workspace_dir": workspace_dir,
                "status":       "hot",
            }

    def get_or_restore(self, thread_id: str) -> str:
        """Resume paused e2b sandbox. Returns state label."""
        from e2b_code_interpreter import Sandbox  # noqa: PLC0415

        with self._lock:
            entry = self._entries.get(thread_id)

        if not entry:
            # Not in memory — check if sandbox_id was persisted on disk
            sandbox_id = self._load_sandbox_id(thread_id)
            if not sandbox_id:
                return "cold"
            # Reconnect to paused snapshot from disk
            log.info("[%s] Reconnecting to paused e2b snapshot from disk (id: %s)", thread_id, sandbox_id)
            t = time.perf_counter()
            sbx = Sandbox.connect(sandbox_id)
            resume_ms = round((time.perf_counter() - t) * 1000, 2)
            log.info("[%s] e2b snapshot reconnected in %.0f ms (cold→hot via snapshot)", thread_id, resume_ms)
            with self._lock:
                self._entries[thread_id] = {
                    "sandbox":       sbx,
                    "sandbox_id":    sandbox_id,
                    "workspace_dir": WORKSPACES_DIR / thread_id,
                    "status":        "hot",
                }
            return f"cold→hot via snapshot ({resume_ms}ms)"

        with self._lock:
            entry = self._entries[thread_id]
            if entry["status"] == "warm":
                t = time.perf_counter()
                sbx = Sandbox.connect(entry["sandbox_id"])
                entry["sandbox"] = sbx
                entry["status"]  = "hot"
                if "workspace_dir" not in entry:
                    entry["workspace_dir"] = WORKSPACES_DIR / thread_id
                resume_ms = round((time.perf_counter() - t) * 1000, 2)
                log.info("[%s] e2b sandbox resumed in %.0f ms (warm→hot)", thread_id, resume_ms)
                return f"warm→hot ({resume_ms}ms)"

            return "hot"

    def suspend(self, thread_id: str) -> None:
        """Pause e2b sandbox — full kernel state saved to e2b cloud storage."""
        with self._lock:
            entry = self._entries.get(thread_id)
            if not entry or entry["status"] != "hot":
                return
            t = time.perf_counter()
            entry["sandbox"].beta_pause()
            entry["status"] = "warm"
            entry["sandbox"] = None  # sandbox object is gone server-side
            pause_ms = round((time.perf_counter() - t) * 1000, 2)
            log.info("[%s] e2b sandbox paused in %.0f ms (snapshot saved)", thread_id, pause_ms)

    def exec_code(self, thread_id: str, code: str) -> tuple[str, float]:
        with self._lock:
            entry = self._entries.get(thread_id)
            if not entry or not entry.get("sandbox"):
                raise RuntimeError(f"[{thread_id}] No active e2b sandbox for exec")
            sbx = entry["sandbox"]

        t = time.perf_counter()
        execution = sbx.run_code(code)
        exec_ms = round((time.perf_counter() - t) * 1000, 2)

        # stdout  → print() output
        # results → bare expressions evaluated by Jupyter kernel (e.g. df.head())
        stdout  = "\n".join(str(o) for o in execution.logs.stdout).strip()
        results = "\n".join(r.text for r in execution.results if r.text).strip()
        stderr  = "\n".join(str(o) for o in execution.logs.stderr).strip()

        parts = [p for p in [stdout, results] if p]
        output = "\n".join(parts) or "(executed, no output)"
        if stderr:
            output += f"\n[stderr]: {stderr}"
        if execution.error:
            output += f"\n[error]: {execution.error.name}: {execution.error.value}"

        return output, exec_ms

    def status(self, thread_id: str) -> str:
        entry = self._entries.get(thread_id)
        if not entry:
            return "cold"
        return entry["status"]

    def save_namespace(self, thread_id: str) -> None:
        """Serialize kernel globals → namespace.pkl in sandbox → download to workspace.
        Runs in background: parallel with next OpenAI call, or as a post-response task.
        """
        with self._lock:
            entry = self._entries.get(thread_id)
            if not entry or not entry.get("sandbox"):
                return
            sbx = entry["sandbox"]
            workspace_dir = entry["workspace_dir"]
        self._save_namespace(thread_id, sbx, workspace_dir)

    # ── internals ─────────────────────────────────────────────────────────────

    def _sandbox_id_path(self, thread_id: str) -> Path:
        return WORKSPACES_DIR / thread_id / ".e2b_sandbox_id"

    def _save_sandbox_id(self, thread_id: str, workspace_dir: Path, sandbox_id: str) -> None:
        try:
            (workspace_dir / ".e2b_sandbox_id").write_text(sandbox_id)
            log.info("[%s] Sandbox id persisted to disk (%s)", thread_id, sandbox_id)
        except Exception as e:
            log.warning("[%s] Could not persist sandbox_id: %s", thread_id, e)

    def _load_sandbox_id(self, thread_id: str) -> str | None:
        path = self._sandbox_id_path(thread_id)
        if not path.exists():
            return None
        try:
            return path.read_text().strip()
        except Exception:
            return None

    def _save_namespace(self, thread_id: str, sbx, workspace_dir: Path) -> None:
        """Dump kernel globals to namespace.pkl inside sandbox, then download to workspace."""
        save_code = (
            "import pickle as _p, types as _t\n"
            "_skip = {'In','Out','get_ipython','exit','quit','open','_p','_t','_skip','_ns','_failed','_k','_v'}\n"
            "_ns = {}\n"
            "_failed = []\n"
            "for _k, _v in list(globals().items()):\n"
            "    if _k.startswith('_') or _k in _skip or isinstance(_v, _t.ModuleType):\n"
            "        continue\n"
            "    try:\n"
            "        _p.dumps(_v)\n"
            "        _ns[_k] = _v\n"
            "    except Exception:\n"
            "        _failed.append(_k)\n"
            "try:\n"
            "    _p.dump(_ns, open('namespace.pkl', 'wb'))\n"
            "except Exception as _e:\n"
            "    print(f'[namespace save error: {_e}]')\n"
        )
        try:
            sbx.run_code(save_code)
            pkl_bytes = sbx.files.read("namespace.pkl", format="bytes")
            (workspace_dir / "namespace.pkl").write_bytes(pkl_bytes)
        except Exception as e:
            log.warning("[%s] Could not save namespace pickle: %s", thread_id, e)

    def on_startup(self) -> None:
        pass  # e2b manages sandbox lifecycle in its cloud; nothing to clean locally

    def cleanup_all(self) -> None:
        with self._lock:
            entries = dict(self._entries)
        for thread_id, entry in entries.items():
            sbx = entry.get("sandbox")
            if sbx:
                try:
                    sbx.kill()
                    log.info("[%s] e2b sandbox killed on shutdown", thread_id)
                except Exception as e:
                    log.warning("[%s] Could not kill e2b sandbox: %s", thread_id, e)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_runtime() -> RuntimeBackend:
    """Select backend from SANDBOX_RUNTIME env var (default: docker)."""
    backend = os.getenv("SANDBOX_RUNTIME", "docker").lower()
    if backend == "e2b":
        api_key = os.getenv("E2B_API_KEY")
        if not api_key:
            raise RuntimeError("E2B_API_KEY env var is required when SANDBOX_RUNTIME=e2b")
        log.info("Using e2b cloud sandbox backend")
        return E2BRuntime(api_key=api_key)
    log.info("Using Docker sandbox backend")
    return DockerRuntime()
