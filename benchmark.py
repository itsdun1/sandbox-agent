import asyncio
import time
import os
import io
import zipfile
import shutil
import tempfile
import subprocess
from e2b_code_interpreter import Sandbox

# ── CONFIG ─────────────────────────────────────────────────────────────────
E2B_API_KEY = "e2b_b358c359df5bb956cb98b3be3a2f2773ab339c81"   # <-- replace with your key
os.environ["E2B_API_KEY"] = E2B_API_KEY
CSV_FILE    = "transactions.csv"

# 5 sequential agent tasks simulating a real accounting thread
TASKS = [
    "import pandas as pd\ndf = pd.read_csv('transactions.csv')\nprint(f'Total transactions: {len(df)}')\nprint(f'Total amount: ${df[\"amount\"].sum():,.2f}')",
    "import pandas as pd\ndf = pd.read_csv('transactions.csv')\npending = df[df['status'] == 'pending']\nprint(f'Pending transactions: {len(pending)}')\nprint(pending[['id','date','amount']].head())",
    "import pandas as pd\ndf = pd.read_csv('transactions.csv')\ngrouped = df.groupby('category')['amount'].sum().sort_values(ascending=False)\nprint('Totals by category:')\nprint(grouped.to_string())",
    "import pandas as pd\ndf = pd.read_csv('transactions.csv')\ntop5 = df.nlargest(5, 'amount')[['id','date','description','amount']]\nprint('Top 5 transactions:')\nprint(top5.to_string())",
    "import pandas as pd\ndf = pd.read_csv('transactions.csv')\ndf['suspicious'] = df['amount'] > 10000\nflagged = df[df['suspicious']][['id','date','amount','category']]\nprint(f'Suspicious transactions (>$10k): {len(flagged)}')\nprint(flagged.head().to_string())"
]

# ── HELPERS ─────────────────────────────────────────────────────────────────

def zip_workspace(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder):
            for file in files:
                path = os.path.join(root, file)
                zf.write(path, os.path.relpath(path, folder))
    return buf.getvalue()

def unzip_workspace(data, folder):
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(folder)

# ── APPROACH 1: S3 Simulation (save/restore workspace on every message) ────

class FakeS3:
    """Simulates S3 using local disk. Adds realistic delay."""
    store = {}

    @classmethod
    def upload(cls, key, data):
        time.sleep(1.5)   # realistic S3 upload latency for workspace
        cls.store[key] = data

    @classmethod
    def download(cls, key):
        time.sleep(2.0)   # realistic S3 download latency for workspace
        return cls.store.get(key)

async def run_s3_approach(thread_id: str):
    """Current approach: sync workspace from S3 before every message."""
    latencies = []
    workspace_key = f"workspaces/{thread_id}"

    # Initial workspace setup
    tmp = tempfile.mkdtemp()
    shutil.copy(CSV_FILE, os.path.join(tmp, CSV_FILE))
    FakeS3.upload(workspace_key, zip_workspace(tmp))
    shutil.rmtree(tmp)

    for i, task in enumerate(TASKS):
        t0 = time.time()

        # Restore workspace from S3
        workspace = tempfile.mkdtemp()
        data = FakeS3.download(workspace_key)
        unzip_workspace(data, workspace)

        # Run code locally (simulates agent execution)
        result = subprocess.run(
            ["python", "-c", task],
            capture_output=True, text=True, cwd=workspace
        )

        # Agent writes result to workspace (this is what grows the workspace)
        with open(os.path.join(workspace, f"result_{i+1}.txt"), "w") as f:
            f.write(result.stdout)

        print(f"    Output: {result.stdout[:80].strip()}...")

        # Save workspace back to S3
        FakeS3.upload(workspace_key, zip_workspace(workspace))
        shutil.rmtree(workspace)

        latency = time.time() - t0
        latencies.append(latency)
        print(f"  [S3]  Thread {thread_id} | Message {i+1} | {latency:.2f}s")

    return latencies

# ── APPROACH 2: e2b Pause/Resume ───────────────────────────────────────────

async def run_e2b_approach(thread_id: str):
    """Proposed approach: sandbox stays paused between messages."""
    latencies = []
    sbx = None

    for i, task in enumerate(TASKS):
        t0 = time.time()

        if sbx is None:
            # First message: create sandbox and upload CSV
            sbx = Sandbox.create()
            with open(CSV_FILE, "rb") as f:
                sbx.files.write(CSV_FILE, f.read())
        else:
            # Subsequent messages: resume paused sandbox (connect auto-resumes)
            sbx = sbx.connect()

        # Run the task and capture output
        execution = sbx.run_code(task)
        output = "\n".join([str(o) for o in execution.logs.stdout])
        print(f"    Output: {output[:80].strip()}...")

        # Agent writes result to workspace file (same as S3 approach)
        sbx.files.write(f"result_{i+1}.txt", output)

        # Pause sandbox after each message
        sbx.beta_pause()

        latency = time.time() - t0
        latencies.append(latency)
        print(f"  [e2b] Thread {thread_id} | Message {i+1} | {latency:.2f}s")

    # Cleanup
    if sbx:
        try:
            sbx.kill()
        except:
            pass

    return latencies

# ── MAIN ───────────────────────────────────────────────────────────────────

async def main():
    NUM_THREADS = 3  # increase to stress test

    print("=" * 60)
    print(f"Benchmarking {NUM_THREADS} concurrent threads, {len(TASKS)} messages each")
    print("=" * 60)

    # ── S3 Approach ──
    print("\n📦 Running S3 Sync Approach...")
    t0 = time.time()
    s3_results = await asyncio.gather(*[
        run_s3_approach(f"thread_{i}") for i in range(NUM_THREADS)
    ])
    s3_total = time.time() - t0
    s3_flat = [l for r in s3_results for l in r]

    # ── e2b Approach ──
    print("\n⚡ Running e2b Pause/Resume Approach...")
    t1 = time.time()
    e2b_results = await asyncio.gather(*[
        run_e2b_approach(f"thread_{i}") for i in range(NUM_THREADS)
    ])
    e2b_total = time.time() - t1
    e2b_flat = [l for r in e2b_results for l in r]

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<35} {'S3 Sync':>10} {'e2b Pause/Resume':>18}")
    print("-" * 60)
    print(f"{'Avg latency per message':<35} {sum(s3_flat)/len(s3_flat):>9.2f}s {sum(e2b_flat)/len(e2b_flat):>17.2f}s")
    print(f"{'Total time (' + str(NUM_THREADS) + ' threads)':<35} {s3_total:>9.2f}s {e2b_total:>17.2f}s")
    print(f"{'Speedup':<35} {'':>10} {s3_total/e2b_total:>16.1f}x")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())