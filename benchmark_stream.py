#!/usr/bin/env python3
"""
benchmark_stream.py — compare non-streaming vs streaming endpoint.

Usage:
  python benchmark_stream.py <thread_id> "<message>"

Measures:
  Non-streaming: total time until full JSON response
  Streaming:     TTFT (time to first token) + total time

The TTFT gap is the UX improvement streaming adds.
"""

import sys
import time
import json
import httpx

BASE = "http://localhost:8000"


def bench_non_stream(thread_id: str, message: str) -> dict:
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{BASE}/thread/{thread_id}/message",
        json={"message": message},
        timeout=120.0,
    )
    total_ms = round((time.perf_counter() - t0) * 1000)
    data = resp.json()
    return {
        "total_ms":   total_ms,
        "server_ms":  data.get("total_ms"),
        "timings":    data.get("timings_ms"),
        "response":   data.get("response", "")[:120],
    }


def bench_stream(thread_id: str, message: str) -> dict:
    t0 = time.perf_counter()
    ttft_ms = None
    total_ms = None
    server_timings = None
    tokens = []

    with httpx.stream(
        "POST",
        f"{BASE}/thread/{thread_id}/message/stream",
        json={"message": message},
        timeout=120.0,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])

            if event["type"] == "token":
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - t0) * 1000)
                tokens.append(event["text"])

            elif event["type"] == "done":
                total_ms = round((time.perf_counter() - t0) * 1000)
                server_timings = event.get("timings_ms")

            elif event["type"] == "status":
                elapsed = round((time.perf_counter() - t0) * 1000)
                print(f"  [{elapsed:>5}ms] status: {event['text']}")

            elif event["type"] == "code_run":
                elapsed = round((time.perf_counter() - t0) * 1000)
                print(f"  [{elapsed:>5}ms] code_run ({event['exec_ms']}ms exec)")

            elif event["type"] == "error":
                print(f"  ERROR: {event['message']}")

    return {
        "ttft_ms":    ttft_ms,
        "total_ms":   total_ms,
        "server_ms":  server_timings and server_timings.get("openai_total"),
        "timings":    server_timings,
        "response":   "".join(tokens)[:120],
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python benchmark_stream.py <thread_id> '<message>'")
        sys.exit(1)

    thread_id = sys.argv[1]
    message   = sys.argv[2]

    print(f"\n{'─'*60}")
    print(f"Thread:  {thread_id}")
    print(f"Message: {message}")
    print(f"{'─'*60}\n")

    # ── Non-streaming ──────────────────────────────────────────
    print("[ Non-streaming ]")
    ns = bench_non_stream(thread_id, message)
    print(f"  Total (client-side): {ns['total_ms']} ms")
    print(f"  Total (server-side): {ns['server_ms']} ms")
    print(f"  Response preview:    {ns['response']}…")
    if ns["timings"]:
        t = ns["timings"]
        print(f"  Breakdown:  restore={t.get('sandbox_restore')}ms  "
              f"openai={t.get('openai_total')}ms  "
              f"exec={t.get('exec_total')}ms  "
              f"overhead={t.get('overhead')}ms")

    print()

    # ── Streaming ──────────────────────────────────────────────
    print("[ Streaming ]")
    st = bench_stream(thread_id, message)
    print(f"  TTFT (first token):  {st['ttft_ms']} ms  ← UX improvement")
    print(f"  Total (client-side): {st['total_ms']} ms")
    print(f"  Response preview:    {st['response']}…")
    if st["timings"]:
        t = st["timings"]
        print(f"  Breakdown:  restore={t.get('sandbox_restore')}ms  "
              f"openai={t.get('openai_total')}ms  "
              f"exec={t.get('exec_total')}ms  "
              f"overhead={t.get('overhead')}ms")

    print()

    # ── Comparison ─────────────────────────────────────────────
    if st["ttft_ms"] and ns["total_ms"]:
        gap = ns["total_ms"] - st["ttft_ms"]
        pct = round(gap / ns["total_ms"] * 100)
        print(f"{'─'*60}")
        print(f"  TTFT vs non-stream total: {st['ttft_ms']} ms vs {ns['total_ms']} ms")
        print(f"  User sees first word {gap} ms earlier ({pct}% of wait time saved)")
        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
