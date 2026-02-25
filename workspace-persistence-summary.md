# Workspace Persistence for AI Agents — Full Summary

## The Company: Aqqrue
- Accounting/finance AI agent platform
- Backend: Python (API layer)
- Agent runtime: Go (code execution with AI — faster startup than Python)
- Autonomous agent like Claude Code / Codex — not just a chatbot with RAG

## The Problem
The agent needs a **workspace** per task (files, code it writes, results, memory files, user uploads).
Currently syncing workspace **to/from S3 on every message** in a thread — this is the bottleneck.

### Workspace Details
- Size: **2-10MB** per workspace
- Contains: memory files, user-uploaded files, agent-generated code and docs
- **One workspace per task** (for isolation)
- **Each task can have multiple runs** (each message = new run)
- All runs in a thread share the same workspace
- **No parallel sub-runs** — only one run at a time per thread
- Runs are sequential within a thread

## The Solution Architecture

### Core Idea
Stop syncing S3 on every message. Keep workspace alive between messages. Only hit S3 at session boundaries.

### Tiered Approach (handles 1000 concurrent users)
```
Tier 1: HOT (actively chatting)
  → Container/VM running, full memory
  → ~50-100 concurrent at any time
  → Instant response

Tier 2: WARM (chatted in last 15 min)
  → Paused in memory (docker pause)
  → Still holding memory but no CPU
  → ~200-300 at any time
  → Resume in ~10ms

Tier 3: COLD (idle > 15 min)
  → Workspace FILES saved to shared storage (EFS) or S3
  → Container destroyed, memory freed
  → Could be thousands
  → Restore: ~500ms-1s (new container + mount files)
```

### Why Not Keep All 1000 Paused?
Docker paused container still uses ~50-100MB RAM each.
1000 paused = 50-100GB RAM wasted. Not feasible.

## System Architecture

```
Users
  ↓
Load Balancer (external)
  ↓
Python API Containers (stateless, any pod)
  ↓
Redis Lookup: "thread-abc → go-agent-2"
  ↓
Go Agent Pods (stateful, hold paused workspaces)
  ├── Pod-1: [vm/container paused] [vm/container paused]
  ├── Pod-2: [vm/container paused]
  └── Pod-3: [idle, ready for new threads]
```

### Smart Routing (thread affinity)
- Python API checks Redis for thread→pod mapping
- If exists: route directly to that pod (skip LB)
- If not: pick least loaded pod, save mapping in Redis
- Use **Kubernetes StatefulSet + Headless Service** for stable pod names

### Python API (Router)
```python
async def handle_message(thread_id: str, message: str):
    target_pod = await redis.get(f"thread:{thread_id}:pod")

    if target_pod:
        try:
            response = await httpx.post(f"http://{target_pod}:8080/run", json={...})
            return response.json()
        except ConnectionError:
            # pod died — fall back to restore
            await redis.delete(f"thread:{thread_id}:pod")

    # new thread or fallback — assign new pod
    pod = await pick_least_loaded_pod()
    await redis.set(f"thread:{thread_id}:pod", pod)
    response = await httpx.post(f"http://{pod}:8080/run", json={...})
    return response.json()
```

### Go Agent (Core Loop)
```go
func (a *Agent) Run(task string, workspace *Workspace) (string, error) {
    messages := []Message{
        {Role: "system", Content: "You are an agent..."},
        {Role: "user", Content: task},
    }
    for {
        response := a.llmClient.Chat(messages)
        if response.HasCodeBlock() {
            code := response.ExtractCode()
            result, err := workspace.Execute(code)
            messages = append(messages, Message{Role: "assistant", Content: response.Text})
            messages = append(messages, Message{Role: "user", Content: fmt.Sprintf("Result:\n%s", result)})
        } else {
            return response.Text, nil
        }
    }
}
```

### Workspace Manager (Go)
```go
type WorkspaceManager struct {
    activeContainers map[string]*Container  // threadID → running/paused container
    timers           map[string]*time.Timer
    mu               sync.RWMutex
}

func (wm *WorkspaceManager) GetWorkspace(threadID string) (*Container, error) {
    wm.mu.RLock()
    container, exists := wm.activeContainers[threadID]
    wm.mu.RUnlock()

    if exists {
        // WARM: container paused on this pod — just unpause
        container.Unpause()  // ~10ms, no S3
        return container, nil
    }

    // COLD: need to restore — create new container, load files
    container = wm.CreateNewContainer(threadID)
    wm.RestoreFilesFromStorage(threadID, container)  // EFS or S3
    wm.activeContainers[threadID] = container
    return container, nil
}

func (wm *WorkspaceManager) AfterRun(threadID string) {
    // DON'T destroy — just pause
    wm.activeContainers[threadID].Pause()  // ~10ms
    wm.ResetIdleTimer(threadID, 15*time.Minute)
}

func (wm *WorkspaceManager) Evict(threadID string) {
    container := wm.activeContainers[threadID]
    // save files to shared storage
    container.CopyFilesTo("/shared-storage/workspaces/" + threadID)
    container.Destroy()  // free memory
    delete(wm.activeContainers, threadID)
    redis.Del("thread:" + threadID + ":pod")
}
```

## Storage Options Comparison

| Option | Latency | Cross-Pod | Cost | Notes |
|--------|---------|-----------|------|-------|
| S3 every message | 500ms-5s | ✅ | Low | Current approach (the bottleneck) |
| EFS (shared filesystem) | 10-50ms per file read | ✅ any pod | Medium | Simplest architecture, no routing needed |
| Docker pause/unpause | ~10ms | ❌ same pod only | RAM cost | Need smart routing |
| Firecracker snapshot (local) | ~4-10ms | ❌ same host | Low | Best performance, complex setup |
| Docker CRIU checkpoint | ~200-500ms | ✅ any host | Low | Experimental, flaky, NOT recommended |

## For the Demo (Docker-based)

### Stack
- **Python FastAPI** — API layer + smart router
- **Go service** — agent runtime (multiple pods)
- **Redis** — thread→pod mapping + idle timers
- **Shared volume** (simulating EFS) — workspace files
- **Docker** — containers as sandboxes
- **Docker Compose** — local orchestration

### Docker Compose Structure
```yaml
services:
  api:
    build: ./api          # Python FastAPI
    ports: ["8000:8000"]
    depends_on: [redis]

  agent-1:
    build: ./agent        # Go agent
    hostname: agent-1
    volumes:
      - workspaces:/workspaces    # shared storage (simulates EFS)
      - /var/run/docker.sock:/var/run/docker.sock  # to manage sandbox containers

  agent-2:
    build: ./agent
    hostname: agent-2
    volumes:
      - workspaces:/workspaces
      - /var/run/docker.sock:/var/run/docker.sock

  agent-3:
    build: ./agent
    hostname: agent-3
    volumes:
      - workspaces:/workspaces
      - /var/run/docker.sock:/var/run/docker.sock

  redis:
    image: redis:alpine
    ports: ["6379:6379"]

volumes:
  workspaces:             # shared volume simulating EFS
```

### Kubernetes (Production)
```yaml
# Headless service for direct pod addressing
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  clusterIP: None            # headless
  selector:
    app: go-agent

---
# StatefulSet for stable pod names
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: go-agent
spec:
  serviceName: "agent-service"
  replicas: 5
  template:
    spec:
      containers:
        - name: agent
          image: go-agent:latest
          volumeMounts:
            - name: workspaces
              mountPath: /workspaces
      volumes:
        - name: workspaces
          persistentVolumeClaim:
            claimName: workspace-efs  # EFS shared across all pods
```

## Production Upgrade Path
For production, swap Docker containers for **Firecracker microVMs** via **Kata Containers**:
- 5MB per VM vs 50MB per container
- Built-in snapshot with lazy page loading (~4ms restore)
- Same Kubernetes integration via Kata runtime class
- This is what e2b, Modal, Fly.io, AWS Lambda use
- Requires bare metal nodes (for KVM/nested virtualization)

## Key Numbers to Remember
| Metric | Value |
|--------|-------|
| Firecracker boot | <125ms |
| Firecracker memory overhead | <5MB per VM |
| Firecracker snapshot restore (local) | ~4-10ms |
| Docker pause/unpause | ~10ms |
| Docker CRIU restore | ~200-500ms (unreliable) |
| S3 sync (current) | 500ms-5s |
| EFS file read | 10-50ms |
| E2B sandbox resume | ~1 second |
| Workspace size | 2-10MB |

## What to Say on Wednesday
- "Built with Docker to demo the architecture quickly"
- "Docker pause/unpause for active threads, shared storage for files, Redis for routing"
- "In production: Firecracker microVMs via Kata Containers — 10x lighter, built-in snapshots, 4ms restore"
- "Tiered eviction: hot → warm → cold to handle 1000 concurrent without wasting RAM"

## Ashfakh's Hint
> "Think about how people run massive continuous sandboxed workloads in prod"
→ Answer: Firecracker microVMs with cluster-aware orchestration (Kubernetes/Kata)
→ Don't build custom cluster systems — use existing ones to avoid split-brain, consensus, network partition issues
