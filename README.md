# Context Guard

[![PyPI version](https://img.shields.io/pypi/v/context-guards.svg?color=blue)](https://pypi.org/project/context-guards/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Model Context Protocol](https://img.shields.io/badge/MCP-2.0+-green.svg)](https://modelcontextprotocol.io)
[![Docker](https://img.shields.io/badge/GHCR-Multi--Arch-blue?logo=docker)](https://github.com/Sandeep-Hipparagi/context-guard/pkgs/container/context-guard)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Dual-layer Context Health Guardrail and Adaptive State-Ledger Compression Engine for LLM Agent Harnesses, IDEs, and Terminals.**

---

## 1. Executive Summary

Autonomous LLM agents and multi-turn coding assistants frequently degrade over extended sessions due to four critical context failure modes:

1. **Context Poisoning & Error Propagation**: The agent hallucinates or makes a mistake, the user issues a correction (e.g., *"No, don't use Flask, use FastAPI"*), but subsequent turns keep referencing the poisoned context and repeating the rejected pattern.
2. **Distraction & Echo Loops**: Consecutive turns suffer from repetitive phrasing, circular logic, or verbatim code generation loops that consume context budget without advancing the task.
3. **Directive Clash & Contradiction**: System directives or pinned user constraints are gradually forgotten or violated as context windows fill up.
4. **Noise Ballooning & Context Rot**: High cumulative context token payloads are carried across low-information single-word queries (e.g., *"ok"*, *"continue"*, *"next"*), diluting model attention.

**Context Guard** solves this through a dual-layer architecture:
- **Layer 1 (Deterministic Heuristic Evaluator)**: A zero-latency, rule-based health evaluator that scores context degradation between 0 and 100 (`🟢 Healthy`, `🟡 Degraded`, `🔴 Critical`).
- **Layer 2 (Adaptive State-Ledger Compressor)**: Strips ANSI noise, deep repetitive stack traces, and pleasantries, consolidating older conversation turns into a structured **Active State Ledger** while keeping recent raw turns intact.

Context Guard can be deployed as:
- A **Model Context Protocol (MCP) server** for Cursor, Claude Desktop, Claude Code CLI, Windsurf, Zed, and VS Code.
- A **FastAPI OpenAI-compatible reverse proxy** (`/v1/chat/completions`) with model discovery (`/v1/models`), CORS, and live SSE streaming.
- An **Interactive Visual Dashboard** (`GET /` and `GET /dashboard`) for ambient real-time health monitoring.
- A **Python SDK library** embedded directly into LangGraph, AutoGen, or custom agent loops.

---

## 2. Architecture Flow

```
                      Client Request (OpenAI Payload)
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │       Context Guard Interceptor       │
                └───────────────────┬───────────────────┘
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │   Layer 1: Deterministic Evaluator    │
                │  - Repetition / Echo Loops (+35)      │
                │  - Error Propagation / Poisoning (+45)│
                │  - Directive Clashes (+40)            │
                │  - Noise Ballooning (+25)             │
                └───────────────────┬───────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [Score < 25 (GREEN)]                [Score >= 25 (YELLOW/RED)]
                  │                                   │
                  │ Passes through                    ▼
                  │ unmodified          ┌───────────────────────────┐
                  │                     │ Layer 2: Adaptive         │
                  │                     │ State-Ledger Compressor   │
                  │                     │ - Noise & Trace Stripping │
                  │                     │ - State Ledger Markdown   │
                  │                     │ - Preserves N Raw Turns   │
                  │                     │ - Intervention Directive  │
                  │                     └─────────────┬─────────────┘
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                     Forwarded Upstream Request
               (Groq, OpenRouter, OpenAI, Ollama, vLLM)
                                    │
                                    ▼
               Telemetry Headers Injected on Response:
               - X-Context-Health-Status: 🟢 Healthy | 🟡 Degraded | 🔴 Critical
               - X-Context-Penalty-Score: 0 - 100
               - X-Context-Tokens-Saved: N tokens
```

---

## 3. Quickstart & Installation

### Option 1: Zero-Install via `uvx` (Fastest)

Run the MCP server or proxy directly without installing anything permanently:

```bash
# Run MCP server directly
uvx context-guards-mcp

# Or run the reverse proxy directly
uvx --from context-guards context-guard
```

### Option 2: Install via pip or uv

```bash
# Install from PyPI
pip install context-guards

# Or with uv
uv add context-guards
```

### Option 3: Run via Docker (Multi-Arch)

```bash
# Run the pre-built multi-arch image from GHCR
docker run -d -p 8080:8080 \
  -e OPENAI_BASE_URL="https://api.groq.com/openai/v1" \
  -e GROQ_API_KEY="gsk_..." \
  ghcr.io/sandeep-hipparagi/context-guard:v0.1.0
```

---

## 4. IDE & Editor Integration (MCP Server)

Context Guard natively implements the **Model Context Protocol (MCP)**, exposing context inspection and ledger generation tools directly to modern AI-enabled IDEs and terminals.

### Provided MCP Tools & Prompts:
- `inspect_context_health`: Evaluates conversation messages for failure modes and returns a standardized `HealthReport`.
- `compress_context_buffer`: Automatically compresses bloated older context into a State Ledger.
- `generate_state_ledger`: Extracts goals, hard constraints, active variables, and pending tasks.
- `context_audit`: Prompt template instructing an LLM to self-audit its context health.

---

### Configuration by IDE:

#### 1. Cursor (`.cursor/mcp.json` or `~/.cursor/mcp.json`)
```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uvx",
      "args": ["context-guards-mcp"]
    }
  }
}
```

#### 2. Claude Desktop
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uvx",
      "args": ["context-guards-mcp"]
    }
  }
}
```

#### 3. Claude Code CLI (Terminal)
Add Context Guard to your Claude Code CLI in one command:
```bash
claude mcp add context-guard -- uvx context-guards-mcp
```

#### 4. Windsurf (Codeium) (`~/.codeium/windsurf/mcp_config.json`)
```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uvx",
      "args": ["context-guards-mcp"]
    }
  }
}
```

#### 5. Zed Editor (`~/.config/zed/settings.json`)
```json
{
  "experimental.context_servers": {
    "context-guard": {
      "command": "uvx",
      "args": ["context-guards-mcp"]
    }
  }
}
```

#### 6. VS Code (Cline / Roo Code / Continue)
In Cline / Roo Code `mcpSettings.json`:
```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uvx",
      "args": ["context-guards-mcp"]
    }
  }
}
```

---

## 5. Running as a Drop-in Reverse Proxy

Run Context Guard in front of **Groq**, **OpenRouter**, **OpenAI**, **Ollama**, **vLLM**, or any OpenAI-compatible API.

By default, it routes to **Groq** (`https://api.groq.com/openai/v1`), allowing high-throughput Groq API keys (`gsk_...`) and models like `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`.

### Starting the Proxy:
```bash
# Set your target upstream base URL and key
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export GROQ_API_KEY="gsk_..."

# Start proxy (listening on http://0.0.0.0:8080)
context-guard
```

### Supported Upstream Targets:
| Provider | `OPENAI_BASE_URL` | Auth Header / Key |
| :--- | :--- | :--- |
| **Groq (Default)** | `https://api.groq.com/openai/v1` | `GROQ_API_KEY="gsk_..."` |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY="sk-or-v1-..."` |
| **OpenAI** | `https://api.openai.com/v1` | `OPENAI_API_KEY="sk-..."` |
| **Ollama (Local)** | `http://localhost:11434/v1` | Optional |
| **vLLM (Local)** | `http://localhost:8000/v1` | Optional |
| **LM Studio (Local)** | `http://localhost:1234/v1` | Optional |

### Client Configuration (OpenAI Python SDK):
```python
from openai import OpenAI

# Point client directly to Context Guard reverse proxy
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="gsk_...",  # forwarded intact to upstream
)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Build an API using FastAPI."},
        {"role": "assistant", "content": "Here is the implementation..."},
        {"role": "user", "content": "No, do not use Pydantic v1, use v2."},
    ],
)
```

---

## 6. Interactive Visual Diagnostic Dashboard

Context Guard embeds a dependency-free, ambient visual diagnostic dashboard accessible at:
👉 **`http://localhost:8080/`** or **`http://localhost:8080/dashboard`**

### Dashboard Capabilities:
- **Live Radial Health Gauge**: Dynamic penalty score (0–100) and color-coded status (`🟢 Healthy`, `🟡 Degraded`, `🔴 Critical`).
- **Failure Mode Badges**: Real-time breakdown of detected Poisoning, Distraction, Confusion, and Directive Clash issues.
- **State-Ledger Preview**: Visual cards displaying the Pinned Goal, Hard Constraints, Active State, and Token Reduction metrics.
- **Interactive Payload Tester**: Test sample payloads or paste live conversation JSON to visualize evaluation and compression immediately.

---

## 7. Python Library SDK Integration

Context Guard can be embedded directly into custom agent loops, LangGraph, AutoGen, or CrewAI:

```python
import asyncio
from context_guard.compressors import ContextCompressor
from context_guard.evaluators import DeterministicEvaluator


async def main():
    messages = [
        {
            "role": "user",
            "content": "Goal: Build high-throughput event pipeline with Redis.",
        },
        {"role": "assistant", "content": "Setting up consumer group... " * 15},
        {"role": "user", "content": "Never use blocking keys."},
        {
            "role": "assistant",
            "content": "Updated code without blocking keys... " * 15,
        },
        {"role": "user", "content": "What is our current memory consumption?"},
    ]

    # 1. Evaluate context health
    evaluator = DeterministicEvaluator()
    report = evaluator.evaluate(
        messages, pinned_constraints=["Never use blocking keys"]
    )
    print(f"Status: {report.status.value} (Score: {report.penalty_score}/100)")
    print(f"Action: {report.recommended_action}")

    # 2. Compress context if degraded
    if report.penalty_score >= 25:
        compressor = ContextCompressor(preserve_recent_turns=3)
        compressed = await compressor.compress(messages)
        print(f"Reduced tokens by: {compressed.token_reduction_pct}%")
        print("\nGenerated State Ledger:")
        print(compressed.state_ledger_markdown)

        # 3. Format final payload for model inference
        next_payload = compressor.format_for_inference(compressed)


asyncio.run(main())
```

---

## 8. Telemetry Header Reference

Every response forwarded through the reverse proxy includes real-time telemetry headers (with full CORS support for browser clients):

| Header | Example Value | Description |
| :--- | :--- | :--- |
| `X-Context-Health-Status` | `🟢 Healthy`<br>`🟡 Degraded`<br>`🔴 Critical` | Categorical health status of conversation history prior to upstream forwarding. |
| `X-Context-Penalty-Score` | `0` to `100` | Cumulative penalty score calculated across all detected failure modes. |
| `X-Context-Tokens-Saved` | `384` | Estimated token count saved by noise stripping and State Ledger compression ($0$ if uncompressed). |

---

## 9. Interactive Terminal Simulation

A standalone simulation script is included to test multi-turn degradation and state-ledger compression locally without needing an API key:

```bash
python examples/simulate_session.py
```

---

## 10. Development & Testing

```bash
# Clone the repository
git clone https://github.com/Sandeep-Hipparagi/context-guard.git
cd context-guard

# Install dependencies using uv
uv sync --extra dev

# Run full test suite (evaluators, compressors, proxy, mcp)
uv run pytest

# Check formatting and style
uv run ruff check .
uv run ruff format --check .
```

---

## 11. License

Context Guard is open-source software licensed under the [Apache-2.0 License](LICENSE).
