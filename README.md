# Context Guard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Model Context Protocol](https://img.shields.io/badge/MCP-2.0+-green.svg)](https://modelcontextprotocol.io)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Dual-layer Context Health Guardrail and Adaptive State-Ledger Compression Engine for LLM Agent Harnesses.**

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
- A **FastAPI OpenAI-compatible reverse proxy** (`/v1/chat/completions`) with live streaming support.
- A **Model Context Protocol (MCP) server** for Cursor, Claude Code, and AGY.
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
               (e.g., OpenAI, Anthropic, vLLM, Ollama)
                                    │
                                    ▼
               Telemetry Headers Injected on Response:
               - X-Context-Health-Status: 🟢 Healthy | 🟡 Degraded | 🔴 Critical
               - X-Context-Penalty-Score: 0 - 100
               - X-Context-Tokens-Saved: N tokens
```

---

## 3. Quickstart Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/context-guard.git
cd context-guard

# Install dependencies using uv
uv sync --all-extras
```

---

### Option A: Running as an MCP Server

Context Guard natively implements the **Model Context Protocol (MCP)**, exposing context inspection and ledger generation tools directly to IDEs like **Cursor**, **Claude Code**, and **Gemini CLI**.

#### Tools Provided:
- `inspect_context_health`: Evaluates conversation messages for failure modes and returns a standardized `HealthReport`.
- `compress_context_buffer`: Automatically compresses bloated older context into a State Ledger.
- `generate_state_ledger`: Extracts goals, hard constraints, active variables, and pending tasks.
- `context_audit`: Prompt template instructing an LLM to self-audit its context health.

#### Cursor Configuration (`~/.cursor/mcp.json` or project `.cursor/mcp.json`):
```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/context-guard", "context-guard-mcp"]
    }
  }
}
```

#### Claude Desktop Configuration (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "context-guard": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/context-guard", "context-guard-mcp"]
    }
  }
}
```

---

### Option B: Running as a Drop-in Reverse Proxy

Run Context Guard in front of OpenRouter, OpenAI, vLLM, Ollama, or any OpenAI-compatible API. By default, it routes to **OpenRouter** (`https://openrouter.ai/api/v1`), allowing you to use OpenRouter keys (`sk-or-v1-...`) and models like `openai/gpt-5-mini`.

Point your agent harness or client's `base_url` to `http://localhost:8080/v1`.

#### Running via CLI:
```bash
# Configure OpenRouter (or override with any OpenAI-compatible upstream)
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_API_KEY="sk-or-v1-..."

# Launch the proxy (listening on 0.0.0.0:8080)
uv run context-guard
```

#### Running via Docker / Docker Compose:
```bash
# Launch container
docker compose up -d

# Verify health check
curl http://localhost:8080/health
# {"status":"ok","service":"context-guard"}
```

#### Client Configuration Example:
```python
from openai import OpenAI

# Point client to Context Guard reverse proxy
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-or-v1-...",  # forwarded intact to OpenRouter
)

response = client.chat.completions.create(
    model="openai/gpt-5-mini",
    messages=[
        {"role": "user", "content": "Build an API using FastAPI."},
        {"role": "assistant", "content": "Here is the implementation..."},
        {"role": "user", "content": "No, do not use Pydantic v1, use v2."},
    ],
)
```

---

### Option C: Python Library SDK Integration

You can integrate Context Guard directly into custom agent loops or LangGraph orchestrators:

```python
import asyncio
from context_guard.evaluators import DeterministicEvaluator
from context_guard.compressors import ContextCompressor


async def main():
    messages = [
        {"role": "user", "content": "Goal: Build high-throughput event pipeline with Redis."},
        {"role": "assistant", "content": "Setting up consumer group... " * 15},
        {"role": "user", "content": "Never use blocking keys."},
        {"role": "assistant", "content": "Updated code without blocking keys... " * 15},
        {"role": "user", "content": "What is our current memory consumption?"},
    ]

    # 1. Evaluate context health
    evaluator = DeterministicEvaluator()
    report = evaluator.evaluate(messages, pinned_constraints=["Never use blocking keys"])
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

## 4. Telemetry Header Reference

Every response forwarded through the reverse proxy includes real-time telemetry headers:

| Header | Example Value | Description |
| :--- | :--- | :--- |
| `X-Context-Health-Status` | `🟢 Healthy`<br>`🟡 Degraded`<br>`🔴 Critical` | Categorical health status of the conversation history prior to upstream forwarding. |
| `X-Context-Penalty-Score` | `0` to `100` | Cumulative penalty score calculated across all detected failure modes. |
| `X-Context-Tokens-Saved` | `384` | Estimated token count saved by noise stripping and State Ledger compression ($0$ if uncompressed). |

---

## 5. Development & Testing

```bash
# Run full test suite (evaluators, compressors, proxy, mcp)
uv run pytest

# Check formatting and style
uv run ruff check .
uv run ruff format --check .

# Auto-fix formatting
uv run ruff format .
```

---

## 6. License

Context Guard is open-source software licensed under the [Apache-2.0 License](LICENSE).
