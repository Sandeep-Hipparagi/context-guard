# Context Guard

Context Guard is a dual-layer Context Health Guardrail and Compression Engine for LLM agent harnesses.

It acts as an interceptor and state-ledger compressor, deployable as:
- A Python package / SDK library
- A FastAPI reverse-proxy middleware (`/v1/chat/completions`)
- An MCP (Model Context Protocol) Server exposing context inspection tools

## Project Structure

```
context_guard/
  ├── core/             # Health evaluator, scoring, state ledger models
  ├── evaluators/       # Deterministic heuristics & SLM/judge interface
  ├── compressors/      # Trimming & State Ledger summarizer
  ├── proxy/            # FastAPI reverse-proxy middleware (/v1/chat/completions)
  └── mcp/              # MCP Server entrypoint exposing context inspection tools
tests/
  ├── test_evaluators.py
  ├── test_compressors.py
  └── test_proxy.py
```

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Format and lint
uv run ruff check .
uv run ruff format .
```
