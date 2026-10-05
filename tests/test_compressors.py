"""Comprehensive test suite for Context Guard adaptive compressor engine."""

import pytest

from context_guard.compressors import (
    ContextCompressor,
    HeuristicLedgerExtractor,
    LLMLedgerExtractor,
)
from context_guard.core.models import StateLedger


def test_noise_stripping_ansi_stacktraces_pleasantries():
    """Verify strip_noise removes ANSI codes, collapses long traces, and trims pleasantries."""
    compressor = ContextCompressor()

    # 1. ANSI escape sequences
    ansi_text = "\x1b[31;1mError in worker node\x1b[0m: connection timed out."
    cleaned_ansi = compressor.strip_noise(ansi_text)
    assert "\x1b[" not in cleaned_ansi
    assert cleaned_ansi == "Error in worker node: connection timed out."

    # 2. Conversational pleasantries
    pleasantry_text = "Sure, I can help with that! Here is the implementation of the auth service."
    cleaned_pleasantry = compressor.strip_noise(pleasantry_text)
    assert "Sure, I can help with that!" not in cleaned_pleasantry
    assert cleaned_pleasantry == "Here is the implementation of the auth service."

    ai_model_text = "As an AI language model, I recommend using argon2 for password hashing."
    cleaned_ai = compressor.strip_noise(ai_model_text)
    assert "As an AI language model" not in cleaned_ai
    assert "I recommend using argon2 for password hashing." in cleaned_ai

    # 3. Stack trace collapsing (>10 lines)
    long_trace = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 10, in <module>\n'
        "    app.run()\n"
        '  File "core/app.py", line 45, in run\n'
        "    self.init_services()\n"
        '  File "core/app.py", line 78, in init_services\n'
        "    db.connect()\n"
        '  File "db/driver.py", line 22, in connect\n'
        "    socket.create_connection()\n"
        '  File "net/socket.py", line 88, in create_connection\n'
        "    handshake()\n"
        '  File "net/ssl.py", line 120, in handshake\n'
        "    verify_cert()\n"
        "ConnectionRefusedError: Failed to connect to server on port 5432"
    )

    cleaned_trace = compressor.strip_noise(long_trace)
    assert "Traceback (most recent call last):" in cleaned_trace
    assert "frames collapsed" in cleaned_trace
    assert "ConnectionRefusedError: Failed to connect to server on port 5432" in cleaned_trace


@pytest.mark.asyncio
async def test_heuristic_extractor_direct():
    """Verify HeuristicLedgerExtractor directly parses goals, constraints, and active state."""
    extractor = HeuristicLedgerExtractor()
    history = [
        {"role": "user", "content": "Goal: Setup PostgreSQL database with Docker."},
        {"role": "assistant", "content": "Configuring docker-compose.yml on port=5432."},
        {"role": "user", "content": "Never use default passwords and avoid root user."},
    ]

    ledger = await extractor.extract_ledger(history)
    assert "Setup PostgreSQL" in ledger.pinned_goal
    assert any("never use default" in c.lower() for c in ledger.hard_constraints)
    assert "PostgreSQL" in ledger.active_state.get("technologies", [])
    assert "docker-compose.yml" in ledger.active_state.get("files", [])


@pytest.mark.asyncio
async def test_short_conversation_untouched():
    """Verify that conversation turns <= preserve_recent_turns are untouched (reduction = 0%)."""
    compressor = ContextCompressor(preserve_recent_turns=3)

    messages = [
        {"role": "system", "content": "You are a senior engineer."},
        {"role": "user", "content": "How do I configure pytest?"},
        {"role": "assistant", "content": "Add a pyproject.toml with tool.pytest.ini_options."},
    ]

    compressed = await compressor.compress(messages)

    assert compressed.compression_ratio == 1.0
    assert compressed.token_reduction_pct == 0.0
    assert len(compressed.recent_raw_turns) == 2
    assert compressed.state_ledger_markdown == ""

    # Verify format_for_inference preserves messages exactly
    formatted = compressor.format_for_inference(compressed)
    assert len(formatted) == 3
    assert formatted[0]["role"] == "system"
    assert formatted[1]["role"] == "user"
    assert formatted[2]["role"] == "assistant"


@pytest.mark.asyncio
async def test_long_conversation_compression_and_reduction():
    """Verify older turns are condensed into State Ledger while recent turns remain verbatim."""
    compressor = ContextCompressor(preserve_recent_turns=3)

    # Build a long multi-turn dialogue with bloated older context
    messages = [
        {"role": "system", "content": "You are an assistant helping build a FastAPI backend."},
        {
            "role": "user",
            "content": (
                "Pinned Goal: Build a distributed caching microservice.\n"
                "We must have strict schema validation and never use eval.\n"
                "The configuration uses port=8000 and database=postgresql."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Sure, I can help with that! Architecture setup with Redis and Docker:\n"
                + "import os\nimport sys\n# Detailed architecture setup...\n" * 20
            ),
        },
        {
            "role": "user",
            "content": "Can you also make sure we avoid global state in main.py?",
        },
        {
            "role": "assistant",
            "content": (
                "Certainly! Here is how to encapsulate dependencies without global state:\n"
                + "class Container:\n    pass\n" * 20
            ),
        },
        {"role": "user", "content": "What is the status of the Redis cluster connection pool?"},
        {
            "role": "assistant",
            "content": "The Redis connection pool is configured with max_connections=20.",
        },
        {"role": "user", "content": "Could you add health check endpoints?"},
    ]

    compressed = await compressor.compress(messages)

    # 1. Verification of token reduction (> 30%)
    assert compressed.token_reduction_pct > 30.0
    assert compressed.compressed_tokens < compressed.original_tokens

    # 2. State Ledger presence and content
    assert "[ACTIVE CONTEXT STATE LEDGER]" in compressed.state_ledger_markdown
    assert compressed.state_ledger.pinned_goal != ""
    assert len(compressed.recent_raw_turns) == 3

    # 3. Recent 3 turns must remain verbatim
    assert (
        compressed.recent_raw_turns[0]["content"]
        == "What is the status of the Redis cluster connection pool?"
    )
    assert compressed.recent_raw_turns[2]["content"] == "Could you add health check endpoints?"

    # 4. Formatted payload structure
    formatted = compressor.format_for_inference(
        compressed, next_user_prompt="Also check memory usage"
    )
    assert formatted[0]["role"] == "system"
    assert "FastAPI backend" in formatted[0]["content"]
    assert "[ACTIVE CONTEXT STATE LEDGER]" in formatted[0]["content"]
    assert formatted[-1]["role"] == "user"
    assert formatted[-1]["content"] == "Also check memory usage"


@pytest.mark.asyncio
async def test_llm_extractor_fallback_on_error():
    """Verify LLMLedgerExtractor falls back gracefully to HeuristicLedgerExtractor on failure."""

    # Callable that simulates network failure / timeout
    async def failing_completion(prompt: str) -> str:
        raise ConnectionError("LLM API endpoint connection timed out.")

    llm_extractor = LLMLedgerExtractor(completion_callable=failing_completion)
    compressor = ContextCompressor(extractor=llm_extractor)

    messages = [
        {"role": "user", "content": "Implement auth using FastAPI and PostgreSQL."},
        {"role": "assistant", "content": "Sure! Here is the auth configuration."},
        {"role": "user", "content": "Never use md5 for hashing passwords."},
        {"role": "assistant", "content": "Understood, using bcrypt instead."},
        {"role": "user", "content": "What is our database configuration?"},
        {"role": "assistant", "content": "The database is PostgreSQL on port 5432."},
        {"role": "user", "content": "What are our remaining tasks?"},
    ]

    compressed = await compressor.compress(messages)

    # Should not raise exception and should produce valid state ledger
    assert isinstance(compressed.state_ledger, StateLedger)
    assert compressed.state_ledger.pinned_goal != ""
    assert any("never use md5" in c.lower() for c in compressed.state_ledger.hard_constraints)


def test_collapse_repeated_lines_helper():
    """Verify collapse_repeated_lines collapses 120 repeated error lines to 1 line + marker."""
    from context_guard.compressors import collapse_repeated_lines

    repeated_text = "\n".join(["ERROR: connection reset by peer"] * 120)
    collapsed = collapse_repeated_lines(repeated_text)

    expected = "ERROR: connection reset by peer\n[repeated 120 times]"
    assert collapsed == expected


@pytest.mark.asyncio
async def test_single_turn_bloated_lines_compression():
    """Verify single turn with 120 repeated lines is compressed and yields token reduction."""
    compressor = ContextCompressor(preserve_recent_turns=3)

    repeated_error = "\n".join(["ERROR: connection reset by peer"] * 120)
    messages = [
        {"role": "user", "content": repeated_error},
    ]

    compressed = await compressor.compress(messages)

    assert compressed.compressed_tokens < compressed.original_tokens
    assert compressed.token_reduction_pct > 80.0
    assert len(compressed.recent_raw_turns) == 1
    assert "[repeated 120 times]" in compressed.recent_raw_turns[0]["content"]
    assert "ERROR: connection reset by peer" in compressed.recent_raw_turns[0]["content"]
