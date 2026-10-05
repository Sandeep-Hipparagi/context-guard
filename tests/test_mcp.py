"""Comprehensive test suite for Context Guard MCP server and tools."""

import pytest

from context_guard.mcp.server import (
    compress_context_buffer,
    context_audit,
    generate_state_ledger,
    inspect_context_health,
    mcp_server,
)


@pytest.mark.asyncio
async def test_mcp_server_registration():
    """Verify MCP server registers all expected tools and prompt templates."""
    tools = await mcp_server.list_tools()
    tool_names = {tool.name for tool in tools}

    assert "inspect_context_health" in tool_names
    assert "compress_context_buffer" in tool_names
    assert "generate_state_ledger" in tool_names

    prompts = await mcp_server.list_prompts()
    prompt_names = {prompt.name for prompt in prompts}
    assert "context_audit" in prompt_names


@pytest.mark.asyncio
async def test_inspect_context_health_tool():
    """Verify inspect_context_health tool evaluates conversation health correctly."""
    clean_messages = [
        {"role": "user", "content": "How do I create a virtual environment with uv?"},
        {"role": "assistant", "content": "Run `uv venv` to create a virtual environment."},
    ]

    report = await inspect_context_health(messages=clean_messages)
    assert report["status"] == "🟢 Healthy"
    assert report["penalty_score"] == 0
    assert report["detected_issues"] == []
    assert report["recommended_action"] == "Proceed as normal."

    # Test via mcp_server.call_tool
    mcp_result = await mcp_server.call_tool(
        "inspect_context_health",
        {"messages": clean_messages},
    )
    assert not mcp_result.is_error
    assert mcp_result.structured_content is not None
    assert mcp_result.structured_content["status"] == "🟢 Healthy"


@pytest.mark.asyncio
async def test_compress_context_buffer_tool():
    """Verify compress_context_buffer tool returns state ledger and token reduction."""
    bloated_messages = [
        {"role": "system", "content": "You are a backend assistant."},
        {
            "role": "user",
            "content": (
                "Goal: Implement a Redis caching layer for FastAPI.\n"
                "Never use pickle serialization and avoid global state."
            ),
        },
        {
            "role": "assistant",
            "content": "Sure, I can help with that!\n" + "import redis\n# setup details...\n" * 20,
        },
        {"role": "user", "content": "How should we handle connection timeouts?"},
        {
            "role": "assistant",
            "content": "Use socket_timeout=2.0 in the connection pool configuration.",
        },
        {"role": "user", "content": "What is the next step?"},
    ]

    result = await compress_context_buffer(messages=bloated_messages, preserve_recent_turns=2)
    assert result["tokens_saved"] > 0
    assert result["token_reduction_pct"] > 20.0
    assert "[ACTIVE CONTEXT STATE LEDGER]" in result["state_ledger_markdown"]
    assert len(result["compressed_messages"]) > 0

    # Test via mcp_server.call_tool
    mcp_result = await mcp_server.call_tool(
        "compress_context_buffer",
        {"messages": bloated_messages, "preserve_recent_turns": 2},
    )
    assert not mcp_result.is_error
    assert mcp_result.structured_content["tokens_saved"] > 0


@pytest.mark.asyncio
async def test_generate_state_ledger_tool():
    """Verify generate_state_ledger tool extracts goals, constraints, and files accurately."""
    messages = [
        {"role": "user", "content": "Goal: Migrate SQLite database to PostgreSQL."},
        {
            "role": "assistant",
            "content": "Checking database migrations in alembic/env.py and config.py.",
        },
        {"role": "user", "content": "Never alter tables without a backup and avoid downtime."},
    ]

    result = await generate_state_ledger(messages=messages)
    ledger = result["state_ledger"]
    markdown = result["state_ledger_markdown"]

    assert "Migrate SQLite" in ledger["pinned_goal"]
    assert any("never alter tables" in c.lower() for c in ledger["hard_constraints"])
    assert "PostgreSQL" in ledger["active_state"].get("technologies", [])
    assert "[ACTIVE CONTEXT STATE LEDGER]" in markdown

    # Test via mcp_server.call_tool
    mcp_result = await mcp_server.call_tool(
        "generate_state_ledger",
        {"messages": messages},
    )
    assert not mcp_result.is_error
    assert "Migrate SQLite" in mcp_result.structured_content["state_ledger"]["pinned_goal"]


@pytest.mark.asyncio
async def test_context_audit_prompt():
    """Verify context_audit prompt template generates structured auditing instructions."""
    prompt_text = context_audit(recent_turn_count=7)
    assert "last 7 turns" in prompt_text
    assert "Poisoning" in prompt_text
    assert "Distraction" in prompt_text
    assert "Clash" in prompt_text
    assert "Confusion" in prompt_text

    # Test via mcp_server.get_prompt
    mcp_prompt = await mcp_server.get_prompt("context_audit", {"recent_turn_count": 4})
    assert mcp_prompt.messages is not None
    assert len(mcp_prompt.messages) > 0
    assert "last 4 turns" in mcp_prompt.messages[0].content.text
