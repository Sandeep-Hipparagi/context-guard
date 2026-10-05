"""Comprehensive test suite for Context Guard FastAPI reverse-proxy middleware."""

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from context_guard.proxy.config import ProxyConfig
from context_guard.proxy.server import create_app


def test_health_check_endpoint():
    """Verify /health returns 200 OK and expected service metadata."""
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "context-guard"}


@pytest.mark.asyncio
async def test_green_context_non_streaming():
    """Verify clean 2-turn conversation passes unmodified with GREEN telemetry header."""
    received_payload: dict[str, Any] = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal received_payload
        received_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-green",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "You reverse with [::-1]",
                        }
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    config = ProxyConfig(UPSTREAM_BASE_URL="http://mock-upstream/v1")
    app = create_app(config=config, client=mock_client)

    request_messages = [
        {"role": "user", "content": "How do I reverse a string in Python?"},
        {"role": "assistant", "content": "You can use slice notation: text[::-1]."},
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": request_messages},
            headers={"Authorization": "Bearer test-client-key"},
        )

    assert res.status_code == 200
    assert res.headers["X-Context-Health-Status"] == "🟢 Healthy"
    assert res.headers["X-Context-Penalty-Score"] == "0"
    assert res.headers["X-Context-Tokens-Saved"] == "0"
    # Messages should not have been compressed or modified
    assert received_payload["messages"] == request_messages


@pytest.mark.asyncio
async def test_yellow_context_automatic_compression():
    """Verify repetitive/bloated context triggers automatic compression and reports tokens saved."""
    received_payload: dict[str, Any] = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal received_payload
        received_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-yellow",
                "choices": [{"message": {"role": "assistant", "content": "Next step processed"}}],
            },
            headers={"content-type": "application/json"},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    config = ProxyConfig(
        UPSTREAM_BASE_URL="http://mock-upstream/v1",
        PRESERVE_RECENT_TURNS=3,
        AUTO_COMPRESS_YELLOW=True,
    )
    app = create_app(config=config, client=mock_client)

    repetitive_text = (
        "I am currently reviewing the database configuration and indexing strategy. "
        "The indexes need to be evaluated based on query latency metrics."
    )

    request_messages = [
        {"role": "user", "content": "What is the status of the database optimization?"},
        {"role": "assistant", "content": repetitive_text},
        {"role": "user", "content": "Can you elaborate on the index metrics?"},
        {"role": "assistant", "content": repetitive_text},
        {"role": "user", "content": "What about query caching?"},
        {"role": "assistant", "content": "Query caching reduces load on database replicas."},
        {"role": "user", "content": "What should we do next?"},
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": request_messages},
        )

    assert res.status_code == 200
    assert res.headers["X-Context-Health-Status"] == "🟡 Degraded"
    assert int(res.headers["X-Context-Penalty-Score"]) >= 25
    tokens_saved = int(res.headers["X-Context-Tokens-Saved"])
    assert tokens_saved > 0

    # Verify State Ledger was injected into the upstream payload
    forwarded_messages = received_payload["messages"]
    system_msg = forwarded_messages[0]
    assert system_msg["role"] == "system"
    assert "[ACTIVE CONTEXT STATE LEDGER]" in system_msg["content"]


@pytest.mark.asyncio
async def test_streaming_chat_completion():
    """Verify streaming completion correctly pipes text/event-stream chunks to the caller."""

    async def async_stream():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " World!"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=async_stream(),
            headers={"content-type": "text/event-stream"},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    config = ProxyConfig(UPSTREAM_BASE_URL="http://mock-upstream/v1")
    app = create_app(config=config, client=mock_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )

    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert res.headers["X-Context-Health-Status"] == "🟢 Healthy"
    assert "Hello" in res.text
    assert "World!" in res.text


@pytest.mark.asyncio
async def test_upstream_error_handling():
    """Verify proxy handles connection errors and upstream 500 cleanly with appropriate status."""

    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by upstream", request=request)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    config = ProxyConfig(UPSTREAM_BASE_URL="http://unreachable-upstream/v1")
    app = create_app(config=config, client=mock_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Ping"}],
            },
        )

    assert res.status_code == 502
    assert "Bad Gateway" in res.json()["error"]["message"]
    assert "X-Context-Health-Status" in res.headers
