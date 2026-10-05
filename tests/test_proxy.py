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


def test_default_upstream_is_groq():
    """Verify default upstream base URL is Groq and trailing slashes are stripped."""
    config = ProxyConfig()
    assert config.UPSTREAM_BASE_URL == "https://api.groq.com/openai/v1"
    assert config.DEFAULT_MODEL == "llama-3.1-8b-instant"

    config_with_slash = ProxyConfig(UPSTREAM_BASE_URL="https://api.groq.com/openai/v1/")
    assert config_with_slash.UPSTREAM_BASE_URL == "https://api.groq.com/openai/v1"


def test_openai_base_url_env_override(monkeypatch):
    """Verify OPENAI_BASE_URL environment variable overrides the default upstream."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom-upstream.ai/api/v1/")
    config = ProxyConfig()
    assert config.UPSTREAM_BASE_URL == "https://custom-upstream.ai/api/v1"


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


@pytest.mark.asyncio
async def test_openrouter_routing_and_header_forwarding():
    """Verify OpenRouter key, URL resolution, and optional headers are correctly forwarded."""
    captured_request: dict[str, Any] = {}

    def openrouter_mock_handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "gen-openrouter-12345",
                "model": "openai/gpt-5-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello from OpenRouter gpt-5-mini!",
                        }
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    # Use default OpenRouter configuration with trailing slash to test stripping
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(openrouter_mock_handler))
    config = ProxyConfig(
        UPSTREAM_BASE_URL="https://openrouter.ai/api/v1/",
        OPENROUTER_HTTP_REFERER="https://my-app.internal",
        OPENROUTER_SITE_TITLE="Context Guard Tests",
    )
    app = create_app(config=config, client=mock_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello OpenRouter"}],
            },
            headers={
                "Authorization": "Bearer sk-or-v1-abcdef1234567890",
                "HTTP-Referer": "https://client-override.app",
                "X-Title": "Client Override Title",
            },
        )

    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"] == "Hello from OpenRouter gpt-5-mini!"
    assert res.headers["X-Context-Health-Status"] == "🟢 Healthy"

    # Verify upstream URL has no double slashes
    assert captured_request["url"] == "https://openrouter.ai/api/v1/chat/completions"

    # Verify headers forwarded intact
    assert captured_request["headers"]["authorization"] == "Bearer sk-or-v1-abcdef1234567890"
    assert captured_request["headers"]["http-referer"] == "https://client-override.app"
    assert captured_request["headers"]["x-title"] == "Client Override Title"

    # Verify model parameter preserved
    assert captured_request["body"]["model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_groq_routing_and_auth_forwarding():
    """Verify Groq API key, URL resolution, and default model handling."""
    captured_request: dict[str, Any] = {}

    def groq_mock_handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["headers"] = dict(request.headers)
        captured_request["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-groq-abc123",
                "object": "chat.completion",
                "model": "llama-3.1-8b-instant",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from Groq llama-3.1-8b-instant!",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    # Use default Groq configuration with trailing slash in URL
    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_mock_handler))
    config = ProxyConfig(UPSTREAM_BASE_URL="https://api.groq.com/openai/v1/")
    app = create_app(config=config, client=mock_client)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Hello Groq"}],
            },
            headers={"Authorization": "Bearer gsk_99999999999999999999"},
        )

    assert res.status_code == 200
    assert res.json()["choices"][0]["message"]["content"] == "Hello from Groq llama-3.1-8b-instant!"
    assert res.headers["X-Context-Health-Status"] == "🟢 Healthy"

    # Verify upstream URL has no double slashes and targets Groq endpoint
    assert captured_request["url"] == "https://api.groq.com/openai/v1/chat/completions"

    # Verify Groq authorization header forwarded intact without being dropped or replaced
    assert captured_request["headers"]["authorization"] == "Bearer gsk_99999999999999999999"

    # Verify Groq model identifier preserved
    assert captured_request["body"]["model"] == "llama-3.1-8b-instant"


@pytest.mark.asyncio
async def test_single_turn_repeated_lines_compression():
    """Verify single turn with 120 repeated error lines triggers Degraded and collapses upstream."""
    received_payload: dict[str, Any] = {}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal received_payload
        received_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-collapsed",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Identified repeated connection reset errors.",
                        }
                    }
                ],
            },
            headers={"content-type": "application/json"},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    config = ProxyConfig(UPSTREAM_BASE_URL="http://mock-upstream/v1")
    app = create_app(config=config, client=mock_client)

    repeated_error = "\n".join(["ERROR: connection reset by peer"] * 120)
    request_messages = [
        {"role": "user", "content": repeated_error},
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post(
            "/v1/chat/completions",
            json={"model": "llama-3.1-8b-instant", "messages": request_messages},
        )

    assert res.status_code == 200
    assert res.headers["X-Context-Health-Status"] in ("🟡 Degraded", "🔴 Critical")
    assert int(res.headers["X-Context-Tokens-Saved"]) > 0

    # Verify forwarded payload contains collapsed content
    forwarded_content = received_payload["messages"][0]["content"]
    assert "[repeated 120 times]" in forwarded_content
    assert "ERROR: connection reset by peer" in forwarded_content
    assert len(forwarded_content.splitlines()) < 10


def test_dashboard_endpoints():
    """Verify GET / and GET /dashboard return 200 OK with HTML content."""
    app = create_app()
    with TestClient(app) as client:
        # Test root endpoint
        res_root = client.get("/")
        assert res_root.status_code == 200
        assert "text/html" in res_root.headers["content-type"]
        assert "Context-Guard" in res_root.text
        assert "Diagnostic Dashboard" in res_root.text

        # Test /dashboard endpoint
        res_dash = client.get("/dashboard")
        assert res_dash.status_code == 200
        assert "text/html" in res_dash.headers["content-type"]
        assert "Context-Guard" in res_dash.text


@pytest.mark.asyncio
async def test_api_inspect_endpoint():
    """Verify POST /api/inspect returns health report, state ledger, and telemetry headers."""
    app = create_app()
    request_messages = [
        {"role": "user", "content": "Goal: Build a cache service. Never use eval."},
        {"role": "assistant", "content": "Configuring Redis cache on port 6379."},
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post("/api/inspect", json={"messages": request_messages})

    assert res.status_code == 200
    assert res.headers["X-Context-Health-Status"] == "🟢 Healthy"
    data = res.json()
    assert "health_report" in data
    assert "state_ledger" in data
    assert "metrics" in data
    assert data["health_report"]["status"] == "🟢 Healthy"
