"""FastAPI reverse-proxy middleware inspecting and compressing context."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from context_guard.compressors import ContextCompressor
from context_guard.core.models import HealthReport, HealthStatus
from context_guard.evaluators import DeterministicEvaluator
from context_guard.proxy.config import ProxyConfig

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

INTERVENTION_DIRECTIVE = (
    "[SYSTEM INTERVENTION - CONTEXT RECTIFICATION]\n"
    "CRITICAL: The conversation context previously exhibited directive contradiction, "
    "persistent errors, or hallucination loops. You MUST strictly adhere to the user's "
    "latest directives and hard constraints. Disregard any previously rejected patterns."
)


def _inject_telemetry_headers(
    response: Response, report: HealthReport, tokens_saved: int
) -> Response:
    """Inject telemetry headers preserving UTF-8 encoding in ASGI raw_headers."""
    response.raw_headers.append((b"x-context-health-status", report.status.value.encode("utf-8")))
    response.raw_headers.append(
        (b"x-context-penalty-score", str(report.penalty_score).encode("utf-8"))
    )
    response.raw_headers.append((b"x-context-tokens-saved", str(tokens_saved).encode("utf-8")))
    return response


def create_app(
    config: ProxyConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Create and configure Context Guard reverse proxy FastAPI application."""
    proxy_config = config or ProxyConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "client", None) is not None:
            yield
        else:
            timeout = httpx.Timeout(60.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                app.state.client = http_client
                yield

    app = FastAPI(
        title="Context Guard Proxy",
        description="Dual-layer Context Health Guardrail and Compression Reverse Proxy",
        lifespan=lifespan,
    )
    app.state.config = proxy_config
    if client is not None:
        app.state.client = client

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "x-context-health-status",
            "x-context-penalty-score",
            "x-context-tokens-saved",
        ],
    )

    evaluator = DeterministicEvaluator()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok", "service": "context-guard"}

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(request: Request) -> Response:
        """List available models, proxying upstream or returning default configuration."""
        upstream_base = proxy_config.UPSTREAM_BASE_URL.rstrip("/")
        upstream_url = f"{upstream_base}/models"
        auth_header = request.headers.get("authorization")
        if not auth_header and proxy_config.UPSTREAM_API_KEY:
            auth_header = f"Bearer {proxy_config.UPSTREAM_API_KEY}"

        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        http_client: httpx.AsyncClient | None = getattr(request.app.state, "client", None)
        if http_client is None:
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

        try:
            upstream_resp = await http_client.get(upstream_url, headers=headers)
            if upstream_resp.status_code == 200:
                return Response(
                    content=upstream_resp.content,
                    status_code=200,
                    media_type="application/json",
                )
        except Exception:
            pass

        fallback_data = {
            "object": "list",
            "data": [
                {
                    "id": proxy_config.DEFAULT_MODEL,
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "context-guard",
                },
                {
                    "id": "llama-3.3-70b-versatile",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "groq",
                },
                {
                    "id": "llama-3.1-8b-instant",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "groq",
                },
            ],
        }
        return JSONResponse(content=fallback_data, status_code=200)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        """Serve the interactive diagnostic visual dashboard."""
        if INDEX_HTML_PATH.exists():
            return HTMLResponse(content=INDEX_HTML_PATH.read_text(encoding="utf-8"))
        return HTMLResponse(
            content=(
                "<!DOCTYPE html><html><body><h1>Context-Guard Dashboard</h1>"
                "<p>Dashboard HTML not found.</p></body></html>"
            ),
            status_code=200,
        )

    @app.post("/api/inspect")
    async def inspect_context(request: Request) -> Response:
        """Inspect context health, extract state ledger, and compute compression."""
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'messages' must be an array",
                        "type": "invalid_request_error",
                    }
                },
            )

        pinned_constraints = body.get("pinned_constraints")
        report = evaluator.evaluate(messages, pinned_constraints=pinned_constraints)
        compressor = ContextCompressor(preserve_recent_turns=proxy_config.PRESERVE_RECENT_TURNS)
        compressed = await compressor.compress(messages)
        formatted = compressor.format_for_inference(compressed)
        tokens_saved = max(0, compressed.original_tokens - compressed.compressed_tokens)

        result = {
            "health_report": report.model_dump(),
            "state_ledger": compressed.state_ledger.model_dump(),
            "state_ledger_markdown": compressed.state_ledger_markdown,
            "compressed_messages": formatted,
            "metrics": {
                "original_tokens": compressed.original_tokens,
                "compressed_tokens": compressed.compressed_tokens,
                "tokens_saved": tokens_saved,
                "compression_ratio": compressed.compression_ratio,
                "token_reduction_pct": compressed.token_reduction_pct,
            },
        }
        resp = JSONResponse(content=result)
        return _inject_telemetry_headers(resp, report, tokens_saved)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        """Inspect context health, apply adaptive compression, and forward to upstream."""
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
            )

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "'messages' must be an array",
                        "type": "invalid_request_error",
                    }
                },
            )

        if not body.get("model"):
            body["model"] = proxy_config.DEFAULT_MODEL

        # 1. Run Context Health Evaluation
        report = evaluator.evaluate(messages)
        tokens_saved = 0

        # 2. Adaptive compression / Intervention handling
        compressor = ContextCompressor(preserve_recent_turns=proxy_config.PRESERVE_RECENT_TURNS)

        if report.status == HealthStatus.YELLOW and proxy_config.AUTO_COMPRESS_YELLOW:
            compressed = await compressor.compress(messages)
            body["messages"] = compressor.format_for_inference(compressed)
            tokens_saved = max(0, compressed.original_tokens - compressed.compressed_tokens)

        elif report.status == HealthStatus.RED and proxy_config.INTERVENTION_ON_RED:
            compressed = await compressor.compress(messages)
            formatted = compressor.format_for_inference(compressed)
            if formatted and formatted[0].get("role") == "system":
                formatted[0]["content"] = f"{INTERVENTION_DIRECTIVE}\n\n{formatted[0]['content']}"
            else:
                formatted.insert(0, {"role": "system", "content": INTERVENTION_DIRECTIVE})
            body["messages"] = formatted
            tokens_saved = max(0, compressed.original_tokens - compressed.compressed_tokens)

        # 3. Upstream Routing and Authorization
        upstream_base = proxy_config.UPSTREAM_BASE_URL.rstrip("/")
        upstream_url = f"{upstream_base}/chat/completions"
        auth_header = request.headers.get("authorization")
        if not auth_header and proxy_config.UPSTREAM_API_KEY:
            auth_header = f"Bearer {proxy_config.UPSTREAM_API_KEY}"

        headers = {"Content-Type": "application/json"}
        if auth_header:
            headers["Authorization"] = auth_header

        referer = (
            request.headers.get("http-referer")
            or request.headers.get("referer")
            or proxy_config.OPENROUTER_HTTP_REFERER
        )
        if referer:
            headers["HTTP-Referer"] = referer

        site_title = request.headers.get("x-title") or proxy_config.OPENROUTER_SITE_TITLE
        if site_title:
            headers["X-Title"] = site_title

        http_client: httpx.AsyncClient | None = getattr(request.app.state, "client", None)
        if http_client is None:
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

        is_stream = bool(body.get("stream", False))

        # 4. Streaming Forwarding
        if is_stream:

            async def stream_generator() -> AsyncIterator[bytes]:
                try:
                    async with http_client.stream(
                        "POST", upstream_url, json=body, headers=headers
                    ) as upstream_resp:
                        async for chunk in upstream_resp.aiter_bytes():
                            yield chunk
                except httpx.RequestError as exc:
                    err_msg = json.dumps(
                        {
                            "error": {
                                "message": f"Upstream failure: {exc}",
                                "type": "bad_gateway",
                            }
                        }
                    )
                    yield f"data: {err_msg}\n\n".encode()

            resp = StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
            return _inject_telemetry_headers(resp, report, tokens_saved)

        # 5. Non-streaming Forwarding
        try:
            upstream_resp = await http_client.post(upstream_url, json=body, headers=headers)
            content_type = upstream_resp.headers.get("content-type", "application/json")
            resp = Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type=content_type,
            )
            return _inject_telemetry_headers(resp, report, tokens_saved)
        except httpx.RequestError as exc:
            resp = JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": f"Bad Gateway: Unable to reach upstream: {exc}",
                        "type": "bad_gateway",
                    }
                },
            )
            return _inject_telemetry_headers(resp, report, tokens_saved)

    return app


# Default app instance
app = create_app()


def main() -> None:
    """Run Context Guard proxy using uvicorn."""
    import uvicorn

    config = ProxyConfig()
    uvicorn.run(
        "context_guard.proxy.server:app",
        host=config.CONTEXT_GUARD_HOST,
        port=config.CONTEXT_GUARD_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
