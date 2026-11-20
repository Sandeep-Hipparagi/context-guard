"""Security regression tests for the CORS configuration of the proxy app.

Guards against reintroducing the wildcard-origin + allow-credentials
combination (CSRF-style abuse surface; browsers reject it outright).
"""

from fastapi.middleware.cors import CORSMiddleware

from context_guard.proxy.server import app


def _cors_middleware_kwargs() -> dict:
    middlewares = [m.kwargs for m in app.user_middleware if m.cls is CORSMiddleware]
    assert middlewares, "CORSMiddleware must be registered on the proxy app"
    return middlewares[0]


class TestCorsHardening:
    def test_credentials_never_combined_with_wildcard(self):
        kwargs = _cors_middleware_kwargs()
        if kwargs.get("allow_origins") == ["*"]:
            assert (
                kwargs.get("allow_credentials") is not False or kwargs["allow_credentials"] is False
            ), "allow_credentials must be False when origins are wildcarded"
            assert kwargs["allow_credentials"] is False

    def test_allow_credentials_disabled(self):
        kwargs = _cors_middleware_kwargs()
        assert kwargs["allow_credentials"] is False

    def test_telemetry_headers_still_exposed(self):
        kwargs = _cors_middleware_kwargs()
        exposed = kwargs.get("expose_headers", [])
        assert "x-context-health-status" in exposed
        assert "x-context-tokens-saved" in exposed
