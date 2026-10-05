"""CLI entrypoint for Context Guard reverse proxy server."""

import uvicorn

from context_guard.proxy.config import ProxyConfig


def main() -> None:
    """Run Context Guard proxy using uvicorn."""
    config = ProxyConfig()
    uvicorn.run(
        "context_guard.proxy.server:app",
        host=config.CONTEXT_GUARD_HOST,
        port=config.CONTEXT_GUARD_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
