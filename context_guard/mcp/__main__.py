"""CLI entrypoint for Context Guard MCP server."""

import sys

from context_guard.mcp.server import run_mcp_server


def main() -> None:
    """Run Context Guard MCP server over stdio transport."""
    transport = "stdio"
    if "--sse" in sys.argv:
        transport = "sse"
    run_mcp_server(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
