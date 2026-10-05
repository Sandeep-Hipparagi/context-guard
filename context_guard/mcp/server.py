"""Model Context Protocol (MCP) server for Context Guard."""

from typing import Any, Literal

try:
    from mcp.server.mcpserver import MCPServer

    FastMCP = MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

from context_guard.compressors import ContextCompressor, HeuristicLedgerExtractor
from context_guard.evaluators import DeterministicEvaluator

mcp_server = FastMCP(
    name="context-guard",
    instructions=(
        "Context Guard MCP Server provides context health inspection, state-ledger extraction, "
        "and adaptive conversation compression tools."
    ),
)


@mcp_server.tool(
    name="inspect_context_health",
    description="Inspect context health for poisoning, distraction, clash, and confusion.",
)
async def inspect_context_health(
    messages: list[dict[str, Any]],
    pinned_constraints: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate context health and return standardized HealthReport dictionary."""
    evaluator = DeterministicEvaluator()
    sanitized: list[dict[str, str]] = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if isinstance(m, dict)
    ]
    report = evaluator.evaluate(sanitized, pinned_constraints=pinned_constraints)
    return report.model_dump()


@mcp_server.tool(
    name="compress_context_buffer",
    description="Compress conversation buffer into state ledger and formatted inference payload.",
)
async def compress_context_buffer(
    messages: list[dict[str, Any]],
    preserve_recent_turns: int = 3,
) -> dict[str, Any]:
    """Compress older turns into State Ledger while preserving recent turns."""
    compressor = ContextCompressor(preserve_recent_turns=preserve_recent_turns)
    sanitized: list[dict[str, str]] = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if isinstance(m, dict)
    ]
    compressed = await compressor.compress(sanitized)
    tokens_saved = max(0, compressed.original_tokens - compressed.compressed_tokens)
    formatted = compressor.format_for_inference(compressed)

    return {
        "state_ledger_markdown": compressed.state_ledger_markdown,
        "compressed_messages": formatted,
        "tokens_saved": tokens_saved,
        "token_reduction_pct": compressed.token_reduction_pct,
        "original_tokens": compressed.original_tokens,
        "compressed_tokens": compressed.compressed_tokens,
        "compression_ratio": compressed.compression_ratio,
    }


@mcp_server.tool(
    name="generate_state_ledger",
    description="Extract state ledger tracking pinned goals, constraints, and active state.",
)
async def generate_state_ledger(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extract StateLedger and format its markdown representation."""
    extractor = HeuristicLedgerExtractor()
    sanitized: list[dict[str, str]] = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if isinstance(m, dict)
    ]
    ledger = await extractor.extract_ledger(sanitized)
    compressor = ContextCompressor()
    markdown = compressor._render_ledger_markdown(ledger)

    return {
        "state_ledger": ledger.model_dump(),
        "state_ledger_markdown": markdown,
    }


@mcp_server.prompt(
    name="context_audit",
    description=(
        "Prompt template instructing an LLM to evaluate its current session history "
        "for poisoning, distraction, confusion, or clashing instructions."
    ),
)
def context_audit(recent_turn_count: int = 5) -> str:
    """Generate self-audit prompt template for LLM context inspection."""
    return (
        f"Please review the last {recent_turn_count} turns of our session history. "
        "Perform a strict Context Health Audit across the following 4 dimensions:\n\n"
        "1. **Poisoning / Error Propagation**: Have user corrections or rejections been ignored "
        "or repeated in subsequent turns?\n"
        "2. **Distraction / Echo Loops**: Are there repetitive phrases, circular responses, "
        "or code duplications?\n"
        "3. **Clash / Directive Contradiction**: Have any pinned directives, constraints, or "
        "negative instructions been violated?\n"
        "4. **Confusion / Noise Ballooning**: Has the conversation accumulated unnecessary context "
        "payload without tangible progress?\n\n"
        "Provide a concise diagnostic assessment: (1) Health Status (Healthy, Degraded, Critical), "
        "(2) Identified failure points with evidence, and "
        "(3) Recommended remediation or state reset."
    )


def run_mcp_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Run Context Guard MCP server."""
    mcp_server.run(transport=transport)
