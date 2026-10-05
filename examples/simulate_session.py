#!/usr/bin/env python3
"""Interactive Terminal Simulation Script for Context-Guard.

Demonstrates real-time Context Health degradation and State-Ledger compression
across a multi-turn developer session without requiring an upstream API key.
"""

import asyncio
import sys
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from context_guard.compressors import ContextCompressor
from context_guard.core.models import HealthReport
from context_guard.evaluators import DeterministicEvaluator


def print_banner() -> None:
    print("=" * 72)
    print("  🛡️  CONTEXT-GUARD — Interactive Session Simulation")
    print("  Dual-Layer Context Health Guardrail & State-Ledger Compression")
    print("=" * 72)
    print()


def print_health_card(stage_title: str, report: HealthReport) -> None:
    print(f"\n┌─ [ {stage_title} ] " + "─" * (65 - len(stage_title)))
    print(f"│ Status:             {report.status.value}")
    print(f"│ Penalty Score:      {report.penalty_score} / 100")
    print(f"│ Estimated Buffer:   ~{report.estimated_tokens} tokens")
    print(f"│ Recommended Action: {report.recommended_action}")

    if report.detected_issues:
        print("│ Detected Failure Modes:")
        for idx, issue in enumerate(report.detected_issues, 1):
            print(f"│   {idx}. [{issue.mode.upper()}] (Penalty: {issue.penalty})")
            print(f"│      {issue.description}")
            for ev in issue.evidence[:2]:
                print(f"│        • {ev}")
    else:
        print("│ Detected Failure Modes: None (Clean Context)")
    print("└" + "─" * 70)


async def main() -> None:
    print_banner()

    evaluator = DeterministicEvaluator()
    compressor = ContextCompressor(preserve_recent_turns=3)

    session_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an expert backend engineer assisting with architecture. "
                "Hard constraint: Never use eval() or unsafe serialization."
            ),
        }
    ]

    # -------------------------------------------------------------
    # STAGE 1: Clean Initial Architecture Exchange (🟢 Healthy)
    # -------------------------------------------------------------
    print("▶ Simulating Turn 1-2: Initial Architecture Definition...")
    session_messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Goal: Build a distributed caching microservice.\n"
                    "We need strict schema validation with Pydantic and Redis connection pooling.\n"
                    "Hard Constraint: Never use eval()."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I can help with that! Here is the initial setup using Redis and Pydantic.\n"
                    "We configure connection pool with max_connections=20 on port=6379."
                ),
            },
        ]
    )

    report_stage1 = evaluator.evaluate(session_messages, pinned_constraints=["Never use eval()"])
    print_health_card("STAGE 1: Clean Architecture Exchange", report_stage1)
    await asyncio.sleep(0.5)

    # -------------------------------------------------------------
    # STAGE 2: Distraction & Repetitive Echo Loop (🟡 Degraded)
    # -------------------------------------------------------------
    print("\n▶ Simulating Turn 3-6: Repetitive Status Inquiries & Echo Loop...")
    echo_response = (
        "I am currently reviewing the database configuration and indexing strategy. "
        "The indexes need to be evaluated based on query latency metrics."
    )
    session_messages.extend(
        [
            {"role": "user", "content": "What is the status of the database migration?"},
            {"role": "assistant", "content": echo_response},
            {"role": "user", "content": "Can you elaborate on the index metrics?"},
            {"role": "assistant", "content": echo_response},
        ]
    )

    report_stage2 = evaluator.evaluate(session_messages, pinned_constraints=["Never use eval()"])
    print_health_card("STAGE 2: Echo Loop & Distraction", report_stage2)
    await asyncio.sleep(0.5)

    # -------------------------------------------------------------
    # STAGE 3: Error Rejection & Directive Contradiction (🔴 Critical)
    # -------------------------------------------------------------
    print("\n▶ Simulating Turn 7-10: User Rejection & Persistent Framework Contradiction...")
    session_messages.extend(
        [
            {
                "role": "user",
                "content": "No, wrong approach! Don't use Flask anywhere, use FastAPI instead.",
            },
            {
                "role": "assistant",
                "content": (
                    "Here is the service using Flask:\n"
                    "from flask import Flask\n"
                    "app = Flask(__name__)"
                ),
            },
            {
                "role": "user",
                "content": "Stop doing that! I explicitly instructed no Flask in this codebase.",
            },
            {
                "role": "assistant",
                "content": (
                    "Understood. Here is the Flask setup updated with routes:\n"
                    "from flask import Flask\n"
                    "app = Flask(__name__)"
                ),
            },
        ]
    )

    report_stage3 = evaluator.evaluate(
        session_messages,
        pinned_constraints=["Never use eval()", "Do not use Flask"],
    )
    print_health_card("STAGE 3: Contradiction & Poisoning", report_stage3)
    await asyncio.sleep(0.5)

    # -------------------------------------------------------------
    # STAGE 4: Adaptive State-Ledger Compression Engine
    # -------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  ⚙️  EXECUTING ADAPTIVE STATE-LEDGER COMPRESSION")
    print("=" * 72)

    compressed = await compressor.compress(session_messages)
    formatted = compressor.format_for_inference(compressed)
    tokens_saved = max(0, compressed.original_tokens - compressed.compressed_tokens)

    print("\n📊 COMPRESSION METRICS:")
    print(f"  • Pre-Compression Tokens:  {compressed.original_tokens}")
    print(f"  • Post-Compression Tokens: {compressed.compressed_tokens}")
    print(f"  • Tokens Saved:            {tokens_saved}")
    print(f"  • Token Reduction:         {compressed.token_reduction_pct}%")
    print(f"  • Compression Ratio:       {compressed.compression_ratio}")

    print("\n📋 EXTRACTED STATE LEDGER MARKDOWN:")
    print("-" * 50)
    print(compressed.state_ledger_markdown)
    print("-" * 50)

    print("\n🚀 FINAL FORWARDED PAYLOAD STRUCTURE (Turn preservation + System Ledger):")
    for i, msg in enumerate(formatted):
        role = msg["role"].upper()
        content_preview = msg["content"].replace("\n", " ")
        if len(content_preview) > 90:
            content_preview = content_preview[:87] + "..."
        print(f"  [{i}] {role:9s}: {content_preview}")

    print("\n✅ Simulation completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
