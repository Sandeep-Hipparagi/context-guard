"""Context compressor engine implementing noise stripping and state-ledger compression."""

import re
from typing import Any

from context_guard.compressors.extractor import (
    BaseLedgerExtractor,
    HeuristicLedgerExtractor,
)
from context_guard.compressors.models import CompressedContext
from context_guard.core.models import StateLedger

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

STACK_TRACE_PATTERN = re.compile(
    r"(Traceback \(most recent call last\):\n"
    r"(?:[ ]+File [^\n]+\n[ ]+[^\n]+\n)+)"
    r"([A-Za-z0-9_]+Error:[^\n]+)",
    re.MULTILINE,
)

PLEASANTRIES_PATTERN = re.compile(
    r"(?i)\b(?:as an ai language model,?\s*|"
    r"sure,?\s*(?:i can (?:help|assist) with that|here is the|let me help)[!\.]*\s*|"
    r"certainly,?\s*(?:here is the|i can help)?[!\.]*\s*|"
    r"of course,?\s*(?:here is the|i would be happy to)?[!\.]*\s*|"
    r"i('d| would) be happy to help (?:you )?with that[!\.]*\s*|"
    r"hello! how can i help you today\??\s*)",
)


def estimate_tokens_fast(messages: list[dict[str, Any]]) -> int:
    """Fast heuristic token counter (~4 characters/token)."""
    if not messages:
        return 0
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "") or "")
        content = msg.get("content", "")
        content_str = content if isinstance(content, str) else str(content or "")
        total_chars += len(role) + len(content_str)
    return max(0, (total_chars + 3) // 4)


class ContextCompressor:
    """Adaptive Context Compressor using State Ledger extraction and turn preservation."""

    def __init__(
        self,
        preserve_recent_turns: int = 3,
        extractor: BaseLedgerExtractor | None = None,
    ) -> None:
        self.preserve_recent_turns = preserve_recent_turns
        self.extractor = extractor or HeuristicLedgerExtractor()

    def strip_noise(self, text: str) -> str:
        """Deterministically strip ANSI codes, bloated stack traces, and pleasantries."""
        if not text:
            return ""

        # 1. Strip ANSI escape sequences
        cleaned = ANSI_ESCAPE_PATTERN.sub("", text)

        # 2. Collapse repetitive stack traces > 10 lines down to root exception
        def _collapse_trace(match: re.Match[str]) -> str:
            frames = match.group(1)
            root_err = match.group(2)
            lines = [line for line in frames.split("\n") if line.strip()]
            if len(lines) > 10:
                collapsed = len(lines) - 2
                return (
                    f"Traceback (most recent call last):\n"
                    f"  ... [{collapsed} frames collapsed] ...\n"
                    f"{root_err}"
                )
            return match.group(0)

        cleaned = STACK_TRACE_PATTERN.sub(_collapse_trace, cleaned)

        # 3. Trim conversational pleasantries
        cleaned = PLEASANTRIES_PATTERN.sub("", cleaned).strip()

        return cleaned

    def _render_ledger_markdown(self, ledger: StateLedger) -> str:
        """Render a StateLedger into a standardized compact Markdown block."""
        constraints_str = (
            "\n".join(f"- {c}" for c in ledger.hard_constraints)
            if ledger.hard_constraints
            else "- None"
        )

        if ledger.active_state:
            active_items = []
            for k, v in ledger.active_state.items():
                if isinstance(v, list):
                    active_items.append(f"- {k}: {', '.join(str(i) for i in v)}")
                else:
                    active_items.append(f"- {k}: {v}")
            active_str = "\n".join(active_items)
        else:
            active_str = "- None"

        pending_action = ledger.pending_questions[0] if ledger.pending_questions else "None"

        return (
            "[ACTIVE CONTEXT STATE LEDGER]\n"
            f"Pinned Goal: {ledger.pinned_goal}\n"
            "Hard Constraints:\n"
            f"{constraints_str}\n"
            "Active State:\n"
            f"{active_str}\n"
            f"Pending Action: {pending_action}"
        )

    def format_for_inference(
        self,
        compressed: CompressedContext,
        next_user_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the compressed context payload for LLM inference."""
        payload: list[dict[str, str]] = []

        system_parts: list[str] = []
        if compressed.system_directive:
            system_parts.append(compressed.system_directive.strip())
        if compressed.state_ledger_markdown:
            system_parts.append(compressed.state_ledger_markdown.strip())

        if system_parts:
            payload.append({"role": "system", "content": "\n\n".join(system_parts)})

        payload.extend(compressed.recent_raw_turns)

        if next_user_prompt:
            payload.append({"role": "user", "content": next_user_prompt})

        return payload

    async def compress(
        self,
        messages: list[dict[str, str]],
        existing_ledger: StateLedger | None = None,
    ) -> CompressedContext:
        """Compress message context using state ledger extraction and turn preservation."""
        original_tokens = estimate_tokens_fast(messages)

        system_messages = [m for m in messages if m.get("role") == "system"]
        conv_turns = [m for m in messages if m.get("role") != "system"]
        system_directive = "\n\n".join(
            m.get("content", "") for m in system_messages if m.get("content")
        )

        # Skip compression if conversation turns are within preservation limit
        if len(conv_turns) <= self.preserve_recent_turns:
            ledger = existing_ledger or StateLedger(pinned_goal="")
            return CompressedContext(
                system_directive=system_directive,
                state_ledger=ledger,
                state_ledger_markdown="",
                recent_raw_turns=conv_turns,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                token_reduction_pct=0.0,
            )

        # Split turns: older turns to compress, recent turns to preserve raw
        older_turns = conv_turns[: -self.preserve_recent_turns]
        recent_raw_turns = conv_turns[-self.preserve_recent_turns :]

        cleaned_older = [
            {"role": m.get("role", ""), "content": self.strip_noise(m.get("content", ""))}
            for m in older_turns
        ]

        # Extract updated state ledger
        ledger = await self.extractor.extract_ledger(cleaned_older, existing_ledger)
        ledger_markdown = self._render_ledger_markdown(ledger)

        # Estimate compressed tokens
        temp_compressed = CompressedContext(
            system_directive=system_directive,
            state_ledger=ledger,
            state_ledger_markdown=ledger_markdown,
            recent_raw_turns=recent_raw_turns,
            original_tokens=original_tokens,
            compressed_tokens=0,
            compression_ratio=1.0,
            token_reduction_pct=0.0,
        )

        formatted_payload = self.format_for_inference(temp_compressed)
        compressed_tokens = estimate_tokens_fast(formatted_payload)

        ratio = round(compressed_tokens / original_tokens, 4) if original_tokens > 0 else 1.0
        reduction_pct = round(max(0.0, (1.0 - ratio) * 100), 2) if original_tokens > 0 else 0.0

        return CompressedContext(
            system_directive=system_directive,
            state_ledger=ledger,
            state_ledger_markdown=ledger_markdown,
            recent_raw_turns=recent_raw_turns,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            token_reduction_pct=reduction_pct,
        )
