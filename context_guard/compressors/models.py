"""Data models for context compression and state ledger representation."""

from pydantic import BaseModel, Field

from context_guard.core.models import StateLedger


class CompressedContext(BaseModel):
    """Result of context compression retaining state ledger and recent turns."""

    system_directive: str
    state_ledger: StateLedger
    state_ledger_markdown: str
    recent_raw_turns: list[dict[str, str]] = Field(default_factory=list)
    original_tokens: int = Field(ge=0)
    compressed_tokens: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0)
    token_reduction_pct: float = Field(ge=0.0, le=100.0)
