"""Core data models for Context Guard health evaluation and state tracking."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Context health status indicator."""

    GREEN = "🟢 Healthy"
    YELLOW = "🟡 Degraded"
    RED = "🔴 Critical"


class StateLedger(BaseModel):
    """Structured state ledger tracking goals, constraints, and progress."""

    pinned_goal: str
    hard_constraints: list[str] = Field(default_factory=list)
    active_state: dict[str, Any] = Field(default_factory=dict)
    pending_questions: list[str] = Field(default_factory=list)
    last_updated_turn: int = 0


class FailureModeDetail(BaseModel):
    """Details of a specific context failure mode detected during evaluation."""

    mode: str  # "poisoning", "distraction", "confusion", "clash"
    penalty: int
    description: str
    evidence: list[str] = Field(default_factory=list)


class HealthReport(BaseModel):
    """Comprehensive health assessment report of conversation context."""

    status: HealthStatus
    penalty_score: int = Field(ge=0, le=100)
    detected_issues: list[FailureModeDetail] = Field(default_factory=list)
    recommended_action: str
    estimated_tokens: int = Field(ge=0, default=0)
