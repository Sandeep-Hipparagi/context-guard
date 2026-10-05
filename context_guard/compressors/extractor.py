"""State ledger extraction interfaces and implementations."""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from context_guard.core.models import StateLedger

CONSTRAINT_PATTERN = re.compile(
    r"\b(?:don't|do not|never|must not|cannot|should not|"
    r"strict(?:ly)?|must have|forbidden to|avoid)\s+([^.,;\n]+)",
    re.IGNORECASE,
)

TECH_PATTERN = re.compile(
    r"\b(FastAPI|Flask|Django|React|Vue|Next\.js|Pytest|PostgreSQL|"
    r"Redis|Docker|Kubernetes|SQLite|Pydantic|SQLAlchemy|Uvicorn)\b",
    re.IGNORECASE,
)

FILE_PATH_PATTERN = re.compile(
    r"\b([a-zA-Z0-9_\-./\\]+\.(?:py|ts|js|json|html|css|yaml|yml|md|txt|sql))\b",
    re.IGNORECASE,
)

CONFIG_KEYVAL_PATTERN = re.compile(
    r"\b([a-zA-Z0-9_]{3,})\s*[:=]\s*([a-zA-Z0-9_\-./:]+)",
)


class BaseLedgerExtractor(ABC):
    """Abstract base class for state ledger extractors."""

    @abstractmethod
    async def extract_ledger(
        self,
        history: list[dict[str, str]],
        existing_ledger: StateLedger | None = None,
    ) -> StateLedger:
        """Extract or update a StateLedger from conversation history."""


class HeuristicLedgerExtractor(BaseLedgerExtractor):
    """Deterministic, zero-API-cost state ledger extractor using regex and heuristics."""

    async def extract_ledger(
        self,
        history: list[dict[str, str]],
        existing_ledger: StateLedger | None = None,
    ) -> StateLedger:
        """Extract state ledger deterministically without external LLM calls."""
        # 1. Goal extraction
        pinned_goal = existing_ledger.pinned_goal if existing_ledger else ""
        if not pinned_goal:
            # First user message is commonly the primary goal
            for msg in history:
                if msg.get("role") == "user" and msg.get("content"):
                    first_line = msg["content"].strip().split("\n")[0]
                    pinned_goal = first_line[:120].strip()
                    break

        # 2. Hard constraints
        hard_constraints: list[str] = (
            list(existing_ledger.hard_constraints) if existing_ledger else []
        )
        for msg in history:
            content = msg.get("content", "")
            for match in CONSTRAINT_PATTERN.finditer(content):
                constraint = match.group(0).strip()
                if len(constraint) > 5 and constraint not in hard_constraints:
                    hard_constraints.append(constraint[:100])

        # 3. Active state
        active_state: dict[str, Any] = dict(existing_ledger.active_state) if existing_ledger else {}
        detected_tech: set[str] = set(active_state.get("technologies", []))
        detected_files: set[str] = set(active_state.get("files", []))

        for msg in history:
            content = msg.get("content", "")
            for tech in TECH_PATTERN.findall(content):
                detected_tech.add(tech)
            for file_path in FILE_PATH_PATTERN.findall(content):
                detected_files.add(file_path)
            for key, val in CONFIG_KEYVAL_PATTERN.findall(content):
                if key.lower() not in {"http", "https", "true", "false", "none"}:
                    active_state[key] = val

        if detected_tech:
            active_state["technologies"] = sorted(detected_tech)
        if detected_files:
            active_state["files"] = sorted(detected_files)

        # 4. Pending questions / tasks
        pending_questions: list[str] = []
        user_msgs = [m for m in history if m.get("role") == "user" and m.get("content")]
        if user_msgs:
            last_content = user_msgs[-1]["content"].strip()
            triggers = ["what", "how", "why", "can", "could", "please"]
            if "?" in last_content or any(last_content.lower().startswith(w) for w in triggers):
                pending_questions.append(last_content.split("\n")[0][:120])

        return StateLedger(
            pinned_goal=pinned_goal or "Assist user with technical task",
            hard_constraints=hard_constraints[:10],
            active_state=active_state,
            pending_questions=pending_questions,
            last_updated_turn=len(history),
        )


class LLMLedgerExtractor(BaseLedgerExtractor):
    """LLM-backed state ledger extractor with automatic fallback to heuristics."""

    def __init__(
        self,
        completion_callable: Callable[[str], Awaitable[str]] | None = None,
        fallback_extractor: BaseLedgerExtractor | None = None,
    ) -> None:
        self.completion_callable = completion_callable
        self.fallback_extractor = fallback_extractor or HeuristicLedgerExtractor()

    async def extract_ledger(
        self,
        history: list[dict[str, str]],
        existing_ledger: StateLedger | None = None,
    ) -> StateLedger:
        """Attempt LLM extraction, falling back cleanly to heuristics on error."""
        if not self.completion_callable:
            return await self.fallback_extractor.extract_ledger(history, existing_ledger)

        existing_repr = existing_ledger.model_dump_json() if existing_ledger else "None"
        prompt = (
            "You are a state ledger extraction engine. Analyze the conversation history and output "
            "a valid JSON object adhering strictly to this schema:\n"
            "{\n"
            '  "pinned_goal": "string",\n'
            '  "hard_constraints": ["string"],\n'
            '  "active_state": {"key": "value"},\n'
            '  "pending_questions": ["string"],\n'
            '  "last_updated_turn": int\n'
            "}\n\n"
            f"Existing Ledger: {existing_repr}\n\n"
            f"History:\n{json.dumps(history, indent=2)}\n\n"
            "Respond ONLY with raw JSON."
        )

        try:
            raw_response = await self.completion_callable(prompt)
            # Clean possible markdown fence
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            return StateLedger.model_validate(parsed)
        except Exception:
            # Graceful fallback to heuristic extraction
            return await self.fallback_extractor.extract_ledger(history, existing_ledger)
