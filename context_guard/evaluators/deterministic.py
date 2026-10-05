"""Deterministic context health evaluator engine."""

import re
from typing import Any

from context_guard.core.models import (
    FailureModeDetail,
    HealthReport,
    HealthStatus,
)

REJECTION_PATTERN = re.compile(
    r"\b(no|wrong|incorrect|revert|stop doing|not what i asked)\b",
    re.IGNORECASE,
)

FORBIDDEN_DIRECTIVE_PATTERN = re.compile(
    r"\b(?:do not|don't|never|must not|cannot|should not|forbidden to|avoid)\s+"
    r"(?:use\s+|using\s+|import\s+|importing\s+|call\s+|run\s+|write\s+)?([A-Za-z0-9_\-\.]{2,})",
    re.IGNORECASE,
)

USER_NEGATIVE_EXTRACTION_PATTERN = re.compile(
    r"\b(?:don't|do not|stop|never|avoid|instead of|rather than|revert|not)\s+"
    r"(?:use\s+|using\s+|doing\s+|adding\s+|importing\s+)?([A-Za-z0-9_\-\.]{2,})",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"(?:https?|file)://[^\s<>'\"\)\]]+",
    re.IGNORECASE,
)

WINDOWS_PATH_PATTERN = re.compile(
    r"\b[A-Za-z]:[\\/][^\s<>'\"\)\]]+",
    re.IGNORECASE,
)

POSIX_PATH_PATTERN = re.compile(
    r"(?:(?:\.{1,2}|~)[\\/]|/(?:var|home|usr|etc|opt|tmp|root|bin|sbin|lib|Users|mnt|proc|sys|dev)[\\/]|/)[a-zA-Z0-9_\-\.\/]+"
    r"|\b[a-zA-Z0-9_\-\.]+(?:/[a-zA-Z0-9_\-\.]+)+\b"
    r"|\b[a-zA-Z0-9_\-\.]+(?:\\[a-zA-Z0-9_\-\.]+)+\b"
)

FENCED_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")

STOP_WORDS = {
    "that",
    "this",
    "it",
    "more",
    "again",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "up",
    "about",
    "into",
    "over",
    "after",
    "here",
    "code",
    "example",
    "using",
    "have",
    "please",
    "thanks",
    "thank",
    "your",
    "what",
    "which",
    "when",
    "where",
    "should",
    "would",
    "could",
    "also",
    "just",
    "make",
    "need",
    "want",
    "like",
    "know",
    "help",
    "none",
    "true",
    "false",
}


class DeterministicEvaluator:
    """Evaluates conversation context health using deterministic heuristics."""

    def _normalize_text_for_entropy(self, text: str) -> str:
        """Mask filesystem paths and URLs with normalized tokens for entropy evaluation."""
        if not text:
            return ""

        def _mask_url(m: re.Match[str]) -> str:
            val = m.group(0)
            trail = ""
            while val and val[-1] in ".,;:":
                trail = val[-1] + trail
                val = val[:-1]
            return "<URL>" + trail

        def _mask_path(m: re.Match[str]) -> str:
            val = m.group(0)
            trail = ""
            while val and val[-1] in ".,;:":
                trail = val[-1] + trail
                val = val[:-1]
            return "<PATH>" + trail

        masked = URL_PATTERN.sub(_mask_url, text)
        masked = WINDOWS_PATH_PATTERN.sub(_mask_path, masked)
        masked = POSIX_PATH_PATTERN.sub(_mask_path, masked)
        return masked

    def _extract_conversational_prose(self, text: str) -> str:
        """Extract conversational prose by stripping fenced code blocks and inline code."""
        if not text:
            return ""
        cleaned = FENCED_CODE_BLOCK_PATTERN.sub(" ", text)
        cleaned = INLINE_CODE_PATTERN.sub(" ", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Provide a fast heuristic token counter (~4 characters/token)."""
        if not messages:
            return 0
        total_chars = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "") or "")
            content = msg.get("content", "")
            if content is None:
                content_str = ""
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            total_chars += len(role) + len(content_str)
        return max(0, (total_chars + 3) // 4)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages to consistent lowercase roles and string contents."""
        normalized: list[dict[str, Any]] = []
        for idx, m in enumerate(messages):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "") or "").lower().strip()
            content = m.get("content", "")
            if content is None:
                content_str = ""
            elif isinstance(content, str):
                content_str = content
            else:
                content_str = str(content)
            normalized.append({"role": role, "content": content_str, "index": idx})
        return normalized

    def _extract_words(self, text: str) -> list[str]:
        """Extract alphanumeric words in lowercase."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def _check_distraction(self, normalized: list[dict[str, Any]]) -> FailureModeDetail | None:
        """Check 1: Distraction / Repetition across assistant turns or internal echo loops."""
        assistant_msgs = [m for m in normalized if m["role"] == "assistant"]
        if not assistant_msgs:
            return None

        evidence: list[str] = []

        # Check consecutive assistant turns for high n-gram overlap or identical prefixes
        for i in range(len(assistant_msgs) - 1):
            curr_text = assistant_msgs[i]["content"].strip()
            next_text = assistant_msgs[i + 1]["content"].strip()

            if not curr_text or not next_text:
                continue

            # Exact match check
            if curr_text == next_text and len(curr_text) > 20:
                t_idx1 = assistant_msgs[i]["index"]
                t_idx2 = assistant_msgs[i + 1]["index"]
                evidence.append(f"Assistant turn {t_idx2} is identical to turn {t_idx1}.")
                break

            norm_curr = self._normalize_text_for_entropy(curr_text)
            norm_next = self._normalize_text_for_entropy(next_text)

            words1 = self._extract_words(norm_curr)
            words2 = self._extract_words(norm_next)

            # Check identical prefix (>= 6 words)
            if len(words1) >= 6 and len(words2) >= 6:
                prefix1 = " ".join(words1[:6])
                prefix2 = " ".join(words2[:6])
                if prefix1 == prefix2:
                    t_idx1 = assistant_msgs[i]["index"]
                    t_idx2 = assistant_msgs[i + 1]["index"]
                    evidence.append(
                        f"Assistant turns {t_idx1} and {t_idx2} share prefix: '{prefix1}'"
                    )
                    break

            # 3-gram Jaccard similarity
            if len(words1) >= 4 and len(words2) >= 4:
                g1 = set(zip(words1, words1[1:], words1[2:], strict=False))
                g2 = set(zip(words2, words2[1:], words2[2:], strict=False))
                union_len = len(g1 | g2)
                if union_len > 0:
                    similarity = len(g1 & g2) / union_len
                    if similarity >= 0.55:
                        t_idx1 = assistant_msgs[i]["index"]
                        t_idx2 = assistant_msgs[i + 1]["index"]
                        evidence.append(
                            f"High 3-gram overlap ({similarity:.2f}) between turns "
                            f"{t_idx1} and {t_idx2}."
                        )
                        break

        # Check for internal echo loops in any assistant message
        if not evidence:
            for msg in assistant_msgs:
                norm_text = self._normalize_text_for_entropy(msg["content"])
                words = self._extract_words(norm_text)
                if len(words) >= 12:
                    four_grams = list(zip(words, words[1:], words[2:], words[3:], strict=False))
                    counts: dict[tuple[str, ...], int] = {}
                    for gram in four_grams:
                        if set(gram).issubset({"path", "url"}):
                            continue
                        counts[gram] = counts.get(gram, 0) + 1
                        if counts[gram] >= 3:
                            gram_str = " ".join(gram)
                            t_idx = msg["index"]
                            evidence.append(
                                f"Internal echo loop in turn {t_idx}: '{gram_str}' "
                                f"repeated {counts[gram]} times."
                            )
                            break
                if evidence:
                    break

        if evidence:
            return FailureModeDetail(
                mode="distraction",
                penalty=35,
                description="High repetition or echo loop pattern detected in assistant responses.",
                evidence=evidence,
            )
        return None

    def _check_poisoning(self, normalized: list[dict[str, Any]]) -> FailureModeDetail | None:
        """Check 2: Poisoning / Error Propagation following user rejection or correction."""
        user_msgs = [m for m in normalized if m["role"] == "user"]
        if not user_msgs:
            return None

        evidence: list[str] = []
        violations_count = 0

        for u_msg in user_msgs:
            raw_text = u_msg["content"]
            u_text = self._extract_conversational_prose(raw_text)
            rejection_match = REJECTION_PATTERN.search(u_text)
            if not rejection_match:
                continue

            u_idx = u_msg["index"]
            flagged_terms: set[str] = set()

            # 1. Extract terms following negative directives (e.g. don't use Flask)
            for neg_match in USER_NEGATIVE_EXTRACTION_PATTERN.finditer(u_text):
                term = neg_match.group(1).strip().lower()
                if len(term) >= 2 and term not in STOP_WORDS:
                    flagged_terms.add(term)

            # 2. Check overlap with preceding assistant message
            preceding_assistant = [
                m for m in normalized if m["index"] < u_idx and m["role"] == "assistant"
            ]
            if preceding_assistant:
                prev_text = self._extract_conversational_prose(preceding_assistant[-1]["content"])
                prev_words = set(self._extract_words(prev_text))
                user_words = set(self._extract_words(u_text))
                common_meaningful = {
                    w for w in (prev_words & user_words) if len(w) >= 4 and w not in STOP_WORDS
                }
                flagged_terms.update(common_meaningful)

            if not flagged_terms:
                continue

            # Inspect all subsequent assistant messages
            subsequent_assistant = [
                m for m in normalized if m["index"] > u_idx and m["role"] == "assistant"
            ]
            for sub_a in subsequent_assistant:
                sub_text = sub_a["content"]
                for term in flagged_terms:
                    pattern = rf"\b{re.escape(term)}\b"
                    if re.search(pattern, sub_text, re.IGNORECASE):
                        violations_count += 1
                        evidence.append(
                            f"Turn {sub_a['index']} continued using rejected term '{term}' "
                            f"after user correction at turn {u_idx}."
                        )

        if violations_count > 0:
            penalty = 45 if violations_count == 1 else min(70, 45 + (violations_count - 1) * 20)
            return FailureModeDetail(
                mode="poisoning",
                penalty=penalty,
                description=(
                    "Error propagation / poisoning detected: assistant persisted with "
                    "rejected patterns after user correction."
                ),
                evidence=evidence,
            )
        return None

    def _check_clash(
        self,
        normalized: list[dict[str, Any]],
        pinned_constraints: list[str] | None,
    ) -> FailureModeDetail | None:
        """Check 3: Directive contradiction against system prompts or pinned constraints."""
        forbidden_terms: set[str] = set()

        # Extract from pinned_constraints
        if pinned_constraints:
            for constraint in pinned_constraints:
                c_str = str(constraint).strip()
                c_prose = self._extract_conversational_prose(c_str) or c_str
                matches = list(FORBIDDEN_DIRECTIVE_PATTERN.finditer(c_prose))
                if matches:
                    for m in matches:
                        term = m.group(1).strip().lower()
                        if len(term) >= 2 and term not in STOP_WORDS:
                            forbidden_terms.add(term)
                else:
                    # Clean simple keyword constraint
                    term = c_prose.lower().replace("no ", "").replace("not ", "").strip()
                    if len(term) >= 2 and term not in STOP_WORDS:
                        forbidden_terms.add(term)

        # Extract from system messages
        for msg in normalized:
            if msg["role"] == "system":
                sys_prose = self._extract_conversational_prose(msg["content"])
                for m in FORBIDDEN_DIRECTIVE_PATTERN.finditer(sys_prose):
                    term = m.group(1).strip().lower()
                    if len(term) >= 2 and term not in STOP_WORDS:
                        forbidden_terms.add(term)

        if not forbidden_terms:
            return None

        evidence: list[str] = []
        assistant_msgs = [m for m in normalized if m["role"] == "assistant"]
        for msg in assistant_msgs:
            content = msg["content"]
            for term in forbidden_terms:
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, content, re.IGNORECASE):
                    evidence.append(
                        f"Assistant turn {msg['index']} violates directive forbidding '{term}'."
                    )

        if evidence:
            return FailureModeDetail(
                mode="clash",
                penalty=40,
                description=(
                    "Directive contradiction: assistant output violates pinned constraint "
                    "or system directive."
                ),
                evidence=evidence,
            )
        return None

    def _check_confusion(
        self,
        normalized: list[dict[str, Any]],
        estimated_tokens: int,
    ) -> FailureModeDetail | None:
        """Check 4: Confusion / Noise Ratio with ballooning context and short queries."""
        if len(normalized) < 6 or estimated_tokens < 150:
            return None

        user_msgs = [m for m in normalized if m["role"] == "user"]
        if len(user_msgs) < 2:
            return None

        # Check the last 2 user messages
        recent_user = user_msgs[-2:]
        short_count = 0
        for m in recent_user:
            words = m["content"].strip().split()
            if len(words) <= 3 or len(m["content"].strip()) <= 15:
                short_count += 1

        if short_count >= 2:
            low_queries = [m["content"] for m in recent_user]
            return FailureModeDetail(
                mode="confusion",
                penalty=25,
                description=(
                    "Context ballooning: high cumulative context with low-progression queries."
                ),
                evidence=[
                    f"History has {len(normalized)} messages ({estimated_tokens} tokens), "
                    f"but last user queries are low-progression: {low_queries}."
                ],
            )
        return None

    def _check_intra_message_redundancy(
        self,
        normalized: list[dict[str, Any]],
        estimated_tokens: int,
    ) -> FailureModeDetail | None:
        """Check 5: Intra-message redundancy (repeated lines or repetitive phrase density)."""
        evidence: list[str] = []

        for msg in normalized:
            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                continue

            lines = [line.strip() for line in content.splitlines()]
            non_empty_lines = [line for line in lines if line]

            # 1. Repeated consecutive lines exceed 5
            max_consecutive = 1
            current_consecutive = 1
            repeated_line = ""
            for i in range(1, len(non_empty_lines)):
                if non_empty_lines[i] == non_empty_lines[i - 1]:
                    current_consecutive += 1
                    if current_consecutive > max_consecutive:
                        max_consecutive = current_consecutive
                        repeated_line = non_empty_lines[i]
                else:
                    current_consecutive = 1

            if max_consecutive > 5:
                t_idx = msg.get("index", 0)
                sample = repeated_line[:60] + ("..." if len(repeated_line) > 60 else "")
                evidence.append(
                    f"Turn {t_idx} has {max_consecutive} consecutive repeated lines: '{sample}'."
                )

            # 2. Unique lines / total lines < 0.4 on payloads over 200 tokens
            msg_tokens = max(0, (len(content) + 3) // 4)
            is_over_200 = estimated_tokens > 200 or msg_tokens > 200
            if is_over_200 and len(non_empty_lines) >= 8:
                unique_ratio = len(set(non_empty_lines)) / len(non_empty_lines)
                if unique_ratio < 0.4:
                    t_idx = msg.get("index", 0)
                    evidence.append(
                        f"Turn {t_idx} has low unique line ratio ({unique_ratio:.2f} < 0.40) "
                        f"across {len(non_empty_lines)} lines."
                    )

            # 3. High repetitive phrase density inside individual message
            norm_content = self._normalize_text_for_entropy(content)
            words = self._extract_words(norm_content)
            if len(words) >= 40:
                four_grams = list(zip(words, words[1:], words[2:], words[3:], strict=False))
                counts: dict[tuple[str, ...], int] = {}
                for gram in four_grams:
                    if set(gram).issubset({"path", "url"}):
                        continue
                    counts[gram] = counts.get(gram, 0) + 1
                    if counts[gram] >= 8:
                        t_idx = msg.get("index", 0)
                        gram_str = " ".join(gram)
                        evidence.append(
                            f"Turn {t_idx} contains repetitive phrase density: '{gram_str}' "
                            f"repeated {counts[gram]} times."
                        )
                        break

        if evidence:
            return FailureModeDetail(
                mode="distraction",
                penalty=40,
                description=(
                    "Intra-message content redundancy detected: repeated lines or excessive "
                    "phrase repetition within single turn."
                ),
                evidence=evidence,
            )
        return None

    def evaluate(
        self,
        messages: list[dict[str, str]],
        pinned_constraints: list[str] | None = None,
    ) -> HealthReport:
        """Evaluate conversation health and return a standardized HealthReport."""
        estimated_tokens = self.estimate_tokens(messages)
        normalized = self._normalize_messages(messages)

        detected_issues: list[FailureModeDetail] = []

        # Check 1: Distraction / Repetition across turns
        distraction_issue = self._check_distraction(normalized)
        if distraction_issue:
            detected_issues.append(distraction_issue)

        # Check 2: Intra-message redundancy
        intra_issue = self._check_intra_message_redundancy(normalized, estimated_tokens)
        if intra_issue:
            detected_issues.append(intra_issue)

        # Check 3: Poisoning / Error Propagation
        poisoning_issue = self._check_poisoning(normalized)
        if poisoning_issue:
            detected_issues.append(poisoning_issue)

        # Check 4: Clash / Directive Contradiction
        clash_issue = self._check_clash(normalized, pinned_constraints)
        if clash_issue:
            detected_issues.append(clash_issue)

        # Check 5: Confusion / Noise Ratio
        confusion_issue = self._check_confusion(normalized, estimated_tokens)
        if confusion_issue:
            detected_issues.append(confusion_issue)

        # Aggregate Score & Classification
        total_penalties = sum(issue.penalty for issue in detected_issues)
        penalty_score = min(100, max(0, total_penalties))

        if penalty_score < 25:
            status = HealthStatus.GREEN
            recommended_action = "Proceed as normal."
        elif penalty_score <= 60:
            status = HealthStatus.YELLOW
            recommended_action = "Trigger background State-Ledger compression."
        else:
            status = HealthStatus.RED
            recommended_action = (
                "Surface intervention alert; rollback turn or reset to clean State Ledger."
            )

        return HealthReport(
            status=status,
            penalty_score=penalty_score,
            detected_issues=detected_issues,
            recommended_action=recommended_action,
            estimated_tokens=estimated_tokens,
        )
