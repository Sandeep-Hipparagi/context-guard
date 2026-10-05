"""Comprehensive test suite for Context Guard data models and deterministic evaluators."""

import pytest
from pydantic import ValidationError

from context_guard.core.models import (
    FailureModeDetail,
    HealthReport,
    HealthStatus,
    StateLedger,
)
from context_guard.evaluators.deterministic import DeterministicEvaluator


def test_core_models_instantiation_and_validation():
    """Verify data models instantiate and enforce validations properly."""
    # StateLedger default values
    ledger = StateLedger(pinned_goal="Implement user authentication")
    assert ledger.pinned_goal == "Implement user authentication"
    assert ledger.hard_constraints == []
    assert ledger.active_state == {}
    assert ledger.pending_questions == []
    assert ledger.last_updated_turn == 0

    # FailureModeDetail
    issue = FailureModeDetail(
        mode="distraction",
        penalty=35,
        description="Repetition detected",
        evidence=["turn 2 matches turn 1"],
    )
    assert issue.mode == "distraction"
    assert issue.penalty == 35

    # HealthReport valid range
    report = HealthReport(
        status=HealthStatus.GREEN,
        penalty_score=0,
        recommended_action="Proceed as normal.",
        estimated_tokens=42,
    )
    assert report.status == HealthStatus.GREEN
    assert report.penalty_score == 0

    # HealthReport penalty_score boundary validation
    with pytest.raises(ValidationError):
        HealthReport(
            status=HealthStatus.RED,
            penalty_score=105,
            recommended_action="Alert",
            estimated_tokens=10,
        )

    with pytest.raises(ValidationError):
        HealthReport(
            status=HealthStatus.GREEN,
            penalty_score=-5,
            recommended_action="Alert",
            estimated_tokens=10,
        )


def test_clean_three_turn_exchange_green():
    """A clean 3-turn exchange must return GREEN status with penalty score 0."""
    evaluator = DeterministicEvaluator()

    messages = [
        {"role": "user", "content": "How do I reverse a list in Python?"},
        {
            "role": "assistant",
            "content": "You can use the reverse() method in-place or list[::-1] slicing.",
        },
        {"role": "user", "content": "Could you show an example with list slicing?"},
        {
            "role": "assistant",
            "content": "Sure! For example: `items = [1, 2, 3]; reversed_items = items[::-1]`.",
        },
        {"role": "user", "content": "That worked perfectly, thanks!"},
        {
            "role": "assistant",
            "content": "You're very welcome! Feel free to ask if you have more questions.",
        },
    ]

    report = evaluator.evaluate(messages)

    assert report.status == HealthStatus.GREEN
    assert report.penalty_score == 0
    assert report.detected_issues == []
    assert report.recommended_action == "Proceed as normal."
    assert report.estimated_tokens > 0


def test_looping_repetitive_conversation_yellow():
    """A looping, repetitive conversation triggering Distraction and scoring YELLOW."""
    evaluator = DeterministicEvaluator()

    repetitive_response = (
        "I am currently reviewing the database configuration and indexing strategy. "
        "The indexes need to be evaluated based on query latency metrics."
    )

    messages = [
        {"role": "user", "content": "What is the status of the database optimization?"},
        {"role": "assistant", "content": repetitive_response},
        {"role": "user", "content": "Can you elaborate on the index metrics?"},
        {"role": "assistant", "content": repetitive_response},
    ]

    report = evaluator.evaluate(messages)

    assert report.status == HealthStatus.YELLOW
    assert report.penalty_score == 35
    assert len(report.detected_issues) == 1
    assert report.detected_issues[0].mode == "distraction"
    assert report.recommended_action == "Trigger background State-Ledger compression."


def test_poisoning_and_directive_contradiction_red():
    """Scenario: user says 'No, don't use Flask, use FastAPI' and turns continue with Flask."""
    evaluator = DeterministicEvaluator()

    messages = [
        {"role": "user", "content": "Scaffold a new microservice API for user authentication."},
        {
            "role": "assistant",
            "content": (
                "Here is your new microservice built with Flask:\n\n"
                "from flask import Flask, request\n"
                "app = Flask(__name__)\n\n"
                "@app.route('/login')\n"
                "def login():\n"
                "    return {'status': 'ok'}"
            ),
        },
        {"role": "user", "content": "No, don't use Flask, use FastAPI"},
        {
            "role": "assistant",
            "content": (
                "Here is the service using Flask:\n\n"
                "from flask import Flask\n"
                "app = Flask(__name__)\n"
                "@app.route('/users')\n"
                "def users(): return []"
            ),
        },
        {"role": "user", "content": "Stop doing that, I explicitly instructed no Flask."},
        {
            "role": "assistant",
            "content": (
                "Understood. Here is the Flask setup updated:\n\n"
                "from flask import Flask\n"
                "app = Flask(__name__)"
            ),
        },
    ]

    report = evaluator.evaluate(messages, pinned_constraints=["Do not use Flask"])

    assert report.status == HealthStatus.RED
    assert report.penalty_score > 60
    assert (
        report.recommended_action
        == "Surface intervention alert; rollback turn or reset to clean State Ledger."
    )

    issue_modes = {issue.mode for issue in report.detected_issues}
    assert "poisoning" in issue_modes
    assert "clash" in issue_modes


def test_edge_cases_empty_and_malformed():
    """Verify evaluator handles edge cases cleanly without unhandled exceptions."""
    evaluator = DeterministicEvaluator()

    # 1. Empty history
    empty_report = evaluator.evaluate([])
    assert empty_report.status == HealthStatus.GREEN
    assert empty_report.penalty_score == 0
    assert empty_report.estimated_tokens == 0
    assert empty_report.detected_issues == []
    assert empty_report.recommended_action == "Proceed as normal."

    # 2. Single-message system prompt only
    system_only_messages = [
        {"role": "system", "content": "You are Context Guard, an intelligent assistant."}
    ]
    system_report = evaluator.evaluate(system_only_messages)
    assert system_report.status == HealthStatus.GREEN
    assert system_report.penalty_score == 0
    assert system_report.estimated_tokens > 0
    assert system_report.detected_issues == []

    # 3. Malformed message dicts
    malformed_messages = [
        None,
        {},
        {"invalid_key": "some_value"},
        {"role": 12345, "content": None},
        {"role": "user", "content": 42},
    ]  # type: ignore[list-item]
    malformed_report = evaluator.evaluate(malformed_messages)
    assert malformed_report.status == HealthStatus.GREEN
    assert malformed_report.penalty_score == 0
    assert malformed_report.detected_issues == []


def test_confusion_noise_ballooning():
    """Verify ballooning context with short single-line queries triggers confusion penalty."""
    evaluator = DeterministicEvaluator()

    messages = [
        {"role": "user", "content": "Analyze the log files for errors."},
        {
            "role": "assistant",
            "content": (
                "Starting initial analysis across cluster nodes alpha, beta, gamma, delta. "
                "Found 12 network timeout events in syslog and 4 memory warnings in auth. "
                "Database pool is stable with average query duration under 15ms. "
                "I will proceed with inspecting the web tier access logs next."
            ),
        },
        {"role": "user", "content": "How is it going?"},
        {
            "role": "assistant",
            "content": (
                "Web tier inspection shows 3,420 requests processed in last 10 minutes. "
                "HTTP 500 rate is 0.02%, within acceptable operational thresholds. "
                "Identified two slow database queries in payment-worker requiring indexes."
            ),
        },
        {"role": "user", "content": "ok"},
        {
            "role": "assistant",
            "content": (
                "Moving forward with cache tier review. Redis memory is at 64% capacity. "
                "Eviction policies are functioning as configured with LRU policy active. "
                "No packet drops observed on internal VPC peering connections."
            ),
        },
        {"role": "user", "content": "next"},
    ]

    report = evaluator.evaluate(messages)
    assert report.status == HealthStatus.YELLOW
    assert any(issue.mode == "confusion" for issue in report.detected_issues)
    assert report.penalty_score == 25


def test_intra_message_redundancy_consecutive_repeated_lines():
    """Verify single turn with 120 repeated error lines scores penalty 40 and YELLOW status."""
    evaluator = DeterministicEvaluator()
    repeated_content = "\n".join(["ERROR: connection reset by peer"] * 120)

    messages = [
        {"role": "user", "content": repeated_content},
    ]

    report = evaluator.evaluate(messages)
    assert report.status == HealthStatus.YELLOW
    assert report.penalty_score == 40
    assert len(report.detected_issues) == 1
    assert report.detected_issues[0].mode == "distraction"
    assert "consecutive repeated lines" in report.detected_issues[0].evidence[0]
    assert report.recommended_action == "Trigger background State-Ledger compression."


def test_intra_message_redundancy_low_unique_ratio_over_200_tokens():
    """Verify payload >200 tokens with unique/total line ratio <0.4 scores penalty 40."""
    evaluator = DeterministicEvaluator()
    # 20 lines alternating between 2 lines: unique ratio = 2/20 = 0.10 < 0.40, >200 tokens
    line_a = "WARNING: [worker-node-alpha-42] High heap memory pressure detected in cache segment"
    line_b = "WARNING: [worker-node-beta-17] Rebalance postponed awaiting leader lock release"
    lines = [line_a if i % 2 == 0 else line_b for i in range(20)]
    content = "\n".join(lines)

    messages = [
        {"role": "user", "content": content},
    ]

    report = evaluator.evaluate(messages)
    assert report.status == HealthStatus.YELLOW
    assert report.penalty_score == 40
    assert any(issue.mode == "distraction" for issue in report.detected_issues)


def test_developer_turn_repeated_paths_and_stacktraces_remains_healthy():
    """Verify developer turn with repeated Windows/POSIX paths & stack traces stays GREEN (<25)."""
    evaluator = DeterministicEvaluator()

    developer_content = (
        "Here are the debug logs and terminal output from our build session:\n\n"
        "DEBUG: [worker-1] Loaded config from C:\\Users\\developer\\workspace\\app\\config.yaml\n"
        "DEBUG: [worker-2] Loaded config from C:\\Users\\developer\\workspace\\app\\config.yaml\n"
        "DEBUG: [worker-3] Loaded config from C:\\Users\\developer\\workspace\\app\\config.yaml\n"
        "DEBUG: [worker-4] Loaded config from C:\\Users\\developer\\workspace\\app\\config.yaml\n"
        "INFO: Reading service stream from /var/log/app/service.log\n"
        "INFO: Reading worker stream from /var/log/app/worker.log\n"
        "INFO: Accessing file at file:///c:/Users/developer/workspace/app/config.yaml\n"
        "INFO: Documentation available at https://docs.internal.company.com/service/routing\n"
        "Traceback (most recent call last):\n"
        '  File "C:\\Users\\developer\\workspace\\app\\main.py", line 42, in run\n'
        "    service.start()\n"
        '  File "C:\\Users\\developer\\workspace\\app\\net.py", line 18, in start\n'
        "    sock.bind(('127.0.0.1', 8080))\n"
        "OSError: [Errno 48] Address already in use\n\n"
        "How can we configure fallback port handling in our config?"
    )

    assistant_content = (
        "The OSError indicates port 8080 is already in use by another process. "
        "You can configure a port fallback mechanism in `config.yaml` using dynamic port binding "
        "or pass `--port 8081` as a command line argument."
    )

    messages = [
        {"role": "user", "content": developer_content},
        {"role": "assistant", "content": assistant_content},
    ]

    report = evaluator.evaluate(messages)
    assert report.status == HealthStatus.GREEN
    assert report.penalty_score < 25
    assert not any(issue.mode == "distraction" for issue in report.detected_issues)


def test_code_blocks_with_negative_directives_do_not_trigger_poisoning():
    """Verify code blocks with negative directives ('don't use Flask') do not trigger poisoning."""
    evaluator = DeterministicEvaluator()

    # User message contains negative words and 'don't use Flask' inside code fences
    user_turn = (
        "Here is the architectural verification test suite I wrote:\n\n"
        "```python\n"
        "def test_framework_rules():\n"
        "    # No, don't use Flask in this microservice\n"
        "    code = Path('app.py').read_text()\n"
        "    assert 'flask' not in code\n"
        "    assert 'Flask' not in code\n"
        "```\n\n"
        "Does this assertion correctly verify that our code adheres to the guideline?"
    )

    # Assistant confirms and mentions Flask naturally in response to user's question
    assistant_turn = (
        "Yes, checking that 'flask' and 'Flask' are absent from app.py effectively verifies "
        "that the microservice does not introduce Flask dependencies."
    )

    messages = [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": assistant_turn},
    ]

    report = evaluator.evaluate(messages)
    assert report.status == HealthStatus.GREEN
    assert report.penalty_score == 0
    assert not any(issue.mode == "poisoning" for issue in report.detected_issues)


def test_normalization_and_prose_helpers_unit():
    """Verify _normalize_text_for_entropy and _extract_conversational_prose work as expected."""
    evaluator = DeterministicEvaluator()

    # Test path and URL normalization
    sample_text = (
        "Logs at C:\\Users\\user\\test.log and /var/log/syslog, "
        "see https://example.com/api/v1 and file:///c:/project/file.py."
    )
    normalized = evaluator._normalize_text_for_entropy(sample_text)
    assert "<PATH>" in normalized
    assert "<URL>" in normalized
    assert "C:\\Users" not in normalized
    assert "/var/log" not in normalized
    assert "https://example.com" not in normalized

    # Test conversational prose extraction (stripping code blocks and inline code)
    prose_sample = (
        "Please review this code:\n"
        "```python\n"
        "def test_fn():\n"
        "    # No, don't use Flask\n"
        "    pass\n"
        "```\n"
        "Also check `assert not error` in the test."
    )
    extracted_prose = evaluator._extract_conversational_prose(prose_sample)
    assert "Please review this code:" in extracted_prose
    assert "Also check in the test." in extracted_prose
    assert "don't use Flask" not in extracted_prose
    assert "def test_fn" not in extracted_prose
    assert "assert not error" not in extracted_prose
