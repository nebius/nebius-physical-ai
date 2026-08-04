"""Guardrail: agent + insights source must not embed answers, data, or infra.

Every value in an agent reply must originate from a live tool observation
(insights query, S3 listing, validate/plan) — never a constant. This test scans
production source and fails on seeded run names / answer lists and on hardcoded
insights endpoint/port/token literals. Such values belong in tests only.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "npa"
_INSIGHTS_DIR = _SRC / "workbench" / "insights"

# Demo run names that were only ever seeded into an S3 store / used as test
# fixtures. They must never appear in production source.
_SEEDED_RUN_NAMES = ("candidate-4gpu", "hardened-4gpu", "baseline-2gpu", "run-4gpu", "run-h100")


def _agent_sources() -> list[Path]:
    # Scan both historical CLI modules/shims and the shipped backend package.
    # Moving logic out of cli/ must never move it out of this guard's scope.
    return sorted(
        [
            *(_SRC / "cli").glob("agent*.py"),
            *(_SRC / "agent_backend").glob("*.py"),
        ]
    )


def _insights_sources() -> list[Path]:
    return sorted(_INSIGHTS_DIR.glob("*.py"))


def _all_sources() -> list[Path]:
    return _agent_sources() + _insights_sources()


def test_no_seeded_run_names_in_production_source() -> None:
    offenders: list[str] = []
    for path in _all_sources():
        text = path.read_text(encoding="utf-8")
        for name in _SEEDED_RUN_NAMES:
            if name in text:
                offenders.append(f"{path.name}: {name!r}")
    assert not offenders, f"seeded run names leaked into src/: {offenders}"


def test_no_hardcoded_metric_answer_lists() -> None:
    # A literal list of run ids as an answer (e.g. ["candidate-4gpu", ...]) is a
    # fabrication smell. Fail on any list literal mixing the seeded names.
    for path in _all_sources():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\[[^\]]*candidate-4gpu[^\]]*\]", text), path.name
        assert not re.search(r"\[[^\]]*hardened-4gpu[^\]]*\]", text), path.name


def test_no_hardcoded_insights_endpoint_or_port_in_agent() -> None:
    # The agent resolves the insights endpoint/token from config/env, never a
    # literal URL, port, or bearer token embedded in agent source.
    for path in _agent_sources():
        text = path.read_text(encoding="utf-8")
        assert ":8793" not in text, f"{path.name} hardcodes the insights port"
        assert not re.search(r"https?://[^\"'\s]*insights", text, re.IGNORECASE), path.name
        # No embedded long token/secret literal.
        assert not re.search(r"INSIGHTS_TOKEN\s*=\s*[\"'][A-Za-z0-9._-]{12,}[\"']", text), path.name


def test_insights_endpoint_and_token_are_env_resolved() -> None:
    # The resolution helper must read from the environment, not string literals.
    agent = (_SRC / "cli" / "agent.py").read_text(encoding="utf-8")
    assert "NPA_INSIGHTS_ENDPOINT" in agent
    assert "NPA_INSIGHTS_STORE_URI" in agent
    assert "INSIGHTS_TOKEN" in agent  # token env var name only, resolved at runtime


def test_which_runs_used_4_gpus_still_routes_to_insights() -> None:
    # Guard the working GPU query: it must NOT match a grounded run-listing intent
    # (falls through so the insights loop answers it from real data).
    from npa.cli import agent_chat
    from npa.cli import agent_semantic_router as sr

    assert agent_chat.match_chat_intent("which runs used 4 gpus") not in agent_chat._RUN_LISTING_INTENTS
    result = sr.classify_intent_semantic(
        "which runs used 4 gpus",
        known_intents=frozenset(agent_chat.INTENT_APIS.keys()),
        model_call=lambda *a, **k: {"choices": [{"message": {"content": "{}"}}], "usage": {}},
    )
    assert result["mode"] == sr.MODE_ACTION
    assert result["tokens"] == 0
