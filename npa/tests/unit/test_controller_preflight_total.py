from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from npa import controller_ownership as ownership
from npa.controller_ownership import ControllerOwner


def _owner(alias: str = "demo", suffix: str = "a") -> ControllerOwner:
    return ControllerOwner(
        project_alias=alias,
        project_id=f"project-{suffix}",
        cluster_id=f"cluster-{suffix}",
        cluster_name=f"cluster-{suffix}",
        context=f"ctx-{suffix}",
        context_fingerprint=f"fingerprint-{suffix}",
    )


def test_multiple_legacy_owners_return_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "projects": {
                    "one": {"controller_owner": _owner("one", "a").to_dict()},
                    "two": {"controller_owner": _owner("two", "b").to_dict()},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ownership, "CONFIG_PATH", path)
    status, reason = ownership.controller_preflight("one", "ctx-a")
    assert status == "blocked"
    assert "Multiple legacy controller owners" in reason


def test_corrupt_owner_yaml_returns_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("projects: [unterminated", encoding="utf-8")
    monkeypatch.setattr(ownership, "CONFIG_PATH", path)
    status, reason = ownership.controller_preflight("demo", "ctx-a")
    assert status == "blocked"
    assert "could not be read" in reason


def test_explicit_candidate_disagreement_returns_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    existing = _owner("other", "b")
    candidate = _owner("demo", "a")
    monkeypatch.setattr(ownership, "controller_owner", lambda **_kwargs: existing)
    monkeypatch.setattr(ownership, "load_cluster_state", lambda _context: object())
    monkeypatch.setattr(ownership, "resolve_controller_candidate", lambda *_args: candidate)
    monkeypatch.setattr(ownership, "verify_live_controller_candidate", lambda value: value)
    status, reason = ownership.controller_preflight("demo", "ctx-a")
    assert status == "blocked"
    assert "shared controller belongs" in reason


def test_valid_unique_owner_returns_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = _owner("demo", "a")
    monkeypatch.setattr(ownership, "controller_owner", lambda **_kwargs: candidate)
    monkeypatch.setattr(ownership, "load_cluster_state", lambda _context: object())
    monkeypatch.setattr(ownership, "resolve_controller_candidate", lambda *_args: candidate)
    monkeypatch.setattr(ownership, "verify_live_controller_candidate", lambda value: value)
    status, reason = ownership.controller_preflight("demo", "ctx-a")
    assert status == "ready"
    assert candidate.cluster_id in reason


def test_untyped_owner_failure_keeps_context_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs):
        raise RuntimeError("unexpected parser failure")

    monkeypatch.setattr(ownership, "controller_owner", fail)
    status, reason = ownership.controller_preflight("demo", "ctx-a")
    assert status == "unknown"
    assert "project='demo'" in reason
    assert "context='ctx-a'" in reason
    assert "RuntimeError" in reason
