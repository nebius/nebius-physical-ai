"""Shared helpers for live npa.workflow infra tests."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pytest
from typer.testing import Result

from npa.clients.config import resolve_project_storage

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECS_DIR = REPO_ROOT / "npa" / "workflows" / "workbench" / "npa-workflows"

ALL_GOLDEN_SPECS = sorted(
    [
        "vlm-eval-single.yaml",
        "tokenfactory-rollout-judge.yaml",
        "tokenfactory-cosmos-gate.yaml",
        "sim2real-vlm-rl.yaml",
        "bdd100k-pipeline.yaml",
    ]
)

DYNAMIC_SPECS = frozenset({"sim2real-vlm-rl.yaml", "tokenfactory-cosmos-gate.yaml"})

_LEAK_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}"),
    re.compile(r"(?i)nebius_api_key\s*[:=]\s*['\"]?[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)hf_[a-z0-9]{20,}"),
)


def assume_decision_for(name: str, *, mode: str = "promote") -> str:
    if name in DYNAMIC_SPECS:
        return "loop_back" if mode == "loop" else "promote_checkpoint"
    return ""


def live_bucket(e2e_project: str | None) -> str:
    storage = resolve_project_storage(e2e_project)
    raw = storage.checkpoint_bucket or ""
    if not raw:
        pytest.fail("checkpoint_bucket is not configured for live npa.workflow tests")
    parsed = urlparse(raw if "://" in raw else f"s3://{raw}")
    bucket = parsed.netloc if parsed.scheme == "s3" else raw.split("/")[0]
    if not bucket:
        pytest.fail(f"could not resolve live bucket from {raw!r}")
    return bucket


def materialize_live_spec(
    tmp_path: Path,
    name: str,
    *,
    bucket: str,
    run_id: str,
) -> Path:
    """Copy a golden spec with the live bucket and a unique e2e prefix."""

    text = (SPECS_DIR / name).read_text(encoding="utf-8")
    text = text.replace("bucket: example-bucket", f"bucket: {bucket}")
    marker = f"npa-workflow-e2e/{run_id}"
    # Keep per-spec prefix tokens but anchor runs under a shared e2e root.
    text = re.sub(
        r'(prefix:\s*")([^"]*)(")',
        lambda m: f'{m.group(1)}{marker}/{name.replace(".yaml", "")}{m.group(3)}',
        text,
        count=1,
    )
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def live_credential_markers() -> list[str]:
    """Collect credential substrings that must never appear in CLI output."""

    markers: list[str] = []
    try:
        from npa.clients.credentials import load_credentials

        storage = load_credentials().get("storage") or {}
        for key in ("aws_access_key_id", "aws_secret_access_key"):
            value = storage.get(key)
            if isinstance(value, str) and len(value) >= 8:
                markers.append(value)
    except Exception:
        pass
    for env_key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "HF_TOKEN",
        "NEBIUS_API_KEY",
    ):
        value = os.environ.get(env_key, "")
        if value and len(value) >= 8:
            markers.append(value)
    return markers


def assert_no_credential_leakage(
    text: str,
    *,
    extra_forbidden: Iterable[str] | None = None,
) -> None:
    """Fail when CLI output contains secrets or live credential material."""

    for pattern in _LEAK_PATTERNS:
        match = pattern.search(text)
        assert match is None, f"credential pattern leaked: {match.group(0)[:32]!r}"
    for marker in extra_forbidden or ():
        if marker and len(marker) >= 8 and marker in text:
            raise AssertionError("live credential marker leaked in CLI output")


def assert_cli_ok(result: Result, *, forbidden: Iterable[str] | None = None) -> None:
    assert result.exit_code == 0, result.output
    assert_no_credential_leakage(result.output, extra_forbidden=forbidden)


def parse_json_output(result: Result, *, forbidden: Iterable[str] | None = None) -> Any:
    assert_cli_ok(result, forbidden=forbidden)
    return json.loads(result.output)


def parse_json_payload(result: Result, forbidden: Iterable[str]) -> dict[str, Any]:
    payload = parse_json_output(result, forbidden=forbidden)
    assert isinstance(payload, dict)
    return payload
