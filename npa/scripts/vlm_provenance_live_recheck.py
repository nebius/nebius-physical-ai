#!/usr/bin/env python3
"""Verify an operator-owned GPU VLM endpoint and retain a fail-closed receipt."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

import httpx
import pytest

from npa.workbench.vlm_eval import (
    DEFAULT_TIMEOUT_S,
    _chat_completions_url,
    select_rollout_frames,
)
from token_factory_live_recheck import (
    PROVENANCE_SUITE,
    Results,
    source_hashes,
    write_receipt,
)


def _private_path(value: str, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Live paths must be absolute")
    path = path.resolve()
    if path.is_relative_to(root):
        raise ValueError("Live configuration and evidence must be outside the checkout")
    return path


def _load_config(root: Path) -> dict[str, str]:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        raise ValueError(
            "NPA_INTEGRATION_E2E=1 is required; refusing skipped verification"
        )
    value = os.environ.get("NPA_VLM_PROVENANCE_LIVE_CONFIG", "")
    if not value:
        raise ValueError("NPA_VLM_PROVENANCE_LIVE_CONFIG is required")
    path = _private_path(value, root)
    if path.stat().st_mode & 0o077:
        raise ValueError("Live configuration must be owner-only")
    config = json.loads(path.read_text())
    required = (
        "input_path",
        "output_path",
        "endpoint_url",
        "model",
        "expected_served_model",
        "task",
    )
    if not isinstance(config, dict) or any(
        not isinstance(config.get(key), str) or not config[key].strip()
        for key in required
    ):
        raise ValueError("Live configuration is missing required string fields")
    return config


def _check_local_artifacts(config: dict[str, str], root: Path) -> None:
    if any("://" in config[key] for key in ("input_path", "output_path")):
        raise ValueError("This lane requires local fixtures and a local JSON result")
    source = _private_path(config["input_path"], root)
    output = _private_path(config["output_path"], root)
    if output.suffix != ".json" or output.exists():
        raise ValueError("Use a fresh local JSON result filename")
    if not output.parent.is_dir() or output.parent.stat().st_mode & 0o077:
        raise ValueError("The output directory must exist and be owner-only")
    if not select_rollout_frames(source, frame_selection="keyframes", max_frames=8):
        raise ValueError("Live fixture must contain decodable frames")


def _check_endpoint(config: dict[str, str]) -> None:
    parsed = urlsplit(config["endpoint_url"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an HTTP endpoint without embedded credentials or query")
    key_env = config.get("api_key_env", "VLM_EVAL_API_KEY")
    if not isinstance(key_env, str) or not os.environ.get(key_env, "").strip():
        raise ValueError(
            "The configured endpoint credential environment variable is required"
        )
    endpoint = _chat_completions_url(config["endpoint_url"]).removesuffix(
        "/chat/completions"
    )
    response = httpx.get(
        endpoint + "/models",
        headers={"Authorization": f"Bearer {os.environ[key_env]}"},
        timeout=DEFAULT_TIMEOUT_S,
        follow_redirects=False,
    )
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("Endpoint returned an invalid model listing")
    if not any(
        isinstance(row, dict) and row.get("id") == config["expected_served_model"]
        for row in models
    ):
        raise ValueError("Endpoint does not advertise the expected served model")


def _run_live_tests(target: Path, results: Results) -> int:
    # Pytest tracebacks can expose endpoint URLs or operator fixture contents.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return int(
            pytest.main(
                [
                    PROVENANCE_SUITE,
                    "-q",
                    "--tb=no",
                    "--basetemp",
                    str(target / "pytest"),
                    "-o",
                    "addopts=",
                ],
                plugins=[results],
            )
        )


def _verify(root: Path, target: Path, receipt: dict, results: Results) -> int:
    receipt["phase"] = "configuration"
    config = _load_config(root)
    receipt["phase"] = "fixtures"
    _check_local_artifacts(config, root)
    receipt["phase"] = "endpoint"
    _check_endpoint(config)
    receipt["endpoint_preflight_passed"] = True
    # A model listing is readiness only; only executed inference can pass.
    receipt["phase"] = "inference"
    return _run_live_tests(target, results)


def _receipt(root: Path) -> dict:
    hashes = source_hashes(root)
    for name in (PROVENANCE_SUITE, "npa/scripts/vlm_provenance_live_recheck.py"):
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {
        "schema": "npa.vlm_provenance.live_recheck.v1",
        "commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_file_sha256": hashes,
        "suites": [PROVENANCE_SUITE],
        "resource_lifecycle": "Operator provisions and cleans up the endpoint; runner performs inference only.",
        "required_live": True,
        "endpoint_preflight_passed": False,
        "passed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the configured provenance suite with sanitized failure reporting.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero only when every required live test passed and evidence was saved.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        parser.error("Run from the repository root using npa/.venv/bin/python")
    try:
        target = _private_path(str(args.evidence_dir), root)
        target.mkdir(parents=True, mode=0o700, exist_ok=False)
        receipt = _receipt(root)
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"passed": False, "failure": "Private evidence setup failed"}))
        return 1
    results = Results((PROVENANCE_SUITE,), provider_contract=False)
    exit_code = 2
    try:
        exit_code = _verify(root, target, receipt, results)
    except (OSError, ValueError, httpx.HTTPError, TypeError, KeyError):
        receipt["failure"] = (
            "Live verification failed; see phase for the failed prerequisite"
        )
    receipt.update(
        completed_at=datetime.now(timezone.utc).isoformat(),
        pytest_exit_code=exit_code,
        counts=results.summary(),
        tests=list(results.reports.values()),
        passed=results.complete(exit_code),
    )
    try:
        write_receipt(target / "receipt.json", receipt)
    except (OSError, ValueError, TypeError):
        print(json.dumps({"passed": False, "failure": "Private receipt write failed"}))
        return 1
    print(json.dumps({key: receipt[key] for key in ("passed", "counts")}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
