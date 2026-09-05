#!/usr/bin/env python3
"""Run every migration live suite and emit one sanitized fail-closed receipt.

Use npa/.venv/bin/python from the repository root. The same entrypoint runs in
the protected GitHub job or in operator automation; the receipt distinguishes
them. Raw pytest output and provider response bodies are never published.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import pytest

from npa.guardrails.confidentiality import compile_builtin_nebius_infra, scan_text

SUITES = (
    "npa/tests/e2e/test_token_factory_e2e.py",
    "npa/tests/e2e/test_hosted_rollout_e2e.py",
    "npa/tests/e2e/test_agent_token_factory_e2e.py",
)


class Results:
    def __init__(self) -> None:
        self.collected: list[str] = []
        self.reports: dict[str, dict[str, Any]] = {}
        self.contract: dict[str, Any] = {}
        self.collection_errors = 0

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        row = self.reports.setdefault(report.nodeid, {
            "nodeid": report.nodeid, "executed": False, "passed": False,
            "failed": False, "skipped": False,
        })
        row["executed"] |= report.when == "call"
        row["failed"] |= report.failed
        row["skipped"] |= report.skipped or hasattr(report, "wasxfail")
        row["passed"] |= report.when == "call" and report.passed
        for name, value in report.user_properties:
            if name == "provider_contract":
                self.contract = value

    def summary(self) -> dict[str, int]:
        rows = list(self.reports.values())
        return {
            "collected": len(self.collected),
            "executed": sum(row["executed"] for row in rows),
            "passed": sum(row["passed"] and not row["failed"] and not row["skipped"] for row in rows),
            "failed": sum(row["failed"] for row in rows),
            "skipped": sum(row["skipped"] for row in rows),
            "collection_errors": self.collection_errors,
        }

    def complete(self, exit_code: int) -> bool:
        counts = self.summary()
        all_suites = all(any(
            node.startswith((suite + "::", suite.removeprefix("npa/") + "::"))
            for node in self.collected
        ) for suite in SUITES)
        return bool(
            exit_code == 0 and all_suites and counts["collected"] > 0
            and counts["collected"] == counts["executed"] == counts["passed"]
            and counts["failed"] == counts["skipped"] == counts["collection_errors"] == 0
            and self.contract.get("passed")
        )


def write_receipt(path: Path, receipt: dict) -> None:
    if scan_text(json.dumps(receipt), compile_builtin_nebius_infra(), source="provider-receipt"):
        raise ValueError("Refusing receipt that failed the infrastructure confidentiality guard")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Create exclusively: a second invocation must have its own evidence path.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")


def source_hashes(root: Path) -> dict[str, str]:
    """Link tested runtime bytes to a later docs-only final commit if needed."""
    paths = set((root / "npa/src").rglob("*.py"))
    paths.update(root / suite for suite in SUITES)
    paths.update(root / path for path in (
        "npa/tests/conftest.py", "npa/tests/e2e/conftest.py",
        "npa/scripts/token_factory_live_recheck.py", "npa/scripts/audit_agent_capabilities.py",
        "npa/pyproject.toml",
    ))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        parser.error("Run from the repository root using npa/.venv/bin/python")
    target = args.evidence_dir.resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(target, 0o700)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    scope = os.environ.get("NPA_TF_RECHECK_SCOPE", "authorized-account")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", scope):
        parser.error("NPA_TF_RECHECK_SCOPE must be a non-identifying lowercase role label")
    execution = os.environ.get("NPA_TF_RECHECK_EXECUTION", "local-manual")
    if execution not in {"local-manual", "operator-automation"}:
        parser.error("NPA_TF_RECHECK_EXECUTION must be local-manual or operator-automation")
    endpoint = os.environ.get("NEBIUS_TOKEN_FACTORY_BASE_URL", "") or os.environ.get(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
    )
    receipt: dict[str, Any] = {
        "schema": "npa.token_factory.live_recheck.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": sha,
        "execution": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else
                     execution,
        "github_run_id": os.environ.get("GITHUB_RUN_ID") if os.environ.get("GITHUB_ACTIONS") == "true" else None,
        "scope_role": "authorized-account",
        "scope_label_sha256": hashlib.sha256(scope.encode()).hexdigest(),
        "endpoint_sha256": hashlib.sha256(endpoint.rstrip("/").encode()).hexdigest(),
        "scope_limit": "One configured credential and endpoint; no cross-account or region coverage implied.",
        "python_version": sys.version.split()[0],
        "pytest_version": pytest.__version__,
        "source_file_sha256": source_hashes(root),
        "required_live": True,
        "credential_present": bool(os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()),
        "suites": list(SUITES),
        "passed": False,
    }
    results = Results()
    exit_code = 2
    if not receipt["credential_present"]:
        receipt["failure"] = "Required live Token Factory job needs NEBIUS_TOKEN_FACTORY_KEY; refusing skipped verification."
    else:
        # Failure tracebacks can contain provider bodies or endpoint URLs. Keep
        # only allowlisted observations/node outcomes in the exported receipt.
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                exit_code = int(pytest.main([
                    *SUITES, "--require-token-factory-live", "-q", "--tb=no",
                    "--basetemp", str(target / "pytest"), "-o", "addopts=",
                ], plugins=[results]))
        except Exception as exc:
            receipt["failure"] = "pytest invocation failed before normal completion"
            receipt["error_type"] = type(exc).__name__
            exit_code = 3
        receipt["passed"] = results.complete(exit_code)
    receipt.update(
        completed_at=datetime.now(timezone.utc).isoformat(),
        pytest_exit_code=exit_code, counts=results.summary(),
        tests=list(results.reports.values()), provider_contract=results.contract,
    )
    write_receipt(target / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "counts": receipt["counts"],
                      "failure": receipt.get("failure"), "evidence": "receipt.json"}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
