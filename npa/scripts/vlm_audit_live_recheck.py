"""Run configured hosted VLM audits and retain private evidence without skipped proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from npa.workbench.vlm_eval import (
    PREFERENCE_COMPARISON_RESULT_FILENAME,
    VlmPreferenceComparisonRequest,
)
from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    TokenFactoryClient,
    TokenFactoryError,
    resolve_config,
)
from npa.guardrails.confidentiality import compile_builtin_nebius_infra, scan_text
from npa.live_verification.vlm_audit_controls import (
    PREFERENCE_MODELS,
    audit_controls,
    generated_preference_config,
)

SUITES = ("npa/tests/e2e/test_vlm_audits_live.py",)
CONFIG_ENV = "NPA_VLM_AUDIT_LIVE_CONFIG"


class _AuditConfigurationError(ValueError):
    """Carry only a fixed public configuration failure reason."""


def _write_private(path: Path, body: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(body)


def _operator_config() -> dict:
    configured_path = os.environ.get(CONFIG_ENV, "")
    if not configured_path:
        raise _AuditConfigurationError("missing_audit_configuration")
    path = Path(configured_path)
    metadata = path.stat()
    if not path.is_file() or metadata.st_mode & 0o077 or metadata.st_uid != os.getuid():
        raise _AuditConfigurationError("audit_configuration_must_be_owner_only")
    return json.loads(path.read_text())


def _prepare_config(
    target: Path, *, generated: bool = False, audit_kind: str | None = None
) -> Path:
    if audit_kind not in (None, "preference") or (generated and audit_kind is None):
        raise _AuditConfigurationError("unavailable_or_missing_audit_kind")
    config = (
        generated_preference_config(target / "controls")
        if generated
        else _operator_config()
    )
    if set(config["cases"]) != {"blinded-preference"}:
        raise _AuditConfigurationError("unavailable_audit_case")
    case = config["cases"]["blinded-preference"]
    controls = audit_controls(case)
    for index, control in enumerate(controls.values()):
        output = target / "blinded-preference"
        if "controls" in case:
            output = output / str(index)
        _prepare_control(control, output)
    destination = target / "audit-config.json"
    _write_private(destination, json.dumps(config))
    return destination


def _prepare_control(control: dict, output: Path) -> None:
    request = dict(control["request"])
    request["output_path"] = str(output)
    options = VlmPreferenceComparisonRequest(**request)
    if not os.environ.get(options.api_key_env, "").strip():
        raise _AuditConfigurationError("missing_audit_credential")
    expectations = control.get("expectations", {})
    mapped = expectations.get("mapped_preferences")
    if (
        not isinstance(mapped, list)
        or len(mapped) != 2
        or any(
            value not in ("baseline", "candidate", "tie", "unresolved")
            for value in mapped
        )
        or type(expectations.get("escalation_required")) is not bool
    ):
        raise _AuditConfigurationError("missing_frozen_preference_expectations")
    control["request"] = asdict(options)


def _scheduled_preflight() -> None:
    key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    config = resolve_config(api_key=key, base_url=DEFAULT_BASE_URL, environ={})
    available = TokenFactoryClient(config).list_models()
    if not set(PREFERENCE_MODELS).issubset(available):
        raise _AuditConfigurationError("required_preference_model_not_advertised")


def _test_environment(config_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "NPA_CI_SHARD_INDEX",
        "NPA_CI_TOTAL_SHARDS",
        "NPA_DRY_RUN",
    ):
        environment.pop(name, None)
    environment.update(
        NPA_INTEGRATION_E2E="1", NPA_VLM_AUDIT_LIVE_CONFIG=str(config_path)
    )
    return environment


def _execute(root: Path, target: Path, config_path: Path) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--pytest-child",
        str(target),
    ]
    descriptor = os.open(
        target / "pytest.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w") as log:
        return subprocess.run(
            command,
            cwd=root,
            env=_test_environment(config_path),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


class _NoExpectedFailures:
    def __init__(self) -> None:
        self.observed = False
        self.collected = 0
        self.executed = 0
        self.deselected = 0
        self.results: dict[str, set[str]] = {}

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = len(session.items)

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.observed |= hasattr(report, "wasxfail")
        self.executed += report.when == "call"
        results = self.results.setdefault(report.nodeid, set())
        if report.failed:
            results.add("failed")
        if report.skipped:
            results.add("skipped")
        if report.when == "call" and report.passed:
            results.add("passed")

    def counts(self) -> dict[str, int]:
        return {
            "collected": self.collected,
            "executed": self.executed,
            "deselected": self.deselected,
            "passed": sum(row == {"passed"} for row in self.results.values()),
            "failed": sum("failed" in row for row in self.results.values()),
            "skipped": sum("skipped" in row for row in self.results.values()),
        }


def _run_tests(target: Path) -> int:
    failures = _NoExpectedFailures()
    code = pytest.main(
        [
            *SUITES,
            "-q",
            "--tb=short",
            "-o",
            "addopts=",
            "-o",
            "xfail_strict=true",
            "--junitxml",
            str(target / "pytest.xml"),
            "--basetemp",
            str(target / "pytest"),
        ],
        plugins=[failures],
    )
    _write_private(
        target / "execution.json",
        json.dumps(
            {
                **failures.counts(),
                "xfail": failures.observed,
            }
        ),
    )
    incomplete = (
        failures.collected == 0
        or failures.collected != failures.executed
        or failures.deselected
        or failures.observed
    )
    return 1 if incomplete else int(code)


def _verify(
    root: Path,
    target: Path,
    receipt: dict,
    *,
    generated: bool = False,
    audit_kind: str | None = None,
) -> None:
    config_path = _prepare_config(target, generated=generated, audit_kind=audit_kind)
    receipt["audit_kind"] = "preference"
    receipt["configuration_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    if generated:
        _scheduled_preflight()
        receipt["credential_and_catalog_preflight_passed"] = True
    exit_code = _execute(root, target, config_path)
    receipt["pytest_exit_code"] = exit_code
    execution = json.loads((target / "execution.json").read_text())
    counts = {field: execution[field] for field in receipt["counts"]}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise _AuditConfigurationError("invalid_execution_counts")
    receipt["counts"] = counts
    config = json.loads(config_path.read_text())
    expected = sum(len(audit_controls(case)) for case in config["cases"].values())
    receipt["passed"] = (
        exit_code == 0
        and counts["collected"] == counts["executed"] == counts["passed"] == expected
        and counts["collected"] == execution["collected"]
        and counts["failed"] == counts["skipped"] == counts["deselected"] == 0
        and not execution["xfail"]
    )
    receipt["outcomes"] = _public_comparisons(target)


def _public_comparisons(target: Path) -> list[dict]:
    summaries = []
    statuses = {
        "judge_error",
        "unresolved",
        "low_confidence",
        "order_disagreement_or_nondeterminism",
        "consistent_candidate_preference",
        "consistent_baseline_preference",
        "consistent_tie",
    }
    directory = target / "blinded-preference"
    for path in directory.rglob(PREFERENCE_COMPARISON_RESULT_FILENAME):
        index = 0 if path.parent == directory else int(path.parent.name)
        content = path.read_bytes()
        report = json.loads(content)
        summaries.append(
            {
                "control_index": index,
                "artifact_sha256": hashlib.sha256(content).hexdigest(),
                "status": report["status"]
                if report["status"] in statuses
                else "unknown",
                "escalation_required": report["escalation_required"] is True,
                "requests_counterbalanced": report["requests_counterbalanced"] is True,
                "first_order": _public_order(report["first_order"]),
                "reversed_order": _public_order(report["reversed_order"]),
            }
        )
    return sorted(summaries, key=lambda summary: summary["control_index"])


def _public_order(outcome: dict) -> dict:
    provider = outcome.get("provider") or {}
    verdict = outcome.get("verdict") or {}
    error = outcome.get("error") or {}
    error_types = {
        "transport_error",
        "provider_http_status_error",
        "provider_response_decode_error",
        "response_contract_error",
    }
    return {
        "returned_model_sha256": hashlib.sha256(
            str(provider.get("returned_model", "")).encode()
        ).hexdigest(),
        "raw_response_sha256": hashlib.sha256(
            str(provider.get("raw_response", "")).encode()
        ).hexdigest(),
        "preference": verdict.get("preference")
        if verdict.get("preference") in ("A", "B", "tie", "unresolved")
        else None,
        "confidence": verdict.get("confidence")
        if verdict.get("confidence") in ("high", "medium", "low")
        else None,
        "error_type": error.get("error_type")
        if error.get("error_type") in error_types
        else None,
    }


def _prepare_evidence_directory(root: Path, path: Path) -> Path:
    try:
        target = path.resolve()
    except RuntimeError as exc:
        # Python 3.12 reports symlink loops as RuntimeError, not OSError.
        # Normalize only that pathlib failure, not arbitrary runtime bugs.
        if not str(exc).startswith("Symlink loop from "):
            raise
        raise _AuditConfigurationError("invalid_audit_evidence_directory") from None
    except (OSError, ValueError):
        raise _AuditConfigurationError("invalid_audit_evidence_directory") from None
    if target.is_relative_to(root):
        raise _AuditConfigurationError("audit_evidence_must_be_outside_checkout")
    try:
        target.mkdir(parents=True, mode=0o700)
    except (OSError, ValueError):
        raise _AuditConfigurationError("audit_evidence_directory_unavailable") from None
    return target


def main(argv: list[str] | None = None) -> int:
    """Execute the configured audit suite and emit a sanitized summary.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero only when every collected audit executes successfully without skips.
    Raises:
        RuntimeError: An unexpected implementation error occurs, not a path failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-kind",
        help="Generated audit kind; this standalone runner supports preference.",
    )
    parser.add_argument(
        "--generated-controls",
        action="store_true",
        help="Run frozen local visual controls with the protected Token Factory key.",
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    receipt = _new_receipt(root)
    try:
        target = _prepare_evidence_directory(root, args.evidence_dir)
    except _AuditConfigurationError as exc:
        receipt["failure"] = str(exc)
        print(
            json.dumps({key: receipt[key] for key in ("passed", "counts", "failure")})
        )
        return 1
    receipt["control_source"] = (
        "generated-visual-contract-v1" if args.generated_controls else "operator"
    )
    _complete_receipt(
        root,
        target,
        receipt,
        generated=args.generated_controls,
        audit_kind=args.audit_kind,
    )
    _write_receipt(target / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "counts", "failure")}))
    return 0 if receipt["passed"] else 1


def _complete_receipt(
    root: Path,
    target: Path,
    receipt: dict,
    *,
    generated: bool,
    audit_kind: str | None = None,
) -> None:
    try:
        _verify(root, target, receipt, generated=generated, audit_kind=audit_kind)
    except _AuditConfigurationError as exc:
        receipt["failure"] = str(exc)
    except (ValueError, KeyError, TypeError, OSError):
        receipt["failure"] = "audit_configuration_or_execution_failed"
    except TokenFactoryError:
        receipt["failure"] = "audit_provider_preflight_failed"
    receipt["passed"] = receipt["passed"] and receipt["failure"] is None
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()


def _write_receipt(path: Path, receipt: dict) -> None:
    body = json.dumps(receipt, indent=2) + "\n"
    if scan_text(body, compile_builtin_nebius_infra(), source="audit-receipt"):
        raise ValueError("Refusing audit receipt that failed confidentiality checks")
    _write_private(path, body)


def _new_receipt(root: Path) -> dict:
    sources = [
        *SUITES,
        "npa/scripts/vlm_audit_live_recheck.py",
        "npa/tests/conftest.py",
        "npa/tests/e2e/conftest.py",
        ".github/workflows/token-factory-live.yml",
    ]
    sources.extend(
        str(path.relative_to(root)) for path in (root / "npa/src").rglob("*.py")
    )
    return {
        "schema": "npa.vlm_audit.live_recheck.v1",
        "required_live": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_file_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(sources)
        },
        "suites": list(SUITES),
        "passed": False,
        "failure": None,
        "pytest_exit_code": None,
        "counts": {
            "collected": 0,
            "executed": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
        },
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--pytest-child":
        raise SystemExit(_run_tests(Path(sys.argv[2])))
    raise SystemExit(main())
