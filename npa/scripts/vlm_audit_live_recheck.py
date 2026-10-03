"""Run configured hosted VLM audits and retain private evidence without skipped proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from npa.workbench import vlm_eval
from npa.workbench.vlm_eval import (
    HOSTED_RESPONSE_PARSER_VERSION,
    JUDGE_COMPARISON_RESULT_FILENAME,
    VlmJudgeComparisonRequest,
)
from npa.workflows.vlm_grade_evidence import vlm_grade_block_details
from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    TokenFactoryClient,
    TokenFactoryError,
    resolve_config,
)
from npa.guardrails.confidentiality import compile_builtin_nebius_infra, scan_text
from npa.literal_values import require_boolean, require_integer, require_number
from npa.live_verification.vlm_audit_controls import (
    PAIRED_MODELS,
    audit_controls,
    configured_audit_cases,
    generated_paired_config,
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


def _prepare_config(target: Path, *, generated: bool = False) -> Path:
    config = (
        generated_paired_config(target / "controls")
        if generated
        else _operator_config()
    )
    try:
        selected = configured_audit_cases(
            config, available_cases=("paired-judges",), required_kind="paired"
        )
    except ValueError:
        raise _AuditConfigurationError("invalid_audit_case_selection") from None
    case = config["cases"][selected[0]]
    controls = audit_controls(case)
    for index, control in enumerate(controls.values()):
        output = target / "paired-judges"
        if "controls" in case:
            output = output / str(index)
        _prepare_control(control, output)
    destination = target / "audit-config.json"
    _write_private(destination, json.dumps(config))
    return destination


def _prepare_control(control: dict, output: Path) -> None:
    request = dict(control["request"])
    request["output_path"] = str(output)
    options = VlmJudgeComparisonRequest(**request)
    if not os.environ.get(options.api_key_env, "").strip():
        raise _AuditConfigurationError("missing_audit_credential")
    expectations = control.get("expectations", {})
    if not isinstance(expectations, dict):
        raise _AuditConfigurationError("missing_frozen_judge_expectations")
    required = ("primary.result.passed", "secondary.result.passed")
    optional = (
        "passed",
        "escalation_required",
        "requests_differ_only_by_model",
        "operational_rate_estimated",
    )
    try:
        for field in (*required, *(name for name in optional if name in expectations)):
            require_boolean(expectations.get(field), field=field)
    except ValueError:
        raise _AuditConfigurationError("missing_frozen_judge_expectations") from None
    control["request"] = request


def _scheduled_preflight() -> None:
    key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    config = resolve_config(api_key=key, base_url=DEFAULT_BASE_URL, environ={})
    available = TokenFactoryClient(config).list_models()
    if not set(PAIRED_MODELS).issubset(available):
        raise _AuditConfigurationError("required_paired_model_not_advertised")


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
        self.passed: set[str] = set()
        self.failed: set[str] = set()
        self.skipped: set[str] = set()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.failed.add(report.nodeid)
        elif report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.collected = len(session.items)

    def pytest_deselected(self, items: list[pytest.Item]) -> None:
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.observed |= hasattr(report, "wasxfail")
        self.executed += report.when == "call"
        if report.failed:
            self.failed.add(report.nodeid)
        elif report.skipped:
            self.skipped.add(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)

    def counts(self) -> dict[str, int]:
        return {
            "collected": self.collected,
            "executed": self.executed,
            "deselected": self.deselected,
            "passed": len(self.passed - self.failed - self.skipped),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
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
        or failures.failed
        or failures.skipped
    )
    return 1 if incomplete else int(code)


def _counts(execution: dict) -> dict[str, int]:
    fields = ("collected", "executed", "passed", "failed", "skipped", "deselected")
    if not isinstance(execution, dict):
        raise _AuditConfigurationError("invalid_audit_execution_counts")
    try:
        require_boolean(execution.get("xfail"), field="xfail")
        return {
            field: require_integer(execution.get(field), field=field, minimum=0)
            for field in fields
        }
    except ValueError:
        raise _AuditConfigurationError("invalid_audit_execution_counts") from None


def _verify(
    root: Path, target: Path, receipt: dict, *, generated: bool = False
) -> None:
    config_path = _prepare_config(target, generated=generated)
    config_bytes = _read_private_artifact(config_path, target)
    receipt["configuration_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    paired_bindings = _paired_bindings(config)
    if generated:
        _scheduled_preflight()
        receipt["credential_and_catalog_preflight_passed"] = True
    exit_code = _execute(root, target, config_path)
    receipt["pytest_exit_code"] = exit_code
    execution = json.loads(_read_private_artifact(target / "execution.json", target))
    counts = _counts(execution)
    receipt["counts"] = counts
    if _read_private_artifact(config_path, target) != config_bytes:
        raise _AuditConfigurationError("audit_configuration_changed")
    expected = sum(len(audit_controls(case)) for case in config["cases"].values())
    outcomes = _public_comparisons(target)
    receipt["outcomes"] = outcomes
    complete = _artifacts_complete(config, outcomes, target, paired_bindings)
    if not complete:
        receipt["failure"] = "missing_or_unexpected_audit_artifacts"
    receipt["passed"] = (
        exit_code == 0
        and counts["collected"] == counts["executed"] == counts["passed"] == expected
        and counts["failed"] == counts["skipped"] == counts["deselected"] == 0
        and not execution["xfail"]
        and complete
        and all(
            outcome[judge]["error_type"] is None
            for outcome in outcomes
            for judge in ("primary", "secondary")
        )
        and all(outcome["status"] != "judge_identity_collision" for outcome in outcomes)
    )


def _paired_bindings(config: dict) -> list[dict]:
    return [
        _paired_control_binding(control)
        for control in audit_controls(config["cases"]["paired-judges"]).values()
    ]


def _paired_control_binding(control: dict) -> dict:
    options = dict(control["request"])
    request = VlmJudgeComparisonRequest(**options)
    models = vlm_eval._comparison_models(request.primary_model, request.secondary_model)
    values = asdict(request)
    for field in ("primary_model", "secondary_model", "rubric_path"):
        values.pop(field)
    values["rubric"] = vlm_eval._load_rubric(
        rubric=request.rubric, rubric_path=request.rubric_path
    )
    with vlm_eval._materialized_input(request.input_path) as local:
        if "input_sha256" in control:
            if (
                hashlib.sha256(local.read_bytes()).hexdigest()
                != control["input_sha256"]
            ):
                raise _AuditConfigurationError("audit_control_input_changed")
        context = vlm_eval._comparison_context(local_input=local, **values)
    common = vlm_eval._common_hosted_request(
        prompt=context.prompt, frames=context.frames
    )
    return {
        "report": _paired_report_binding(context, common),
        "judges": [
            _paired_judge_binding(context, vlm_eval._request_for_model(common, model))
            for model in models
        ],
    }


def _paired_report_binding(context, common: dict) -> dict:
    return {
        "schema_version": vlm_eval.JUDGE_COMPARISON_SCHEMA_VERSION,
        "deployment_status": "audit_only",
        "operational_rate_estimated": False,
        "requests_differ_only_by_model": True,
        "input_path": context.input_path,
        "output_path": context.output_path,
        "result_uri": vlm_eval.judge_comparison_result_uri_for(context.output_path),
        "task": context.task,
        "rubric": context.rubric,
        "success_threshold": context.success_threshold,
        "frame_selection": context.frame_selection,
        "frame_count": len(context.frames),
        "shared_frame_sha256": [
            hashlib.sha256(frame.data).hexdigest() for frame in context.frames
        ],
        "common_request_sha256": vlm_eval._sha256_json(common),
        "limitations": list(vlm_eval._comparison_limitations()),
    }


def _paired_judge_binding(context, request: dict) -> dict:
    evidence = asdict(vlm_eval._comparison_request_evidence(request, context))
    evidence.pop("requested_at")
    return {
        "model": request["model"],
        "transport_request_sha256": vlm_eval._sha256_json(request),
        "request": json.loads(json.dumps(evidence)),
    }


def _artifacts_complete(
    config: dict, outcomes: list[dict], target: Path, paired_bindings: list[dict]
) -> bool:
    controls = audit_controls(config["cases"]["paired-judges"])
    if [outcome["control_index"] for outcome in outcomes] != list(range(len(controls))):
        return False
    for index, control in enumerate(controls.values()):
        path = (
            Path(control["request"]["output_path"]) / JUDGE_COMPARISON_RESULT_FILENAME
        )
        _require_artifact_scope(path, target)
        content = _read_private_artifact(path, target)
        if hashlib.sha256(content).hexdigest() != outcomes[index]["artifact_sha256"]:
            return False
        report = json.loads(content)
        if not _retained_matches(control, report):
            return False
        _require_paired_binding(report, paired_bindings[index])
    return True


def _require_paired_binding(report: dict, binding: dict) -> None:
    for field, expected in binding["report"].items():
        _require_bound_value(report.get(field), expected)
    if "mean_score" in report:
        raise _AuditConfigurationError("paired_request_binding_mismatch")
    for judge, expected in zip(
        ("primary", "secondary"), binding["judges"], strict=True
    ):
        _require_judge_binding(report[judge], expected, binding["report"])
    if all(
        report[judge].get("result") is not None for judge in ("primary", "secondary")
    ):
        delta = round(
            report["secondary"]["result"]["score"]
            - report["primary"]["result"]["score"],
            4,
        )
        _require_bound_value(report.get("score_delta_secondary_minus_primary"), delta)


def _require_judge_binding(outcome: dict, expected: dict, report_binding: dict) -> None:
    for field in ("model", "transport_request_sha256"):
        _require_bound_value(outcome.get(field), expected[field])
    result = outcome.get("result")
    if result is None:
        return
    for field in (
        "input_path",
        "output_path",
        "result_uri",
        "task",
        "rubric",
        "success_threshold",
        "frame_selection",
        "frame_count",
    ):
        _require_bound_value(result.get(field), report_binding[field])
    request = dict(result["evidence"]["request"])
    request.pop("requested_at", None)
    _require_bound_value(request, expected["request"])


def _require_bound_value(actual, expected) -> None:
    if type(expected) is bool:
        require_boolean(actual, field="paired_binding")
    elif type(expected) is int:
        require_integer(actual, field="paired_binding")
    elif type(expected) is float:
        require_number(actual, field="paired_binding")
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise _AuditConfigurationError("paired_request_binding_mismatch")


def _read_private_artifact(path: Path, target: Path) -> bytes:
    parts = path.relative_to(target).parts
    if not parts or any(part in (".", "..") for part in parts):
        raise _AuditConfigurationError("invalid_audit_artifact_path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(target, flags)
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_mode & 0o077
                or metadata.st_uid != os.getuid()
            ):
                raise _AuditConfigurationError("invalid_audit_artifact_file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read()
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _require_artifact_scope(path: Path, target: Path) -> None:
    relative = path.relative_to(target)
    current = target
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise _AuditConfigurationError("invalid_audit_artifact_path")


def _retained_matches(control: dict, report: dict) -> bool:
    for field, expected in control["expectations"].items():
        actual = report
        for part in field.split("."):
            actual = actual[part]
        if type(expected) is bool:
            require_boolean(actual, field=field)
        if actual != expected:
            return False
    for judge in ("primary", "secondary"):
        outcome = report[judge]
        if outcome["model"] != control["request"][f"{judge}_model"]:
            return False
        if outcome.get("result") is not None:
            _validate_retained_judge(outcome)
    return True


def _validate_retained_judge(outcome: dict) -> None:
    result = outcome["result"]
    # Reuse the retained-evidence contract, not the promotion action. The paired
    # report remains audit-only even when both individual evidence records validate.
    if (
        result.get("backend") != "api"
        or result.get("model") != outcome["model"]
        or vlm_grade_block_details(result)
    ):
        raise _AuditConfigurationError("invalid_retained_judge_evidence")
    provider = result["evidence"]["provider"]
    if (
        provider["status_code"] != 200
        or provider["parser_version"] != HOSTED_RESPONSE_PARSER_VERSION
        or not isinstance(provider["provider_request_id"], str)
        or not provider["provider_request_id"].strip()
    ):
        raise _AuditConfigurationError("invalid_retained_judge_evidence")
    usage = provider["usage"]
    raw_usage = json.loads(provider["raw_response"]).get("usage")
    if not isinstance(usage, dict) or not isinstance(raw_usage, dict):
        raise _AuditConfigurationError("invalid_retained_judge_evidence")
    try:
        for field in ("prompt_tokens", "completion_tokens"):
            require_integer(usage.get(field), field=field, minimum=1)
            require_integer(raw_usage.get(field), field=field, minimum=1)
    except ValueError:
        raise _AuditConfigurationError("invalid_retained_judge_evidence") from None


def _public_comparisons(target: Path) -> list[dict]:
    summaries = []
    directory = target / "paired-judges"
    _require_artifact_scope(directory, target)
    for path in directory.rglob(JUDGE_COMPARISON_RESULT_FILENAME):
        _require_artifact_scope(path, target)
        index = 0 if path.parent == directory else int(path.parent.name)
        content = _read_private_artifact(path, target)
        summary = _public_comparison(json.loads(content))
        summary.update(
            control_index=index, artifact_sha256=hashlib.sha256(content).hexdigest()
        )
        summaries.append(summary)
    return sorted(summaries, key=lambda summary: summary["control_index"])


def _public_comparison(report: dict) -> dict:
    statuses = {
        "judge_error",
        "judge_disagreement",
        "judge_identity_collision",
        "judges_agree_passed",
        "judges_agree_needs_iteration",
    }
    if report["status"] not in statuses:
        raise _AuditConfigurationError("invalid_audit_report_status")
    summary = {
        "status": report["status"],
        "passed": require_boolean(report["passed"], field="passed"),
        "escalation_required": require_boolean(
            report["escalation_required"], field="escalation_required"
        ),
        "primary": _public_judge(report["primary"]),
        "secondary": _public_judge(report["secondary"]),
    }
    _validate_summary_status(summary)
    return summary


def _validate_summary_status(summary: dict) -> None:
    primary, secondary = summary["primary"], summary["secondary"]
    if primary["error_type"] is not None or secondary["error_type"] is not None:
        status, passed, escalation = "judge_error", False, True
    elif summary["status"] == "judge_identity_collision":
        status, passed, escalation = "judge_identity_collision", False, True
    elif primary["passed"] != secondary["passed"]:
        status, passed, escalation = "judge_disagreement", False, True
    else:
        passed = primary["passed"]
        status = "judges_agree_passed" if passed else "judges_agree_needs_iteration"
        escalation = False
    if (summary["status"], summary["passed"], summary["escalation_required"]) != (
        status,
        passed,
        escalation,
    ):
        raise _AuditConfigurationError("inconsistent_audit_report_status")


def _public_judge(outcome: dict) -> dict:
    result, error, score, passed = _judge_result(outcome)
    provider = _judge_provider(outcome, result, error)
    return {
        "requested_model_sha256": hashlib.sha256(
            str(outcome.get("model", "")).encode()
        ).hexdigest(),
        "returned_model_sha256": hashlib.sha256(
            str(provider.get("returned_model", "")).encode()
        ).hexdigest(),
        "raw_response_sha256": hashlib.sha256(
            str(provider.get("raw_response", "")).encode()
        ).hexdigest(),
        "score": score,
        "passed": passed,
        "error_type": error.get("error_type"),
    }


def _judge_provider(outcome: dict, result: dict, error: dict) -> dict:
    if not error:
        evidence = result.get("evidence")
        if not isinstance(evidence, dict):
            raise _AuditConfigurationError("invalid_audit_judge_provider")
        provider = evidence.get("provider")
    else:
        provider = error.get("provider")
    if provider is None and error:
        return {}
    if not isinstance(provider, dict):
        raise _AuditConfigurationError("invalid_audit_judge_provider")
    for field in ("returned_model", "raw_response"):
        value = provider.get(field)
        if value is not None and not isinstance(value, str):
            raise _AuditConfigurationError("invalid_audit_judge_provider")
        if not error and (not value or not value.strip()):
            raise _AuditConfigurationError("invalid_audit_judge_provider")
    if not error and provider["returned_model"] != outcome.get("model"):
        raise _AuditConfigurationError("invalid_audit_judge_provider")
    return provider


def _judge_result(outcome: dict) -> tuple[dict, dict, int | float | None, bool | None]:
    if not isinstance(outcome, dict):
        raise _AuditConfigurationError("invalid_audit_judge_summary")
    result = outcome.get("result")
    error = outcome.get("error")
    if (result is None) == (error is None):
        raise _AuditConfigurationError("invalid_audit_judge_summary")
    if result is not None:
        if not isinstance(result, dict):
            raise _AuditConfigurationError("invalid_audit_judge_summary")
        passed = require_boolean(result.get("passed"), field="judge.result.passed")
        score = require_number(result.get("score"), field="score", minimum=0, maximum=1)
        return result, {}, score, passed
    error_types = {
        "transport_error",
        "provider_http_status_error",
        "provider_response_decode_error",
        "response_contract_error",
    }
    if not isinstance(error, dict) or error.get("error_type") not in error_types:
        raise _AuditConfigurationError("invalid_audit_judge_error")
    return {}, error, None, None


def main(argv: list[str] | None = None) -> int:
    """Execute the configured audit suite and emit a sanitized summary.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero only when every collected audit executes successfully without skips.
    Raises:
        SystemExit: The evidence directory is invalid or cannot be created.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-kind", help="Explicit generated audit selection: paired."
    )
    parser.add_argument(
        "--generated-controls",
        action="store_true",
        help="Run frozen local visual controls with the protected Token Factory key.",
    )
    args = parser.parse_args(argv)
    if (args.generated_controls and args.audit_kind != "paired") or (
        args.audit_kind is not None and args.audit_kind != "paired"
    ):
        parser.error("generated audits require supported --audit-kind paired")
    root = Path(__file__).resolve().parents[2]
    try:
        target = _evidence_target(args.evidence_dir, root)
    except _AuditConfigurationError as exc:
        parser.error(str(exc))
    receipt = _new_receipt(root)
    receipt["audit_kind"] = "paired"
    receipt["control_source"] = (
        "generated-visual-contract-v1" if args.generated_controls else "operator"
    )
    _complete_receipt(root, target, receipt, generated=args.generated_controls)
    _write_receipt(target / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("passed", "counts", "failure")}))
    return 0 if receipt["passed"] else 1


def _evidence_target(path: Path, root: Path) -> Path:
    try:
        target = path.resolve()
    except (OSError, RuntimeError):
        # Python 3.12 reports symlink loops as RuntimeError, including the path.
        # Limit that catch to resolution; unrelated runtime bugs must propagate.
        raise _AuditConfigurationError("audit_evidence_path_invalid") from None
    if target.is_relative_to(root):
        raise _AuditConfigurationError("audit evidence must be outside the checkout")
    try:
        target.mkdir(parents=True, mode=0o700)
    except OSError:
        raise _AuditConfigurationError("audit_evidence_directory_unavailable") from None
    return target


def _complete_receipt(
    root: Path, target: Path, receipt: dict, *, generated: bool
) -> None:
    try:
        _verify(root, target, receipt, generated=generated)
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
