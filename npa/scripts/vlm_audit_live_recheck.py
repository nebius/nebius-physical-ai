"""Run configured hosted VLM audits and retain private evidence without skipped proof."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator

from npa.workbench import vlm_eval
from npa.workflows import vlm_grade_evidence
from npa.workbench.vlm_eval import (
    HOSTED_RESPONSE_PARSER_VERSION,
    PREFERENCE_COMPARISON_RESULT_FILENAME,
    JUDGE_COMPARISON_RESULT_FILENAME,
    VlmJudgeComparisonRequest,
    VlmPreferenceComparisonRequest,
)
from npa.clients.token_factory import (
    DEFAULT_BASE_URL,
    TokenFactoryClient,
    TokenFactoryError,
    resolve_config,
)
from npa.guardrails.confidentiality import compile_builtin_nebius_infra, scan_text
from npa.literal_values import require_boolean, require_integer, require_number
from npa.workbench.vlm_eval.preference_schema import preference_response_schema
from npa.live_verification.vlm_preference_evidence import (
    frozen_preference_expectations,
    validate_preference_report,
)
from npa.live_verification.vlm_audit_controls import (
    PREFERENCE_MODELS,
    PAIRED_MODELS,
    AUDIT_CASES_BY_KIND,
    generated_paired_config,
    audit_controls,
    configured_audit_cases,
    generated_preference_config,
)

SUITES = ("npa/tests/e2e/test_vlm_audits_live.py",)
CONFIG_ENV = "NPA_VLM_AUDIT_LIVE_CONFIG"
_VERDICT_VALIDATOR = Draft202012Validator(preference_response_schema())


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
    if audit_kind not in (None, *AUDIT_CASES_BY_KIND) or (
        generated and audit_kind is None
    ):
        raise _AuditConfigurationError("unavailable_or_missing_audit_kind")
    config = (
        (
            generated_preference_config
            if audit_kind == "preference"
            else generated_paired_config
        )(target / "controls")
        if generated
        else _operator_config()
    )
    try:
        case_name = configured_audit_cases(
            config,
            available_cases=tuple(AUDIT_CASES_BY_KIND.values()),
            required_kind=audit_kind,
        )[0]
    except ValueError:
        reason = (
            "invalid_audit_case_selection"
            if audit_kind == "paired"
            else "unavailable_audit_case"
        )
        raise _AuditConfigurationError(reason) from None
    _prepare_case_controls(target, config["cases"][case_name], case_name)
    destination = target / "audit-config.json"
    _write_private(destination, json.dumps(config))
    return destination


def _prepare_case_controls(target: Path, case: dict, case_name: str) -> None:
    controls = audit_controls(case)
    for index, control in enumerate(controls.values()):
        output = target / case_name
        if "controls" in case:
            output = output / str(index)
        if case_name == "paired-judges":
            _prepare_paired_control(control, output)
        else:
            _prepare_control(control, output)


def _prepare_control(control: dict, output: Path) -> None:
    if not isinstance(control.get("request"), dict):
        raise _AuditConfigurationError("invalid_audit_request")
    request = dict(control["request"])
    request["output_path"] = str(output)
    options = VlmPreferenceComparisonRequest(**request)
    if not isinstance(options.api_key_env, str):
        raise _AuditConfigurationError("invalid_audit_request")
    if not os.environ.get(options.api_key_env, "").strip():
        raise _AuditConfigurationError("missing_audit_credential")
    expectations = _preference_expectations(control)
    control["request"] = asdict(options)
    bindings = frozen_preference_expectations(
        options,
        source_sha256=(control.get("baseline_sha256"), control.get("candidate_sha256")),
    )
    for field, expected in bindings.items():
        if field in expectations and expectations[field] != expected:
            raise _AuditConfigurationError("invalid_frozen_preference_expectations")
        expectations[field] = expected


def _preference_expectations(control: dict) -> dict:
    expectations = control.get("expectations", {})
    if not isinstance(expectations, dict):
        raise _AuditConfigurationError("missing_frozen_preference_expectations")
    mapped = expectations.get("mapped_preferences")
    try:
        require_boolean(
            expectations.get("escalation_required"),
            field="expectations.escalation_required",
        )
    except ValueError:
        raise _AuditConfigurationError(
            "missing_frozen_preference_expectations"
        ) from None
    if (
        not isinstance(mapped, list)
        or len(mapped) != 2
        or any(
            value not in ("baseline", "candidate", "tie", "unresolved")
            for value in mapped
        )
    ):
        raise _AuditConfigurationError("missing_frozen_preference_expectations")
    return expectations


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

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed or report.skipped:
            self.results.setdefault(report.nodeid, set()).add(
                "failed" if report.failed else "skipped"
            )

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
    configured_bytes = _read_artifact(target, config_path)
    receipt["configuration_sha256"] = hashlib.sha256(configured_bytes).hexdigest()
    config = json.loads(configured_bytes)
    kind, reports = _configured_reports(config)
    paired_bindings = _paired_bindings(config) if kind == "paired" else None
    receipt["audit_kind"] = kind
    if generated:
        (_scheduled_paired_preflight if kind == "paired" else _scheduled_preflight)()
        receipt["credential_and_catalog_preflight_passed"] = True
    receipt["pytest_exit_code"] = _execute(root, target, config_path)
    receipt["outcomes"] = _public_comparisons(target, reports, paired_bindings)
    _accept_execution(target, receipt, kind, reports)
    if _read_artifact(target, config_path) != configured_bytes:
        receipt["passed"] = False
        raise _AuditConfigurationError("audit_configuration_changed_during_execution")


def _configured_reports(config: dict) -> tuple[str, tuple]:
    case_name = configured_audit_cases(
        config, available_cases=tuple(AUDIT_CASES_BY_KIND.values())
    )[0]
    kind = next(kind for kind, case in AUDIT_CASES_BY_KIND.items() if case == case_name)
    controls = audit_controls(config["cases"][case_name])
    filename = (
        JUDGE_COMPARISON_RESULT_FILENAME
        if kind == "paired"
        else PREFERENCE_COMPARISON_RESULT_FILENAME
    )
    reports = tuple(
        (
            index,
            Path(control["request"]["output_path"]) / filename,
            control["expectations"],
        )
        for index, control in enumerate(controls.values())
    )
    return kind, reports


def _accept_execution(target: Path, receipt: dict, kind: str, reports: tuple) -> None:
    execution = json.loads(_read_artifact(target, target / "execution.json"))
    try:
        counts = _counts(execution)
    except _AuditConfigurationError:
        reason = (
            "invalid_audit_execution_counts"
            if kind == "paired"
            else "invalid_execution_counts"
        )
        raise _AuditConfigurationError(reason) from None
    receipt["counts"] = counts
    if any("failure" in outcome for outcome in receipt["outcomes"]):
        raise _AuditConfigurationError("audit_artifact_contract_failed")
    _require_stable_artifacts(target, reports, receipt["outcomes"])
    expected = len(reports)
    receipt["passed"] = (
        receipt["pytest_exit_code"] == 0
        and counts["collected"] == counts["executed"] == counts["passed"] == expected
        and counts["failed"] == counts["skipped"] == counts["deselected"] == 0
        and not execution["xfail"]
    )


def _read_artifact(target: Path, path: Path) -> bytes:
    return _read_private_artifact(path, target)


def _paired_bindings(config: dict) -> list[dict]:
    return [
        _paired_control_binding(control)
        for control in audit_controls(config["cases"]["paired-judges"]).values()
    ]


def _paired_control_binding(control: dict) -> dict:
    request = _paired_options(control["request"])
    models = vlm_eval._comparison_models(request.primary_model, request.secondary_model)
    values = asdict(request)
    for field in ("primary_model", "secondary_model", "rubric_path"):
        values.pop(field)
    values["rubric"] = (
        _read_control_file(Path(request.rubric_path)).decode("utf-8").strip()
        if request.rubric_path
        else vlm_eval._load_rubric(rubric=request.rubric, rubric_path="")
    )
    with vlm_eval._materialized_input(request.input_path) as local:
        if "input_sha256" in control:
            if (
                hashlib.sha256(_read_control_file(local)).hexdigest()
                != control["input_sha256"]
            ):
                raise _AuditConfigurationError("audit_control_input_changed")
        context = vlm_eval._comparison_context(
            local_input=local, metadata_reader=_read_task_metadata, **values
        )
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


def _paired_options(values: dict) -> VlmJudgeComparisonRequest:
    request = VlmJudgeComparisonRequest(**values)
    for field in fields(request):
        if field.type in (str, "str") and not isinstance(
            getattr(request, field.name), str
        ):
            raise _AuditConfigurationError("invalid_audit_request")
    try:
        require_integer(request.max_frames, field="max_frames", minimum=1)
        require_number(
            request.success_threshold, field="success_threshold", minimum=0, maximum=1
        )
        require_number(request.timeout_s, field="timeout_s", minimum=0)
        if request.timeout_s == 0:
            raise ValueError("timeout_s must be positive")
    except ValueError:
        raise _AuditConfigurationError("invalid_audit_request") from None
    return request


def _read_control_file(path: Path) -> bytes:
    # Control inputs are operator-selected, unlike retained output artifacts.
    # Do not require private media permissions, but never block on special files.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise _AuditConfigurationError("invalid_audit_control_file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _read_task_metadata(path: Path) -> bytes:
    content = _read_control_file(path)
    if path.suffix == ".json" and not isinstance(
        json.loads(content.decode("utf-8")), dict
    ):
        raise _AuditConfigurationError("invalid_audit_task_metadata")
    return content


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


def _require_stable_artifacts(
    target: Path, reports: tuple, outcomes: list[dict]
) -> None:
    expected = {path for _, path, _ in reports}
    for (_, path, _), summary in zip(reports, outcomes, strict=True):
        if (
            hashlib.sha256(_read_artifact(target, path)).hexdigest()
            != summary["artifact_sha256"]
        ):
            raise _AuditConfigurationError("missing_or_unexpected_audit_artifacts")
    for directory, filename in {
        (path.parent.parent if path.parent.name.isdigit() else path.parent, path.name)
        for path in expected
    }:
        if set(directory.rglob(filename)) != {
            path for path in expected if path.name == filename
        }:
            raise _AuditConfigurationError("missing_or_unexpected_audit_artifacts")


def _public_comparisons(
    target: Path,
    reports: tuple[tuple[int, Path, dict], ...] | None = None,
    paired_bindings: list[dict] | None = None,
) -> list[dict]:
    if reports is None:
        directory = target / "paired-judges"
        reports = tuple(
            (0 if path.parent == directory else int(path.parent.name), path, {})
            for path in directory.rglob(JUDGE_COMPARISON_RESULT_FILENAME)
        )
        reports = tuple(sorted(reports, key=lambda row: row[0]))
    summaries = []
    for index, path, expectations in reports:
        summary = {"control_index": index}
        try:
            content = _read_artifact(target, path)
            summary["artifact_sha256"] = hashlib.sha256(content).hexdigest()
            report = json.loads(content)
            formatter = (
                _public_paired_comparison
                if path.name == JUDGE_COMPARISON_RESULT_FILENAME
                else _public_preference_comparison
            )
            summary.update(formatter(report))
            if not _artifact_accepted(report, summary, expectations):
                summary["failure"] = "audit_artifact_did_not_meet_contract"
            if paired_bindings is not None:
                _require_paired_binding(report, paired_bindings[index])
        except (OSError, ValueError, KeyError, TypeError):
            summary["failure"] = "invalid_or_missing_audit_artifact"
        summaries.append(summary)
    return summaries


def _artifact_accepted(report: dict, summary: dict, expectations: dict) -> bool:
    # Typed errors are valid retained outcomes, never successful live proof.
    if (
        not _successful_audit_orders(report, summary)
        or report["deployment_status"] != "audit_only"
        or require_boolean(
            report["operational_rate_estimated"],
            field="report.operational_rate_estimated",
        )
    ):
        return False
    for field, expected in expectations.items():
        actual = report
        for component in field.split("."):
            actual = actual[component]
        _require_bound_value(actual, expected)
    if "first_order" in summary:
        verdicts = tuple(
            _validate_retained_preference_order(report[order], report["model"])
            for order in ("first_order", "reversed_order")
        )
        validate_preference_report(report, verdicts)
    else:
        for judge in ("primary", "secondary"):
            _validate_retained_judge(report[judge])
    return True


def _validate_retained_judge(outcome: dict) -> None:
    result = outcome["result"]
    # Evidence validation is not a promotion action; the report remains audit-only.
    if (
        result.get("backend") != "api"
        or result.get("model") != outcome["model"]
        or vlm_grade_evidence.vlm_paired_audit_block_details(result)
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
    _validate_retained_usage(provider)


def _validate_retained_usage(provider: dict) -> dict:
    raw = provider["raw_response"]
    data = json.loads(
        raw,
        object_pairs_hook=vlm_grade_evidence._unique_provider_fields,
        parse_constant=_reject_nonfinite_response_value,
    )
    if not isinstance(data, dict):
        raise _AuditConfigurationError("invalid_retained_provider_evidence")
    usage, raw_usage = provider["usage"], data.get("usage")
    if not isinstance(usage, dict) or not isinstance(raw_usage, dict):
        raise _AuditConfigurationError("invalid_retained_provider_evidence")
    for field in ("prompt_tokens", "completion_tokens"):
        require_integer(usage.get(field), field=field, minimum=1)
        require_integer(raw_usage.get(field), field=field, minimum=1)
    if usage != raw_usage or provider["raw_response_sha256"] != vlm_eval._sha256_text(
        raw
    ):
        raise _AuditConfigurationError("invalid_retained_provider_evidence")
    return data


def _reject_nonfinite_response_value(value: str) -> None:
    raise _AuditConfigurationError("invalid_retained_provider_evidence")


def _validate_retained_preference_order(outcome: dict, model: str):
    provider = outcome["provider"]
    data = _validate_retained_usage(provider)
    vlm_grade_evidence._validate_transport(provider)
    if (
        provider["status_code"] != 200
        or not isinstance(provider["provider_request_id"], str)
        or not provider["provider_request_id"].strip()
    ):
        raise _AuditConfigurationError("invalid_retained_preference_evidence")
    response = vlm_eval._VlmBackendResponse(
        data,
        provider["raw_response"],
        provider["status_code"],
        provider["provider_request_id"],
        provider["latency_s"],
    )
    verdict, choice = vlm_eval._strict_preference_verdict(response, model)
    vlm_grade_evidence._validate_provider_metadata(
        provider, data, choice, vlm_eval.PREFERENCE_RESPONSE_PARSER_VERSION
    )
    if json.loads(json.dumps(asdict(verdict))) != outcome["verdict"]:
        raise _AuditConfigurationError("invalid_retained_preference_evidence")
    return verdict


def _public_preference_comparison(report: dict) -> dict:
    statuses = {
        "judge_error",
        "unresolved",
        "low_confidence",
        "order_disagreement_or_nondeterminism",
        "consistent_candidate_preference",
        "consistent_baseline_preference",
        "consistent_tie",
    }
    if not isinstance(report, dict) or report["status"] not in statuses:
        raise ValueError("Invalid audit status")
    return {
        "status": report["status"],
        "escalation_required": require_boolean(
            report["escalation_required"], field="report.escalation_required"
        ),
        "requests_counterbalanced": require_boolean(
            report["requests_counterbalanced"], field="report.requests_counterbalanced"
        ),
        "first_order": _public_order(report["first_order"]),
        "reversed_order": _public_order(report["reversed_order"]),
    }


def _public_order(outcome: dict) -> dict:
    error_types = {
        "transport_error",
        "provider_http_status_error",
        "provider_response_decode_error",
        "response_contract_error",
    }
    if not isinstance(outcome, dict):
        raise ValueError("Invalid audit order")
    provider = outcome.get("provider")
    verdict = outcome.get("verdict")
    error = outcome.get("error")
    if (verdict is None) == (error is None):
        raise ValueError("Audit order needs exactly one verdict or error")
    if verdict is not None and not _VERDICT_VALIDATOR.is_valid(verdict):
        raise ValueError("Invalid audit verdict")
    if error is not None and (
        not isinstance(error, dict) or error.get("error_type") not in error_types
    ):
        raise ValueError("Invalid audit error")
    if provider is None:
        if error is None:
            raise ValueError("Successful audit order needs provider evidence")
        provider = {}
    if not isinstance(provider, dict):
        raise ValueError("Invalid audit provider evidence")
    returned_model = provider.get("returned_model")
    raw_response = provider.get("raw_response")
    if any(
        value is not None and not isinstance(value, str)
        for value in (returned_model, raw_response)
    ):
        raise ValueError("Invalid audit provider evidence")
    if error is None and (not returned_model or not raw_response):
        raise ValueError("Successful audit order needs provider evidence")
    return {
        "returned_model_sha256": hashlib.sha256(
            (returned_model if returned_model is not None else "").encode()
        ).hexdigest(),
        "raw_response_sha256": hashlib.sha256(
            (raw_response if raw_response is not None else "").encode()
        ).hexdigest(),
        "preference": verdict["preference"] if verdict is not None else None,
        "confidence": verdict["confidence"] if verdict is not None else None,
        "error_type": error["error_type"] if error is not None else None,
    }


def _prepare_paired_control(control: dict, output: Path) -> None:
    if not isinstance(control.get("request"), dict):
        raise _AuditConfigurationError("invalid_audit_request")
    request = dict(control["request"])
    request["output_path"] = str(output)
    options = _paired_options(request)
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
    control["request"] = asdict(options)
    for judge in ("primary", "secondary"):
        expectations[f"{judge}.model"] = request[f"{judge}_model"]


def _scheduled_paired_preflight() -> None:
    key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    config = resolve_config(api_key=key, base_url=DEFAULT_BASE_URL, environ={})
    available = TokenFactoryClient(config).list_models()
    if not set(PAIRED_MODELS).issubset(available):
        raise _AuditConfigurationError("required_paired_model_not_advertised")


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


def _public_paired_comparison(report: dict) -> dict:
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


def _public_comparison(report: dict) -> dict:
    if "primary" in report or "secondary" in report:
        return _public_paired_comparison(report)
    return _public_preference_comparison(report)


def _successful_audit_orders(report: dict, summary: dict) -> bool:
    if "first_order" in summary:
        return (
            summary["first_order"]["error_type"] is None
            and summary["reversed_order"]["error_type"] is None
            and summary["requests_counterbalanced"] is True
        )
    return (
        all(summary[judge]["error_type"] is None for judge in ("primary", "secondary"))
        and summary["status"] != "judge_identity_collision"
        and require_boolean(
            report["requests_differ_only_by_model"],
            field="report.requests_differ_only_by_model",
        )
    )


def _prepare_evidence_directory(root: Path, path: Path) -> Path:
    try:
        try:
            target = path.resolve(strict=True)
        except FileNotFoundError:
            # Outputs may be new, but existing prefixes must resolve without loops.
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


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-kind", help="Generated audit kind: paired or preference."
    )
    parser.add_argument(
        "--generated-controls",
        action="store_true",
        help="Run frozen local visual controls with the protected Token Factory key.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute the configured audit suite and emit a sanitized summary.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero only when every collected audit executes successfully without skips.
    Raises:
        RuntimeError: An unexpected implementation error occurs, not a path failure.
    """
    args = _arguments(argv)
    root = Path(__file__).resolve().parents[2]
    receipt = _new_receipt(root)
    receipt["audit_kind"] = (
        args.audit_kind if args.audit_kind in AUDIT_CASES_BY_KIND else None
    )
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
