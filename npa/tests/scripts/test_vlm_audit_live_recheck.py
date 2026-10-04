"""Test configured audit lane failure gates without treating mocks as live evidence."""

import importlib.util
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from PIL import Image

from npa.workbench import vlm_eval


def _runner():
    path = Path(__file__).resolve().parents[2] / "scripts/vlm_audit_live_recheck.py"
    spec = importlib.util.spec_from_file_location("audit_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(monkeypatch, tmp_path):
    for name in ("first.png", "second.png"):
        Image.new("RGB", (2, 2), "black").save(tmp_path / name)
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {
                "cases": {
                    "blinded-preference": {
                        "request": {
                            "baseline_path": str(tmp_path / "first.png"),
                            "candidate_path": str(tmp_path / "second.png"),
                            "model": "vision/model",
                            "task": "Compare the visible shapes.",
                        },
                        "expectations": {
                            "mapped_preferences": ["tie", "tie"],
                            "escalation_required": False,
                        },
                    }
                }
            }
        )
    )
    config.chmod(0o600)
    monkeypatch.setenv("NPA_VLM_AUDIT_LIVE_CONFIG", str(config))
    monkeypatch.setenv("VLM_EVAL_API_KEY", "synthetic-test-credential")
    return config


def _valid_report(request):
    options = vlm_eval.VlmPreferenceComparisonRequest(**request)
    rubric = vlm_eval._load_rubric(
        rubric=options.rubric, rubric_path=options.rubric_path
    )
    context = vlm_eval._preference_context(
        options,
        vlm_eval.preference_comparison_result_uri_for(options.output_path),
        rubric,
        *vlm_eval._load_preference_pair(options),
    )
    requests, orders = vlm_eval._preference_requests(context)
    response = _preference_response()
    outcomes = tuple(
        vlm_eval._parse_preference_outcome(
            context,
            *order[:3],
            transport,
            vlm_eval._preference_request_evidence(context, transport, order[3]),
            response,
        )
        for transport, order in zip(requests, orders, strict=True)
    )
    report = vlm_eval._build_preference_report(
        context,
        outcomes,
        vlm_eval._assert_counterbalanced_requests(
            requests, context.baseline, context.candidate
        ),
    )
    return json.loads(json.dumps(asdict(report)))


def _preference_response():
    verdict = {
        "preference": "tie",
        "confidence": "high",
        "observable_support": ["Same visible shape."],
        "critical_defects": {"A": ["None visible."], "B": ["None visible."]},
        "uncertainty": "Synthetic unit-test evidence, not inference.",
    }
    provider = _preference_provider(verdict)
    return vlm_eval._VlmBackendResponse(
        json.loads(provider["raw_response"]),
        provider["raw_response"],
        200,
        provider["provider_request_id"],
        0.1,
    )


def _preference_provider(verdict):
    usage = {"prompt_tokens": 1, "completion_tokens": 1}
    raw = json.dumps(
        {
            "model": "vision/model",
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(verdict)}}
            ],
            "usage": usage,
        }
    )
    return {
        "returned_model": "vision/model",
        "raw_response": raw,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "status_code": 200,
        "finish_reason": "stop",
        "provider_request_id": "synthetic-request",
        "latency_s": 0.1,
        "parser_version": "npa_vlm_preference_hosted_json_v1",
        "usage": usage,
    }


def _write_report(runner, output, report):
    output.mkdir(parents=True, exist_ok=True)
    (output / runner.PREFERENCE_COMPARISON_RESULT_FILENAME).write_text(
        json.dumps(report)
    )


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("config", "missing_audit_configuration"),
        ("key", "missing_audit_credential"),
        ("permissions", "audit_configuration_must_be_owner_only"),
        ("expectations", "missing_frozen_preference_expectations"),
    ],
)
def test_preconditions_fail_before_pytest(monkeypatch, tmp_path, condition, reason):
    runner = _runner()
    config = _config(monkeypatch, tmp_path)
    if condition == "config":
        monkeypatch.delenv(runner.CONFIG_ENV)
    elif condition == "key":
        monkeypatch.delenv("VLM_EVAL_API_KEY")
    elif condition == "permissions":
        config.chmod(0o644)
    else:
        value = json.loads(config.read_text())
        value["cases"]["blinded-preference"]["expectations"] = {}
        config.write_text(json.dumps(value))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == reason
    assert not receipt["passed"]
    assert not any(receipt["counts"].values())


@pytest.mark.parametrize(
    "xml,exit_code,passed",
    [
        (
            '<testsuites><testsuite><testcase name="audit"/></testsuite></testsuites>',
            0,
            True,
        ),
        (
            "<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>",
            0,
            False,
        ),
        (
            "<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>",
            1,
            False,
        ),
        ("<testsuites><testsuite/></testsuites>", 0, False),
        ("<testsuites><testsuite><testcase/></testsuite></testsuites>", 2, False),
    ],
)
def test_receipt_requires_real_execution_counts(
    monkeypatch, tmp_path, xml, exit_code, passed
):
    runner = _runner()
    original = _config(monkeypatch, tmp_path)
    before = original.read_bytes()

    def execute(root, target, config):
        prepared = json.loads(config.read_text())["cases"]["blinded-preference"][
            "request"
        ]
        assert prepared["output_path"] == str(target / "blinded-preference")
        (target / "pytest.xml").write_text(xml)
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1 if "<testcase" in xml else 0,
                    "executed": 1 if "<testcase" in xml else 0,
                    "deselected": 0,
                    "xfail": False,
                    "passed": int(
                        "<testcase" in xml
                        and "<skipped" not in xml
                        and "<failure" not in xml
                    ),
                    "failed": int("<failure" in xml),
                    "skipped": int("<skipped" in xml),
                }
            )
        )
        _write_report(runner, target / "blinded-preference", _valid_report(prepared))
        return exit_code

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == (0 if passed else 1)
    receipt = target / "receipt.json"
    assert json.loads(receipt.read_text())["passed"] is passed
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert original.read_bytes() == before
    assert "synthetic-test-credential" not in receipt.read_text()
    assert str(tmp_path) not in receipt.read_text()
    assert runner.main(["--evidence-dir", str(target)]) == 1
    assert json.loads(receipt.read_text())["passed"] is passed


def test_operator_request_defaults_are_resolved_before_live_execution(
    monkeypatch, tmp_path
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    value = json.loads(path.read_text())
    del value["cases"]["blinded-preference"]["request"]["model"]
    path.write_text(json.dumps(value))
    prepared = runner._prepare_config(tmp_path)
    request = json.loads(prepared.read_text())["cases"]["blinded-preference"]["request"]
    assert request["model"] == "MiniMaxAI/MiniMax-M3"
    assert request["api_key_env"] == "VLM_EVAL_API_KEY"
    assert request["timeout_s"] > 0


def test_subprocess_environment_cannot_select_partial_or_dry_run_coverage(monkeypatch):
    runner = _runner()
    for name in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "NPA_CI_SHARD_INDEX",
        "NPA_CI_TOTAL_SHARDS",
        "NPA_DRY_RUN",
    ):
        monkeypatch.setenv(name, "polluted-parent")
    environment = runner._test_environment(Path("audit-config.json"))
    assert "polluted-parent" not in environment.values()
    assert environment["NPA_INTEGRATION_E2E"] == "1"
    assert environment[runner.CONFIG_ENV] == "audit-config.json"


def test_actual_entrypoint_missing_config_fails_without_skips(tmp_path):
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment.pop("NPA_VLM_AUDIT_LIVE_CONFIG", None)
    result = subprocess.run(
        [
            sys.executable,
            "npa/scripts/vlm_audit_live_recheck.py",
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    summary = json.loads(result.stdout)
    assert summary["failure"] == "missing_audit_configuration"
    assert summary["counts"]["skipped"] == summary["counts"]["collected"] == 0


@pytest.mark.parametrize("nested", [False, True])
def test_actual_evidence_symlink_loop_is_sanitized(tmp_path, nested):
    loop = tmp_path / "private-evidence-loop"
    loop.symlink_to(loop.name)
    target = loop / "output" if nested else loop
    result = subprocess.run(
        [sys.executable, str(_runner().__file__), "--evidence-dir", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["failure"] == "invalid_audit_evidence_directory"
    assert str(tmp_path) not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("boundary", ["config", "generated-controls"])
def test_private_configuration_symlink_loops_fail_closed(
    monkeypatch, tmp_path, capsys, boundary
):
    runner = _runner()
    target = tmp_path / "run"
    target.mkdir(mode=0o700)
    loop = target / (
        "controls" if boundary == "generated-controls" else "private-config"
    )
    loop.symlink_to(loop.name)
    monkeypatch.setenv(runner.CONFIG_ENV, str(loop))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    receipt = runner._new_receipt(Path(__file__).resolve().parents[3])
    runner._complete_receipt(
        Path(__file__).resolve().parents[3],
        target,
        receipt,
        generated=boundary == "generated-controls",
        audit_kind="preference",
    )
    assert receipt["passed"] is False
    assert receipt["failure"] == "audit_configuration_or_execution_failed"
    assert str(tmp_path) not in json.dumps(receipt)
    assert str(tmp_path) not in capsys.readouterr().out


def test_unrelated_resolve_runtime_error_is_not_swallowed(monkeypatch, tmp_path):
    runner = _runner()

    def fail(_):
        raise RuntimeError("unrelated runtime bug")

    monkeypatch.setattr(Path, "resolve", fail)
    with pytest.raises(RuntimeError, match="unrelated runtime bug"):
        runner._prepare_evidence_directory(tmp_path, tmp_path / "run")


@pytest.mark.parametrize(
    "mode", ["xpass", "xfail", "skip", "deselection", "setup", "teardown", "missing"]
)
def test_actual_incomplete_execution_cannot_pass_lane(tmp_path, mode):
    test_path = tmp_path / "test_expected.py"
    if mode == "deselection":
        test_path.write_text(
            "def test_first():\n    pass\ndef test_second():\n    pass\n"
        )
        (tmp_path / "conftest.py").write_text(
            "def pytest_collection_modifyitems(config, items):\n"
            "    config.hook.pytest_deselected(items=[items.pop()])\n"
        )
    elif mode in ("setup", "teardown"):
        failure = "    pytest.fail('fixture boundary')\n"
        body = failure + "    yield\n" if mode == "setup" else "    yield\n" + failure
        test_path.write_text(
            "import pytest\n@pytest.fixture(autouse=True)\ndef boundary():\n"
            + body
            + "def test_control():\n    pass\n"
        )
    elif mode == "missing":
        test_path.write_text("def test_control():\n    pass\n")
        (tmp_path / "conftest.py").write_text(
            "def pytest_runtestloop(session):\n    return True\n"
        )
    else:
        marker = "xfail(strict=False)" if mode == "xpass" else "skip(reason='contract')"
        if mode == "xfail":
            marker = "xfail(reason='contract')"
        body = "assert False" if mode == "xfail" else "pass"
        test_path.write_text(
            f"import pytest\n@pytest.mark.{marker}\ndef test_control():\n    {body}\n"
        )
    runner_path = (
        Path(__file__).resolve().parents[2] / "scripts/vlm_audit_live_recheck.py"
    )
    code = """
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('audit_recheck', sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.SUITES = (sys.argv[2],)
raise SystemExit(runner._run_tests(Path(sys.argv[3])))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(runner_path), str(test_path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    expected = {
        "xpass": "1 xpassed",
        "skip": "1 skipped",
        "deselection": "1 deselected",
        "xfail": "1 xfailed",
        "setup": "1 error",
        "teardown": "1 error",
        "missing": "no tests ran",
    }
    assert expected[mode] in result.stdout
    assert result.returncode == 1
    assert (tmp_path / "pytest.xml").is_file()


@pytest.mark.parametrize(
    "field", ["collected", "executed", "passed", "failed", "skipped", "deselected"]
)
@pytest.mark.parametrize("bad_count", [True, False, -1, "1", None, 1.0, [], {}])
def test_malformed_execution_counts_fail_closed(
    monkeypatch, tmp_path, field, bad_count
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1,
                    "executed": 1,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "deselected": 0,
                    "xfail": False,
                    field: bad_count,
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "run"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "invalid_execution_counts"
    assert receipt["passed"] is False


@pytest.mark.parametrize(
    "value", [0, 1, None, "false", "private-invalid-boolean", 0.0, [], {}]
)
def test_escalation_expectation_requires_literal_boolean(monkeypatch, tmp_path, value):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["blinded-preference"]["expectations"]["escalation_required"] = value
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "run"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    assert json.loads(body)["failure"] == "missing_frozen_preference_expectations"
    assert "private-invalid-boolean" not in body


@pytest.mark.parametrize(
    "boundary", ["xfail", "escalation_required", "requests_counterbalanced"]
)
@pytest.mark.parametrize(
    "value", [0, 1, None, "false", "private-invalid-boolean", 0.0, [], {}, "missing"]
)
def test_execution_and_report_booleans_fail_closed(
    monkeypatch, tmp_path, boundary, value
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        execution = {
            "collected": 1,
            "executed": 1,
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
            "xfail": False,
        }
        request = json.loads(config.read_text())["cases"]["blinded-preference"][
            "request"
        ]
        report = _valid_report(request)
        evidence = execution if boundary == "xfail" else report
        if value == "missing":
            del evidence[boundary]
        else:
            evidence[boundary] = value
        (target / "execution.json").write_text(json.dumps(execution))
        output = target / "blinded-preference"
        output.mkdir()
        (output / runner.PREFERENCE_COMPARISON_RESULT_FILENAME).write_text(
            json.dumps(report)
        )
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "run"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["failure"] == (
        "invalid_execution_counts"
        if boundary == "xfail"
        else "audit_artifact_contract_failed"
    )
    assert receipt["passed"] is False
    assert "private-invalid-boolean" not in body


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "status",
        "preference",
        "confidence",
        "missing_preference",
        "null_confidence",
        "empty_verdict",
        "both_verdict_and_error",
        "neither",
        "unknown_error",
        "typed_error",
        "malformed_provider",
        "nullable_error_model",
        "missing_support",
        "wrong_expectation",
        "not_counterbalanced",
        "not_audit_only",
        "config_changed",
        "symlink_loop",
        "parent_symlink",
        "fifo",
        "directory",
        "execution_fifo",
        "config_fifo",
        "missing_execution",
        "malformed_execution",
        "none",
    ],
)
def test_receipt_acceptance_requires_each_frozen_artifact(monkeypatch, tmp_path, fault):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    control = config["cases"]["blinded-preference"]
    config["cases"]["blinded-preference"] = {
        "controls": {f"control-{index}": deepcopy(control) for index in range(3)}
    }
    path.write_text(json.dumps(config))

    def execute(root, target, config_path):
        configured = json.loads(config_path.read_text())
        for index, control in enumerate(
            configured["cases"]["blinded-preference"]["controls"].values()
        ):
            report = _valid_report(control["request"])
            output = Path(control["request"]["output_path"])
            if index == 1:
                order = report["first_order"]
                if fault == "missing":
                    _write_report(runner, output.with_name("9"), report)
                    continue
                if fault in {"preference", "confidence"}:
                    order["verdict"][fault] = "private-invalid-outcome"
                elif fault == "status":
                    report["status"] = "private-invalid-outcome"
                elif fault == "missing_preference":
                    del order["verdict"]["preference"]
                elif fault == "null_confidence":
                    order["verdict"]["confidence"] = None
                elif fault == "missing_support":
                    del order["verdict"]["observable_support"]
                elif fault == "empty_verdict":
                    order["verdict"] = {}
                elif fault in {
                    "both_verdict_and_error",
                    "unknown_error",
                    "typed_error",
                    "nullable_error_model",
                }:
                    order["error"] = {"error_type": "transport_error"}
                    if fault != "both_verdict_and_error":
                        order["verdict"] = None
                        order["provider"] = None
                    if fault == "unknown_error":
                        order["error"]["error_type"] = "private-invalid-outcome"
                    if fault == "nullable_error_model":
                        order["provider"] = {
                            "returned_model": None,
                            "raw_response": "synthetic failure",
                        }
                    report["status"] = "judge_error"
                    report["escalation_required"] = True
                elif fault == "neither":
                    order["verdict"] = None
                elif fault == "malformed_provider":
                    order["provider"] = False
                elif fault == "wrong_expectation":
                    report["mapped_preferences"] = ["candidate", "candidate"]
                elif fault == "not_counterbalanced":
                    report["requests_counterbalanced"] = False
                elif fault == "not_audit_only":
                    report["deployment_status"] = "accepted"
                elif fault == "symlink_loop":
                    output.mkdir(parents=True)
                    report_path = output / runner.PREFERENCE_COMPARISON_RESULT_FILENAME
                    report_path.symlink_to(report_path.name)
                    continue
                elif fault == "parent_symlink":
                    outside = tmp_path / "outside"
                    _write_report(runner, outside, report)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.symlink_to(outside, target_is_directory=True)
                    continue
                elif fault in {"fifo", "directory"}:
                    output.mkdir(parents=True)
                    report_path = output / runner.PREFERENCE_COMPARISON_RESULT_FILENAME
                    if fault == "fifo":
                        os.mkfifo(report_path)
                    else:
                        report_path.mkdir()
                    continue
            _write_report(runner, output, report)
        if fault == "config_changed":
            configured["cases"]["blinded-preference"]["controls"].pop("control-2")
            config_path.write_text(json.dumps(configured))
        if fault != "missing_execution":
            (target / "execution.json").write_text(
                json.dumps(
                    {
                        "collected": "3" if fault == "malformed_execution" else 3,
                        "executed": 3,
                        "passed": 3,
                        "failed": 0,
                        "skipped": 0,
                        "deselected": 0,
                        "xfail": False,
                    }
                )
            )
        if fault in {"execution_fifo", "config_fifo"}:
            fifo = (
                target / "execution.json" if fault == "execution_fifo" else config_path
            )
            fifo.unlink()
            os.mkfifo(fifo)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "run"
    assert runner.main(
        ["--audit-kind", "preference", "--evidence-dir", str(target)]
    ) == (0 if fault == "none" else 1)
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["passed"] is (fault == "none")
    assert [row["control_index"] for row in receipt["outcomes"]] == [0, 1, 2]
    assert "failure" not in receipt["outcomes"][0]
    assert "failure" not in receipt["outcomes"][2]
    if fault in {"typed_error", "nullable_error_model"}:
        order = receipt["outcomes"][1]["first_order"]
        assert order["preference"] is None
        assert order["error_type"] == "transport_error"
    assert "private-invalid-outcome" not in body
    assert str(tmp_path) not in body


@pytest.mark.parametrize("kind", [None, "unknown-private-value"])
def test_generated_selector_cannot_substitute_another_lane(monkeypatch, tmp_path, kind):
    runner = _runner()
    monkeypatch.setattr(
        runner, "generated_preference_config", lambda *_: pytest.fail("wrong lane")
    )
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    arguments = ["--generated-controls", "--evidence-dir", str(tmp_path / "run")]
    if kind is not None:
        arguments += ["--audit-kind", kind]
    assert runner.main(arguments) == 1
    receipt = json.loads((tmp_path / "run/receipt.json").read_text())
    assert receipt["failure"] == "unavailable_or_missing_audit_kind"
    assert receipt["passed"] is False and not any(receipt["counts"].values())
    assert "unknown-private-value" not in json.dumps(receipt)


def test_operator_config_cannot_smuggle_an_unavailable_case(monkeypatch, tmp_path):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["paired-judges"] = config["cases"]["blinded-preference"]
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "run"
    assert (
        runner.main(["--audit-kind", "preference", "--evidence-dir", str(target)]) == 1
    )
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "unavailable_audit_case"
    assert receipt["passed"] is False and not any(receipt["counts"].values())


def test_missing_execution_report_fails_closed(monkeypatch, tmp_path):
    runner = _runner()
    _config(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_execute", lambda *_: 0)
    target = tmp_path / "run"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "audit_configuration_or_execution_failed"
    assert receipt["passed"] is False


@pytest.mark.parametrize(
    "kind,case", [("preference", "blinded-preference"), ("paired", "paired-judges")]
)
@pytest.mark.parametrize("malformed", [None, [], "private-invalid-case"])
def test_case_shape_is_rejected_before_execution_with_sanitized_receipt(
    monkeypatch, tmp_path, kind, case, malformed
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    path.write_text(json.dumps({"audit_kind": kind, "cases": {case: malformed}}))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "case-shape"
    assert runner.main(["--audit-kind", kind, "--evidence-dir", str(target)]) == 1
    text = (target / "receipt.json").read_text()
    receipt = json.loads(text)
    assert receipt["passed"] is False and not any(receipt["counts"].values())
    assert receipt["failure"] == (
        "invalid_audit_case_selection" if kind == "paired" else "unavailable_audit_case"
    )
    assert "private-invalid-case" not in text and str(tmp_path) not in text


@pytest.mark.parametrize("malformed", [None, [], "private-invalid-expectations"])
def test_preference_expectations_shape_fails_with_sanitized_receipt(
    monkeypatch, tmp_path, malformed
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["blinded-preference"]["expectations"] = malformed
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "expectations-shape"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    text = (target / "receipt.json").read_text()
    receipt = json.loads(text)
    assert receipt["failure"] == "missing_frozen_preference_expectations"
    assert receipt["passed"] is False and not any(receipt["counts"].values())
    assert "private-invalid-expectations" not in text and str(tmp_path) not in text


def _mutate_preference_raw_provider(report, fault):
    provider = report["first_order"]["provider"]
    raw = json.loads(provider["raw_response"])
    choice = raw["choices"][0]
    if fault == "invalid_json":
        provider["raw_response"] = "private-invalid-response"
    elif fault in {"outer_null", "outer_list", "empty_object"}:
        provider["raw_response"] = json.dumps(
            {"outer_null": None, "outer_list": [], "empty_object": {}}[fault]
        )
    else:
        if fault == "wrong_model":
            raw["model"] = "private-wrong-model"
        elif fault == "wrong_finish":
            choice["finish_reason"] = "length"
        elif fault == "refusal":
            choice["message"]["refusal"] = "private-refusal"
        elif fault == "missing_usage":
            raw.pop("usage")
        elif fault == "boolean_usage":
            raw["usage"]["prompt_tokens"] = True
        elif fault == "changed_verdict":
            verdict = json.loads(choice["message"]["content"])
            verdict["preference"] = "A"
            choice["message"]["content"] = json.dumps(verdict)
        elif fault in {"invalid_content", "encoded_content", "wrapped_content"}:
            choice["message"]["content"] = {
                "invalid_content": "{}",
                "encoded_content": json.dumps(choice["message"]["content"]),
                "wrapped_content": json.dumps(
                    {"response": choice["message"]["content"]}
                ),
            }[fault]
        provider["raw_response"] = json.dumps(raw)
    provider["raw_response_sha256"] = hashlib.sha256(
        provider["raw_response"].encode()
    ).hexdigest()
    if fault == "wrong_digest":
        provider["raw_response_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "invalid_json",
        "outer_null",
        "outer_list",
        "empty_object",
        "wrong_model",
        "wrong_finish",
        "refusal",
        "missing_usage",
        "boolean_usage",
        "changed_verdict",
        "invalid_content",
        "encoded_content",
        "wrapped_content",
        "wrong_digest",
    ],
)
def test_retained_preference_response_is_verified_before_receipt_acceptance(
    monkeypatch, tmp_path, capsys, fault
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        request = json.loads(config.read_text())["cases"]["blinded-preference"][
            "request"
        ]
        report = _valid_report(request)
        _mutate_preference_raw_provider(report, fault)
        _write_report(runner, target / "blinded-preference", report)
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1,
                    "executed": 1,
                    "passed": 1,
                    "failed": 0,
                    "skipped": 0,
                    "deselected": 0,
                    "xfail": False,
                }
            )
        )
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "evidence"
    assert runner.main(
        ["--audit-kind", "preference", "--evidence-dir", str(target)]
    ) == (0 if fault == "none" else 1)
    text = (target / "receipt.json").read_text()
    receipt = json.loads(text)
    assert receipt["passed"] is (fault == "none")
    assert receipt["counts"]["passed"] == 1
    assert "private-" not in text + capsys.readouterr().out
    assert str(tmp_path) not in text


@pytest.fixture(autouse=True)
def _private_artifact_creation():
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def _production_preference_report(request):
    response = _preference_response()
    with patch.object(
        vlm_eval, "_post_with_readiness_retry", return_value=response
    ) as post:
        report = vlm_eval.compare_vlm_preference(
            vlm_eval.VlmPreferenceComparisonRequest(**request)
        )
    assert post.call_count == 2
    return json.loads(json.dumps(asdict(report)))


def _assert_preference_binding(
    monkeypatch,
    tmp_path,
    mutate,
    *,
    passed=False,
    updates=None,
    replace_pixels=False,
    metadata_kind=None,
):
    runner = _runner()
    config_path = _config(monkeypatch, tmp_path)
    if metadata_kind:
        _directory_preference_inputs(config_path, tmp_path, metadata_kind)

    def execute(root, target, config):
        request = json.loads(config.read_text())["cases"]["blinded-preference"][
            "request"
        ]
        request.update(updates or {})
        if replace_pixels:
            Image.new("RGB", (2, 2), "red").save(request["baseline_path"])
        report = _production_preference_report(request)
        mutate(report)
        _write_report(runner, target / "blinded-preference", report)
        _write_passing_execution(target)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "evidence"
    assert runner.main(
        ["--audit-kind", "preference", "--evidence-dir", str(target)]
    ) == (0 if passed else 1)
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["passed"] is passed
    assert receipt["counts"]["passed"] == 1
    assert str(tmp_path) not in body and "private-" not in body


def _write_passing_execution(target):
    (target / "execution.json").write_text(
        json.dumps(
            {
                "collected": 1,
                "executed": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "deselected": 0,
                "xfail": False,
            }
        )
    )


def _directory_preference_inputs(config_path, tmp_path, metadata_kind):
    config = json.loads(config_path.read_text())
    request = config["cases"]["blinded-preference"]["request"]
    for arm in ("baseline", "candidate"):
        directory = tmp_path / arm
        directory.mkdir()
        Path(request[f"{arm}_path"]).rename(directory / "frame.png")
        request[f"{arm}_path"] = str(directory)
        metadata = directory / "manifest.json"
        if metadata_kind == "fifo":
            os.mkfifo(metadata)
        else:
            metadata.write_text(
                {"list": "[]", "null": "null", "integer": "1"}[metadata_kind]
            )
    config_path.write_text(json.dumps(config))


@pytest.mark.timeout(5)
@pytest.mark.parametrize("kind", ["list", "null", "integer", "fifo"])
def test_preference_does_not_discover_rollout_task_metadata(
    monkeypatch, tmp_path, kind
):
    _assert_preference_binding(
        monkeypatch, tmp_path, lambda report: None, passed=True, metadata_kind=kind
    )


def test_repeated_first_order_is_not_counterbalance(monkeypatch, tmp_path):
    _assert_preference_binding(
        monkeypatch,
        tmp_path,
        lambda report: report.update(reversed_order=deepcopy(report["first_order"])),
    )


@pytest.mark.parametrize("changed", [False, True])
def test_full_preference_report_binds_distinct_sources_even_with_identical_pixels(
    monkeypatch, tmp_path, changed
):
    updates = (
        {
            "baseline_path": str(tmp_path / "second.png"),
            "candidate_path": str(tmp_path / "first.png"),
        }
        if changed
        else {}
    )
    _assert_preference_binding(
        monkeypatch, tmp_path, lambda report: None, passed=not changed, updates=updates
    )


@pytest.mark.parametrize("field", ["task", "rubric"])
def test_full_preference_report_binds_actual_config(monkeypatch, tmp_path, field):
    _assert_preference_binding(
        monkeypatch,
        tmp_path,
        lambda report: None,
        updates={field: "different private instructions"},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "wrong-schema"),
        ("task", "different private task"),
        ("rubric", "different private rubric"),
        ("deployment_status", "promoted"),
        ("operational_rate_estimated", True),
        ("requests_counterbalanced", False),
        ("agreement_eligible", False),
        ("agreement_eligible", 1),
        ("escalation_required", True),
        ("status", "consistent_baseline_preference"),
        ("mapped_preferences", ["baseline", "baseline"]),
        ("normalized_baseline_sha256", "0" * 64),
        ("normalized_candidate_sha256", "0" * 64),
        ("unordered_pair_sha256", "0" * 64),
        ("limitations", []),
        ("generated_at", None),
        ("mean_score", 0),
    ],
)
def test_full_preference_report_requires_canonical_metadata(
    monkeypatch, tmp_path, field, value
):
    _assert_preference_binding(
        monkeypatch, tmp_path, lambda report: report.update({field: value})
    )


@pytest.mark.parametrize("order", ["first_order", "reversed_order"])
@pytest.mark.parametrize(
    "fault",
    [
        "empty_transport",
        "generation",
        "prompt",
        "frame",
        "manifest",
        "digest",
        "A_arm",
        "B_arm",
        "order_id",
        "verdict_mapping",
        "whitespace_id",
        "missing_field",
    ],
)
def test_full_preference_report_binds_each_order(monkeypatch, tmp_path, order, fault):
    _assert_preference_binding(
        monkeypatch,
        tmp_path,
        lambda report: _mutate_preference_binding(report[order], fault),
    )


def _mutate_preference_binding(order, fault):
    request = order["request"]
    manifest = request["request_manifest"]
    if fault in ("A_arm", "B_arm"):
        order[fault] = "candidate" if order[fault] == "baseline" else "baseline"
    elif fault == "order_id":
        order[fault] = "private-wrong-order"
    elif fault == "empty_transport":
        order["transport_request"] = {}
    elif fault == "generation":
        order["transport_request"]["temperature"] = 1
        manifest["generation_parameters"]["temperature"] = 1
    elif fault == "prompt":
        request["prompt_sha256"] = manifest["prompt_sha256"] = "f" * 64
    elif fault == "frame":
        request["frames"][0]["sha256"] = "f" * 64
        manifest["frames"] = deepcopy(request["frames"])
    elif fault == "manifest":
        manifest["schema_version"] = "wrong-schema"
    elif fault == "missing_field":
        del order["request"]["frames"]
    elif fault in ("verdict_mapping", "whitespace_id"):
        _mutate_preference_binding_response(order, fault)
    request["request_manifest_sha256"] = vlm_eval._sha256_json(manifest)
    order["transport_request_sha256"] = vlm_eval._sha256_json(
        order["transport_request"]
    )
    if fault == "digest":
        order["transport_request_sha256"] = "0" * 64


def _mutate_preference_binding_response(order, fault):
    provider = order["provider"]
    raw = json.loads(provider["raw_response"])
    if fault == "verdict_mapping":
        order["verdict"]["preference"] = "A"
        raw["choices"][0]["message"]["content"] = json.dumps(order["verdict"])
    else:
        raw["id"] = provider["provider_request_id"] = "   "
    provider["raw_response"] = json.dumps(raw)
    provider["raw_response_sha256"] = vlm_eval._sha256_text(provider["raw_response"])


def test_preference_inputs_are_frozen_before_child_changes_pixels(
    monkeypatch, tmp_path
):
    _assert_preference_binding(
        monkeypatch, tmp_path, lambda report: None, replace_pixels=True
    )


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "task",
        "rubric",
        "rubric_path",
        "baseline_path",
        "candidate_path",
        "endpoint_url",
        "api_key_env",
    ],
)
@pytest.mark.parametrize("value", [None, [], {}, True, 1, 1.0])
def test_malformed_preference_request_fails_before_child(
    monkeypatch, tmp_path, capsys, field, value
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["blinded-preference"]["request"][field] = value
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    assert not any(json.loads(body)["counts"].values())
    assert str(tmp_path) not in body + capsys.readouterr().out


@pytest.mark.timeout(5)
@pytest.mark.parametrize("field", ["baseline_path", "candidate_path", "rubric_path"])
@pytest.mark.parametrize("kind", ["fifo", "loop"])
def test_nonregular_preference_inputs_fail_before_execution(
    monkeypatch, tmp_path, capsys, field, kind
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    invalid = tmp_path / "private-invalid-input"
    if kind == "fifo":
        os.mkfifo(invalid)
    else:
        invalid.symlink_to(invalid.name)
    config = json.loads(path.read_text())
    config["cases"]["blinded-preference"]["request"][field] = str(invalid)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    assert not any(json.loads(body)["counts"].values())
    assert str(tmp_path) not in body + capsys.readouterr().out


@pytest.mark.parametrize(
    "value", [None, [], {}, True, "120", 0, -1, float("inf"), float("nan")]
)
def test_preference_timeout_requires_literal_positive_number(
    monkeypatch, tmp_path, capsys, value
):
    test_malformed_preference_request_fails_before_child(
        monkeypatch, tmp_path, capsys, "timeout_s", value
    )


@pytest.mark.parametrize("kind", ["paired", "preference"])
def test_request_pair_list_is_not_coerced_into_configuration(
    monkeypatch, tmp_path, kind
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    control = config["cases"]["blinded-preference"]
    control["request"] = list(control["request"].items())
    config["cases"] = {runner.AUDIT_CASES_BY_KIND[kind]: control}
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "invalid_audit_request"
    assert not receipt["passed"] and not any(receipt["counts"].values())
