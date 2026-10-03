"""Test configured audit lane failure gates without treating mocks as live evidence."""

import importlib.util
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from PIL import Image

from npa.workbench import vlm_eval
from npa.live_verification.vlm_audit_controls import audit_controls


def _runner():
    path = Path(__file__).resolve().parents[2] / "scripts/vlm_audit_live_recheck.py"
    spec = importlib.util.spec_from_file_location("audit_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(monkeypatch, tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "control.png")
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps(
            {
                "cases": {
                    "paired-judges": {
                        "request": {
                            "input_path": str(tmp_path / "control.png"),
                            "primary_model": "first/model",
                            "secondary_model": "second/model",
                        },
                        "expectations": {
                            "primary.result.passed": False,
                            "secondary.result.passed": False,
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


def _report(target, index=0, config=None):
    if config is None:
        config = json.loads((target / "audit-config.json").read_text())
    controls = list(audit_controls(config["cases"]["paired-judges"]).values())
    request = controls[min(index, len(controls) - 1)]["request"]
    if index >= len(controls):
        request = {**request, "output_path": str(target / "paired-judges" / str(index))}
    with patch.object(
        vlm_eval,
        "_post_with_readiness_retry",
        side_effect=lambda **kwargs: _synthetic_response(kwargs["request"]["model"]),
    ):
        return asdict(
            vlm_eval.compare_vlm_judges(vlm_eval.VlmJudgeComparisonRequest(**request))
        )


def _synthetic_response(model):
    body = {
        "id": "synthetic-request",
        "model": model,
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {"success": False, "score": 0, "rationale": "synthetic"}
                    )
                },
            }
        ],
    }
    return vlm_eval._VlmBackendResponse(body, json.dumps(body), 200, None, 0.1)


@pytest.mark.parametrize(
    "mutation", ["wrong-kind", "unknown-kind", "mixed", "unknown-case"]
)
def test_invalid_audit_selection_cannot_execute_or_pass(
    monkeypatch, tmp_path, mutation
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    if mutation == "wrong-kind":
        config["audit_kind"] = "preference"
    elif mutation == "unknown-kind":
        config["audit_kind"] = "unknown"
    elif mutation == "mixed":
        config["cases"]["blinded-preference"] = config["cases"]["paired-judges"]
    else:
        config["cases"] = {"unknown": config["cases"]["paired-judges"]}
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--audit-kind", "paired", "--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "invalid_audit_case_selection"
    assert receipt["passed"] is False and not any(receipt["counts"].values())


def _write_passing_execution(target, count):
    (target / "execution.json").write_text(
        json.dumps(
            {
                "collected": count,
                "executed": count,
                "passed": count,
                "failed": 0,
                "skipped": 0,
                "deselected": 0,
                "xfail": False,
            }
        )
    )
    (target / "execution.json").chmod(0o600)


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("config", "missing_audit_configuration"),
        ("key", "missing_audit_credential"),
        ("permissions", "audit_configuration_must_be_owner_only"),
        ("expectations", "missing_frozen_judge_expectations"),
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
        value["cases"]["paired-judges"]["expectations"] = {}
        config.write_text(json.dumps(value))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == reason
    assert not receipt["passed"]
    assert not any(receipt["counts"].values())


@pytest.mark.parametrize(
    "counts,exit_code,passed",
    [
        (
            {"collected": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0},
            0,
            True,
        ),
        (
            {"collected": 1, "executed": 0, "passed": 0, "failed": 0, "skipped": 1},
            0,
            False,
        ),
        (
            {"collected": 1, "executed": 1, "passed": 0, "failed": 1, "skipped": 0},
            1,
            False,
        ),
        (
            {"collected": 0, "executed": 0, "passed": 0, "failed": 0, "skipped": 0},
            0,
            False,
        ),
        (
            {"collected": 1, "executed": 1, "passed": 1, "failed": 0, "skipped": 0},
            2,
            False,
        ),
    ],
)
def test_receipt_requires_real_execution_counts(
    monkeypatch, tmp_path, counts, exit_code, passed
):
    runner = _runner()
    original = _config(monkeypatch, tmp_path)
    before = original.read_bytes()

    def execute(root, target, config):
        prepared = json.loads(config.read_text())["cases"]["paired-judges"]["request"]
        assert prepared["output_path"] == str(target / "paired-judges")
        (target / "execution.json").write_text(
            json.dumps({**counts, "deselected": 0, "xfail": False})
        )
        (target / "execution.json").chmod(0o600)
        directory = target / "paired-judges"
        directory.mkdir()
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).write_text(
            json.dumps(_report(target))
        )
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).chmod(0o600)
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
    with pytest.raises(SystemExit) as raised:
        runner.main(["--evidence-dir", str(target)])
    assert raised.value.code == 2


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


@pytest.mark.parametrize("generated", [False, True])
def test_actual_entrypoint_symlink_loop_target_is_sanitized(tmp_path, generated):
    loop = tmp_path / "private-output-loop"
    loop.symlink_to(loop.name)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts/vlm_audit_live_recheck.py"),
        "--evidence-dir",
        str(loop / "evidence"),
    ]
    if generated:
        command.extend(["--generated-controls", "--audit-kind", "paired"])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "audit_evidence_path_invalid" in result.stderr
    assert "Traceback" not in result.stderr
    assert str(tmp_path) not in result.stdout + result.stderr
    assert loop.name not in result.stdout + result.stderr


def test_symlink_loop_config_fails_closed_without_private_paths(
    monkeypatch, tmp_path, capsys
):
    runner = _runner()
    loop = tmp_path / "private-config-loop"
    loop.symlink_to(loop.name)
    monkeypatch.setenv(runner.CONFIG_ENV, str(loop))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = (target / "receipt.json").read_text()
    assert json.loads(receipt)["failure"] == "audit_configuration_or_execution_failed"
    assert not any(json.loads(receipt)["counts"].values())
    public = receipt + capsys.readouterr().out
    assert str(tmp_path) not in public
    assert loop.name not in public


def test_symlink_loop_input_fails_before_execution(monkeypatch, tmp_path, capsys):
    runner = _runner()
    config = _config(monkeypatch, tmp_path)
    loop = tmp_path / "private-input-loop"
    loop.symlink_to(loop.name)
    value = json.loads(config.read_text())
    case = value["cases"]["paired-judges"]
    case["request"]["input_path"] = str(loop)
    # Freezing the actual input rejects the loop before child/provider execution.
    case["input_sha256"] = "0" * 64
    config.write_text(json.dumps(value))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = (target / "receipt.json").read_text()
    counts = json.loads(receipt)["counts"]
    assert not any(counts.values())
    assert not (target / "pytest.log").exists()
    public = receipt + capsys.readouterr().out
    assert str(tmp_path) not in public
    assert loop.name not in public


def test_generated_control_loop_setup_is_sanitized(monkeypatch, tmp_path):
    runner = _runner()
    (tmp_path / "controls").symlink_to("controls")
    monkeypatch.setattr(
        runner, "_scheduled_preflight", lambda: pytest.fail("must not contact provider")
    )
    receipt = {"passed": False, "failure": None}
    runner._complete_receipt(tmp_path, tmp_path, receipt, generated=True)
    assert receipt["failure"] == "audit_configuration_or_execution_failed"
    assert not receipt["passed"]
    assert str(tmp_path) not in json.dumps(receipt)


def test_unrelated_runtime_errors_are_not_suppressed(monkeypatch, tmp_path):
    runner = _runner()

    def broken(*args, **kwargs):
        raise RuntimeError("unrelated runtime defect")

    monkeypatch.setattr(runner, "_verify", broken)
    with pytest.raises(RuntimeError, match="unrelated runtime defect"):
        runner.main(["--evidence-dir", str(tmp_path / "evidence")])


def test_unavailable_evidence_directory_is_sanitized(tmp_path, capsys):
    runner = _runner()
    unavailable = tmp_path / "private-existing-file"
    unavailable.touch()
    with pytest.raises(SystemExit) as raised:
        runner.main(["--evidence-dir", str(unavailable)])
    assert raised.value.code == 2
    public = capsys.readouterr().err
    assert "audit_evidence_directory_unavailable" in public
    assert str(unavailable) not in public


@pytest.mark.parametrize(
    "mode", ["xpass", "skip", "deselection", "setup", "teardown", "collection"]
)
def test_actual_incomplete_execution_cannot_pass_lane(tmp_path, mode):
    test_path = _incomplete_test(tmp_path, mode)
    result = _run_incomplete_test(tmp_path, test_path)
    expected = {
        "xpass": "1 xpassed",
        "skip": "1 skipped",
        "deselection": "1 deselected",
        "setup": "1 error",
        "teardown": "1 error",
        "collection": "1 error",
    }
    assert expected[mode] in result.stdout
    assert result.returncode == 1
    counts = json.loads((tmp_path / "execution.json").read_text())
    if mode in ("setup", "teardown", "collection"):
        assert counts["failed"] == 1
        assert counts["passed"] == 0
    if mode == "teardown":
        assert counts["executed"] == 1
    elif mode in ("setup", "collection"):
        assert counts["executed"] == 0


def _incomplete_test(tmp_path, mode):
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
        body = "    raise ValueError('fixture error')\n"
        if mode == "teardown":
            body = "    yield\n" + body
        test_path.write_text(
            "import pytest\n@pytest.fixture(autouse=True)\ndef fixture():\n"
            + body
            + "def test_control():\n    pass\n"
        )
    elif mode == "collection":
        test_path.write_text("raise ValueError('collection error')\n")
    else:
        marker = "xfail(strict=False)" if mode == "xpass" else "skip(reason='contract')"
        test_path.write_text(
            f"import pytest\n@pytest.mark.{marker}\ndef test_control():\n    pass\n"
        )
    return test_path


def _run_incomplete_test(tmp_path, test_path):
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
    return subprocess.run(
        [sys.executable, "-c", code, str(runner_path), str(test_path), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )


@pytest.mark.parametrize(
    "field", ["collected", "executed", "passed", "failed", "skipped", "deselected"]
)
@pytest.mark.parametrize("invalid", [True, -1, 1.0, "1", None])
def test_report_counts_reject_invalid_values(field, invalid):
    runner = _runner()
    execution = dict.fromkeys(
        ("collected", "executed", "passed", "failed", "skipped", "deselected"), 0
    )
    execution.update(xfail=False)
    execution[field] = invalid
    with pytest.raises(ValueError, match="invalid_audit_execution_counts"):
        runner._counts(execution)


def test_report_counts_require_explicit_xfail_boolean():
    runner = _runner()
    execution = dict.fromkeys(
        ("collected", "executed", "passed", "failed", "skipped", "deselected"), 0
    )
    with pytest.raises(ValueError, match="invalid_audit_execution_counts"):
        runner._counts(execution)


@pytest.mark.parametrize("invalid", [0, 1, None, "", "false", [], {}])
def test_falsy_malformed_xfail_cannot_pass_receipt(monkeypatch, tmp_path, invalid):
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
                    "xfail": invalid,
                }
            )
        )
        (target / "execution.json").chmod(0o600)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "invalid_audit_execution_counts"
    assert receipt["passed"] is False


@pytest.mark.parametrize(
    "field,invalid",
    [
        (field, value)
        for field in ("passed", "escalation_required", "primary", "secondary")
        for value in (0, 1, None, "", "false", [], {})
    ]
    + [
        (field, value)
        for field in ("primary_score", "secondary_score")
        for value in (
            True,
            False,
            "1",
            None,
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            10**1000,
        )
    ],
)
def test_malformed_summary_scalar_fails_with_sanitized_receipt(
    monkeypatch, tmp_path, capsys, field, invalid
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        _write_passing_execution(target, 1)
        report = _report(target)
        if field in ("primary", "secondary"):
            report[field]["result"]["passed"] = invalid
        elif field.endswith("_score"):
            report[field.removesuffix("_score")]["result"]["score"] = invalid
        else:
            report[field] = invalid
        directory = target / "paired-judges"
        directory.mkdir()
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).write_text(
            json.dumps(report)
        )
        (directory / runner.JUDGE_COMPARISON_RESULT_FILENAME).chmod(0o600)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "private-evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = (target / "receipt.json").read_text()
    assert json.loads(receipt)["failure"] == "audit_configuration_or_execution_failed"
    assert json.loads(receipt)["passed"] is False
    assert str(tmp_path) not in receipt + capsys.readouterr().out


@pytest.mark.parametrize(
    "field",
    [
        "primary.result.passed",
        "secondary.result.passed",
        "passed",
        "escalation_required",
        "requests_differ_only_by_model",
        "operational_rate_estimated",
    ],
)
@pytest.mark.parametrize("invalid", [0, None, "false"])
def test_frozen_boolean_expectations_reject_coercion(
    monkeypatch, tmp_path, field, invalid
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["paired-judges"]["expectations"][field] = invalid
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = json.loads((target / "receipt.json").read_text())
    assert receipt["failure"] == "missing_frozen_judge_expectations"
    assert not any(receipt["counts"].values())


@pytest.mark.parametrize("malformed", [False, 0, "", []])
def test_boolean_containers_fail_with_bounded_errors(monkeypatch, malformed):
    runner = _runner()
    with pytest.raises(ValueError, match="invalid_audit_execution_counts"):
        runner._counts(malformed)
    with pytest.raises(ValueError, match="invalid_audit_judge_summary"):
        runner._public_judge({"result": malformed})
    monkeypatch.setenv("VLM_EVAL_API_KEY", "synthetic-key")
    control = {
        "request": {
            "input_path": "unused.png",
            "primary_model": "first/model",
            "secondary_model": "second/model",
        },
        "expectations": malformed,
    }
    with pytest.raises(ValueError, match="missing_frozen_judge_expectations"):
        runner._prepare_control(control, Path("unused"))


@pytest.mark.parametrize(
    "fault",
    [
        "missing_first",
        "wrong_index",
        "extra",
        "unknown_status",
        "inconsistent_status",
        "no_outcome",
        "both_outcomes",
        "unknown_error",
        "typed_error",
        "symlink",
        "directory_symlink",
        "wrong_expectation",
        "wrong_model",
        "mutated_config",
        "fifo",
        "readable_by_others",
        "changed_bytes",
        "execution_fifo",
        "execution_symlink",
        "config_fifo",
        "config_symlink",
    ],
)
def test_green_pytest_counts_cannot_mask_retained_artifact_faults(
    monkeypatch, tmp_path, capsys, fault
):
    runner = _runner()
    _three_controls(monkeypatch, tmp_path)
    if fault == "changed_bytes":
        _change_bytes_after_first_read(monkeypatch, runner)

    def execute(root, target, config):
        _write_passing_execution(target, 3)
        prepared = json.loads(config.read_text())
        _child_report_fault(config, target, fault)
        if fault == "mutated_config":
            _mutate_frozen_expectations(config)
        _write_fault_artifacts(runner, target, fault, prepared)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "private-evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt_text = (target / "receipt.json").read_text()
    receipt = json.loads(receipt_text)
    assert receipt["counts"]["passed"] == (0 if fault.startswith("execution_") else 3)
    assert receipt["passed"] is False
    assert str(tmp_path) not in receipt_text + capsys.readouterr().out
    if fault == "missing_first":
        assert [row["control_index"] for row in receipt["outcomes"]] == [1, 2]
    elif fault == "typed_error":
        assert receipt["outcomes"][0]["primary"]["score"] is None
        assert receipt["outcomes"][0]["primary"]["error_type"] == "transport_error"
    elif fault == "changed_bytes":
        assert receipt["failure"] == "missing_or_unexpected_audit_artifacts"
        assert [row["control_index"] for row in receipt["outcomes"]] == [0, 1, 2]


def _three_controls(monkeypatch, tmp_path):
    config_path = _config(monkeypatch, tmp_path)
    case = json.loads(config_path.read_text())["cases"]["paired-judges"]
    controls = {str(index): copy.deepcopy(case) for index in range(3)}
    config_path.write_text(
        json.dumps({"cases": {"paired-judges": {"controls": controls}}})
    )


def _child_report_fault(config, target, fault):
    if not fault.startswith(("execution_", "config_")):
        return
    path = target / "execution.json" if fault.startswith("execution_") else config
    content = path.read_bytes()
    path.unlink()
    if fault.endswith("fifo"):
        os.mkfifo(path, mode=0o600)
    else:
        elsewhere = target / "private-report.json"
        elsewhere.write_bytes(content)
        elsewhere.chmod(0o600)
        path.symlink_to(elsewhere)


def _change_bytes_after_first_read(monkeypatch, runner):
    original = runner._read_private_artifact
    observed = set()

    def read(path, target):
        content = original(path, target)
        if path.name == runner.JUDGE_COMPARISON_RESULT_FILENAME and not observed:
            observed.add(path)
            path.write_bytes(content + b"\n")
        return content

    monkeypatch.setattr(runner, "_read_private_artifact", read)


def _write_fault_artifacts(runner, target, fault, prepared):
    if fault == "directory_symlink":
        elsewhere = target / "private-elsewhere"
        elsewhere.mkdir()
        (target / "paired-judges").symlink_to(elsewhere, target_is_directory=True)
    for index in range(4 if fault == "extra" else 3):
        if index == 0 and fault == "missing_first":
            continue
        directory = (
            target
            / "paired-judges"
            / str(7 if index == 0 and fault == "wrong_index" else index)
        )
        directory.mkdir(parents=True)
        report = (
            _fault_report(fault, target, index, prepared)
            if index == 0 or fault == "mutated_config"
            else _report(target, index, prepared)
        )
        path = directory / runner.JUDGE_COMPARISON_RESULT_FILENAME
        if index == 0 and fault == "fifo":
            os.mkfifo(path, mode=0o600)
        elif index == 0 and fault == "symlink":
            private = target / "private-artifact.json"
            private.write_text(json.dumps(report))
            path.symlink_to(private)
        else:
            path.write_text(json.dumps(report))
            path.chmod(0o644 if index == 0 and fault == "readable_by_others" else 0o600)


def _fault_report(fault, target, index, prepared):
    report = _report(target, index, prepared)
    if fault in ("wrong_expectation", "mutated_config"):
        report.update(status="judges_agree_passed", passed=True)
        for judge in ("primary", "secondary"):
            report[judge]["result"].update(passed=True, score=1)
    elif fault == "wrong_model":
        report["primary"]["model"] = "different/model"
    elif fault == "unknown_status":
        report["status"] = "private-invalid-status"
    elif fault == "inconsistent_status":
        report["status"] = "judges_agree_passed"
    elif fault == "no_outcome":
        report["primary"] = {}
    elif fault == "both_outcomes":
        report["primary"]["error"] = {"error_type": "transport_error"}
    elif fault in ("typed_error", "unknown_error"):
        report["primary"] = {
            "result": None,
            "error": {
                "error_type": "transport_error"
                if fault == "typed_error"
                else "private-invalid-error"
            },
        }
        report.update(status="judge_error", escalation_required=True)
    return report


def _mutate_frozen_expectations(config):
    mutable = json.loads(config.read_text())
    for control in mutable["cases"]["paired-judges"]["controls"].values():
        control["expectations"] = {
            "primary.result.passed": True,
            "secondary.result.passed": True,
        }
    config.write_text(json.dumps(mutable))


def test_descriptor_reader_rejects_parent_symlink_without_path_precheck(tmp_path):
    runner = _runner()
    outside = tmp_path / "outside"
    outside.mkdir()
    private = outside / "private.json"
    private.write_text("private-content")
    private.chmod(0o600)
    target = tmp_path / "evidence"
    target.mkdir()
    (target / "parent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        runner._read_private_artifact(target / "parent/private.json", target)


@pytest.mark.parametrize("kind", [None, "preference", "unknown"])
def test_generated_kind_cannot_silently_select_another_lane(
    monkeypatch, tmp_path, capsys, kind
):
    runner = _runner()
    monkeypatch.setattr(
        runner, "_prepare_config", lambda *_a, **_k: pytest.fail("must not prepare")
    )
    target = tmp_path / "evidence"
    command = ["--generated-controls", "--evidence-dir", str(target)]
    if kind is not None:
        command.extend(["--audit-kind", kind])
    with pytest.raises(SystemExit) as raised:
        runner.main(command)
    assert raised.value.code == 2
    assert "supported --audit-kind paired" in capsys.readouterr().err
    assert not target.exists()


@pytest.mark.parametrize("judge", ["primary", "secondary"])
@pytest.mark.parametrize(
    "field,malformed",
    [("evidence", value) for value in (None, [], "private-invalid-evidence")]
    + [("evidence.provider", value) for value in (None, [], "private-provider")]
    + [
        ("evidence.provider.returned_model", value)
        for value in (None, True, "", " ", "wrong/model")
    ]
    + [("evidence.provider.raw_response", value) for value in (None, [], "", " ")],
)
def test_malformed_provider_artifacts_cannot_pass_receipt(
    monkeypatch, tmp_path, capsys, judge, field, malformed
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        _write_passing_execution(target, 1)
        report = _report(target)
        value = report[judge]["result"]
        components = field.split(".")
        for component in components[:-1]:
            value = value[component]
        value[components[-1]] = malformed
        directory = target / "paired-judges"
        directory.mkdir()
        path = directory / runner.JUDGE_COMPARISON_RESULT_FILENAME
        path.write_text(json.dumps(report))
        path.chmod(0o600)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "evidence"
    assert runner.main(["--audit-kind", "paired", "--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["passed"] is False and receipt["counts"]["passed"] == 1
    assert receipt["failure"] == "invalid_audit_judge_provider"
    assert "private-" not in body + capsys.readouterr().out
    assert str(tmp_path) not in body


@pytest.mark.parametrize("judge", ["primary", "secondary"])
@pytest.mark.parametrize(
    "fault",
    [
        "non_json",
        "wrong_hash",
        "wrong_model",
        "truncated",
        "refused",
        "fenced",
        "score_changed",
        "rubric_changed",
        "manifest_changed",
        "status_missing",
        "status_float",
        "request_id_missing",
        "request_id_whitespace",
    ],
)
def test_retained_semantic_fault_cannot_pass_receipt(
    monkeypatch, tmp_path, judge, fault
):
    def mutate(report):
        result = report[judge]["result"]
        provider = result["evidence"]["provider"]
        if fault == "non_json":
            provider["raw_response"] = "private-not-json"
        elif fault == "wrong_hash":
            provider["raw_response_sha256"] = "0" * 64
        elif fault == "rubric_changed":
            result["rubric"] = "private-changed-rubric"
        elif fault == "manifest_changed":
            result["evidence"]["request"]["request_manifest"]["requested_model"] = (
                "other/model"
            )
        elif fault.startswith("status_"):
            provider["status_code"] = None if fault == "status_missing" else 200.0
        elif fault in ("request_id_missing", "request_id_whitespace"):
            raw = json.loads(provider["raw_response"])
            if fault == "request_id_missing":
                raw.pop("id")
                provider["provider_request_id"] = None
            else:
                raw["id"] = provider["provider_request_id"] = "   "
            _replace_raw(provider, raw)
        else:
            _mutate_completion(provider, fault)

    _assert_retained_rejected(monkeypatch, tmp_path, mutate)


def _mutate_completion(provider, fault):
    raw = json.loads(provider["raw_response"])
    choice = raw["choices"][0]
    if fault == "wrong_model":
        raw["model"] = provider["returned_model"] = "different/model"
    elif fault == "truncated":
        choice["finish_reason"] = provider["finish_reason"] = "length"
    elif fault == "refused":
        choice["message"]["refusal"] = "private-refusal"
    elif fault == "fenced":
        choice["message"]["content"] = (
            "```json\n" + choice["message"]["content"] + "\n```"
        )
        provider["parser_version"] += "+markdown-fence-v1"
    else:
        verdict = json.loads(choice["message"]["content"])
        verdict["score"] = 1
        choice["message"]["content"] = json.dumps(verdict)
    _replace_raw(provider, raw)


def _replace_raw(provider, raw):
    provider["raw_response"] = json.dumps(raw)
    provider["raw_response_sha256"] = vlm_eval._sha256_text(provider["raw_response"])


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("value", [True, False, 1.0, "1", None, 0, -1])
@pytest.mark.parametrize("location", ["raw", "metadata", "both"])
def test_retained_usage_literals_cannot_pass_receipt(
    monkeypatch, tmp_path, field, value, location
):
    def mutate(report):
        provider = report["primary"]["result"]["evidence"]["provider"]
        raw = json.loads(provider["raw_response"])
        if location in ("metadata", "both"):
            provider["usage"][field] = value
        if location in ("raw", "both"):
            raw["usage"][field] = value
        _replace_raw(provider, raw)

    _assert_retained_rejected(monkeypatch, tmp_path, mutate)


def _assert_retained_rejected(
    monkeypatch, tmp_path, mutate, request_updates=None, replace_pixels=False
):
    runner = _runner()
    _config(monkeypatch, tmp_path)

    def execute(root, target, config):
        _write_passing_execution(target, 1)
        altered = json.loads(config.read_text())
        request = altered["cases"]["paired-judges"]["request"]
        request.update(request_updates or {})
        if replace_pixels:
            Image.new("RGB", (2, 2), "red").save(request["input_path"])
        report = _report(target, config=altered)
        mutate(report)
        directory = target / "paired-judges"
        directory.mkdir()
        path = directory / runner.JUDGE_COMPARISON_RESULT_FILENAME
        path.write_text(json.dumps(report))
        path.chmod(0o600)
        return 0

    monkeypatch.setattr(runner, "_execute", execute)
    target = tmp_path / "private-evidence"
    assert runner.main(["--audit-kind", "paired", "--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["counts"]["passed"] == 1
    assert receipt["passed"] is False and receipt["failure"] is not None
    assert str(tmp_path) not in body and "private-" not in body


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "different task"),
        ("rubric", "different rubric"),
        ("success_threshold", 0.4),
        ("frame_selection", "sequence"),
        ("max_frames", 7),
    ],
)
def test_complete_production_report_is_bound_to_frozen_options(
    monkeypatch, tmp_path, field, value
):
    _assert_retained_rejected(
        monkeypatch, tmp_path, lambda report: None, {field: value}
    )


def test_complete_production_report_is_bound_to_pre_execution_pixels(
    monkeypatch, tmp_path
):
    _assert_retained_rejected(
        monkeypatch, tmp_path, lambda report: None, replace_pixels=True
    )


@pytest.mark.parametrize("judge", ["primary", "secondary"])
@pytest.mark.parametrize("field", ["task", "rubric", "frame", "generation"])
def test_internally_rehashed_judge_cannot_change_shared_request(
    monkeypatch, tmp_path, judge, field
):
    def mutate(report):
        result = report[judge]["result"]
        request = result["evidence"]["request"]
        manifest = request["request_manifest"]
        if field in ("task", "rubric"):
            result[field] = "different private instructions"
            request["prompt_sha256"] = vlm_eval._sha256_text(
                vlm_eval._comparison_prompt(
                    result["task"],
                    result["rubric"],
                    result["frame_selection"],
                    result["frame_count"],
                )
            )
            request["rubric_sha256"] = vlm_eval._sha256_text(result["rubric"])
            manifest.update(
                prompt_sha256=request["prompt_sha256"],
                rubric_sha256=request["rubric_sha256"],
            )
        elif field == "frame":
            request["frames"][0]["sha256"] = "f" * 64
            manifest["frames"] = copy.deepcopy(request["frames"])
        else:
            manifest["generation_parameters"]["temperature"] = 1
        request["request_manifest_sha256"] = vlm_eval._sha256_json(manifest)

    _assert_retained_rejected(monkeypatch, tmp_path, mutate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deployment_status", "promoted"),
        ("operational_rate_estimated", True),
        ("operational_rate_estimated", 0),
        ("requests_differ_only_by_model", False),
        ("requests_differ_only_by_model", 1),
        ("schema_version", "wrong-schema"),
        ("common_request_sha256", "0" * 64),
        ("shared_frame_sha256", ["0" * 64]),
        ("task", "different top-level task"),
        ("rubric", "different top-level rubric"),
        ("frame_count", True),
        ("score_delta_secondary_minus_primary", False),
        ("score_delta_secondary_minus_primary", 0.5),
        ("mean_score", 0),
        ("limitations", []),
    ],
)
def test_complete_report_requires_canonical_audit_metadata(
    monkeypatch, tmp_path, field, value
):
    _assert_retained_rejected(
        monkeypatch, tmp_path, lambda report: report.update({field: value})
    )


@pytest.mark.parametrize("judge", ["primary", "secondary"])
def test_complete_report_requires_exact_transport_digest(monkeypatch, tmp_path, judge):
    _assert_retained_rejected(
        monkeypatch,
        tmp_path,
        lambda report: report[judge].update(transport_request_sha256="0" * 64),
    )


@pytest.mark.parametrize(
    "field",
    ["primary_model", "secondary_model", "input_path", "task", "rubric", "rubric_path"],
)
@pytest.mark.parametrize("value", [True, None, 42, [], {}])
def test_malformed_request_strings_fail_before_child_with_receipt(
    monkeypatch, tmp_path, capsys, field, value
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    config["cases"]["paired-judges"]["request"][field] = value
    path.write_text(json.dumps(config))
    _assert_prechild_refusal(monkeypatch, tmp_path, capsys, runner)


@pytest.mark.parametrize("field", ["input_path", "rubric_path"])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink_loop"])
def test_control_hash_and_rubric_read_reject_special_files(
    monkeypatch, tmp_path, capsys, field, kind
):
    runner = _runner()
    path = _config(monkeypatch, tmp_path)
    config = json.loads(path.read_text())
    control = config["cases"]["paired-judges"]
    special = tmp_path / "private-control"
    if kind == "fifo":
        os.mkfifo(special, mode=0o600)
    elif kind == "directory":
        special.mkdir()
    else:
        special.symlink_to(special.name)
    control["request"][field] = str(special)
    if field == "input_path":
        control["input_sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    _assert_prechild_refusal(monkeypatch, tmp_path, capsys, runner)


def _assert_prechild_refusal(monkeypatch, tmp_path, capsys, runner):
    monkeypatch.setattr(runner, "_execute", lambda *_: pytest.fail("must not execute"))
    target = tmp_path / "evidence"
    assert runner.main(["--audit-kind", "paired", "--evidence-dir", str(target)]) == 1
    body = (target / "receipt.json").read_text()
    receipt = json.loads(body)
    assert receipt["passed"] is False
    assert not any(receipt["counts"].values())
    assert str(tmp_path) not in body + capsys.readouterr().out
    assert "private-control" not in body
