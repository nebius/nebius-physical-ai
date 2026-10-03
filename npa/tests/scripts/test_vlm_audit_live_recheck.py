"""Test configured audit lane failure gates without treating mocks as live evidence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _runner():
    path = Path(__file__).resolve().parents[2] / "scripts/vlm_audit_live_recheck.py"
    spec = importlib.util.spec_from_file_location("audit_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(monkeypatch, tmp_path):
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


def test_symlink_loop_input_stays_in_private_log(monkeypatch, tmp_path, capsys):
    runner = _runner()
    config = _config(monkeypatch, tmp_path)
    loop = tmp_path / "private-input-loop"
    loop.symlink_to(loop.name)
    value = json.loads(config.read_text())
    case = value["cases"]["paired-judges"]
    case["request"]["input_path"] = str(loop)
    # The real child verifies input bytes before any provider call.
    case["input_sha256"] = "0" * 64
    config.write_text(json.dumps(value))
    target = tmp_path / "evidence"
    assert runner.main(["--evidence-dir", str(target)]) == 1
    receipt = (target / "receipt.json").read_text()
    counts = json.loads(receipt)["counts"]
    assert counts["collected"] == counts["executed"] == counts["failed"] == 1
    assert counts["skipped"] == counts["deselected"] == 0
    assert "Too many levels of symbolic links" in (target / "pytest.log").read_text()
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
