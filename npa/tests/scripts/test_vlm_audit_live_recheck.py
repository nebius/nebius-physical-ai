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
        prepared = json.loads(config.read_text())["cases"]["paired-judges"]["request"]
        assert prepared["output_path"] == str(target / "paired-judges")
        (target / "pytest.xml").write_text(xml)
        (target / "execution.json").write_text(
            json.dumps(
                {
                    "collected": 1 if "<testcase" in xml else 0,
                    "executed": 1 if "<testcase" in xml else 0,
                    "deselected": 0,
                    "xfail": False,
                }
            )
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
    with pytest.raises(FileExistsError):
        runner.main(["--evidence-dir", str(target)])


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


@pytest.mark.parametrize("mode", ["xpass", "skip", "deselection"])
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
    }
    assert expected[mode] in result.stdout
    assert result.returncode == 1
    assert (tmp_path / "pytest.xml").is_file()
