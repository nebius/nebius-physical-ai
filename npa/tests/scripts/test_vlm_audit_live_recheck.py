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
