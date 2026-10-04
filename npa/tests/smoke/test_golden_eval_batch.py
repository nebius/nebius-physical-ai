"""Tests for batch golden-eval runner."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from npa.deploy.images import CONTAINER_IMAGE_NAMES
from npa.smoke.batch import (
    iter_containers,
    run_all,
    run_container_eval,
    select_containers,
)


def test_iter_containers_excludes_unrunnable_by_default() -> None:
    names = iter_containers(include_foundation=True)
    assert "base-cuda13-b300" not in names
    assert "cosmos3-reason" not in names
    assert "ncore" not in names
    assert "wan2-2" not in names


def test_iter_containers_can_include_each_unrunnable_status() -> None:
    blocked = iter_containers(include_blocked=True)
    needs_update = iter_containers(include_needs_image_update=True)
    assert "base-cuda13-b300" in blocked
    assert "ncore" not in blocked
    assert "base-cuda13-b300" not in needs_update
    assert "ncore" in needs_update


def test_default_selection_records_every_status_exclusion() -> None:
    from npa.smoke.manifest import load_manifest

    selection = select_containers(tools_only=True, include_foundation=False)
    expected = {
        name: spec.golden_eval.status
        for name, spec in load_manifest().items()
        if name in CONTAINER_IMAGE_NAMES
        and spec.golden_eval.status in {"blocked-on-upstream", "needs-image-update"}
    }

    assert {result.name: result.status for result in selection.excluded} == expected
    assert all(result.skipped and result.skip_reason for result in selection.excluded)


def test_needs_image_update_exclusions_can_only_shrink() -> None:
    from npa.smoke.manifest import load_manifest

    reviewed_baseline = {
        "antioch",
        "cosmos-curate",
        "cosmos-evaluator",
        "cosmos3",
        "cosmos3-ray-serve",
        "curobo",
        "genesis",
        "gymnasium-robotics",
        "isaac-arena",
        "isaac-lab",
        "lerobot",
        "lerobot-vla-jepa",
        "lerobot-vlm-rl",
        "libero",
        "loop-eval",
        "ncore",
        "openpi",
        "reference-policy",
        "robocasa",
        "robotwin",
        "sam3",
        "sonic",
        "wan2-2",
    }
    current = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status == "needs-image-update"
    }

    assert current <= reviewed_baseline, (
        "new default batch exclusions need explicit review: "
        f"{sorted(current - reviewed_baseline)}"
    )


def test_iter_containers_tools_only_matches_image_names() -> None:
    from npa.smoke.manifest import load_manifest

    names = iter_containers(tools_only=True, include_foundation=False)
    unrunnable = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status in {"blocked-on-upstream", "needs-image-update"}
    }
    expected = set(CONTAINER_IMAGE_NAMES) - unrunnable
    assert set(names) == expected


def test_run_container_eval_dry_run() -> None:
    result = run_container_eval("retargeting")
    assert result.ok
    assert result.mode == "dry-run"
    assert "test_retargeting_functional" in result.command


def test_run_all_dry_run_includes_every_tool() -> None:
    from npa.smoke.manifest import load_manifest

    unrunnable = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status in {"blocked-on-upstream", "needs-image-update"}
    }
    expected = set(CONTAINER_IMAGE_NAMES) - unrunnable
    batch = run_all(
        iter_containers(tools_only=True, include_foundation=False),
        execute=False,
        serverless=False,
    )
    assert {r.name for r in batch.results} == expected
    assert batch.ok


def test_run_all_execute_workflow_smoke() -> None:
    batch = run_all(["retargeting"], execute=True, parallel=1)
    assert len(batch.results) == 1
    assert batch.results[0].ok


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_all_serverless_parallel(mock_submit) -> None:
    mock_submit.side_effect = [
        {"ok": True, "job_id": "a", "status": "completed"},
        {"ok": False, "job_id": "b", "status": "failed"},
    ]
    batch = run_all(["retargeting", "fiftyone"], serverless=True, parallel=2)
    assert mock_submit.call_count == 2
    assert not batch.ok
    assert sum(1 for r in batch.ran if r.ok) == 1


@pytest.mark.parametrize(
    ("detail", "expected_ok", "expected_exit_code"),
    [
        pytest.param({"ok": True}, True, 0, id="true"),
        pytest.param({"ok": False}, False, 1, id="false"),
        pytest.param({}, False, 1, id="missing"),
        pytest.param({"ok": "true"}, False, 1, id="true-string"),
        pytest.param({"ok": "false"}, False, 1, id="false-string"),
        pytest.param({"ok": 1}, False, 1, id="one"),
        pytest.param({"ok": 0}, False, 1, id="zero"),
        pytest.param({"ok": None}, False, 1, id="null"),
        pytest.param({"ok": []}, False, 1, id="list"),
        pytest.param({"ok": {}}, False, 1, id="mapping"),
    ],
)
@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_container_eval_serverless_requires_literal_true(
    mock_submit, detail, expected_ok, expected_exit_code
) -> None:
    mock_submit.return_value = detail

    result = run_container_eval("retargeting", serverless=True)

    assert result.ok is expected_ok
    assert result.exit_code == expected_exit_code


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_all_serverless_false_like_strings_do_not_pass(mock_submit) -> None:
    mock_submit.side_effect = [
        {"ok": True, "status": "completed"},
        {"ok": "true", "status": "completed"},
        {"ok": "false", "status": "failed"},
    ]

    batch = run_all(
        ["retargeting", "cosmos2-transfer", "fiftyone"],
        serverless=True,
        parallel=1,
    )

    summary = json.loads(batch.to_json())
    assert summary["passed"] == 1
    assert summary["failed"] == 2


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_container_eval_serverless_submit_error_continues(mock_submit) -> None:
    from npa.clients.serverless import ServerlessClientError

    mock_submit.side_effect = ServerlessClientError("job startup failed")
    result = run_container_eval("cosmos2-transfer", serverless=True)
    assert not result.ok
    assert result.detail.get("error") == "ServerlessClientError"
    assert "job startup failed" in result.detail.get("message", "")


@patch("npa.smoke.serverless_runner.submit_golden_eval")
def test_run_all_serverless_submit_error_does_not_abort_fleet(mock_submit) -> None:
    from npa.clients.serverless import ServerlessClientError

    mock_submit.side_effect = [
        {"ok": True, "job_id": "a", "status": "completed"},
        ServerlessClientError("job startup failed"),
        {"ok": True, "job_id": "c", "status": "completed"},
    ]
    batch = run_all(
        ["retargeting", "cosmos2-transfer", "fiftyone"],
        serverless=True,
        parallel=1,
    )
    assert mock_submit.call_count == 3
    assert len(batch.results) == 3
    assert sum(1 for r in batch.ran if r.ok) == 2
    cosmos = next(r for r in batch.results if r.name == "cosmos2-transfer")
    assert not cosmos.ok


def test_run_all_cli_dry_run() -> None:
    from click.utils import strip_ansi
    from typer.testing import CliRunner

    from npa.cli.main import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["workbench", "golden-eval", "run-all", "--tools-only"],
    )
    assert result.exit_code == 0, strip_ansi(result.output)
    output = strip_ansi(result.output)
    assert "lerobot" in output
    assert '"skipped"' in output
    assert "needs-image-update" in output
    assert '"passed"' in output


@pytest.mark.parametrize("command", ["run", "run-all"])
@pytest.mark.parametrize(
    ("detail", "expected_exit_code"),
    [
        ({"ok": True}, 0),
        ({"ok": False}, 1),
        ({}, 1),
        ({"ok": "false"}, 1),
        ({"ok": "true"}, 1),
        ({"ok": 1}, 1),
        ({"ok": None}, 1),
    ],
)
def test_serverless_cli_requires_literal_true(
    monkeypatch, tmp_path, command, detail, expected_exit_code
) -> None:
    from typer.testing import CliRunner

    from npa.cli.main import app

    monkeypatch.setattr(
        "npa.smoke.serverless_runner.submit_golden_eval", lambda *_a, **_kw: detail
    )
    args = ["workbench", "golden-eval", command, "retargeting", "--serverless"]
    report_path = tmp_path / "results.json"
    if command == "run-all":
        args.extend(["--json-out", str(report_path)])

    result = CliRunner().invoke(app, args)

    assert result.exit_code == expected_exit_code, result.output
    if command == "run-all":
        report = json.loads(report_path.read_text())
        assert report["ok"] is (expected_exit_code == 0)
        assert report["failed"] == expected_exit_code
        assert report["results"][0]["detail"] == detail


def test_run_all_script_dry_run() -> None:
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    python = repo / "npa" / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.skip("venv not present")
    proc = subprocess.run(
        [
            str(python),
            str(repo / "npa" / "scripts" / "run_golden_evals.py"),
            "run-all",
            "--tools-only",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "lerobot" in proc.stdout
    assert '"skipped"' in proc.stdout
    assert "needs-image-update" in proc.stdout


# --------------------------------------------------------------------------------------
# Candidate-image validation
#
# A golden eval could only ever exercise whatever the CANONICAL tag pointed at, so there
# was no way to prove a rebuilt image works before promoting its tag over the one running
# workloads resolve. That forces "promote, then test", which is backwards for an image
# whose whole point is a redistribution claim. run_container_eval now forwards a
# registry/tag override to the serverless submitter.
# --------------------------------------------------------------------------------------


def test_run_container_eval_forwards_registry_and_tag(monkeypatch) -> None:
    from npa.smoke import batch

    seen: dict[str, object] = {}

    def fake_submit(tool, **kwargs):
        seen.update({"tool": tool, **kwargs})
        return {"tool": tool, "ok": True, "status": "COMPLETED", "image": "x"}

    monkeypatch.setattr(
        "npa.smoke.serverless_runner.submit_golden_eval", fake_submit, raising=False
    )
    result = batch.run_container_eval(
        "lerobot",
        serverless=True,
        registry="registry.example/example",
        tag="0.5.1-rtfetch-rc3",
    )

    assert result.ok
    assert seen["registry"] == "registry.example/example"
    assert seen["tag"] == "0.5.1-rtfetch-rc3"


def test_run_container_eval_defaults_to_the_canonical_image(monkeypatch) -> None:
    """Overrides must be opt-in; the default must stay the pinned canonical image."""
    from npa.smoke import batch

    seen: dict[str, object] = {}

    def fake_submit(tool, **kwargs):
        seen.update(kwargs)
        return {"tool": tool, "ok": True, "status": "COMPLETED", "image": "x"}

    monkeypatch.setattr(
        "npa.smoke.serverless_runner.submit_golden_eval", fake_submit, raising=False
    )
    batch.run_container_eval("lerobot", serverless=True)

    assert seen["registry"] is None
    assert seen["tag"] is None


@pytest.mark.parametrize("execute", [False, True])
def test_candidate_overrides_fail_closed_without_serverless(execute: bool) -> None:
    from npa.smoke import batch

    result = batch.run_container_eval("lerobot", execute=execute, tag="candidate-123")

    assert not result.ok
    assert result.exit_code == 2
    assert result.detail["error"] == "CandidateOverrideRequiresServerless"
