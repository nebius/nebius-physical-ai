"""Real subprocess guards for every production-built internal NPA argv."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import json
import yaml

from npa.cli.invocation import internal_cli_argv
from npa.project_destroy import _internal_command_argv


def _without_npa_on_path() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        value
        for value in ("/usr/bin", "/bin")
        if Path(value).is_dir()
    )
    return env


def test_console_and_module_entrypoints_share_version_contract() -> None:
    env = _without_npa_on_path()
    module = subprocess.run(
        internal_cli_argv(("--version",)),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    console = subprocess.run(
        [str(Path(sys.executable).with_name("npa")), "--version"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert module.returncode == console.returncode == 0
    assert module.stdout == console.stdout
    assert module.stderr == console.stderr == ""


def test_every_project_destroy_command_maps_to_real_cli_help() -> None:
    # One help probe per production command path. Help stops before config,
    # credentials, provider, or mutation while exercising Typer registration.
    command_paths = (
        ("workbench", "workflow", "list"),
        ("workbench", "workflow", "cancel"),
        ("agent", "destroy"),
        ("skypilot", "cleanup-controller"),
        ("cluster", "down"),
        ("storage", "bucket", "delete"),
        ("storage", "service-account", "delete"),
        ("cleanup",),
        ("configure",),
        ("destroy",),
    )
    env = _without_npa_on_path()
    for path in command_paths:
        argv = _internal_command_argv(("npa", *path, "--help"))
        result = subprocess.run(
            argv,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, (path, result.stdout, result.stderr)
        assert "Usage:" in result.stdout, path


def test_internal_invocation_is_always_active_interpreter_module() -> None:
    assert internal_cli_argv(("--help",)) == [sys.executable, "-m", "npa", "--help"]
    assert _internal_command_argv(("npa", "--help")) == internal_cli_argv(("--help",))


def test_real_destroy_cli_executes_registered_phases_without_npa_on_path(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    config = home / ".npa" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "default_project": "disposable",
                "projects": {
                    "disposable": {
                        "project_id": "project-disposable",
                        "tenant_id": "tenant-disposable",
                        "region": "region-disposable",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    env = _without_npa_on_path()
    env.update(
        {
            "HOME": str(home),
            "NPA_TEARDOWN_RECEIPT_DIR": str(tmp_path / "receipts"),
            "NPA_OPERATION_JOURNAL_DIR": str(tmp_path / "operations"),
        }
    )
    result = subprocess.run(
        internal_cli_argv(
            ("destroy", "--project", "disposable", "--all", "--yes", "--json")
        ),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    # This intentionally has no provider credentials. The real child process
    # must execute the first inventory phase, preserve its primary failure, and
    # still execute the independent local-cleanup phase. The gated disposable-
    # project lifecycle test proves the successful provider-backed path.
    assert result.returncode == 2, (result.stdout, result.stderr)
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial"
    assert [item["phase"] for item in payload["phases"]] == [
        "workflows",
        "agents",
        "controller",
        "clusters",
        "bucket",
        "storage_iam",
        "local_cleanup",
        "forget_alias",
        "final_audit",
    ]
    by_phase = {item["phase"]: item for item in payload["phases"]}
    workflows = by_phase["workflows"]
    assert workflows["status"] == "partial"
    assert len(workflows["errors"]) == 1
    assert workflows["errors"][0].startswith(
        "workflow inventory returned ambiguous JSON: command failed (exit 1); "
        "primary_stderr:"
    )
    assert workflows["evidence"]["command_results"] == [
        {
            "argv": internal_cli_argv(tuple(workflows["commands"][0][1:])),
            "exit_code": 1,
            "stderr_kind": "text",
            "stderr_summary": (
                "Error: S3 bucket is not configured. Pass --s3-bucket, "
                "--workflow-s3-uri, or configure project storage."
            ),
            "stdout_kind": "empty",
            "stdout_summary": "",
        }
    ]
    assert by_phase["local_cleanup"]["status"] == "completed"
    assert by_phase["local_cleanup"]["evidence"]["command_results"][0][
        "exit_code"
    ] == 0
    assert by_phase["controller"]["blocked_by"] == ["workflows"]
