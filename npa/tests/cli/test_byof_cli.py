"""CLI coverage for npa workbench byof."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
import re
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from npa.cli.main import app
from npa.cli.workbench import byof as byof_cli
from npa.cli.workbench.byof import build_byof_argv
from npa.sdk.workbench import byof as byof_sdk
from npa.orchestration.npa_workflow.robotwin_preflight import (
    MATERIALIZED_KUBECONFIG_ENV,
    MATERIALIZED_SKYPILOT_CONFIG_ENV,
    PUBLIC_CONTEXT_ENV,
    TRANSPORT_CONTEXT_ENV,
    encode_transport,
    load_runtime_authorization,
)

runner = CliRunner()


def test_byof_runner_resolves_from_staged_npa_source(
    tmp_path: Path, monkeypatch
) -> None:
    staged_module = tmp_path / "src/npa/cli/workbench/byof.py"
    staged_module.parent.mkdir(parents=True)
    staged_module.touch()
    staged_runner = tmp_path / "scripts/run_byof_repo.py"
    staged_runner.parent.mkdir()
    staged_runner.touch()
    monkeypatch.delenv("NPA_REPO_ROOT", raising=False)
    monkeypatch.setattr(byof_cli, "__file__", str(staged_module))

    assert byof_cli._script_path() == staged_runner


def test_byof_registered_in_workbench_help() -> None:
    result = runner.invoke(app, ["workbench", "--help"])
    assert result.exit_code == 0
    assert re.search(r"│\s+byof\s+", result.output)


def test_byof_run_help() -> None:
    result = runner.invoke(app, ["workbench", "byof", "run", "--help"])
    assert result.exit_code == 0
    assert "--repo-url" in result.output
    assert "--workload" in result.output
    assert "--base-profile" in result.output
    assert "--repo-auth" in result.output
    assert "--repo-token-env" in result.output
    assert "--runtime-contex" in result.output


def test_byof_run_dry_run_json() -> None:
    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/example/repo.git",
            "--repo-ref",
            "main",
            "--workload",
            "container-verify",
            "--dry-run",
            "--output",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["script"] == "npa/scripts/run_byof_repo.py"
    assert "--repo-url" in payload["argv"]
    assert "https://github.com/example/repo.git" in payload["argv"]
    assert payload["ladder"] == "docs/architecture/oss-onboarding-ladder.md"


def test_byof_ladder_and_status() -> None:
    ladder = runner.invoke(app, ["workbench", "byof", "ladder", "--output", "json"])
    assert ladder.exit_code == 0
    ladder_payload = json.loads(ladder.output)
    assert ladder_payload["tiers"][0]["tier"] == 0

    status = runner.invoke(app, ["workbench", "byof", "status", "--output", "json"])
    assert status.exit_code == 0
    status_payload = json.loads(status.output)
    assert status_payload["cli"] == "npa workbench byof"
    assert status_payload["sdk"] == "npa.sdk.workbench.byof"
    assert "workbench.byof.repo" in status_payload["tool_refs"]


def test_build_byof_argv_and_sdk_plan() -> None:
    argv = build_byof_argv(
        repo_url="https://github.com/example/repo.git",
        repo_ref="main",
        workload="container-verify",
        skip_run=True,
    )
    assert "--skip-run" in argv
    assert (
        byof_sdk.plan_argv(
            repo_url="https://github.com/example/repo.git",
            repo_ref="main",
            workload="container-verify",
            skip_run=True,
        )
        == argv
    )


def test_private_byof_sdk_plan_carries_only_token_variable_name() -> None:
    argv = byof_sdk.plan_argv(
        repo_url="https://github.com/example/private.git",
        repo_ref="main",
        repo_auth="github",
        repo_token_env="NPA_BYOF_GITHUB_TOKEN",
        skip_run=True,
    )

    assert argv[argv.index("--repo-auth") + 1] == "github"
    assert argv[argv.index("--repo-token-env") + 1] == "NPA_BYOF_GITHUB_TOKEN"
    assert "secret-canary" not in " ".join(argv)


def test_robotwin_dry_run_carries_only_runtime_context_variable_name(
    monkeypatch,
) -> None:
    variable = "NPA_BYOF_ROBOTWIN_RUNTIME_CONTEXT"
    secret = "runtime-context-secret-canary"
    monkeypatch.setenv(variable, secret)
    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/RoboTwin-Platform/RoboTwin.git",
            "--solution-name",
            "robotwin",
            "--runtime-context-env",
            variable,
            "--dry-run",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    argv = json.loads(result.output)["argv"]
    assert argv[argv.index("--runtime-context-env") + 1] == variable
    assert secret not in " ".join(argv)


def test_robotwin_direct_cli_refuses_before_loading_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TRANSPORT_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(
        byof_cli,
        "_load_runner",
        lambda: pytest.fail("direct RoboTwin CLI loaded the live BYOF runner"),
    )

    result = runner.invoke(
        app,
        [
            "workbench",
            "byof",
            "run",
            "--repo-url",
            "https://github.com/RoboTwin-Platform/RoboTwin.git",
            "--runtime-context-env",
            PUBLIC_CONTEXT_ENV,
        ],
    )

    assert result.exit_code == 2
    assert "only through normal" in result.output


def _robotwin_transport_fixture(tmp_path: Path) -> tuple[list[str], str]:
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "workflows/testing/byof-robotwin.yaml"
        ).read_text(encoding="utf-8")
    )
    config = workflow["config"]
    kubeconfig = tmp_path / "kubeconfig.yaml"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\n"
        "current-context: robotwin-context\n"
        "clusters: [{name: robotwin-cluster, cluster: {server: "
        "https://cluster.example.invalid, certificate-authority-data: Y2E=}}]\n"
        "contexts: [{name: robotwin-context, context: {cluster: "
        "robotwin-cluster, user: robotwin-user}}]\n"
        "users: [{name: robotwin-user, user: {token: portable-test-token}}]\n",
        encoding="utf-8",
    )
    skypilot = tmp_path / "skypilot.yaml"
    skypilot.write_text(
        "kubernetes:\n  allowed_contexts: [robotwin-context]\n",
        encoding="utf-8",
    )
    kubeconfig.chmod(0o600)
    skypilot.chmod(0o600)
    context = tmp_path / "runtime-context.json"
    context.write_text(
        json.dumps(
            {
                "solution": "robotwin",
                "ownership_provenance": "manager-issued",
                "workflow_sha256": "718bb6ae47c8e5e7e761303ebda9e962afa446a6b84030dade7c224cd255ece3",
                "source_revision": "96c1feab536306b50c26af200044fcdf126e8904",
                "curobo_revision": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
                "asset_revision": "785feb15aa4a4f532395ad2b1d2be5f28cb561ad",
                "runtime_lock_sha256": "c42c4037392f51ad6c2473eb3f07843738a4c5147328ace1686ddb9cf553b4ef",
                "bootstrap_image": "registry.example/robotwin-private/npa-robotwin@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "reservation": {
                    "policy": "STRICT",
                    "accelerator": "RTXPRO-6000-BLACKWELL-SERVER-EDITION",
                    "count": 1,
                },
                "license_acceptance": {
                    "nvidia_cuda_eula": True,
                    "nvidia_cudnn_sla": True,
                    "curobo_noncommercial_research_or_evaluation": True,
                    "robotwin2_aggregate_asset_and_output_terms": True,
                },
                "project": "robotwin-project-canary",
                "nebius_profile": "robotwin-profile",
                "kubeconfig": str(kubeconfig),
                "kubernetes_context": "robotwin-context",
                "skypilot_config_path": str(skypilot),
                "bucket": "robotwin-bucket-canary",
                "output_root": "s3://robotwin-bucket-canary/output",
                "run_id": "robotwin-run-canary",
            }
        ),
        encoding="utf-8",
    )
    context.chmod(0o600)
    authorization = load_runtime_authorization({PUBLIC_CONTEXT_ENV: str(context)})
    argv = build_byof_argv(
        repo_url=config["repo_url"],
        repo_ref=config["repo_ref"],
        repo_auth=config["repo_auth"],
        repo_token_env=config["repo_token_env"],
        base_profile=config["base_profile"],
        base_image=config["base_image"],
        workload=config["workload"],
        build_command=config["build_command"],
        smoke_command=config["smoke_command"],
        solution_name=config["solution_name"],
        capability_name=config["capability_name"],
        smoke_artifact_name=config["smoke_artifact_name"],
        runtime_context_env=config["runtime_context_env"],
        task=config["task"],
        iterations=config["iterations"],
        num_envs=config["num_envs"],
        num_demos=config["num_demos"],
        yaml_path=config["resource_profile_yaml"],
        output_root="s3://example-bucket/oss-solutions/robotwin",
        wait_timeout=config["wait_timeout"],
        poll_interval=config["poll_interval"],
    )
    return argv, encode_transport(authorization)


@pytest.mark.parametrize("failure", [None, RuntimeError, SystemExit])
def test_robotwin_worker_transport_materializes_owner_only_and_always_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: type[BaseException] | None,
) -> None:
    runner_module = byof_cli._load_runner()
    argv, transport = _robotwin_transport_fixture(tmp_path)
    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    materialized_paths: list[Path] = []

    expectation = pytest.raises(failure) if failure is not None else nullcontext()
    with expectation:
        with byof_cli._robotwin_runtime_materialization(
            runner_module, argv
        ) as authorization:
            assert TRANSPORT_CONTEXT_ENV not in os.environ
            context = Path(os.environ[PUBLIC_CONTEXT_ENV])
            kubeconfig = Path(os.environ[MATERIALIZED_KUBECONFIG_ENV])
            skypilot = Path(os.environ[MATERIALIZED_SKYPILOT_CONFIG_ENV])
            materialized_paths.extend((context, kubeconfig, skypilot))
            assert authorization is not None
            assert authorization.kubeconfig_source == str(kubeconfig)
            assert authorization.skypilot_config_source == str(skypilot)
            assert stat.S_IMODE(context.parent.stat().st_mode) == 0o700
            assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in materialized_paths)
            if failure is not None:
                raise failure("fixed-test-failure")

    assert all(not path.exists() for path in materialized_paths)
    assert os.environ[TRANSPORT_CONTEXT_ENV] == transport
    assert PUBLIC_CONTEXT_ENV not in os.environ


def test_robotwin_worker_transport_uses_only_authorized_internal_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_runner = byof_cli._load_runner()
    argv, transport = _robotwin_transport_fixture(tmp_path)
    observed: dict[str, object] = {}

    def run_authorized(
        received: list[str], *, authorization: object
    ) -> int:
        observed["argv"] = received
        observed["authorization"] = authorization
        return 0

    monkeypatch.setenv(TRANSPORT_CONTEXT_ENV, transport)
    monkeypatch.delenv(PUBLIC_CONTEXT_ENV, raising=False)
    monkeypatch.setattr(
        byof_cli,
        "_load_runner",
        lambda: SimpleNamespace(
            _parse_args=real_runner._parse_args,
            main=lambda *_args: pytest.fail("worker used the direct script boundary"),
            _run_authorized_robotwin=run_authorized,
        ),
    )

    result = runner.invoke(app, ["workbench", "byof", "run", *argv])

    assert result.exit_code == 0, result.output
    assert observed["argv"] == argv
    assert observed["authorization"] is not None
