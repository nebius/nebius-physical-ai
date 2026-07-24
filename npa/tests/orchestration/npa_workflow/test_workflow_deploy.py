from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from npa.cli.main import app
from npa.orchestration.npa_workflow.deploy import (
    DeployTarget,
    ensure_infra_present,
    parse_deploy_targets,
)
from npa.orchestration.npa_workflow.errors import NpaWorkflowError
from npa.orchestration.npa_workflow.spec import NpaWorkflowSpec

runner = CliRunner()


def _spec(resources: dict[str, Any]) -> NpaWorkflowSpec:
    return NpaWorkflowSpec(
        api_version="npa.workflow/v0.0.1",
        kind="Workflow",
        metadata={"name": "t"},
        config={},
        run_defaults={},
        resources=resources,
        initial="x",
        states={},
    )


def test_parse_targets_bool_and_mapping() -> None:
    spec = _spec(
        {
            "cpu": {"cloud": "kubernetes", "cpus": 4},
            "gpu-a": {"cloud": "kubernetes", "accelerators": "RTXPRO6000:1", "deployIfAbsent": True},
            "gpu-b": {
                "cloud": "kubernetes",
                "accelerators": "H100:1",
                "deployIfAbsent": {
                    "clusterName": "npa-rtxpro-mk8s",
                    "context": "npa-rtxpro-mk8s",
                    "project": "default",
                    "skipS3": False,
                },
            },
        }
    )
    targets = {t.profile: t for t in parse_deploy_targets(spec)}
    assert set(targets) == {"gpu-a", "gpu-b"}
    assert targets["gpu-a"].cluster_name == "npa-cluster"
    assert targets["gpu-a"].resolved_context == "npa-cluster"
    assert targets["gpu-a"].skip_s3 is True
    assert targets["gpu-b"].cluster_name == "npa-rtxpro-mk8s"
    assert targets["gpu-b"].context == "npa-rtxpro-mk8s"
    assert targets["gpu-b"].project == "default"
    assert targets["gpu-b"].skip_s3 is False


def test_parse_targets_false_and_absent_are_ignored() -> None:
    spec = _spec(
        {
            "cpu": {"cloud": "kubernetes"},
            "gpu-off": {"cloud": "kubernetes", "deployIfAbsent": False},
        }
    )
    assert parse_deploy_targets(spec) == []


def test_ensure_infra_present_dedupes_by_context_and_returns_records() -> None:
    calls: list[dict[str, Any]] = []

    def fake_provisioner(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(status="ok", actions=["k8s:ensured terraform cluster npa-rtxpro-mk8s"], warnings=[])

    targets = [
        DeployTarget(profile="gpu-a", cluster_name="npa-rtxpro-mk8s", context="npa-rtxpro-mk8s"),
        DeployTarget(profile="gpu-b", cluster_name="npa-rtxpro-mk8s", context="npa-rtxpro-mk8s"),
    ]
    results = ensure_infra_present(targets, provisioner=fake_provisioner)
    # Same context => provisioned once.
    assert len(calls) == 1
    assert len(results) == 1
    assert results[0]["context"] == "npa-rtxpro-mk8s"
    assert results[0]["status"] == "ok"
    assert results[0]["actions"]


def test_ensure_infra_present_dry_run_passthrough() -> None:
    seen: dict[str, Any] = {}

    def fake_provisioner(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(status="ok", actions=["k8s:dry-run terraform apply deploy/cluster"], warnings=[])

    results = ensure_infra_present(
        [DeployTarget(profile="gpu", cluster_name="c1")],
        dry_run=True,
        provisioner=fake_provisioner,
    )
    assert seen["dry_run"] is True
    assert results[0]["dry_run"] is True


def test_ensure_infra_present_wraps_provisioner_errors() -> None:
    def boom(**_kwargs: Any) -> Any:
        raise RuntimeError("terraform exploded")

    with pytest.raises(NpaWorkflowError) as excinfo:
        ensure_infra_present([DeployTarget(profile="gpu", cluster_name="c1")], provisioner=boom)
    assert "deployIfAbsent failed" in str(excinfo.value)


def test_ensure_infra_present_empty_is_noop() -> None:
    assert ensure_infra_present([]) == []


def test_submit_help_exposes_deploy_if_absent_flag() -> None:
    result = runner.invoke(app, ["workbench", "workflow", "submit", "--help"])
    assert result.exit_code == 0
    assert "deploy-if-absent" in result.output
