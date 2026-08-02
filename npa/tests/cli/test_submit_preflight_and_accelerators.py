"""CLI behavior for the two submit-time gates added after the PAIDF walkthrough."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench import workflow as workflow_cli
from npa.orchestration.skypilot.k8s_gpu_catalog import KubernetesGpuCatalog
from npa.orchestration.skypilot.registry_preflight import ImagePullCheck


runner = CliRunner()

SPEC = {
    "apiVersion": "npa.workflow/v0.0.1",
    "kind": "Workflow",
    "metadata": {"name": "accel-demo"},
    "config": {"bucket": "demo-bucket", "prefix": "demo"},
    "resources": {
        "gpu": {
            "cloud": "kubernetes",
            "accelerators": "RTXPRO6000:1",
            "cpus": 8,
            "memory": "32Gi",
        }
    },
    "initial": "render",
    "states": {
        "render": {
            "resources": "gpu",
            "run": {"shell": "echo render"},
            "next": "done",
        },
        "done": {"terminal": True},
    },
}


@pytest.fixture()
def spec_path(tmp_path: Path) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(SPEC, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def sky_bin(tmp_path: Path) -> str:
    path = tmp_path / "sky"
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


CATALOG_OUTPUT = """Context: npa-cluster
GPU                                   REQUESTABLE_QTY_PER_NODE  UTILIZATION
RTXPRO-6000-BLACKWELL-SERVER-EDITION  1                         2 of 2 free
"""


def _stub_catalog(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_submit_remaps_the_spec_accelerator_onto_the_cluster_name(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)

    overrides = workflow_cli._resolve_submit_accelerators(
        spec_path, infra="k8s/npa-cluster", sky_bin=sky_bin, enabled=True
    )

    assert overrides == {"RTXPRO6000:1": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"}


def test_submit_refuses_two_gpus_per_task_on_single_gpu_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    spec = {**SPEC}
    spec["resources"] = {
        "gpu": {"cloud": "kubernetes", "accelerators": "RTXPRO6000:2"}
    }
    path = tmp_path / "two-gpu.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)

    with pytest.raises(Exception) as excinfo:
        workflow_cli._resolve_submit_accelerators(
            path, infra="k8s/npa-cluster", sky_bin=sky_bin, enabled=True
        )

    assert excinfo.type.__name__ == "Exit"


def test_an_explicit_env_override_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.setenv("NPA_WORKFLOW_GPU_ACCELERATOR", "H100:1")
    called = False

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        nonlocal called
        called = True
        return subprocess.CompletedProcess(cmd, 0, stdout=CATALOG_OUTPUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    overrides = workflow_cli._resolve_submit_accelerators(
        spec_path, infra="k8s/npa-cluster", sky_bin=sky_bin, enabled=True
    )

    assert overrides == {}
    assert called is False


def test_an_unreachable_cluster_does_not_block_submit(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no context")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        workflow_cli._resolve_submit_accelerators(
            spec_path, infra="k8s/npa-cluster", sky_bin=sky_bin, enabled=True
        )
        == {}
    )


def test_resolution_is_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)

    assert (
        workflow_cli._resolve_submit_accelerators(
            spec_path, infra="k8s/npa-cluster", sky_bin=sky_bin, enabled=False
        )
        == {}
    )


def test_workflow_gpus_prints_the_export_line(
    monkeypatch: pytest.MonkeyPatch, sky_bin: str
) -> None:
    monkeypatch.setenv("NPA_SKYPILOT_BIN", sky_bin)
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)

    result = runner.invoke(app, ["workbench", "workflow", "gpus", "--context", "npa-cluster"])

    assert result.exit_code == 0, result.output
    assert (
        "export NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"
        in result.output
    )
    assert "requestable per node 1" in result.output


def test_workflow_gpus_resolves_a_spec(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.setenv("NPA_SKYPILOT_BIN", sky_bin)
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)

    result = runner.invoke(
        app,
        ["workbench", "workflow", "gpus", "--context", "npa-cluster", "--spec", str(spec_path)],
    )

    assert result.exit_code == 0, result.output
    assert "RTXPRO6000:1 -> RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in result.output


def _stub_pull(monkeypatch: pytest.MonkeyPatch, checks: list[ImagePullCheck]) -> None:
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls",
        lambda images, **kwargs: checks,
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.resolve_registry_credentials",
        lambda **kwargs: ("iam", "token"),
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.skypilot_render.plan_images",
        lambda *args, **kwargs: [check.image for check in checks],
    )


NEBIUS_IMAGE = "cr.us-central1.nebius.cloud/u000/npa-cosmos2-transfer:2.5.1"


def test_a_forbidden_nebius_image_blocks_submit(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path
) -> None:
    _stub_pull(
        monkeypatch,
        [
            ImagePullCheck(
                image=NEBIUS_IMAGE,
                status="forbidden",
                http_status=403,
                detail="DENIED: permission denied",
                remedy="grant pull access",
            )
        ],
    )

    with pytest.raises(Exception) as excinfo:
        workflow_cli._preflight_submit_images(
            spec_path, options=object(), assume_decision="", enabled=True
        )

    assert excinfo.type.__name__ == "Exit"


def test_a_third_party_registry_failure_only_warns(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # nvcr.io and friends carry their own in-pod credentials; this run's Nebius
    # token says nothing about them, so they must not block submit.
    _stub_pull(
        monkeypatch,
        [
            ImagePullCheck(
                image="nvcr.io/nvidia/pytorch:24.01",
                status="no_credentials",
                http_status=401,
                remedy="supply credentials",
            )
        ],
    )

    workflow_cli._preflight_submit_images(
        spec_path, options=object(), assume_decision="", enabled=True
    )

    assert "warning" in capsys.readouterr().err


def test_pullable_images_pass(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_pull(
        monkeypatch, [ImagePullCheck(image=NEBIUS_IMAGE, status="ok", http_status=200)]
    )

    workflow_cli._preflight_submit_images(
        spec_path, options=object(), assume_decision="", enabled=True
    )

    assert "1 image(s) pullable" in capsys.readouterr().err


def test_preflight_is_skipped_when_disabled(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path
) -> None:
    def explode(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202 - must not run
        raise AssertionError("preflight ran while disabled")

    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls", explode
    )

    workflow_cli._preflight_submit_images(
        spec_path, options=object(), assume_decision="", enabled=False
    )


def test_catalog_helper_reports_max_per_node() -> None:
    catalog = KubernetesGpuCatalog(
        quantities_by_accelerator={"H100": frozenset({1, 2, 8})}
    )

    assert catalog.max_per_node("H100") == 8
    assert catalog.max_per_node("A100") == 0
