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


def test_an_unreachable_cluster_times_out_without_deleting_capacity(
    monkeypatch: pytest.MonkeyPatch,
    spec_path: Path,
    sky_bin: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)

    def fake_run(cmd, **kwargs):  # noqa: ANN001 - test stub
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="no context")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(Exception) as excinfo:
        workflow_cli._resolve_submit_accelerators(
            spec_path,
            infra="k8s/npa-cluster",
            sky_bin=sky_bin,
            enabled=True,
            readiness_timeout=0.003,
            readiness_poll_interval=0.001,
        )

    assert excinfo.type.__name__ == "Exit"
    assert "Capacity was left running" in capsys.readouterr().err


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
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        lambda images, **kwargs: checks,
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


def test_a_third_party_registry_failure_blocks_submit(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path
) -> None:
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

    with pytest.raises(Exception) as excinfo:
        workflow_cli._preflight_submit_images(
            spec_path, options=object(), assume_decision="", enabled=True
        )
    assert excinfo.type.__name__ == "Exit"


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


def test_the_image_check_runs_before_any_provisioning() -> None:
    """A registry missing the workbench images must cost no cluster time.

    The check only needs the registry and the --image overrides, never the
    cluster, so it belongs ahead of deployIfAbsent.
    """

    import inspect

    source = inspect.getsource(workflow_cli.submit_cmd)
    preflight_at = source.index("_preflight_submit_images(")
    provision_at = source.index("if deploy_if_absent:")

    assert preflight_at < provision_at


def test_a_missing_workbench_image_carries_its_build_command(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.deploy.images import supported_tool_version
    from npa.orchestration.skypilot.registry_preflight import check_image_pull

    class Missing:
        def __call__(self, url, headers, timeout):  # noqa: ANN001 - test stub
            if "Authorization" not in headers:
                return 401, {"www-authenticate": 'Bearer realm="https://cr.x/v2/token/",service="cr.x"'}, b""
            if "/v2/token/" in url:
                import json as _json

                return 200, {}, _json.dumps({"token": "t"}).encode()
            return 404, {}, b'{"errors":[{"code":"MANIFEST_UNKNOWN","message":"unknown"}]}'

    check = check_image_pull(
        "cr.eu-north1.nebius.cloud/e000/npa-cosmos-curate:some-old-tag",
        password="token",
        fetcher=Missing(),
    )

    assert check.status == "not_found"
    assert "docker buildx build" in check.remedy
    assert f"npa-cosmos-curate:{supported_tool_version('cosmos-curate')}" in check.remedy
    assert "docker login" in check.remedy


def test_submit_checks_the_project_registry_not_the_first_party_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`npa configure` saves a project registry; the image pins must use it.

    Without this, preflight checked one registry while the run pulled from
    another, and the build command it printed named the wrong destination.
    """

    monkeypatch.setattr(
        "npa.clients.config.resolve_container_registry",
        lambda project=None: "cr.us-central1.nebius.cloud/u00proj",
    )

    assert (
        workflow_cli._resolve_submit_registry("", "test-rtx")
        == "cr.us-central1.nebius.cloud/u00proj"
    )


def test_an_explicit_registry_still_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "npa.clients.config.resolve_container_registry",
        lambda project=None: "cr.us-central1.nebius.cloud/u00proj",
    )

    assert workflow_cli._resolve_submit_registry("cr.explicit/x", "p") == "cr.explicit/x"


def test_npa_registry_env_wins_over_project_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "ghcr.io/nebius/nebius-physical-ai")
    monkeypatch.setattr(
        "npa.clients.config.resolve_container_registry",
        lambda project=None: "cr.us-central1.nebius.cloud/u00proj",
    )

    assert (
        workflow_cli._resolve_submit_registry("", "test-rtx")
        == "ghcr.io/nebius/nebius-physical-ai"
    )


def test_an_unreadable_config_falls_back_to_the_render_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(project=None):  # noqa: ANN001 - test stub
        raise RuntimeError("no config")

    monkeypatch.setattr("npa.clients.config.resolve_container_registry", explode)

    assert workflow_cli._resolve_submit_registry("", "p") == ""
