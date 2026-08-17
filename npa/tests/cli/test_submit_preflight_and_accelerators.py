"""CLI behavior for the two submit-time gates added after the PAIDF walkthrough."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
from typer.testing import CliRunner
import yaml

from npa.cli.main import app
from npa.cli.workbench import workflow as workflow_cli
from npa.orchestration.npa_workflow.submit import load_spec_for_submit
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.skypilot.image_bootstrap_contract import (
    ATTESTATION_LABEL,
    CONTRACT_VERSION,
    ImageContractEvidence,
    store_cached_evidence,
)
from npa.orchestration.skypilot.k8s_gpu_catalog import KubernetesGpuCatalog
from npa.orchestration.skypilot.registry_preflight import ImagePullCheck
from npa.orchestration.skypilot.workflow import SkyPilotSubmitError


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[3]
OPENPI_FOUR_MODE_SPEC = (
    REPO_ROOT
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
    / "openpi-pi05-four-mode.yaml"
)

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


class _RecordingOperation:
    def __init__(self) -> None:
        self.rollback: dict[str, object] | None = None
        self.transitions: list[tuple[str, dict[str, object]]] = []

    def record_rollback(self, **kwargs) -> None:  # noqa: ANN003
        self.rollback = kwargs

    def transition(self, phase: str, **kwargs) -> None:  # noqa: ANN003
        self.transitions.append((phase, kwargs))


def test_prelaunch_reconciliation_failure_does_not_leave_recovery_blocker() -> None:
    operation = _RecordingOperation()
    transaction = type("Transaction", (), {"launch_sequence": 0})()
    error = SkyPilotSubmitError("controller identity mismatch", transaction=transaction)

    workflow_cli._record_workflow_submit_failure(operation, error)

    assert operation.rollback == {
        "attempted": False,
        "completed": True,
        "removed": [],
        "preserved": [],
        "outcomes": [],
    }
    assert operation.transitions[0][0] == "rolled-back"
    assert operation.transitions[0][1]["details"]["launch_attempted"] is False


def test_postlaunch_failure_preserves_recovery_blocker() -> None:
    operation = _RecordingOperation()
    transaction = type("Transaction", (), {"launch_sequence": 1})()
    error = SkyPilotSubmitError("launch indeterminate", transaction=transaction)

    workflow_cli._record_workflow_submit_failure(operation, error)

    assert operation.rollback is None
    assert operation.transitions == [
        ("recovery-required", {"error": "launch indeterminate"})
    ]


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


def test_submit_accelerator_readiness_uses_resolved_config_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    spec = yaml.safe_load(yaml.safe_dump(SPEC))
    spec["config"].update({"gpu_type": "RTXPRO6000", "gpu_count": "8"})
    spec["resources"]["gpu"]["accelerators"] = (
        "{{config.gpu_type}}:{{config.gpu_count}}"
    )
    path = tmp_path / "templated-accelerator.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)
    merged_spec = load_spec_for_submit(path, config_overrides={"gpu_count": "1"})

    overrides = workflow_cli._resolve_submit_accelerators(
        path,
        spec=merged_spec,
        infra="k8s/npa-cluster",
        sky_bin=sky_bin,
        enabled=True,
    )

    assert overrides == {"RTXPRO6000:1": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1"}


@pytest.mark.parametrize(
    ("config_overrides", "expected"),
    [({}, "B200:1"), ({"gpu_type": "H200", "gpu_count": "2"}, "H200:2")],
)
def test_openpi_readiness_uses_fully_resolved_planned_profiles(
    monkeypatch: pytest.MonkeyPatch,
    sky_bin: str,
    config_overrides: dict[str, str],
    expected: str,
) -> None:
    from npa.orchestration.skypilot import k8s_gpu_catalog

    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    seen: list[list[str]] = []

    def wait(requested, **_kwargs):  # noqa: ANN001
        seen.append(list(requested))
        return {
            value: k8s_gpu_catalog.AcceleratorResolution(
                requested=value,
                resolved=value,
                remapped=False,
                catalog=k8s_gpu_catalog.KubernetesGpuCatalog(
                    quantities_by_accelerator={
                        value.rsplit(":", 1)[0]: frozenset(
                            {int(value.rsplit(":", 1)[1])}
                        )
                    }
                ),
            )
            for value in requested
        }

    monkeypatch.setattr(k8s_gpu_catalog, "wait_for_kubernetes_accelerators", wait)
    merged = load_spec_for_submit(
        OPENPI_FOUR_MODE_SPEC, config_overrides=config_overrides
    )

    overrides = workflow_cli._resolve_submit_accelerators(
        OPENPI_FOUR_MODE_SPEC,
        spec=merged,
        infra="k8s/npa-cluster",
        sky_bin=sky_bin,
        enabled=True,
    )

    assert seen == [[expected]]
    assert overrides == {}
    assert "{{config." not in seen[0][0]


def test_submit_refuses_two_gpus_per_task_on_single_gpu_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sky_bin: str
) -> None:
    monkeypatch.delenv("NPA_WORKFLOW_GPU_ACCELERATOR", raising=False)
    spec = {**SPEC}
    spec["resources"] = {"gpu": {"cloud": "kubernetes", "accelerators": "RTXPRO6000:2"}}
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

    result = runner.invoke(
        app, ["workbench", "workflow", "gpus", "--context", "npa-cluster"]
    )

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
        [
            "workbench",
            "workflow",
            "gpus",
            "--context",
            "npa-cluster",
            "--spec",
            str(spec_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "RTXPRO6000:1 -> RTXPRO-6000-BLACKWELL-SERVER-EDITION:1" in result.output


def test_workflow_gpus_json_reports_the_exact_alias_resolution(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, sky_bin: str
) -> None:
    monkeypatch.setenv("NPA_SKYPILOT_BIN", sky_bin)
    _stub_catalog(monkeypatch, CATALOG_OUTPUT)

    result = runner.invoke(
        app,
        [
            "workbench",
            "workflow",
            "gpus",
            "--context",
            "npa-cluster",
            "--spec",
            str(spec_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["spec_resolutions"] == [
        {
            "requested": "RTXPRO6000:1",
            "resolved": "RTXPRO-6000-BLACKWELL-SERVER-EDITION:1",
            "remapped": True,
        }
    ]


def _stub_pull(monkeypatch: pytest.MonkeyPatch, checks: list[ImagePullCheck]) -> None:
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        lambda images, **kwargs: checks,
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.skypilot_render.plan_images",
        lambda *args, **kwargs: [check.image for check in checks],
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.skypilot_render.plan_image_pull_secrets",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        workflow_cli,
        "_preflight_image_bootstrap_contracts",
        lambda *, images, **_kwargs: [
            {
                "image": f"{image.split(':', 1)[0]}@sha256:{'a' * 64}",
                "state": "compatible",
            }
            for image in images
        ],
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


def test_image_preflight_plans_with_submit_config_overrides(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path
) -> None:
    observed: dict[str, str] = {}

    def plan_images(spec, *_args, **_kwargs):
        observed["runtime_image"] = str(spec.config["runtime_image"])
        return []

    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.skypilot_render.plan_images", plan_images
    )
    monkeypatch.setattr(
        "npa.orchestration.npa_workflow.skypilot_render.plan_image_pull_secrets",
        lambda *_args, **_kwargs: {},
    )
    digest = "cr.example/openpi@sha256:" + "b" * 64

    assert (
        workflow_cli._preflight_submit_images(
            spec_path,
            spec=load_spec_for_submit(
                spec_path, config_overrides={"runtime_image": digest}
            ),
            options=object(),
            assume_decision="",
            enabled=True,
        )
        == {}
    )

    assert observed == {"runtime_image": digest}


def test_image_preflight_includes_reject_only_image_and_pull_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reject_image = (
        "cr.us-central1.nebius.cloud/example-registry/npa-cosmos3:reject-only"
    )
    spec_path = tmp_path / "dynamic.yaml"
    spec_path.write_text(
        """
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata: {name: dynamic-image-preflight}
config:
  bucket: example-bucket
  prefix: runs/test
  cosmos3_mode: text2image
  prompt: test
  output_uri: s3://example-bucket/out/
  cosmos3_checkpoint: Cosmos3-Nano
  cosmos3_input_path: ''
  cosmos3_seed: 1
  cosmos3_guidance: 5
  cosmos3_steps: 1
  decision_uri: s3://example-bucket/decision.json
resources:
  cpu: {cloud: kubernetes, cpus: 1, memory: 1Gi}
  gpu: {cloud: kubernetes, accelerators: H100:1, cpus: 1, memory: 1Gi}
initial: route
states:
  route:
    writesDecision: true
    resources: cpu
    run: {shell: echo route}
    transitions:
      - {when: promote_checkpoint, goto: accept}
      - {when: loop_back, goto: reject}
  accept:
    resources: cpu
    run: {shell: echo accept}
    terminal: true
  reject:
    resources: gpu
    toolRef: workbench.cosmos3.generate
    terminal: true
""",
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def check(images, **kwargs):
        observed["images"] = list(images)
        observed["pull_secrets_by_image"] = kwargs["pull_secrets_by_image"]
        return [ImagePullCheck(image=image, status="ok", http_status=200) for image in images]

    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.check_image_pulls_with_credentials",
        check,
    )
    monkeypatch.setattr(
        workflow_cli,
        "_preflight_image_bootstrap_contracts",
        lambda *, images, **_kwargs: [
            {"image": image, "state": "compatible"} for image in images
        ],
    )

    workflow_cli._preflight_submit_images(
        spec_path,
        options=SkypilotRenderOptions(
            image_overrides={"workbench.cosmos3.generate": reject_image}
        ),
        assume_decision="promote_checkpoint",
        enabled=True,
        infra="k8s/example-context",
    )

    assert reject_image in observed["images"]
    assert observed["pull_secrets_by_image"] == {
        reject_image: ("npa-nebius-registry",)
    }


def test_first_party_image_without_attestation_fails_instead_of_probing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = "sha256:" + "a" * 64
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_REGISTRY", "cr.us-central1.nebius.cloud/u000")
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.resolve_registry_credentials",
        lambda *_args, **_kwargs: ("iam", "opaque"),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.fetch_image_config_metadata",
        lambda *_args, **_kwargs: (digest, {}),
    )

    def probe_forbidden(**_kwargs):
        raise AssertionError(
            "first-party missing evidence must not use a runtime probe"
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.image_bootstrap_contract.probe_image_capabilities",
        probe_forbidden,
    )
    with pytest.raises(Exception) as excinfo:
        workflow_cli._preflight_image_bootstrap_contracts(
            images=[NEBIUS_IMAGE],
            pull_checks=[
                ImagePullCheck(
                    image=NEBIUS_IMAGE,
                    status="ok",
                    http_status=200,
                    digest=digest,
                )
            ],
            context="exact-context",
        )
    assert excinfo.type.__name__ == "Exit"


def test_groot_label_and_label_backed_cache_cannot_bypass_runtime_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Canonical and repaired GR00T use one repo, so only selected bytes are evidence."""

    digest = "sha256:" + "b" * 64
    image = "cr.us-central1.nebius.cloud/u000/npa-groot:0.1.0-sky1"
    immutable = image.rsplit(":", 1)[0] + "@" + digest
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("NPA_REGISTRY", "cr.us-central1.nebius.cloud/u000")
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.resolve_registry_credentials",
        lambda *_args, **_kwargs: ("iam", "opaque"),
    )
    monkeypatch.setattr(
        "npa.orchestration.skypilot.registry_preflight.fetch_image_config_metadata",
        lambda *_args, **_kwargs: (
            digest,
            {ATTESTATION_LABEL: CONTRACT_VERSION},
        ),
    )
    cache_path = tmp_path / ".npa" / "cache" / "sky-image-bootstrap.json"
    store_cached_evidence(
        cache_path,
        ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="compatible",
            source="oci_attestation",
        ),
    )
    calls: list[tuple[str, str, str]] = []

    def probe(*, image: str, digest: str, context: str, **_kwargs):
        calls.append((image, digest, context))
        return ImageContractEvidence(
            image=immutable,
            digest=digest,
            contract_version=CONTRACT_VERSION,
            state="compatible",
            source="ephemeral_capability_probe",
            checks=("runtime_capabilities",),
            cleanup="deleted",
        )

    monkeypatch.setattr(
        "npa.orchestration.skypilot.image_bootstrap_contract.probe_image_capabilities",
        probe,
    )

    result = workflow_cli._preflight_image_bootstrap_contracts(
        images=[image],
        pull_checks=[
            ImagePullCheck(
                image=image,
                status="ok",
                http_status=200,
                digest=digest,
            )
        ],
        context="exact-context",
    )

    assert calls == [(image, digest, "exact-context")]
    assert result[0]["source"] == "ephemeral_capability_probe"


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


def test_parse_exact_image_overrides() -> None:
    assert workflow_cli._parse_image_overrides(
        [
            "workbench.fiftyone.curate_augmented=cr.example/fiftyone@sha256:abc",
            "workbench.cosmos_evaluator.evaluate=cr.example/evaluator@sha256:def",
        ]
    ) == {
        "workbench.fiftyone.curate_augmented": "cr.example/fiftyone@sha256:abc",
        "workbench.cosmos_evaluator.evaluate": "cr.example/evaluator@sha256:def",
    }


@pytest.mark.parametrize(
    "values",
    [
        ["missing-equals"],
        ["*=cr.example/all:latest"],
        ["workbench.tool="],
        ["workbench.tool=one", "workbench.tool=two"],
    ],
)
def test_parse_exact_image_overrides_rejects_ambiguous_input(
    values: list[str],
) -> None:
    with pytest.raises(ValueError):
        workflow_cli._parse_image_overrides(values)


def test_catalog_helper_reports_max_per_node() -> None:
    catalog = KubernetesGpuCatalog(
        quantities_by_accelerator={"H100": frozenset({1, 2, 8})}
    )

    assert catalog.max_per_node("H100") == 8
    assert catalog.max_per_node("A100") == 0


def test_image_and_capacity_checks_run_before_any_provisioning() -> None:
    """A registry missing the workbench images must cost no cluster time.

    The check only needs the registry and the --image overrides, never the
    cluster, so it belongs ahead of deployIfAbsent.
    """

    import inspect

    source = inspect.getsource(workflow_cli.submit_cmd)
    capacity_at = source.index("resolved_deploy_plans = plan_infra_present(")
    image_at = source.index("_preflight_submit_images(")
    mutation_at = source.index("records = ensure_infra_present(")

    assert capacity_at < image_at < mutation_at


def test_a_missing_workbench_image_carries_its_build_command(
    monkeypatch: pytest.MonkeyPatch, spec_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from npa.deploy.images import supported_tool_version
    from npa.orchestration.skypilot.registry_preflight import check_image_pull

    class Missing:
        def __call__(self, url, headers, timeout):  # noqa: ANN001 - test stub
            if "Authorization" not in headers:
                return (
                    401,
                    {
                        "www-authenticate": 'Bearer realm="https://cr.x/v2/token/",service="cr.x"'
                    },
                    b"",
                )
            if "/v2/token/" in url:
                import json as _json

                return 200, {}, _json.dumps({"token": "t"}).encode()
            return (
                404,
                {},
                b'{"errors":[{"code":"MANIFEST_UNKNOWN","message":"unknown"}]}',
            )

    check = check_image_pull(
        "cr.eu-north1.nebius.cloud/e000/npa-cosmos-curate:some-old-tag",
        password="token",
        fetcher=Missing(),
    )

    assert check.status == "not_found"
    assert "docker buildx build" in check.remedy
    assert (
        f"npa-cosmos-curate:{supported_tool_version('cosmos-curate')}" in check.remedy
    )
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

    assert (
        workflow_cli._resolve_submit_registry("cr.explicit/x", "p") == "cr.explicit/x"
    )


def test_npa_registry_env_wins_over_project_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
