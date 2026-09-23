from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from npa.workbench.nurec import evidence, runtime_attestation


IMAGE = (
    "nvcr.io/nvidia/nre/nre-ga@"
    "sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284"
)
DIGEST = IMAGE.split("@", 1)[1]


def _job(stage: str) -> tuple[str, str]:
    return f"private-run-0{1 if stage == 'reconstruct' else 2}-{stage}", (
        "41" if stage == "reconstruct" else "42"
    )


def _status(tmp_path: Path, stage: str) -> Path:
    job_name, job_id = _job(stage)
    path = tmp_path / f"{stage}-status.json"
    path.write_text(
        json.dumps(
            {
                "run_id": "private-run",
                "stages": {
                    stage: {
                        "state": "RUNNING",
                        "workflow_state": stage,
                        "managed_job_id": job_id,
                        "job_name": job_name,
                        "job_attribution": "exact",
                        "managed_job_attempts": [
                            {
                                "attempt": 1,
                                "job_id": job_id,
                                "job_name": job_name,
                                "state": "RUNNING",
                            }
                        ],
                    }
                },
            }
        )
    )
    path.chmod(0o600)
    return path


def _runner(
    *,
    image_id: str = IMAGE,
    gpu_limit: str = "1",
    gpu_label: str = "",
    accelerator_label: str = "",
    managed_job_name: str = "private-run-01-reconstruct",
    managed_job_id: str = "41",
):
    def run(command, **kwargs):
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 120
        resource = command[command.index("get") + 1]
        if resource == "pod":
            pod_name = command[command.index("pod") + 1]
            stage = "render" if "render" in pod_name else "reconstruct"
            payload = {
                "metadata": {
                    "name": pod_name,
                    "uid": f"pod-uid-{stage}",
                    "annotations": {
                        "skypilot-managed-job-name": managed_job_name,
                        "skypilot-managed-job-id": managed_job_id,
                    },
                    "labels": {
                        "skypilot-cluster-name": (
                            f"{stage}-{managed_job_id}-privatehash"
                        )
                    },
                },
                "spec": {
                    "nodeName": "node-private",
                    "containers": [
                        {
                            "name": "ray-node",
                            "image": IMAGE,
                            "resources": {"limits": {"nvidia.com/gpu": gpu_limit}},
                        }
                    ],
                },
                "status": {
                    "containerStatuses": [
                        {
                            "name": "ray-node",
                            "imageID": image_id,
                            "state": {"running": {"startedAt": "redacted"}},
                        }
                    ]
                },
            }
        else:
            assert resource == "node"
            payload = {
                "metadata": {
                    "uid": "node-uid-private",
                    "labels": {
                        "nvidia.com/gpu.product": gpu_label
                        or "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition",
                        **(
                            {"skypilot.co/accelerator": accelerator_label}
                            if accelerator_label
                            else {}
                        ),
                    },
                }
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    return run


def _observe(tmp_path: Path, stage: str) -> tuple[dict, Path]:
    path = tmp_path / f"{stage}.json"
    job_name, job_id = _job(stage)
    receipt = runtime_attestation.observe_kubernetes_stage(
        stage=stage,
        pod_name=f"private-{stage}-pod",
        namespace="private-namespace",
        container_name="ray-node",
        expected_image=IMAGE,
        managed_job_name=job_name,
        managed_job_id=job_id,
        workflow_run_id="private-run",
        workflow_status_path=_status(tmp_path, stage),
        output_path=path,
        context="private-context",
        runner=_runner(managed_job_name=job_name, managed_job_id=job_id),
    )
    return receipt, path


def test_observe_and_bundle_two_control_plane_stage_identities(
    tmp_path: Path,
) -> None:
    reconstruct, reconstruct_path = _observe(tmp_path, "reconstruct")
    render, render_path = _observe(tmp_path, "render")

    assert reconstruct["observed_image_digest"] == DIGEST
    assert render["gpu_count"] == 1
    assert "private-" not in reconstruct_path.read_text()
    bundle_path = tmp_path / "bundle.json"
    bundle = runtime_attestation.bundle_runtime_attestations(
        reconstruct_path=reconstruct_path,
        render_path=render_path,
        output_path=bundle_path,
    )

    assert (
        evidence.validate_runtime_attestation(
            bundle,
            expected_image=IMAGE,
            required_stages=("reconstruct", "render"),
        )
        == bundle
    )
    assert set(bundle["stages"]) == {"reconstruct", "render"}
    assert bundle_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("runner", "match"),
    [
        (_runner(image_id="containerd://sha256:" + "0" * 64), "digest differs"),
        (_runner(gpu_limit="2"), "one-GPU"),
        (_runner(gpu_label="NVIDIA-H100-80GB-HBM3"), "RTX PRO 6000"),
        (
            _runner(gpu_label="NVIDIA-RTX-PRO-6000-Blackwell-Workstation-Edition"),
            "Server Edition",
        ),
        (
            _runner(
                accelerator_label="NVIDIA-H100-80GB-HBM3",
            ),
            "conflicting",
        ),
    ],
)
def test_observation_rejects_wrong_runtime_identity(
    runner, match: str, tmp_path: Path
) -> None:
    job_name, job_id = _job("reconstruct")
    with pytest.raises(runtime_attestation.NurecRuntimeAttestationError, match=match):
        runtime_attestation.observe_kubernetes_stage(
            stage="reconstruct",
            pod_name="private-pod",
            namespace="private-namespace",
            container_name="ray-node",
            expected_image=IMAGE,
            managed_job_name=job_name,
            managed_job_id=job_id,
            workflow_run_id="private-run",
            workflow_status_path=_status(tmp_path, "reconstruct"),
            output_path=tmp_path / "receipt.json",
            context="private-context",
            runner=runner,
        )

    assert not (tmp_path / "receipt.json").exists()


def test_observation_rejects_unrelated_managed_job(tmp_path: Path) -> None:
    job_name, job_id = _job("reconstruct")
    with pytest.raises(
        runtime_attestation.NurecRuntimeAttestationError,
        match="not bound",
    ):
        runtime_attestation.observe_kubernetes_stage(
            stage="reconstruct",
            pod_name="private-reconstruct-pod",
            namespace="private-namespace",
            container_name="ray-node",
            expected_image=IMAGE,
            managed_job_name=job_name,
            managed_job_id=job_id,
            workflow_run_id="private-run",
            workflow_status_path=_status(tmp_path, "reconstruct"),
            output_path=tmp_path / "receipt.json",
            context="private-context",
            runner=_runner(managed_job_name=job_name, managed_job_id="40"),
        )


def test_observation_rejects_status_from_another_workflow_stage(tmp_path: Path) -> None:
    job_name, job_id = _job("reconstruct")
    wrong_status = _status(tmp_path, "render")
    with pytest.raises(
        runtime_attestation.NurecRuntimeAttestationError,
        match="exact stage managed job",
    ):
        runtime_attestation.observe_kubernetes_stage(
            stage="reconstruct",
            pod_name="private-reconstruct-pod",
            namespace="private-namespace",
            container_name="ray-node",
            expected_image=IMAGE,
            managed_job_name=job_name,
            managed_job_id=job_id,
            workflow_run_id="private-run",
            workflow_status_path=wrong_status,
            output_path=tmp_path / "receipt.json",
            context="private-context",
            runner=_runner(managed_job_name=job_name, managed_job_id=job_id),
        )


def test_observation_discovers_only_exact_managed_job_stage(tmp_path: Path) -> None:
    job_name, job_id = _job("reconstruct")
    exact = _runner(managed_job_name=job_name, managed_job_id=job_id)

    def runner(command, **kwargs):
        resource = command[command.index("get") + 1]
        if resource != "pods":
            return exact(command, **kwargs)
        pod_command = [
            *command[: command.index("get") + 1],
            "pod",
            "private-reconstruct-pod",
            "--output",
            "json",
        ]
        pod = json.loads(exact(pod_command, **kwargs).stdout)
        unrelated = json.loads(json.dumps(pod))
        unrelated["metadata"]["uid"] = "unrelated"
        unrelated["metadata"]["annotations"]["skypilot-managed-job-id"] = "40"
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"items": [unrelated, pod]}), ""
        )

    receipt = runtime_attestation.observe_kubernetes_stage(
        stage="reconstruct",
        namespace="private-namespace",
        container_name="ray-node",
        expected_image=IMAGE,
        managed_job_name=job_name,
        managed_job_id=job_id,
        workflow_run_id="private-run",
        workflow_status_path=_status(tmp_path, "reconstruct"),
        output_path=tmp_path / "receipt.json",
        context="private-context",
        max_wait_seconds=1,
        poll_seconds=0.1,
        runner=runner,
    )

    assert receipt["status"] == "pass"
    assert receipt["managed_job_id_sha256"]


def test_bundle_rejects_reused_single_stage(tmp_path: Path) -> None:
    _, reconstruct_path = _observe(tmp_path, "reconstruct")

    with pytest.raises(
        runtime_attestation.NurecRuntimeAttestationError,
        match="stage receipt contract differs",
    ):
        runtime_attestation.bundle_runtime_attestations(
            reconstruct_path=reconstruct_path,
            render_path=reconstruct_path,
            output_path=tmp_path / "bundle.json",
        )


def test_bundle_requires_distinct_stage_managed_jobs(tmp_path: Path) -> None:
    reconstruct, reconstruct_path = _observe(tmp_path, "reconstruct")
    _, render_path = _observe(tmp_path, "render")
    render = json.loads(render_path.read_text())
    render["managed_job_id_sha256"] = reconstruct["managed_job_id_sha256"]
    render_path.write_text(json.dumps(render))

    with pytest.raises(
        runtime_attestation.NurecRuntimeAttestationError,
        match="runtime identities differ",
    ):
        runtime_attestation.bundle_runtime_attestations(
            reconstruct_path=reconstruct_path,
            render_path=render_path,
            output_path=tmp_path / "bundle.json",
        )
