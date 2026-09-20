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


def _runner(*, image_id: str = IMAGE, gpu_limit: str = "1", gpu_label: str = ""):
    def run(command, **kwargs):
        assert kwargs["check"] is False
        assert kwargs["timeout"] == 120
        resource = command[command.index("get") + 1]
        if resource == "pod":
            payload = {
                "metadata": {"uid": "pod-uid-private"},
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
                        or "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"
                    },
                }
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    return run


def _observe(tmp_path: Path, stage: str) -> tuple[dict, Path]:
    path = tmp_path / f"{stage}.json"
    receipt = runtime_attestation.observe_kubernetes_stage(
        stage=stage,
        pod_name=f"private-{stage}-pod",
        namespace="private-namespace",
        container_name="ray-node",
        expected_image=IMAGE,
        output_path=path,
        context="private-context",
        runner=_runner(),
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
    ],
)
def test_observation_rejects_wrong_runtime_identity(
    runner, match: str, tmp_path: Path
) -> None:
    with pytest.raises(runtime_attestation.NurecRuntimeAttestationError, match=match):
        runtime_attestation.observe_kubernetes_stage(
            stage="reconstruct",
            pod_name="private-pod",
            namespace="private-namespace",
            container_name="ray-node",
            expected_image=IMAGE,
            output_path=tmp_path / "receipt.json",
            runner=runner,
        )

    assert not (tmp_path / "receipt.json").exists()


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
