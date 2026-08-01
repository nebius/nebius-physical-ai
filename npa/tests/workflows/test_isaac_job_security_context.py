"""Isaac sibling jobs run as root — deliberately, and only that far.

The npa-isaac-lab image defaults to the non-root ``ubuntu`` user, which cannot
traverse the image's ``/isaac-sim`` tree, so ``/isaac-sim/python.sh`` resolves
empty and every Isaac job exits 127. Running the pod as root is the fix. These
tests pin that decision so it stays explicit, and — more importantly — pin its
boundary: root inside the container must not grow into a privileged container,
host namespaces, or host-path mounts.
"""

from __future__ import annotations

from typing import Any

import pytest

from npa.workflows.sim2real import byo_isaac_eval as ev
from npa.workflows.sim2real import byo_isaac_policy_rollout as pr
from npa.workflows.sim2real import byo_isaac_trainer as tr

GPU_PRODUCT = "NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition"


def _trainer_manifest() -> dict[str, Any]:
    return tr.build_isaac_job_manifest(
        job_name="s2r-byo-isaac-train-run1",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=64,
        iterations=10,
        s3_output_uri="s3://b/run1/byo-trainer/",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product=GPU_PRODUCT,
    )


def _rollout_manifest() -> dict[str, Any]:
    return pr.build_isaac_rollout_job_manifest(
        job_name="s2r-byo-isaac-roll-run1-iter0",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        rollout_count=2,
        steps_per_rollout=4,
        checkpoint_uri="s3://b/run1/byo-trainer/j/model_latest.pt",
        out_s3_prefix="s3://b/sim2real-b/run1/byo-rollouts/iter0",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product=GPU_PRODUCT,
    )


def _eval_manifest() -> dict[str, Any]:
    return ev.build_isaac_eval_job_manifest(
        job_name="s2r-byo-isaac-eval-run1",
        run_id="run1",
        image="reg/npa-isaac-lab:2.3.2.post1",
        task="Isaac-Lift-Cube-Franka-v0",
        num_envs=4,
        checkpoint_uri="s3://b/run1/model_latest.pt",
        per_env_s3_uri="s3://b/run1/byo-eval/job/per_env_distances.json",
        s3_endpoint="https://s3.example",
        namespace="default",
        service_account="agent-sa",
        gpu_product=GPU_PRODUCT,
    )


MANIFEST_BUILDERS = {
    "trainer": _trainer_manifest,
    "rollout": _rollout_manifest,
    "eval": _eval_manifest,
}


def _pod_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest["spec"]["template"]["spec"]


@pytest.mark.parametrize("job", sorted(MANIFEST_BUILDERS))
def test_isaac_job_runs_as_root(job: str) -> None:
    pod = _pod_spec(MANIFEST_BUILDERS[job]())
    assert pod["securityContext"]["runAsUser"] == 0, (
        f"{job}: Isaac jobs must run as root; the image's default ubuntu user "
        "cannot traverse /isaac-sim and python.sh resolves empty (exit 127)"
    )


@pytest.mark.parametrize("job", sorted(MANIFEST_BUILDERS))
def test_isaac_job_root_stays_bounded(job: str) -> None:
    """Root in-container is the whole grant: no privileged escalation beyond it."""

    pod = _pod_spec(MANIFEST_BUILDERS[job]())

    for field in ("hostNetwork", "hostPID", "hostIPC"):
        assert not pod.get(field), f"{job}: Isaac jobs must not join the host {field}"

    for volume in pod.get("volumes") or []:
        assert "hostPath" not in volume, (
            f"{job}: Isaac jobs must not mount host paths (found {volume.get('name')!r}); "
            "root in-container plus a host mount is root on the node"
        )

    for container in pod["containers"]:
        container_security = container.get("securityContext") or {}
        assert not container_security.get("privileged"), (
            f"{job}: Isaac containers must not be privileged"
        )
        added = ((container_security.get("capabilities") or {}).get("add")) or []
        assert not added, f"{job}: Isaac containers must not add capabilities, got {added}"
