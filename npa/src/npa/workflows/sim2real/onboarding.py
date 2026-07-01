"""Onboarding service: turn an onboarding spec into a derived plan + smoke job.

Glue between the declarative spec (B1), the auto-derivation (B2), the robot-aware
task variant (B3), and the BYO Isaac trainer. Kept import-light (no torch /
isaaclab) so the CLI can call it without a GPU. The actual k8s submit is injected
as a callable so the plan-building is unit-testable without a cluster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from npa.workflows.sim2real import byo_isaac_trainer as trainer
from npa.workflows.sim2real import isaac_byo_robot_task as robotmod
from npa.workflows.sim2real import onboarding_derive as der
from npa.workflows.sim2real import onboarding_spec as ob

# Conservative smoke defaults: enough iterations to confirm the env builds, the
# robot retargets, and a checkpoint is produced — not to fully learn.
DEFAULT_SMOKE_ITERATIONS = 20
DEFAULT_SMOKE_NUM_ENVS = 64


@dataclass
class OnboardingPlan:
    """Everything the onboarding flow derives from a spec, ready to display/submit."""

    spec: ob.OnboardingSpec
    derived: der.DerivedTaskConfig
    robot_payload: dict[str, Any]
    robot_usd_uri: str  # s3 uri to stage, or "" when the USD loads in-place (CDN/local)
    compat: dict[str, Any]


def robot_payload_from_spec(
    spec: ob.OnboardingSpec, derived: der.DerivedTaskConfig
) -> tuple[dict[str, Any], str]:
    """Build the ``NPA_BYO_ROBOT_SPEC_JSON`` payload + the USD uri to stage.

    The payload is the contract ``isaac_byo_robot_task`` reads in-container
    (articulation overrides + link/joint retarget + gripper). An ``s3://`` asset
    is staged to the in-container path and returned as ``robot_usd_uri``; an
    https/CDN or already-local USD is passed through as ``usd_path`` and loads in
    place (``robot_usd_uri`` empty).
    """

    ri = spec.robot
    arm = list(ri.joint_names)
    grip = list(ri.gripper_joint_names)
    # Full joint list = arm joints then any gripper joints not already listed.
    joint_names = arm + [g for g in grip if g not in arm]

    uri = ri.robot_uri.strip()
    if uri.startswith("s3://"):
        usd_path = robotmod.ROBOT_USD_CONTAINER_PATH if hasattr(robotmod, "ROBOT_USD_CONTAINER_PATH") else "/tmp/npa_robot/robot.usd"
        robot_usd_uri = uri
    else:
        usd_path = uri  # https CDN or container-local path: loads in place
        robot_usd_uri = ""

    home = list(derived.init_joint_pos) or list(ri.home_qpos)

    payload = {
        "robot_source": ri.robot_source,
        "name": ri.name,
        "ee_link": ri.ee_link,
        "base_link": ri.base_link,
        "joint_names": joint_names,
        "gripper_joint_names": grip,
        "finger_links": list(ri.finger_links),
        "n_arm_joints": ri.n_arm_joints or len(arm),
        "n_gripper_joints": ri.n_gripper_joints or len(grip),
        "home_qpos": home,
        "gripper_open": derived.gripper_open,
        "gripper_close": derived.gripper_close,
        "usd_path": usd_path,
    }
    return payload, robot_usd_uri


def build_plan(spec: ob.OnboardingSpec) -> OnboardingPlan:
    """Derive the full onboarding plan (config + payload + compat) from a spec.

    Pure: off-GPU it uses preset reach + the explicit spec morphology (the
    on-cluster wrapper refines action scale / placement from measured USD ranges).
    """

    derived = der.derive_task_config(spec)
    payload, robot_usd_uri = robot_payload_from_spec(spec, derived)
    compat = robotmod.task_robot_compatibility(payload, task_kind=spec.task.skill)
    return OnboardingPlan(
        spec=spec,
        derived=derived,
        robot_payload=payload,
        robot_usd_uri=robot_usd_uri,
        compat=compat,
    )


def smoke_trainer_env(
    plan: OnboardingPlan,
    *,
    iterations: int = DEFAULT_SMOKE_ITERATIONS,
    num_envs: int = DEFAULT_SMOKE_NUM_ENVS,
    entropy_coef: str = trainer.DEFAULT_ENTROPY_COEF,
) -> dict[str, str]:
    """The env a direct BYO trainer smoke job needs (no infra; unit-testable).

    Captures the BYO-robot routing + derived task config + short budget. Cluster
    creds / image / bucket are expected to already be in the operator's env.
    """

    return {
        "NPA_BYO_ROBOT_TASK": "1",
        "NPA_BYO_ROBOT_SPEC_JSON": json.dumps(plan.robot_payload, sort_keys=True),
        "NPA_BYO_TASK_CONFIG_JSON": json.dumps(plan.derived.to_dict(), sort_keys=True),
        "NPA_BYO_ISAAC_ITERATIONS": str(int(iterations)),
        "NPA_BYO_ISAAC_NUM_ENVS": str(int(num_envs)),
        "NPA_BYO_ISAAC_ENTROPY_COEF": str(entropy_coef),
    }


def submit_smoke_job(
    plan: OnboardingPlan,
    *,
    run_id: str,
    image: str,
    bucket: str,
    endpoint: str,
    namespace: str = "default",
    service_account: str = "agent-sa",
    gpu_product: str = trainer.DEFAULT_GPU_PRODUCT,
    iterations: int = DEFAULT_SMOKE_ITERATIONS,
    num_envs: int = DEFAULT_SMOKE_NUM_ENVS,
    entropy_coef: str = trainer.DEFAULT_ENTROPY_COEF,
    kubectl_apply: Callable[[dict[str, Any]], int] | None = None,
) -> dict[str, Any]:
    """Build + submit a short BYO trainer job confirming the robot loads/trains.

    ``kubectl_apply`` is injected (a callable taking the manifest, returning a
    return code) so this is unit-testable without a cluster. Returns the job
    name, output uri, and the apply return code.
    """

    if not image:
        raise ValueError("smoke job requires an Isaac image (ISAAC_IMAGE)")
    if not bucket:
        raise ValueError("smoke job requires an S3 bucket (NPA_SIM2REAL_BUCKET/S3_BUCKET)")

    job_name = f"s2r-onboard-smoke-{run_id}"[:63]
    out_uri = f"s3://{bucket}/sim2real-b/{run_id}/onboard-smoke/{job_name}/"
    manifest = trainer.build_isaac_job_manifest(
        job_name=job_name,
        run_id=run_id,
        image=image,
        task=trainer.DEFAULT_ISAAC_TASK,
        num_envs=num_envs,
        iterations=iterations,
        s3_output_uri=out_uri,
        s3_endpoint=endpoint,
        namespace=namespace,
        service_account=service_account,
        gpu_product=gpu_product,
        entropy_coef=entropy_coef,
        robot_spec=plan.robot_payload,
        robot_usd_uri=plan.robot_usd_uri,
        task_config=plan.derived.to_dict(),
    )
    rc = 0
    if kubectl_apply is not None:
        rc = int(kubectl_apply(manifest))
    return {"job_name": job_name, "out_uri": out_uri, "apply_rc": rc, "manifest": manifest}
