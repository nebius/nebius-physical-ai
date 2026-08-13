"""Small retired-controller artifact compatibility helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


from npa.workflows.sim2real.artifact_upload import (
    upload_run_artifacts,  # noqa: F401 - public engine import surface
)
from npa.workflows.sim2real.config import artifact_uris, byo_seams
from npa.workflows.sim2real.models import (
    Sim2RealLoopConfig,
)
from npa.workflows.sim2real.k8s_components import (
    _component_job_script as _component_job_script,
    _kubernetes_component_env as _kubernetes_component_env,
)
from npa.workflows.sim2real.workflow_transfer import (
    run_real_cosmos_transfer as _run_real_cosmos_transfer,
)
from npa.workflows.sim2real.policy_actions_stage import (
    run_policy_actions_component_from_s3 as run_policy_actions_component_from_s3,
)
from npa.workflows.sim2real.reference_helpers import (
    _signal_diversity_report as _signal_diversity_report,
    _signal_mean_reward as _signal_mean_reward,
    _write_env_manifest as _write_env_manifest,
    _write_train_heldout_split as _write_train_heldout_split,
)
from npa.workflows.sim2real.utils import (
    _utc_now,
    _write_json_artifact,
)
from npa.workflows.sim2real.workflow_state_io import (
    _workflow_state_path,  # noqa: F401 - legacy engine import surface
    emit_active_progress_rerun,  # noqa: F401 - imported by runner from engine
    sync_workflow_state_to_s3,  # noqa: F401 - imported by runner from engine
)

# Isaac Sim app handle — closed only after held-out report upload.
_ISAAC_SIMULATION_APP: Any = None
HELDOUT_VIZ_CAMERA_NAME = "heldout_viz_camera"
DEFAULT_HELDOUT_RENDER_FRAMES = 8
SCHEMA_HELDOUT_RENDERS = "npa.sim2real.heldout_renders.v1"
if TYPE_CHECKING:
    pass


_HELDOUT_DISTANCE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def run_cosmos2_transfer_component_from_s3(
    *,
    input_uri: str,
    output_uri: str,
    augmented_frames_uri: str,
    assets_uri: str = "",
    scene_spec_uri: str = "",
    image: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Legacy entrypoint; the canonical workflow calls ``workflow_transfer``."""

    from npa.workflows.sim2real.cosmos_transfer_stage import (
        run_cosmos_transfer_component,
    )

    return run_cosmos_transfer_component(
        input_uri=input_uri,
        output_uri=output_uri,
        augmented_frames_uri=augmented_frames_uri,
        assets_uri=assets_uri,
        scene_spec_uri=scene_spec_uri,
        image=image,
        run_id=run_id,
        real_runner=_run_real_cosmos_transfer,
    )


def _write_stage(
    local_dir: Path,
    number: int,
    name: str,
    payload: dict[str, Any],
    *,
    filename: str | None = None,
) -> dict[str, Any]:
    path = local_dir / f"stage_{number:02d}_{name}" / (filename or f"{name}.json")
    return _write_json_artifact(path, payload)


def _trigger_payload(config: Sim2RealLoopConfig) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.trigger.v1",
        "stage": 1,
        "run_id": config.run_id,
        "created_at": _utc_now(),
        "trigger_dataset_uri": config.trigger_dataset_uri,
        "trigger_dataset_id": config.trigger_dataset_id,
        "input_format": "lerobot",
        "start_condition": "dataset_landed_in_trigger_path",
        "artifact_root": artifact_uris(config).get("root", ""),
        "byo_seams": byo_seams(config),
    }


def _redacted_config(config: Sim2RealLoopConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir) if config.output_dir else None
    return payload


__all__ = [
    "_redacted_config",
    "_trigger_payload",
    "_write_stage",
    "run_cosmos2_transfer_component_from_s3",
]
