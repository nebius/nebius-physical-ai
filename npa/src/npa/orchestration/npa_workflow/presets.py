"""Explicit, opt-in configuration presets for shipped NPA workflows."""

from __future__ import annotations

from collections.abc import Mapping

from npa.orchestration.npa_workflow.errors import NpaWorkflowError


PUBLIC_FRANKA_LIFT = "public-franka-lift"
PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY = (
    "huyyyyan/pi05-Isaac-sim_Franka_lift_cube"
)
PUBLIC_FRANKA_LIFT_DATASET_REVISION = (
    "42c181e40a43afb1702c29d6f24d5de25219aff8"
)
PUBLIC_FRANKA_LIFT_DATASET_ID = (
    f"{PUBLIC_FRANKA_LIFT_DATASET_REPOSITORY}"
    f"@{PUBLIC_FRANKA_LIFT_DATASET_REVISION}"
)
PUBLIC_FRANKA_LIFT_SOURCE_TASK_ID = "Isaac-Lift-Cube-Franka-IK-Rel-v0"
PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID = "Isaac-Lift-Cube-Franka-v0"


_PRESETS: dict[tuple[str, str], dict[str, str]] = {
    (
        "sim2real",
        PUBLIC_FRANKA_LIFT,
    ): {
        "workflow_preset": PUBLIC_FRANKA_LIFT,
        "dataset_id": PUBLIC_FRANKA_LIFT_DATASET_ID,
        "task_id": PUBLIC_FRANKA_LIFT_CANONICAL_TASK_ID,
        "trigger_uri": (
            "s3://{{config.bucket}}/sim2real-triggers/{{run.id}}/"
            "public-franka-lift/"
        ),
        "seed_manifest_uri": (
            "s3://{{config.bucket}}/sim2real-triggers/{{run.id}}/"
            "public-franka-lift/dataset-manifest.json"
        ),
    }
}


def available_presets(workflow_name: str = "") -> tuple[str, ...]:
    """Return stable public preset names, optionally scoped to one workflow."""

    return tuple(
        sorted(
            preset
            for (workflow, preset) in _PRESETS
            if not workflow_name or workflow == workflow_name
        )
    )


def preset_overrides(
    *, workflow_name: str, preset: str, explicit: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Resolve a preset and merge non-conflicting explicit config overrides.

    Preset-owned identity and trigger keys are intentionally immutable. Operators
    can still tune every production/runtime knob and select ``bucket`` normally.
    """

    requested = str(preset or "").strip()
    overrides = dict(explicit or {})
    if not requested:
        return overrides
    selected = _PRESETS.get((workflow_name, requested))
    if selected is None:
        choices = ", ".join(available_presets(workflow_name)) or "none"
        raise NpaWorkflowError(
            f"workflow {workflow_name!r} has no preset {requested!r}; "
            f"available presets: {choices}"
        )
    conflicts = sorted(
        key
        for key, value in selected.items()
        if key in overrides and str(overrides[key]) != value
    )
    if conflicts:
        raise NpaWorkflowError(
            f"preset {requested!r} owns config keys {conflicts}; remove their "
            "--var overrides or omit --preset to retain a custom/private dataset"
        )
    return {**selected, **overrides}
