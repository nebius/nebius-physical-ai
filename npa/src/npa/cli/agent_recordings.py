"""Rerun recording identity helpers for the NPA agent backend.

Enforces the "no stock Franka/demo `.rrd` as run data" rule
(`skills/tools/npa-agent/SKILL.md`): a Sim2Real run may only be marked
``rerun_ready`` — and its recording served/claimed as run data — when the
recording actually contains run-specific entities (held-out eval, rollouts,
training signal, per-env scores). The stock Franka demo recording contains only
scene geometry (`world/franka/*`, `world/table`, `world/cube`) and must never
masquerade as a run's artifact.

Rerun `.rrd` files embed entity-path strings as UTF-8, so a byte scan is a cheap,
dependency-free way to tell a real run recording from the stock demo. These
helpers are pure/deterministic and unit-test without infra; the module is
embedded verbatim into the agent VM backend (same mechanism as the other agent
modules).
"""

from __future__ import annotations

import re

# Entity-path markers that only appear in a real Sim2Real run recording.
RUN_ENTITY_MARKERS: tuple[bytes, ...] = (
    b"heldout",
    b"rollout",
    b"per_env",
    b"success_rate",
    b"training_signal",
    b"/scores",
    b"/signals",
    b"outer_loop",
    # Physical AI Data Factory recordings use input/ and augmented/ image
    # timelines plus pipeline/* report entities.  ``augmented/`` and
    # ``pipeline/`` never occur in the stock Franka geometry recording.
    b"augmented/",
    b"pipeline/",
    # Neural-reconstruction (NuRec/NRE) run entities: rig-offset novel views
    # rendered from the trained Gaussians, NRE's validation renders, and the
    # Gaussian quality summary. None of these appear in the stock demo recording.
    b"novel_view",
    b"reconstruction/",
    b"gaussians",
    b"nurec",
)

# Markers characteristic of the stock Franka/demo recording (scene geometry only).
DEMO_MARKERS: tuple[bytes, ...] = (
    b"world/franka",
    b"/franka/base",
    b"/franka/gripper",
    b"world/table",
    b"world/cube",
    b"demo/active_camera",
)

_SAFE_RUN_ID_RE = re.compile(r"[^A-Za-z0-9._:-]")

#: Several producers all write ``reports/sim2real.rrd``, so the recording's own
#: name cannot identify it -- the run prefix does. Keeping the producer identity
#: here (rather than in the agent bootstrap) puts it next to the entity-marker
#: scan it belongs with, and keeps the agent module off its size ratchet.
PIPELINE_RECORDING_SUFFIX = "/reports/sim2real.rrd"
NEURAL_RECONSTRUCTION_APP_ID = "neural-reconstruction"

#: Preview entity and viewer note for a NuRec run. A reconstruction has no
#: held-out-simulation camera, so the generic Sim2Real note would be actively
#: misleading.
NEURAL_RECONSTRUCTION_PREVIEW_ENTITY = "novel_view"
#: Camera label for a NuRec run. A reconstruction has no held-out simulation
#: camera, so inheriting the previous run's "heldout-sim" label would contradict
#: the viewer note directly above.
NEURAL_RECONSTRUCTION_CAMERA_LABEL = "novel-view"
NEURAL_RECONSTRUCTION_VIEWER_NOTE = (
    "NuRec / NRE neural-reconstruction recording loaded. Entities: "
    "novel_view/<camera> (views rendered from the trained 3D Gaussians at an "
    "offset rig pose - views the reconstruction was NOT trained on), "
    "reconstruction/<camera> (NRE validation renders), input/<sensor> (the real "
    "capture frames that were reconstructed), gaussians/summary (PSNR / SSIM / "
    "LPIPS), and pipeline/* (per-stage reports, including how the rig->world pose "
    "edge was derived). Scrub the frame timeline to fly the novel-view camera "
    "through the reconstructed scene."
)


def is_pipeline_recording(key: str) -> bool:
    """True for any producer's run-scoped ``reports/sim2real.rrd``."""
    return str(key or "").endswith(PIPELINE_RECORDING_SUFFIX)


def is_neural_reconstruction_recording(key: str) -> bool:
    """True when the recording belongs to a NuRec / NRE reconstruction run.

    Matches the capability id as a path SEGMENT so an unrelated prefix that merely
    contains the phrase is not misclassified.
    """
    return is_pipeline_recording(key) and (
        NEURAL_RECONSTRUCTION_APP_ID + "/"
    ) in str(key or "")


def recording_has_run_entities(data: bytes | None) -> bool:
    """Return True when the recording bytes contain run-specific entity paths."""
    if not data:
        return False
    return any(marker in data for marker in RUN_ENTITY_MARKERS)


def is_stock_demo_recording(data: bytes | None) -> bool:
    """Return True when bytes look like the stock demo (geometry only, no run data)."""
    if not data:
        return False
    if recording_has_run_entities(data):
        return False
    return any(marker in data for marker in DEMO_MARKERS)


def run_recording_basename(run_id: str) -> str:
    """Return a filesystem-safe ``<run_id>.rrd`` basename for run-scoped recordings."""
    token = _SAFE_RUN_ID_RE.sub("_", str(run_id or "").strip())
    token = re.sub(r"\.{2,}", "_", token).strip("._")
    if not token:
        token = "run"
    return f"{token}.rrd"
