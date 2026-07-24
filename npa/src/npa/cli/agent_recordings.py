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
