"""Human-readable capability probes for each container golden eval.

Used by `run_golden_evals.py list --capabilities` and
`docs/security/container-golden-evals.md`. Keys must match
``golden_evals.yaml`` container names.
"""

from __future__ import annotations

# Each value is an ordered list of concrete checks the golden eval runs.
GOLDEN_EVAL_CAPABILITIES: dict[str, list[str]] = {
    "base-cuda13-b300": [
        "torch import + CUDA device available",
        "flash_attn import (Blackwell/CUDA13 stack)",
    ],
    "groot": [
        "Isaac-GR00T repo present",
        "uv available",
        "standalone GR00T inference script runs",
    ],
    "lerobot": [
        "LeRobot package version pin",
        "50-step PushT training run",
        "checkpoint artifact written",
        "policy eval on checkpoint",
        "eval output artifact written",
    ],
    "lerobot-policy": [
        "short LeRobot train step (policy_container train CLI)",
        "short eval on produced checkpoint (policy_container eval CLI)",
    ],
    "lerobot-vlm-rl": [
        "CUDA available",
        "VLM signal batch parse + one RL training step",
    ],
    "genesis": [
        "Genesis import",
        "Franka scene build",
        "physics step",
        "body state readback",
    ],
    "isaac-lab": [
        "Isaac Lab version",
        "headless runtime launch",
        "manipulation env create",
        "env step loop",
    ],
    "cosmos": [
        "Cosmos package version",
        "model load (with safety guardrail enabled)",
        "single text2world inference",
    ],
    "cosmos2-transfer": [
        "cosmos-transfer2.5 inference env (torch cu128 + flash-attn)",
        "real video-to-video world transfer on a bundled robot control example",
        "generated output video produced (capability, not a CUDA probe)",
    ],
    "cosmos3-reason": [
        "real Cosmos-Reason VLM inference on synthetic frames (run_cosmos_reason_vlm)",
        "structured rollout judgment returned (score + success verdict)",
    ],
    "sonic": [
        "entrypoint smoke mode",
        "GPU + image-pull proofs",
        "sonic_smoke_result.json artifact",
    ],
    "retargeting": [
        "motion-lib validate_motion_lib on synthetic payload",
    ],
    "fiftyone": [
        "fiftyone import + version pin",
        "CLI --help",
        "app config (DB-free env smoke)",
    ],
    "lancedb": [
        "FastAPI server start",
        "create table",
        "vector query roundtrip",
        "list tables",
    ],
    "detection-training": [
        "FastAPI server start",
        "/health",
        "/system-info",
    ],
    "envgen": [
        "raw env generation (JSONL contract)",
        "Genesis CUDA env step (mocked in unit gate)",
    ],
    "reference-policy": [
        "delegates to envgen functional checks",
    ],
    "loop-eval": [
        "CUDA available",
        "FrankaPickPlace rollout step",
    ],
    "rerun-viewer": [
        "rerun SDK import + __version__",
    ],
    "foxglove": [
        "static Lichtblick/Foxglove web bundle present (/srv/index.html)",
        "served bundle version pin (VERSION == 1.26.0)",
    ],
}


def capability_rows() -> list[tuple[str, str]]:
    """Return (container, semicolon-separated capabilities) for tabular output."""

    return [(name, "; ".join(checks)) for name, checks in GOLDEN_EVAL_CAPABILITIES.items()]
