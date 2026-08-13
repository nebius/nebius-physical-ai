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
        "real video-to-video world transfer on a runtime-generated procedural control video",
        "generated output video produced (capability, not a CUDA probe)",
    ],
    "cosmos3": [
        "cosmos-framework inference env (torch cu130 + guardrail deps)",
        "real text2image generation with the Cosmos 3 omni model",
        "decodable image artifact produced (capability, not a CUDA probe)",
        "no baked weights: checkpoint fetched with the operator's HF token",
    ],
    "cosmos3-serving": [
        "vLLM-Omni serving stack imports in the pinned build",
        "pin-specific Hugging Face Xet workaround remains justified",
        "real entrypoint assembles the pinned 8-GPU serve command",
        "no model checkpoint files are baked into image-owned trees",
        "separate live evidence: real Cosmos3-Super video generation on 8xH200",
    ],
    "cosmos3-reason": [
        "real Cosmos-Reason VLM inference on synthetic frames (run_cosmos_reason_vlm)",
        "structured rollout judgment returned (score + success verdict)",
    ],
    "cosmos-curate": [
        "real upstream Cosmos Curator stages run in-process (no Ray, no GPU)",
        "canonical curator output written: clips/*.mp4 + metas/v0/*.json",
        "per-clip motion score computed by upstream's MotionFilterStage",
        "no model weights in the image; fetch-models downloads them with HF_TOKEN",
    ],
    "cosmos-evaluator": [
        "upstream Cosmos Evaluator HallucinationProcessor runs on real clips",
        "hallucinated motion discriminated: appearance-only passes, new scene fails",
        "upstream engine and the in-repo port agree on verdict and score",
        "no model weights needed or present (CV check + hosted VLM endpoint)",
    ],
    "sonic": [
        "entrypoint smoke mode",
        "GPU + image-pull proofs",
        "sonic_smoke_result.json artifact",
    ],
    "sonic-mujoco": [
        "MuJoCo EGL rollout of a SONIC checkpoint (cross-simulator check on a policy "
        "trained in Isaac Lab — where a sim-to-sim gap shows before a sim-to-real one)",
        "sonic_eval_results.json artifact",
        "runs on the baked venv: no Isaac Sim download and no EULA acceptance required",
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
    "wan2-2": [
        "pinned Wan source import with OSS CPU dependency base",
        "machine-readable runtime health/version contract",
        "CUDA runtime fetch refuses before operator terms acceptance",
    ],
    "sim2real-control": [
        "canonical compositional stage-adapter module imports",
        "stage CLI exposes the complete 1-through-14 contract",
        "exact baked source and immutable-image checks run before stage work",
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
    "foxglove-embed": [
        "/healthz reports the service and the pinned @foxglove/embed version",
        "real SDK served (FoxgloveViewer class + embed postMessage handshake)",
        "shared NPA glue module served and importing the served SDK",
        "standalone host page served and loading the glue module",
        "HTTP byte range on /data returns 206 with an exact Content-Range",
        "CORS preflight for the Range header answered on /data",
    ],
    "lichtblick": [
        "static Lichtblick (Foxglove-compatible) web bundle present (/srv/index.html)",
        "served bundle version pin (VERSION == 1.26.0)",
    ],
}


def capability_rows() -> list[tuple[str, str]]:
    """Return (container, semicolon-separated capabilities) for tabular output."""

    return [
        (name, "; ".join(checks)) for name, checks in GOLDEN_EVAL_CAPABILITIES.items()
    ]
