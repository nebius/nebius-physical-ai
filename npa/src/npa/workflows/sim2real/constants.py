"""Sim2Real workflow constants, schemas, and reference mappings."""

from __future__ import annotations

# Keep sim2real constants importable without npa.workbench. Purpose-built Isaac
# siblings use the source baked into their immutable image and intentionally omit
# unrelated lancedb/pyarrow/fiftyone dependencies.
DEFAULT_COSMOS_REASON_CACHE = "/tmp/hf_home/cosmos-reason2"
DEFAULT_COSMOS_REASON2_CACHE = "/tmp/hf_home/cosmos-reason2"
DEFAULT_COSMOS_REASON3_CACHE = "/tmp/hf_home/cosmos-reason2-2b"

DEFAULT_S3_ENDPOINT = ""
DEFAULT_BUCKET = ""
DEFAULT_PREFIX = "sim2real-b"
DEFAULT_COSMOS2_TRANSFER_TAG = "2.5.1-skypilot-ready-20260801T053000Z"
DEFAULT_VLM_IMAGE_TAG = "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
# Reference-image pins. Canonical source of truth is pyproject.toml
# ([tool.npa.supported-tools], mirrored in npa/src/npa/deploy/images.py); keep
# these no-registry fallbacks in sync so the staged engine (models.py) and the
# legacy loop (sim2real_loop.py) resolve identical defaults.
DEFAULT_ENVGEN_TAG = "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
DEFAULT_REFERENCE_POLICY_TAG = (
    "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
)
DEFAULT_TRAINER_TAG = "cuda13-b300-0.1.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
DEFAULT_EVAL_TAG = "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
DEFAULT_ISAAC_TAG = "2.3.2.post1"
# Pluggable held-out sim backend. Genesis remains fully supported; Isaac Lab
# (Isaac Sim headless) is the default and requires RT-core GPUs (L40S / RTX Pro).
SIM_BACKEND_GENESIS = "genesis"
SIM_BACKEND_ISAAC = "isaac"
SIM_BACKENDS = (SIM_BACKEND_GENESIS, SIM_BACKEND_ISAAC)
DEFAULT_SIM_BACKEND = SIM_BACKEND_ISAAC
# Default headless Isaac Lab manipulation task for the stock held-out rollout.
DEFAULT_ISAAC_TASK = "Isaac-Lift-Cube-Franka-v0"

DEFAULT_THRESHOLD = 0.50
DEFAULT_INNER_ITERATIONS = 3
DEFAULT_OUTER_ITERATIONS = 3
DEFAULT_LOOP_OF_LOOPS_ITERATIONS = 1
DEFAULT_ROLLOUT_COUNT = 64
DEFAULT_STEPS_PER_ROLLOUT = 32
DEFAULT_HELDOUT_ENVS = 64
DEFAULT_VALIDATION_ENVS = 64
DEFAULT_ENV_COUNT = 10_000
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_ENVGEN_SHARD_COUNT = 16
DEFAULT_K8S_MAX_PARALLEL_GPUS = 16
DEFAULT_ACTION_ENV_LIMIT = 256
# This is the compact VLM-signal adapter/control step size. It is intentionally
# not the Isaac RSL-RL PPO optimizer learning rate, which remains owned by the
# selected Isaac task/agent configuration.
DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE = 0.08
DEFAULT_REFERENCE_VLM_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_REASON2_MODEL = "nvidia/Cosmos-Reason2-8B"
DEFAULT_REASON3_MODEL = "nvidia/Cosmos-Reason2-2B"
DEFAULT_LEROBOT_DATASET_ID = "npa/isaac-lift-cube-franka-seed-v1"
REFERENCE_VLM_ALIASES = {
    "",
    "npa-cosmos3-reason",
    "cosmos3-reason",
    "cosmos-reason",
    "reason2",
    "reason3",
}
DEFAULT_VLM_SEAM_EVIDENCE = (
    f"Dual self-hosted VLM defaults: {DEFAULT_REASON2_MODEL} (Reason2) and "
    f"{DEFAULT_REASON3_MODEL} (Reason3 sibling). Accept gated Hugging Face "
    "licenses before launch; see sim2real-workflow.md."
)
SCHEMA_VLM_EVAL = "npa.sim2real.vlm_eval.v1"
SCHEMA_RL_SIGNAL = "npa.sim2real.rl_signal.v1"
SCHEMA_HELDOUT_REPORT = "npa.sim2real.heldout_eval.v1"
SCHEMA_THRESHOLD_DECISION = "npa.sim2real.threshold_decision.v1"
SCHEMA_E2E_REPORT = "npa.sim2real.e2e_report.v1"

ERROR_SEVERITY = {
    "collision": 0.95,
    "missed_target": 0.85,
    "unstable": 0.7,
    "late_grasp": 0.55,
    "minor_alignment": 0.3,
    "ok": 0.0,
}

CORRECTIVE_TARGETS = {
    "collision": {
        "nl_correction": "Back off from contact and retry with a shallower approach.",
        "action_delta": [-0.12, 0.0, 0.04],
    },
    "missed_target": {
        "nl_correction": "Move the end effector toward the object center before closing.",
        "action_delta": [0.12, 0.02, 0.0],
    },
    "unstable": {
        "nl_correction": "Reduce vertical speed and stabilize before release.",
        "action_delta": [0.0, 0.0, -0.08],
    },
    "late_grasp": {
        "nl_correction": "Close the gripper earlier once the object is centered.",
        "action_delta": [0.03, 0.0, 0.02],
    },
    "minor_alignment": {
        "nl_correction": "Apply a small lateral correction toward the target marker.",
        "action_delta": [0.04, 0.01, 0.0],
    },
    "ok": {
        "nl_correction": "Preserve the current action.",
        "action_delta": [0.0, 0.0, 0.0],
    },
}
