"""Live-submit matrix for npa.workflow twins.

Shared by e2e tests and the operator runner. SkyPilot-only exceptions
(parallel sweeps, burst, sim-to-real monolithic, etc.) are intentionally
absent — see ``npa/workflows/workbench/npa-workflows/README.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmitLiveCase:
    """One npa.workflow twin to submit live through SkyPilot."""

    spec: str
    tier: str  # cpu | gpu | multi
    secret_envs: tuple[str, ...] = ()
    requires_token_factory: bool = False
    plan_only: bool = False
    #: Skip this twin in the bounded daily GPU rotation because it cannot pass as
    #: a standalone submit today (needs a prior workflow's artifact, an input not
    #: staged into the job, or infra the npa.workflow render doesn't yet wire).
    #: The twin stays in the matrix for manual/plan runs; ``skip_reason`` explains
    #: the gap so it can be re-included once fixed.
    rotation_skip: bool = False
    skip_reason: str = ""
    notes: str = ""


SUBMIT_LIVE_MATRIX: tuple[SubmitLiveCase, ...] = (
    # --- CPU / zero-GPU (Token Factory hosted) ---
    SubmitLiveCase(
        "token-factory-caption.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
        notes="Cheapest live path; validates render→submit without a GPU.",
    ),
    SubmitLiveCase(
        "token-factory-generate.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
    ),
    SubmitLiveCase(
        "token-factory-cosmos-reason.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
    ),
    SubmitLiveCase(
        "retargeting.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes="CPU resources in spec; still needs cluster image pull.",
    ),
    # --- Single-tool GPU ---
    SubmitLiveCase(
        "vlm-eval-single.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        rotation_skip=True,
        skip_reason=(
            "vlm_backend=self-hosted, but the npa.workflow render "
            "(catalog workbench.vlm_eval.run) starts only the eval client, not a "
            "vLLM server, so :8000 is never up. Confirmed live: the eval now "
            "waits and reports 'VLM backend not ready after 600s' (readiness "
            "fix) instead of an instant connection-refused. Re-include once the "
            "render injects a `vllm serve` background start for self-hosted steps."
        ),
        notes="Self-hosted VLM; render does not yet stand up the vLLM server.",
    ),
    SubmitLiveCase(
        "vlm-eval-benchmark.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        rotation_skip=True,
        skip_reason=(
            "Confirmed live: fails with 'Expected a JSON file or directory "
            "containing benchmark.json' — the benchmark dataset is not staged "
            "into the job. Re-include once the twin stages/points at a dataset."
        ),
    ),
    SubmitLiveCase(
        "mjlab-eval.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
    ),
    SubmitLiveCase(
        "sonic-train.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NGC_API_KEY"),
    ),
    SubmitLiveCase(
        "sonic-export.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
    ),
    SubmitLiveCase(
        "sonic-eval.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        rotation_skip=True,
        skip_reason=(
            "Consume-only: evaluates an exported ONNX policy from a prior "
            "sonic-export. Not runnable standalone (confirmed live: 'ONNX policy "
            "not found'); sonic-export-eval covers SONIC eval self-contained."
        ),
    ),
    SubmitLiveCase(
        "cosmos3-reason.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
    ),
    SubmitLiveCase(
        "tokenfactory-rollout-judge.yaml",
        "gpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
    ),
    # --- Multi-stage GPU ---
    SubmitLiveCase(
        "sonic-export-eval.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
    ),
    SubmitLiveCase(
        "sonic-locomotion-finetuning.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NGC_API_KEY"),
        notes="retarget → train → mjlab",
    ),
    SubmitLiveCase(
        "bdd100k-pipeline.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes="11-stage AV pipeline; longest wall-clock.",
    ),
    SubmitLiveCase(
        "tokenfactory-cosmos-gate.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        notes="Dynamic gate; needs --assume-decision.",
    ),
    # --- Plan-only / stub twins (do not burn GPUs on stubs) ---
    SubmitLiveCase(
        "physical-ai-data-factory.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        plan_only=True,
        notes=(
            "Physical AI Data Factory blueprint. Dynamic gate (needs "
            "--assume-decision). All stages are real (augment = cosmos2."
            "transfer_execute on GPU; curate/finalize/grade = real run.shell). "
            "Plan-only in CI because a real Cosmos Transfer 2.5 run is heavy "
            "(gated-weight download + diffusion) and needs the npa-cosmos2-transfer "
            "image rebuilt from this branch; live render/submit-prep is validated "
            "without burning a GPU."
        ),
    ),
    SubmitLiveCase(
        "sim2real-vlm-rl.yaml",
        "multi",
        plan_only=True,
        notes="Stub toolRefs; plan-only until engine wiring lands.",
    ),
    SubmitLiveCase(
        "byof.yaml",
        "multi",
        plan_only=True,
        notes="Delegates to run_byof_repo.py; covered by byof live e2e.",
    ),
    SubmitLiveCase(
        "rl-policy-training-sim-success.yaml",
        "multi",
        plan_only=True,
        notes="Partial Isaac twin; plan-only until Hydra parity.",
    ),
)


def selected_submit_cases() -> list[SubmitLiveCase]:
    """Filter SUBMIT_LIVE_MATRIX by env tier / spec allowlists."""

    tiers_raw = os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS", "cpu,gpu,multi")
    tiers = {t.strip().lower() for t in tiers_raw.split(",") if t.strip()}
    specs_raw = os.environ.get("NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS", "")
    specs = {s.strip() for s in specs_raw.split(",") if s.strip()}
    return [
        case
        for case in SUBMIT_LIVE_MATRIX
        if case.tier in tiers and (not specs or case.spec in specs)
    ]


def gpu_submit_cases(
    *, include_plan_only: bool = False, include_skipped: bool = False
) -> list[SubmitLiveCase]:
    """Real-GPU-launching twins, sorted by spec for a deterministic rotation.

    Excludes ``plan_only`` stub twins (they never launch a GPU) and
    ``rotation_skip`` twins (they cannot pass as a standalone submit today — see
    each ``skip_reason``), unless asked, so the daily rotation only ever picks a
    case that actually exercises a GPU and can succeed on its own.
    """

    cases = [
        case
        for case in SUBMIT_LIVE_MATRIX
        if case.tier in {"gpu", "multi"}
        and (include_plan_only or not case.plan_only)
        and (include_skipped or not case.rotation_skip)
    ]
    return sorted(cases, key=lambda c: c.spec)


def rotating_gpu_submit_case(day_index: int) -> SubmitLiveCase | None:
    """Pick one real-GPU twin for ``day_index`` (round-robins over days).

    Lets the daily runner exercise a *different* real GPU workflow E2E each day
    at bounded cost (one managed job) instead of the whole ``gpu and e2e`` blast,
    cycling through every GPU twin over the rotation window.
    """

    cases = gpu_submit_cases()
    if not cases:
        return None
    return cases[day_index % len(cases)]
