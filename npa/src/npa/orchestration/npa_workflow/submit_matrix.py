"""Live-submit matrix for npa.workflow twins.

Shared by e2e tests and the operator runner. SkyPilot-only exceptions (burst,
sim-to-real monolithic, etc.) are intentionally absent — see
``npa/workflows/workbench/npa-workflows/README.md``.

Parallel sweeps are no longer such an exception: ``isaac-lab-rl-sweep.yaml`` is an
``npa.workflow`` spec in this matrix, verified live on four GPUs. The raw SkyPilot
template it was ported from is retained as a reference example (and is still
referenced by docs, a runner script and its own test); retiring it is a separate
change.
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
    #: Submit through the runtime orchestrator (``submit --runtime``) instead of
    #: the one-shot serial path. Required for specs with a ``parallel:`` group or
    #: a loop that must early-exit on the real decision artifact.
    runtime: bool = False
    #: Config overrides applied at submit time (``--var k=v``), e.g. to drive a
    #: gate threshold in one live run.
    config_vars: tuple[tuple[str, str], ...] = ()
    #: Expected number of concurrent tasks in the spec's largest parallel wave
    #: (0 when the spec has no fan-out); asserted from the live job timeline.
    expected_parallel_tasks: int = 0
    #: Workbench tool whose image every task of this spec needs (resolved against
    #: the live registry at submit time). Set for specs whose stages run inside a
    #: baked image instead of the default SkyPilot image + staged npa source.
    image_tool: str = ""
    #: Per-wave deadline for this case, in seconds. 0 = use
    #: ``NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS``. Set it when one case is
    #: much slower than the rest (a 8 GB image pull plus GPU training) so the whole
    #: runtime tier does not have to run with the slowest case's deadline.
    max_wait_seconds: int = 0


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
        "token-factory-parallel-fanout.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
        runtime=True,
        expected_parallel_tasks=3,
        notes=(
            "Cheapest live PARALLEL path: three caption shards launch as one "
            "SkyPilot JobGroup, then an insights barrier. Needs --runtime."
        ),
    ),
    SubmitLiveCase(
        "token-factory-gate-loop.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
        runtime=True,
        config_vars=(("grade_threshold", "0.0"),),
        notes=(
            "Cheapest live RUNTIME-GATE path: the loop reads the real decision "
            "artifact and early-exits on iteration 1 with grade_threshold=0.0 "
            "(raise it above the achievable score to run the full budget)."
        ),
    ),
    SubmitLiveCase(
        "token-factory-trigger-watch.yaml",
        "cpu",
        secret_envs=("NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        requires_token_factory=True,
        runtime=True,
        notes=(
            "Trigger/watch reference: the driver polls the inbox prefix and only "
            "submits the stage once data lands. The live harness seeds the inbox "
            "AFTER the run starts, so the wait is real."
        ),
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
            "vlm_backend=self-hosted. The render now DOES start a background "
            "vLLM server and the client waits up to NPA_VLM_READY_TIMEOUT_S "
            "(=1800s, set by the render) for readiness. Confirmed live on "
            "RTXPRO-6000 (Blackwell/sm_120): the cold start — heavy vLLM+CUDA13 "
            "install, ~16GB Qwen2-VL-7B weight download, and first-run compile — "
            "exceeds even the 30-min window, so it is not a fit for the bounded "
            "daily rotation. Re-include with the model pre-baked/pre-cached into "
            "the image or a faster node. (The old instant connection-refused is "
            "fixed; failure is now a clean, bounded 'not ready' diagnostic.)"
        ),
        notes="Self-hosted VLM; render starts vLLM, but cold start > 30min on this node.",
    ),
    SubmitLiveCase(
        "vlm-eval-benchmark.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Fixed: the twin now uses the `sample` sentinel, which resolves to "
            "the packaged benchmark fixture at its install location (a "
            "repo-relative path did not exist in the rendered job). backend=stub, "
            "so it validates the submit path without a GPU model."
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
        rotation_skip=True,
        skip_reason=(
            "Confirmed live: fails 'SONIC --runtime serverless requires "
            "--project-id'. The twin renders `sonic train --runtime serverless` "
            "INSIDE an already-GPU SkyPilot job; SONIC's runtimes (vm/container/"
            "serverless) all delegate to more infra rather than training in-job, "
            "and serverless needs a project id/creds. Re-include once SONIC train "
            "supports an in-job runtime for the npa.workflow render."
        ),
    ),
    SubmitLiveCase(
        "sonic-export.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        rotation_skip=True,
        skip_reason=(
            "Consume-only + env gap. Confirmed live: 'SONIC ONNX export requires "
            "torch' (job env lacks torch) and it needs a checkpoint.pt from a "
            "prior sonic-train. Re-include once export runs on the torch-bearing "
            "SONIC image against a staged checkpoint."
        ),
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
        rotation_skip=True,
        skip_reason=(
            "Consume-only: the export stage needs a checkpoint.pt from a prior "
            "sonic-train (per-run path is empty on a standalone submit), so it "
            "fails before eval. Re-include once a checkpoint is staged."
        ),
    ),
    SubmitLiveCase(
        "sonic-locomotion-finetuning.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "NGC_API_KEY"),
        rotation_skip=True,
        skip_reason=(
            "retarget → train → mjlab: hits the same SONIC train in-job runtime "
            "gap as sonic-train, and needs a staged motion source. Re-include "
            "once SONIC train has an in-job runtime and inputs are staged."
        ),
        notes="retarget → train → mjlab",
    ),
    SubmitLiveCase(
        "isaac-lab-rl-sweep.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        runtime=True,
        expected_parallel_tasks=4,
        image_tool="isaac-lab",
        # The Isaac Lab image is ~8 GB per node and the variants train on GPU.
        max_wait_seconds=5400,
        # Cost control for the live tier: hold two GPUs at a time instead of four.
        # This also exercises the multi-batch path (4 members / maxConcurrency 2).
        config_vars=(("max_concurrency", "2"),),
        notes=(
            "Parallel GPU reference case (port of the execution:parallel SkyPilot "
            "template): four RSL-RL variants as one JobGroup + ranking barrier. "
            "Needs --runtime and the Isaac Lab image (run branch code on top with "
            "NPA_SRC_OVERLAY=1); cap GPUs with --var max_concurrency=N."
        ),
    ),
    SubmitLiveCase(
        "bdd100k-pipeline.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        rotation_skip=True,
        skip_reason=(
            "11-stage AV pipeline needs the raw-bdd100k demo dataset staged in "
            "the run bucket and is the longest wall-clock twin; not a bounded "
            "daily-rotation fit. Run manually with a staged dataset."
        ),
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
        rotation_skip=True,
        skip_reason=(
            "Confirmed live: FAILED — dynamic Cosmos gate loop needs a staged "
            "scene input and --assume-decision; not runnable as a bounded "
            "standalone rotation submit. Run manually with a staged scene."
        ),
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


def runtime_submit_cases() -> list[SubmitLiveCase]:
    """Selected cases that must be driven by the runtime orchestrator.

    These are the specs with a ``parallel:`` group or a loop that has to
    early-exit on the real decision artifact; they are submitted with
    ``submit --runtime``.
    """

    return [case for case in selected_submit_cases() if case.runtime and not case.plan_only]


def one_shot_submit_cases() -> list[SubmitLiveCase]:
    """Selected cases for the classic one-shot submit path.

    Runtime cases are excluded on purpose: submitting them one-shot would render
    the flattened serial plan, which is valid but proves nothing about
    concurrency or early-exit — and would run a GPU sweep serially. They are
    still covered by the plan-only matrix and by the runtime live test.
    """

    return [case for case in selected_submit_cases() if not case.runtime]
