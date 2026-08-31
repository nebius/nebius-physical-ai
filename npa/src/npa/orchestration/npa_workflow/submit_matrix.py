"""Live-submit matrix for npa.workflow twins.

Shared by e2e tests and the operator runner. SkyPilot-only exceptions (burst,
sim-to-real monolithic, etc.) are intentionally absent — see
``npa/workflows/workbench/npa-workflows/README.md``.

Parallel sweeps are no longer such an exception: ``isaac-lab-rl-sweep.yaml`` is an
``npa.workflow`` spec in this matrix, verified live on four GPUs, and the raw SkyPilot
template it was ported from has been retired.

The raw SkyPilot task catalog is being retired one live-verified twin at a time; the
remaining templates are pinned in
``npa/tests/guardrails/test_skypilot_catalog_retirement.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SubmitLiveCase:
    """One npa.workflow case to render or submit through SkyPilot."""

    spec: str
    tier: str  # cpu | gpu | multi
    #: Secrets this twin's stages actually read. The live test SKIPS the case
    #: when one is missing from the operator env, so listing a secret the
    #: exercised path never consumes turns an unrelated gap in someone's env
    #: into a silently no-op day of the daily GPU rotation. List only what the
    #: twin needs to run; the render's ``SECRET_ENV_HINTS`` cover the rest.
    secret_envs: tuple[str, ...] = ()
    requires_token_factory: bool = False
    plan_only: bool = False
    #: Required, machine-checked explanation for every non-executing matrix
    #: entry. Empty for every real-submit case.
    plan_only_justification: str = ""
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
    #: Explicit workflow preset passed through the same CLI used by operators.
    preset: str = ""
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
    #: Exact toolRef -> image-tool mappings for workflows that require several
    #: distinct workbench images in one run.
    image_overrides: tuple[tuple[str, str], ...] = ()
    #: Per-wave deadline for this case, in seconds. 0 = use
    #: ``NPA_E2E_NPA_WORKFLOW_SUBMIT_MAX_WAIT_SECONDS``. Set it when one case is
    #: much slower than the rest (a 8 GB image pull plus GPU training) so the whole
    #: runtime tier does not have to run with the slowest case's deadline.
    max_wait_seconds: int = 0


SUBMIT_LIVE_MATRIX: tuple[SubmitLiveCase, ...] = (
    SubmitLiveCase(
        "alpamayo2-super-inference.yaml",
        "gpu",
        secret_envs=("HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        image_tool="alpamayo2-super",
        notes=(
            "Real single-B200 Alpamayo 2 Super VLM + diffusion-expert inference; "
            "publishes calibrated trajectory JSON/PNG and provenance. Requires "
            "operator-side PhysicalAI-AV dataset acceptance."
        ),
    ),
    # --- CPU / zero-GPU (Token Factory hosted) ---
    SubmitLiveCase(
        "token-factory-caption.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        notes="Cheapest live path; validates render→submit without a GPU.",
    ),
    SubmitLiveCase(
        "token-factory-generate.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
    ),
    SubmitLiveCase(
        "token-factory-batch-generate.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        notes=(
            "Batch inference is a separate model entitlement from real-time chat, "
            "and the stage waits out a completion window, so this case takes far "
            "longer than the generate case it mirrors."
        ),
    ),
    SubmitLiveCase(
        "token-factory-cosmos-reason.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
    ),
    SubmitLiveCase(
        "token-factory-parallel-fanout.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
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
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
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
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        runtime=True,
        notes=(
            "Trigger/watch reference: the driver polls the inbox prefix and only "
            "submits the stage once data lands. The live harness seeds the inbox "
            "AFTER the run starts, so the wait is real."
        ),
    ),
    SubmitLiveCase(
        "vlm-eval-token-factory.yaml",
        "cpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        notes=(
            "Zero-GPU VLM eval through the hosted `api` backend. This is the VLM eval "
            "case that can always run: vlm-eval-single asks for `self-hosted`, and "
            "nothing in that spec starts a vLLM server (pre-existing gap)."
        ),
    ),
    SubmitLiveCase(
        "scenario-gen-smoke.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Adversarial scenario mining + ranking on the default heuristic adversary "
            "backend, which is GPU-free and needs no seeded inputs: the policy/base-config "
            "URIs are recorded in lineage, not read. The rank stage consumes the manifest "
            "the generate stage wrote."
        ),
    ),
    SubmitLiveCase(
        "sonic-b300-routing-evidence.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "CPU-only fail-closed execution of the installed SONIC accelerator "
            "resolver; publishes a manifest, test report, and time-structured RRD. "
            "Provider recognition and GPU execution remain separate assertions."
        ),
    ),
    SubmitLiveCase(
        "dataset-of-record-smoke.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Dataset-of-record smoke: ingest -> validate -> quality gate -> curate -> "
            "query. CPU-only; the harness seeds real raw sensor records. Dynamic gate, "
            "so it is also in DYNAMIC_SPECS."
        ),
    ),
    SubmitLiveCase(
        "dataset-ingest-curate.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Stricter dataset-of-record variant (completeness_min 0.5, max_corruption_rate "
            "0.1, location filter). Its `register` stage writes to the in-cluster LanceDB "
            "service at http://npa-lancedb.workbench.svc.cluster.local:8686 — deploy it with "
            "`npa workbench lancedb deploy --runtime kubernetes --namespace workbench`."
        ),
    ),
    SubmitLiveCase(
        "insights-smoke.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Insights lineage + metrics store: ingest a run prefix, compare two runs, "
            "render a dashboard. CPU-only. The harness seeds a real dataset manifest and "
            "a decision artifact, the two shapes the ingester recognises."
        ),
    ),
    SubmitLiveCase(
        "insights-aggregate.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes="Insights ingest + dashboard over one run prefix. CPU-only.",
    ),
    SubmitLiveCase(
        "cosmos3-text-to-image.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        image_tool="cosmos3-reason",
        notes=(
            "Clones the Cosmos framework, syncs its uv environment, anonymously downloads "
            "public Cosmos3-Nano with guardrails disabled, and generates an image. Needs the "
            "Cosmos image rather than SkyPilot's default: "
            "transformer_engine links against glibc >= 2.32 (job 301), which no LD_LIBRARY_PATH "
            "can supply."
        ),
    ),
    SubmitLiveCase(
        "cosmos3-generate.yaml",
        "gpu",
        secret_envs=("HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        image_tool="cosmos3",
        notes=(
            "Runs the real Cosmos 3 omni-model generate path in the npa-cosmos3 image. "
            "The image contains the framework but no weights. Cosmos3-Nano is public; "
            "HF_TOKEN is required here only because this workflow keeps gated guardrails on."
        ),
    ),
    SubmitLiveCase(
        "cosmos3-ray-batch.yaml",
        "cpu",
        secret_envs=(
            "NPA_COSMOS3_RAY_TOKEN",
            "HF_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        image_tool="cosmos3-ray-serve",
        notes=(
            "Submits a real prepared batch to a separately deployed persistent "
            "Cosmos3-Nano native Ray Serve service and publishes its structured "
            "outputs and media through S3. The dedicated B200/RTX validation "
            "starts the model-backed service before this client path runs."
        ),
    ),
    SubmitLiveCase(
        "cosmos3-checkpoint-eval.yaml",
        "gpu",
        secret_envs=("HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        image_tool="cosmos3",
        plan_only=True,
        plan_only_justification=(
            "operator-driven gated benchmark is covered by its dedicated live B200 "
            "campaign; the generic rotation cannot supply task-scoped license "
            "acceptance or intentionally launch a 40-image checkpoint matrix"
        ),
        notes=(
            "B200-only guarded checkpoint comparison. The dedicated campaign executes "
            "the primary and consistency phases through this spec with an accepted, "
            "versioned config staged to operator-owned S3."
        ),
    ),
    SubmitLiveCase(
        "cosmos2-transfer.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        image_tool="cosmos2-transfer",
        notes=(
            "The REAL Cosmos-Transfer2.5 model, not a manifest: transfer_execute "
            "conditions on a repository-authored procedural MP4, and --execute makes a "
            "missing runtime a hard error rather than a silent fallback. Replaces a "
            'template that held a GPU to print `"status": "contract_ready"`.'
        ),
    ),
    SubmitLiveCase(
        "isaac-franka-capture-reason.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        image_tool="isaac-lab",
        notes=(
            "Isaac Lab renders Franka frames on a GPU, then a hosted Cosmos3 reasoner plans "
            "from them on CPU. Needs no seeded input: the first stage produces the second's."
        ),
    ),
    SubmitLiveCase(
        "tokenfactory-scene-to-rollout-judge.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        image_tool="lerobot",
        notes=(
            "Three stages, one chain: a hosted reasoner plans from a seeded scene, a GPU rolls "
            "out a policy, and a hosted VLM judges that rollout AGAINST THAT PLAN "
            "(`--task-from` reads the reasoner's artifact). Only the middle stage holds a GPU. "
            "Same LeRobot image requirement as the rollout-judge combo."
        ),
    ),
    SubmitLiveCase(
        "tokenfactory-rollout-judge-combo.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        image_tool="lerobot",
        notes=(
            "The real rollout-judge twin: the GPU stage rolls out a public LeRobot policy in its "
            "own pod and publishes the rendered episodes, then a hosted VLM scores exactly that "
            "prefix with no GPU. Needs a hostable LeRobot image whose torch and torchcodec agree "
            "(NPA_E2E_IMAGE_OVERRIDE_LEROBOT=<registry>/npa-lerobot:0.6.0-k8s-runtime)."
        ),
    ),
    SubmitLiveCase(
        "tokenfactory-train-triage.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        image_tool="lerobot",
        notes=(
            "The producer/consumer combo: LeRobot trains in the stage's own pod (the vendor "
            "image's LeRobot, one step) and publishes the run's checkpoint AND textual "
            "artifacts, then a hosted text model triages that run with no GPU. The train stage "
            "materialises its own dataset from `--dataset-repo-id`, because stages do not share "
            "a filesystem. Requires a SkyPilot-hostable LeRobot image AND one whose torch and "
            "torchcodec agree: run with "
            "NPA_E2E_IMAGE_OVERRIDE_LEROBOT=<registry>/npa-lerobot:0.6.0-k8s-runtime. The 0.5.1 "
            "image fails at training step 0 with a torchcodec ABI mismatch. Six live iterations "
            "and five engine gaps to get here - see EVIDENCE.md \u00a7R32-R33."
        ),
    ),
    SubmitLiveCase(
        "cosmos-fetch.yaml",
        "cpu",
        # setup stages the npa source from S3 with boto3, so the keys are needed even
        # though nothing in this plan touches object storage.
        secret_envs=("HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        config_vars=(
            # The spec's defaults name Cosmos3 assets that are gated behind early access and
            # a licence acceptance. Substituting public ones exercises the identical code
            # path — a real git clone and a real Hugging Face download into the cache — which
            # is what a live run of this twin is meant to prove.
            (
                "cosmos_source_repo",
                "https://github.com/githubtraining/hellogitworld.git",
            ),
            ("cosmos_model_id", "hf-internal-testing/tiny-random-gpt2"),
        ),
        notes=(
            "Cosmos access check then fetch. CPU. Run with public substitutes for the gated "
            "Cosmos3 source repo and checkpoint; the commands, flags and cache layout are "
            "the same ones the retired template invoked."
        ),
    ),
    SubmitLiveCase(
        "sim2real-envgen-shards.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        runtime=True,
        expected_parallel_tasks=2,
        notes=(
            "Shard fan-out: two raw env shards as one JobGroup, then a barrier that splits "
            "the combined catalog 80/20. Replaces a template that read its shard index from "
            "a Kubernetes Job completion index. CPU — generation writes env descriptors, it "
            "does not render. Needs the runtime tier so the group really is concurrent."
        ),
    ),
    SubmitLiveCase(
        "multi-node-probe.yaml",
        "cpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Multi-node reference: `resources.gang.num_nodes` gang-schedules a real "
            "2-node stage, then a single-node stage verifies one report per rank landed "
            "on a distinct host. CPU on purpose — the property is the node count."
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
        # No HF_TOKEN: the served 2B Qwen2-VL is public, so requiring one would
        # skip the twin on an operator env that simply never set it.
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        # A self-hosted VLM cold start is dominated by the vLLM wheel set and the
        # engine's own warmup, both of which land outside the other twins' range.
        max_wait_seconds=2400,
        notes=(
            "Self-hosted vLLM on the job's own GPU. Bounded by serving the 2B "
            "Qwen2-VL (config.vlm_model), installing vLLM with uv, pre-fetching "
            "weights in setup, and having the run script wait for readiness so a "
            "server that dies during startup fails immediately with its log."
        ),
    ),
    SubmitLiveCase(
        "vlm-eval-benchmark.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Labeled sweep on the self-hosted backend, like the template it replaces. The "
            "harness seeds two rollouts with known outcomes plus an S3 benchmark manifest."
        ),
    ),
    SubmitLiveCase(
        "vlm-eval-loop.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        notes=(
            "Rollout-SET scoring plus the aggregate task_success report. `run` scores one "
            "rollout, so this is the capability that let sim-to-real-loop.yaml retire: the "
            "harness seeds several rollout directories and the report must count them all."
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
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "NGC_API_KEY",
        ),
        notes=(
            "Trains in-job (`sonic_runtime: local`). The serverless/vm/container "
            "runtimes delegate to more infrastructure, which a stage that already "
            "holds a GPU cannot provision."
        ),
    ),
    SubmitLiveCase(
        "sonic-export.yaml",
        "gpu",
        # No NGC_API_KEY: the in-job trainer pulls nothing from NGC, and gating
        # on it would skip the twin instead of running it.
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        notes="train (in-job runtime) -> export; self-contained.",
    ),
    SubmitLiveCase(
        "sonic-eval.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        rotation_skip=True,
        skip_reason=(
            "Consume-only by design: a single stage that evaluates an ONNX a "
            "previous `sonic export` wrote, so a standalone submit has nothing "
            "to read ('ONNX policy not found'). SONIC eval IS in the rotation "
            "through sonic-export-eval, which chains train -> export -> eval and "
            "hands each stage the previous one's S3 artifact."
        ),
    ),
    SubmitLiveCase(
        "cosmos3-reason.yaml",
        "gpu",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
    ),
    SubmitLiveCase(
        "nurec-reconstruct.yaml",
        "gpu",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NGC_API_KEY",
        ),
        # No image_tool: the runtime is NVIDIA's vendor NRE container supplied via
        # resources.image / image_id, not an NPA-built workbench image.
        # A ~14 GB NGC image pull on a cold node, then 30k 3DGUT steps and a
        # novel-view render pass. Far slower than the rest of the gpu tier, so it
        # carries its own deadline instead of forcing it on every case.
        max_wait_seconds=5400,
        notes=(
            "NuRec/NRE reconstruction on an RT-core GPU: real NCore V4 capture -> "
            "3DGUT Gaussians -> renderable USDZ -> rig-offset novel views -> "
            "reports/sim2real.rrd. Needs NGC_API_KEY for the nre-ga container and "
            "the public PhysicalAI capture works anonymously."
        ),
    ),
    SubmitLiveCase(
        "content-agents-rigid-object.yaml",
        "gpu",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        requires_token_factory=True,
        image_tool="content-agents",
        notes=(
            "Public zero-vendor-payload NVIDIA Content Agents v0.5.2 adapter: "
            "generated/customer USD -> real OVRTX material + physics pipelines -> "
            "Validation Agent profiles -> rigid-ready USD/USDZ and Isaac Stage-2 manifest."
        ),
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
    # --- Multi-stage multi-GPU training and visualization ---
    SubmitLiveCase(
        "groot-1-7-finetune.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        image_tool="groot",
        notes=(
            "Runs deterministic offline held-out baseline inference, configurable "
            "one-to-many-GPU GR00T training (live evidence used two GPUs), and "
            "post-training inference on the identical split. It emits RRD/MCAP "
            "diagnostics and reports learning outcome separately from pipeline "
            "status; it is not closed-loop or physical-robot task evidence."
        ),
    ),
    # --- Multi-stage GPU ---
    SubmitLiveCase(
        "sonic-export-eval.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        notes=(
            "train -> export -> eval, self-contained: the in-job train runtime "
            "writes checkpoint.pt to S3 and each stage reads the previous "
            "stage's artifact from there."
        ),
    ),
    SubmitLiveCase(
        "sonic-locomotion-finetuning.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NGC_API_KEY",
        ),
        notes=(
            "retarget → train → mjlab. Retargeting consumes the SOMA/G1 motion "
            "clips staged in the run bucket (see SONIC_MOTION_FIXTURE_PREFIX in "
            "the live helpers, overridable with NPA_E2E_SONIC_MOTION_SRC); train "
            "uses the in-job runtime."
        ),
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
            "Two reasons, both structural. (1) Every stage talks to a workbench "
            "SERVICE deployed in-cluster (npa-lancedb:8686, "
            "npa-detection-training:8790); a standalone submit cannot bring "
            "those up, and it also wants the raw-bdd100k demo dataset in the run "
            "bucket. (2) 11 sequential stages, each its own cluster: measured "
            "~2.2 min per stage of provisioning alone on RTXPRO-6000 (from the "
            "3-stage SONIC chain), so ~25 min before any real work — over the "
            "rotation's bounded window once CLIP backfill, three trainings and "
            "three evals are added. Run it manually against a live workbench."
        ),
        notes="11-stage AV pipeline over in-cluster services; longest wall-clock.",
    ),
    SubmitLiveCase(
        "tokenfactory-cosmos-gate.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        notes=(
            "Dynamic gate; the harness seeds the scene frames and passes "
            "--assume-decision (see DYNAMIC_SPECS in the live helpers)."
        ),
    ),
    # --- Phase 4.1 coverage backfill: executable conditioned Cosmos paths ---
    SubmitLiveCase(
        "sim2real-two-step.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        image_tool="cosmos2-transfer",
        notes=(
            "Real conditioned Cosmos-Transfer2.5 path is wired: the dedicated toolRef "
            "fails closed without the vendor runtime or seeded MP4, and envgen resolves "
            "the durable manifest's exact non-empty frames list."
        ),
    ),
    SubmitLiveCase(
        "sim2real-two-step-agent.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        image_tool="cosmos2-transfer",
        notes=(
            "Agent-authored sibling on the dedicated conditioned execute path: the seeded "
            "MP4 drives generation and envgen cycles only over frame URIs in the durable "
            "npa.cosmos2.transfer.v1 manifest."
        ),
    ),
    # --- Plan-only: incomplete hardening blueprints ---
    SubmitLiveCase(
        "adversarial-scenario-hardening.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        plan_only=True,
        plan_only_justification="three execution-contract gaps: missing VM config, disconnected decision, and local-only publish",
        notes=(
            "Plan-only: Isaac Lab train/eval default to a VM config absent on a fresh "
            "worker, the decision stage does not consume the evaluation report, and "
            "publish writes only a local /tmp file instead of release_uri."
        ),
    ),
    SubmitLiveCase(
        "hardening-with-insights.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
        plan_only=True,
        plan_only_justification="inherits the hardening workflow's missing VM config, disconnected decision, and local-only publish",
        notes=(
            "Plan-only: the inherited Isaac Lab train/eval needs a VM config missing on "
            "fresh workers, its decision ignores evaluation, and publish writes only "
            "under /tmp rather than release_uri; later insights stages do not repair "
            "those gaps."
        ),
    ),
    # --- Real PAIDF GPU-daily acceptance ---
    SubmitLiveCase(
        "physical-ai-data-factory.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        runtime=True,
        config_vars=(("n_augmentations", "1"),),
        image_overrides=(
            ("workbench.cosmos2.transfer_execute", "cosmos2-transfer"),
            ("workbench.cosmos_evaluator.evaluate", "cosmos-evaluator"),
            ("workbench.cosmos_curate.curate", "cosmos-curate"),
            ("workbench.fiftyone.curate_augmented", "fiftyone"),
        ),
        max_wait_seconds=7200,
        notes=(
            "Authorized GPU-daily PAIDF acceptance. Uses the pinned real RoboPro "
            "starter input, runs the dynamic gate through the runtime orchestrator, "
            "and asserts real Cosmos Transfer/Evaluator/Curator/FiftyOne and Rerun "
            "artifacts. One augmentation keeps the daily proof decisive."
        ),
    ),
    SubmitLiveCase(
        "paidf-cosmos3.yaml",
        "multi",
        secret_envs=(
            "NEBIUS_TOKEN_FACTORY_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
        ),
        requires_token_factory=True,
        runtime=True,
        config_vars=(("variant_count", "1"), ("variant_parallelism", "1")),
        image_overrides=(
            ("workbench.cosmos3.prepare_video_input", "cosmos3"),
            ("workbench.cosmos3.generate_variants", "cosmos3"),
            ("workbench.cosmos_evaluator.evaluate", "cosmos-evaluator"),
            ("workbench.cosmos_curate.curate", "cosmos-curate"),
            ("workbench.fiftyone.curate_augmented", "fiftyone"),
            ("workbench.nurec.visualize", "rerun-viewer"),
        ),
        notes=(
            "Real dynamic PAIDF Cosmos 3 acceptance using only the repository-owned "
            "synthetic MP4 fixture. Proves source-video-conditioned framework output, "
            "Cosmos Evaluator, Cosmos Curator, FiftyOne Brain, and Rerun evidence."
        ),
    ),
    # --- Plan-only: stubs or separately covered BYOF onboarding flows ---
    SubmitLiveCase(
        "sim2real.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NEBIUS_TOKEN_FACTORY_KEY",
        ),
        runtime=True,
        preset="public-franka-lift",
        expected_parallel_tasks=8,
        rotation_skip=True,
        skip_reason=(
            "The canonical live case requires five immutable component image "
            "digests, a prewarmed Isaac cache PVC, and task-aligned trigger data."
        ),
        notes=(
            "Canonical compositional 14-stage Sim2Real runtime using the explicit "
            "public-franka-lift preset. Repository CI validates the dynamic plan; "
            "an operator live run first stages the pinned public seed and supplies five "
            "immutable component images, an Isaac cache PVC, "
            "and task-aligned trigger data. The reduced real-GPU proof is archived "
            "separately because the component images are project-local."
        ),
    ),
    SubmitLiveCase(
        "byof.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="Delegates to run_byof_repo.py; covered by byof live e2e.",
    ),
    SubmitLiveCase(
        "byof-maniskill.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="BYOF onboarding flow; covered by test_byof_onboarding_live_e2e.py.",
    ),
    SubmitLiveCase(
        "byof-mujoco-playground.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="BYOF onboarding flow; covered by test_byof_onboarding_live_e2e.py.",
    ),
    SubmitLiveCase(
        "byof-robocasa.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="BYOF onboarding flow; covered by test_byof_onboarding_live_e2e.py.",
    ),
    SubmitLiveCase(
        "byof-openpi.yaml",
        "multi",
        secret_envs=("NPA_OPENPI_ACCEPT_GEMMA_TERMS",),
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="OpenPI Polaris B200 inference; covered by test_byof_openpi_polaris_live_e2e.py.",
    ),
    SubmitLiveCase(
        "openpi-pi05-four-mode.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "NPA_OPENPI_ACCEPT_GEMMA_TERMS",
        ),
        plan_only=True,
        plan_only_justification=(
            "the immutable runtime digest is produced by the connected BYOF build and "
            "the dedicated OpenPI live E2E submits the complete graph"
        ),
        notes=(
            "Real direct, cross-pod ClusterIP serve, LoRA optimizer, checkpoint reload, "
            "and held-out evaluation; dedicated E2E uses top-level workflow submit."
        ),
    ),
    SubmitLiveCase(
        "byof-droid-policy-learning.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated BYOF execution is covered by its dedicated live onboarding tier",
        notes="BYOF onboarding flow; covered by test_byof_onboarding_live_e2e.py.",
    ),
    SubmitLiveCase(
        "byof-open-dreamer.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="delegated multi-GPU BYOF execution is covered by its dedicated Open Dreamer live tier",
        notes=(
            "BYOF onboarding flow; the real multi-GPU path is covered by "
            "test_byof_open_dreamer_live_e2e.py."
        ),
    ),
    SubmitLiveCase(
        "byof-ltx2.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "HF_TOKEN",
            "NPA_LTX_ACCEPT_NVIDIA_RUNTIME_TERMS",
        ),
        plan_only=True,
        plan_only_justification=(
            "the shared submit harness cannot establish that its HF token has the "
            "operator-specific gated Lightricks entitlement required at runtime"
        ),
        notes=(
            "The accepted zero-payload digest has passed a real RTX PRO 6000 "
            "text-to-video run. This shared matrix remains plan-only because both "
            "runtime fetches require the submitting operator's gated "
            "Lightricks/LTX-2.5 entitlement; test_ltx2_live_e2e.py owns that live path."
        ),
    ),
    SubmitLiveCase(
        "byof-wan2.2.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        plan_only=True,
        plan_only_justification=(
            "the dedicated RTX PRO live E2E owns the large public model run"
        ),
        notes=(
            "BYOF Wan 2.2 TI2V-5B candidate. Plan-only in the shared matrix: "
            "the real pushed-image RTX PRO generation, decoded MP4, verified RRD, "
            "and S3 evidence gate is test_byof_wan22_live_e2e.py."
        ),
    ),
    SubmitLiveCase(
        "byof-wan2.2-multigpu.yaml",
        "multi",
        secret_envs=(
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        ),
        plan_only=True,
        plan_only_justification=(
            "the dedicated four-B200 live E2E owns the multi-GPU public model run"
        ),
        notes=(
            "Plan-only in the shared submit matrix: the dedicated Wan live E2E "
            "runs the four-B200 TI2V-5B torchrun path and verifies immutable-image reuse, all four NCCL "
            "ranks, FULL_SHARD T5/DiT, Ulysses collectives, S3 topology JSON, "
            "decoded H.264 MP4, and the remotely verified RRD manifest."
        ),
    ),
    SubmitLiveCase(
        "cosmos-synth-fanout-curation.yaml",
        "multi",
        secret_envs=("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN"),
        plan_only=True,
        plan_only_justification="contains a stub FiftyOne state and colliding synthetic output targets",
        notes=(
            "workbench.fiftyone.launch_app is a stub, and both synthetic shard states "
            "currently target the same transfer manifest object; keep plan-only until "
            "both gaps close."
        ),
    ),
    SubmitLiveCase(
        "av-night-scene-hardening.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="terminal FiftyOne launch state remains a stub",
        notes=(
            "The terminal workbench.fiftyone.launch_app state is a stub; retain a full "
            "render preflight until the review toolRef becomes executable."
        ),
    ),
    SubmitLiveCase(
        "rl-policy-training-sim-success.yaml",
        "multi",
        plan_only=True,
        plan_only_justification="partial Isaac twin lacks the required Hydra execution parity",
        notes="Partial Isaac twin; plan-only until Hydra parity.",
    ),
)


def selected_submit_cases() -> list[SubmitLiveCase]:
    """Filter SUBMIT_LIVE_MATRIX by env tier / spec allowlists."""

    tiers_env = "NPA_E2E_NPA_WORKFLOW_SUBMIT_TIERS"
    specs_env = "NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS"
    tiers_raw = os.environ.get(tiers_env, "cpu,gpu,multi")
    tiers = {t.strip().lower() for t in tiers_raw.split(",") if t.strip()}
    specs_raw = os.environ.get(specs_env, "")
    specs = {s.strip() for s in specs_raw.split(",") if s.strip()}
    cases = [
        case
        for case in SUBMIT_LIVE_MATRIX
        if case.tier in tiers and (not specs or case.spec in specs)
    ]
    if not cases and (tiers_env in os.environ or specs_raw.strip()):
        known_specs = ", ".join(sorted({case.spec for case in SUBMIT_LIVE_MATRIX}))
        known_tiers = ", ".join(sorted({case.tier for case in SUBMIT_LIVE_MATRIX}))
        raise ValueError(
            "live submit filters selected no npa.workflow cases: "
            f"{tiers_env}={tiers_raw!r}, {specs_env}={specs_raw!r}. "
            f"Known tiers: {known_tiers}. Known specs: {known_specs}."
        )
    return cases


def gpu_submit_cases(
    *, include_plan_only: bool = False, include_skipped: bool = False
) -> list[SubmitLiveCase]:
    """Real-GPU-launching twins, sorted by spec for a deterministic rotation.

    Excludes ``plan_only`` cases (they never launch a GPU) and
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

    return [
        case for case in selected_submit_cases() if case.runtime and not case.plan_only
    ]


def one_shot_submit_cases() -> list[SubmitLiveCase]:
    """Selected cases for the classic one-shot submit path.

    Runtime cases are excluded on purpose: submitting them one-shot would render
    the flattened serial plan, which is valid but proves nothing about
    concurrency or early-exit — and would run a GPU sweep serially. They are
    still covered by the plan-only matrix and by the runtime live test.
    """

    return [case for case in selected_submit_cases() if not case.runtime]
