---
name: sim2real-operate
description: Operate the compositional 14-stage Sim2Real npa.workflow on Kubernetes through the standard SkyPilot runtime, durable S3 resume ledger, real component images, and artifact audit.
---

# Operate compositional Sim2Real

Use the one canonical spec:

`npa/workflows/workbench/npa-workflows/sim2real.yaml`

It is `npa.workflow/v0.0.1`. Always use `npa workbench workflow ... --runtime`;
there is no direct-Kubernetes Sim2Real submit path. The old materializer and
`k8s_submit` implementation have been removed; the retained CLI command exits
with an actionable migration to this canonical spec.

## Preflight

1. Validate tenant/project/region, bucket, registry, Kubernetes context, Kueue
   admission, Ready RT-core nodes, and the read-only Isaac cache PVC.
2. Require registry-qualified immutable digests for controller, Transfer,
   EnvGen, Reason, Isaac, and viewer images. Confirm each image attests the exact
   source SHA; never use source overlays or best-effort bootstrap.
3. Validate the task-aligned seed manifest, HF/NGC access, S3 read/write, image
   pulls, and primary/side/overhead capture before a full run.
4. Run `validate-spec`, `plan-spec --waves`, scheduler-plan, and submit plan-only
   on the same canonical file.

## Submit and resume

Before provisioning or submitting an Isaac state, load
`skills/atomic/third-party-eula-preflight/SKILL.md`. Isaac acceptance defaults on
for non-interactive submissions; pass `--no-accept-eula` to opt out. An opted-out
run fails before work is created. Optional privacy and telemetry remain disabled.

Submit with `--runtime --resume`. Pass tenant-specific data only through
`--var`, isolated config, and secret envs. For a no-deadline run pass
`--max-wait-seconds 0`; the runtime still records wave/job status in
`<run-root>/npa-workflow/runtime.json`.

The graph owns every stage Job. Isaac rollout/PPO/eval execute their proven
payload inside their already admitted SkyPilot GPU task and must report
`npa_workflow_skypilot_task`; a hidden sibling Job is a contract failure.

Use a deliberate controller restart after Stage 8 and `--resume` to prove the
Stage 8→9 barrier. Also restart during Stage 14 in the integration ladder. The
runtime must adopt/replay complete waves from declared S3 outputs and resubmit
only incomplete work.

## Audit

Require exactly 14 canonical ComponentRecords. Stages 1–11, 13, and 14 are
`WORKS`; Stage 12 alone is `SEAM`. For GPU stages verify workflow Job identity,
immutable digest, source SHA, GPU product, and explicit S3 inputs/outputs.
Verify train/validation/gold digest disjointness, validation-only checkpoint
selection, exact checkpoint SHA/size loaded by gold, strict 5 cm stable
placement, bounded non-degenerate temporal signals, and explicit gold render
lineage.

Download and independently decode non-empty `reports/sim2real.rrd` and
`reports/sim2real.mcap`. Confirm 10 FPS (or configured FPS) timestamps,
primary/side/overhead footage, progress/policy/evaluation evidence, and
checkpoint accessibility. Pipeline completion does not imply policy efficacy;
report measured strict success without weakening the threshold.
