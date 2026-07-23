# Score a Robot in 60 Seconds (No GPU, No Cloud)

**The hook:** before you spin up a single GPU, prove the whole evaluation loop
on your laptop. You'll grade a set of robot rollouts with a vision-language
model "judge" and get a ranked report — using only the sample data that ships
with `npa`. No cloud. No GPU. No credentials.

This is the friendliest possible first taste of the workbench, and it's the same
command you'll later point at a real VLM backend.

## Ingredients

- **Robot:** any — we score rollout clips, not a specific arm.
- **Sim / engine:** none. The `stub` backend runs fully offline.
- **Dataset:** a shipped, labeled benchmark of four robot rollouts.
- **You need:** just `npa` installed (see
  [the guides intro](README.md#before-you-start)).

## Fast path

Run the benchmark against the shipped sample set with the offline `stub`
backend (from the repository root — the `--dataset` path is relative to it):

```bash
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

You should see a ranked report with `accuracy: 1.0` over four labeled rollouts.
That's the entire local loop: a VLM judge scores each rollout, the sweep tries
every threshold/rubric/model combination, and the best config wins.

Want a single-rollout taste instead of a sweep? Try a dry run:

```bash
npa workbench vlm-eval run \
  --input-path ./rollout.json \
  --output-path ./eval.json \
  --backend stub \
  --score 0.9 \
  --dry-run
```

## What just happened

The `vlm-eval` tool is the workbench's evaluation backend. It takes robot
rollout artifacts (frames + metadata), asks a vision-language model whether the
task succeeded, and writes a task-success report. The `stub` backend returns
deterministic scores so you can validate plumbing, CI, and report format with
zero infrastructure.

The exact same command swaps `--backend stub` for `--backend self-hosted`
(a vLLM server you launch) or `--backend api` (a hosted endpoint) once you add
credentials — the report shape never changes.

## Go bigger

- **Real judge, self-hosted:** serve a VLM with vLLM and score real rollout
  directories — see the
  [VLM-Eval Loop Runbook](../cookbooks/vlm-eval-loop-runbook.md).
- **Real judge, hosted API:** point the `api` backend at a hosted, OpenAI-
  compatible VLM (such as Nebius Token Factory) to score with no GPU of your own
  — see the [VLM-Eval Loop Runbook](../cookbooks/vlm-eval-loop-runbook.md).
- **In a pipeline:** `vlm-eval` is the scorer inside the
  [sim-to-real loop](pusht-sim-to-real.md), where it grades a freshly trained
  policy.

## Dig deeper

- Cookbook: [VLM-Eval Loop Runbook](../cookbooks/vlm-eval-loop-runbook.md)
- Workflow YAML: `npa/workflows/workbench/skypilot/vlm-eval.yaml`
- Used as the scorer inside the sim-to-real loop: `skills/workflows/sim-to-real/SKILL.md`
