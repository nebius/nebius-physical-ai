# Teach a Robot to Push a T (PushT, Sim-to-Real)

**The hook:** PushT is the beloved toy benchmark of robot learning — shove a
T-shaped block onto a target. In this guide you run the **whole sim-to-real
loop** end to end with one command: stage a public dataset, train a policy,
evaluate it, collect feedback, and record a Rerun visualization you can scrub
frame by frame.

The best part: you can dry-run the entire spine locally before spending a
single GPU-minute.

## Ingredients

- **Robot:** a simulated planar pusher (the PushT task).
- **Sim / engine:** the workbench **sim-to-real loop** — imitation training plus
  a pluggable evaluator and feedback source.
- **Public dataset:**
  [`lerobot/pusht`](https://huggingface.co/lerobot/pusht) — MIT-licensed, with
  vision, state, action, episode, and timestamp fields. Pinned revision
  `7628202a2180972f291ba1bc6723834921e72c19`.
- **You need:** `npa` installed for the local smoke; Nebius creds + an H100 for
  the live run.

## Fast path (local smoke, no cluster)

Run the same structural spine the live pipeline uses, entirely on your machine,
with typed return objects:

```python
from npa.sdk.workbench import sim_to_real

report = sim_to_real.local_smoke(
    run_id="pusht-hello",
    s3_bucket="your-bucket-name",
    s3_endpoint="https://storage.eu-north1.nebius.cloud",
    s3_prefix="sim-to-real/pusht-hello",
    input_data_uri="s3://your-bucket-name/datasets/lerobot-pusht/",
    policy_image="npa-lerobot-policy:0.1.1",
    gpu="H100:1",
    eval_backend="state-success",
    feedback_source="sim-env",
    feedback_type="scalar",
    vlm_eval_backend="stub",
    attempt_s3_roundtrip=False,
)
print(report.status)
```

This validates the data split, the eval backend, the feedback object, and the
Rerun recording wiring without provisioning anything. It downloads and inspects
the public `lerobot/pusht` metadata (≈206 episodes, ≈25k frames) on the fly.

The report is **tiered**, so read `report.status` and the per-component tiers
literally. In a plain `pip install -e npa` environment the `LeRobotDataset`
import isn't present, so the dataset component reports `PARTIAL`/`BLOCKED` and
`report.status` is `blocked` — that's expected. Install the LeRobot extra (so
`import lerobot` works) for a fully green local smoke; the live H100 run below
uses the policy image and doesn't need LeRobot on your laptop.

## The one-command live run

When you're ready to do it for real on an H100:

```bash
npa/.venv/bin/python npa/scripts/run_sim_to_real_quickstart.py
```

That wrapper renders `sim-to-real-pipeline.yaml`, submits it on `H100:1`, runs
the real training/eval loop, prints the **task-success score** plus the
checkpoint / report / Rerun S3 URIs, and tears down the run-scoped cluster.
Warm runs target about 5-6 minutes for the small proof config.

## What's happening under the hood

```text
lerobot/pusht ─▶ split (train / heldout) ─▶ imitation train ─▶ eval ─▶ feedback
                                                   │                      │
                                                   └──── checkpoint ◀──────┘
                                                          + Rerun .rrd
```

Both the **evaluator** and the **feedback source** are swappable:

- `--eval-backend`: `state-success` (pose predicate), `vlm-frames` (VLM judges
  rendered frames), or `heldout-metrics`.
- `--feedback-source`: `none`, `sim-env`, `vlm`, or `byo-container`.

That's the same `vlm-eval` judge from the
[no-GPU guide](score-a-robot-no-gpu.md), now grading a live policy.

## Look at it

Download the Rerun recording and scrub through demonstrations, the policy
rollout, and per-episode feedback:

```bash
rerun /tmp/npa-sim-to-real-<run-id>/<run-id>.rrd
```

## Bring your own dataset

Point the loop at any `LeRobotDataset` in S3 and keep the same visualization:

```bash
--input-data-uri "s3://your-bucket-name/datasets/my-lerobot-dataset/"
```

This is exactly how you'd plug in the [Franka demos](franka-pick-and-place-genesis.md)
you recorded in Genesis, or a [Reachy 2](reachy2-lerobot-policy.md) dataset.

## Dig deeper

- **14-stage production loop:** [Sim-to-real workflow](sim2real-workflow.md) · [Data contracts](sim2real-data-contracts.md)
- Cookbook (legacy module): [Sim-To-Real Pipeline Runbook](../cookbooks/sim-to-real-pipeline.md)
- Quickstart (legacy H100 proof): [../sim-to-real-quickstart.md](../sim-to-real-quickstart.md)
- Workflow YAML: `npa/workflows/workbench/skypilot/sim-to-real-pipeline.yaml`
- Skill: `skills/workflows/sim-to-real/SKILL.md`
