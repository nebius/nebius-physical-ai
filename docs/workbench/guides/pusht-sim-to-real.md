# PushT: inspect the sim-to-real SDK

[Guides](README.md) · [LeRobot training](reachy2-lerobot-policy.md) · [Sim2Real workflow](sim2real-workflow.md)

PushT uses a planar pusher to move a T-shaped block onto a target. This guide
shows the SDK's **structural smoke**: dataset inspection, splitting, feedback,
and report wiring. For GPU policy training, use the [LeRobot guide](reachy2-lerobot-policy.md)
with a compatible PushT dataset and policy configuration.

## Ingredients

- Install [npa](../../install.md) and allow access to the public Hugging Face dataset.
- The helper defaults to `lerobot/pusht` at revision
  `7628202a2180972f291ba1bc6723834921e72c19`.
- The base NPA install does not include the upstream `lerobot` package.
  Dataset checks can report partial or blocked when that package is unavailable.

## Fast path (local smoke, no cluster)

Run this Python example in the environment where NPA is installed:

```python
from npa.sdk.workbench import sim_to_real

report = sim_to_real.local_smoke(
    run_id="pusht-hello",
    output_dir="./outputs/pusht-hello",
    input_data_uri="hf://datasets/lerobot/pusht",
    s3_bucket="example-bucket",  # artifact layout only; no S3 round trip
    eval_backend="state-success",
    feedback_source="sim-env",
    feedback_type="scalar",
    vlm_eval_backend="stub",
    attempt_s3_roundtrip=False,
)
print(report.status)
for component in report.components:
    print(component.name, component.tier, component.evidence)
print(report.artifacts["local_report_dir"])
```

This downloads dataset content. It uses fixture rollouts and stub VLM feedback;
it produces no trained policy weights and does not test cloud execution.
`attempt_s3_roundtrip=False` skips the explicit storage round-trip probe; the
explicit `hf://datasets/` input also avoids selecting an S3 dataset source.
Read each component's evidence instead of treating the overall status as a
training result. Installing another dependency does not turn this smoke into
real policy training.

## Look at it

Inspect the local report directory. The `rerun` component records the generated
recording and its `view_command`, or explains why only a partial recording was
possible. Open that recording to inspect the documented data and feedback;
fixture rollout data is not a learned policy evaluation.

## Continue with a real workflow

The maintained 14-stage workflow is
[`workflows/main/sim2real.yaml`](../../../workflows/main/sim2real.yaml).
Follow its [runbook](sim2real-workflow.md) and
[data contracts](sim2real-data-contracts.md) for inputs, GPU resources, images,
submission, and quality gates. It has different requirements from this SDK smoke.
The retired `sim-to-real-pipeline.yaml` is not a runnable next step.

A custom dataset must match the selected trainer's observation, action, and
robot schemas. See [customer assets](sim2real-customer-assets.md) before
substituting a different robot or dataset.
