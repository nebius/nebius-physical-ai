# Compose Nebius GPU workloads with Token Factory

Run training or simulation on Nebius GPUs, write the results to S3, then use
Token Factory to caption frames, propose a scene plan, or interpret a run's
reports. The hosted stage runs on a CPU worker and calls the Token Factory API.

```text
GPU training / rollout → S3 artifacts → hosted inference → saved report
```

The artifact format matters as much as the URI: captioning expects images,
text generation expects prompt records, and rollout scoring expects supported
rollout inputs. Sending an arbitrary checkpoint prefix to `generate` does not
turn it into a triage report.

## Prerequisites

Complete the [quickstart](../quickstart.md) and
[Workbench setup](getting-started.md) for the selected GPU workload. You need:

- A configured Nebius project and Kubernetes context with the workflow's GPU
  and CPU capacity.
- S3 credentials with access to the input objects and a writable output prefix.
- A separate Token Factory key, verified with
  `npa workbench token-factory verify`, plus access to the selected model.
- The producer's real inputs, compatible checkpoints, and any required model
  or runtime access. Model listing does not establish inference access.

Credentials belong in the existing NPA credential store or environment. Pass
their names through `workflow submit --secret-env`; keep secrets out of YAML.

## Choose a checked-in pipeline

| Workflow | GPU stage | Hosted stage | Required input |
| --- | --- | --- | --- |
| `tokenfactory-train-triage.yaml` | LeRobot training | Summarize textual training artifacts | Training dataset |
| `tokenfactory-rollout-judge-combo.yaml` | LeRobot policy rollout | Score the produced rollout | Compatible policy checkpoint |
| `tokenfactory-scene-to-rollout-judge.yaml` | LeRobot policy rollout | Plan from scene images, then judge against that plan | Scene images and a compatible policy for the same task |

All paths are under
[`workflows/`](../../workflows/).
The [cookbook](cookbooks/tokenfactory-compute-combos.md) gives current config
keys and launch commands. The older `tokenfactory-rollout-judge.yaml` is a
different pipeline: it reasons about a scene and scores externally supplied
rollouts; it does not produce a GPU rollout.

## Compose through an NPA workflow

The checked-in authoring format is `npa.workflow/v0.0.1`. A state selects a
catalog `toolRef`, resources, input/output URIs, and the next state. NPA renders
and executes it through SkyPilot. Use [the workflow guide](npa-workflow-guide.md)
for the schema and [the tool catalog](npa-workflow-tool-catalog.md) for supported
commands.

Keep each stage's S3 paths in `config`, scoped to the run. A consumer must read
the exact artifact its producer writes. Use the spec's case-sensitive config
keys with `--var`; for example, the rollout combo uses `policy_checkpoint`,
`rollouts_uri`, and `scores_uri`.

Validate with `workflow validate-spec`, inspect resolved commands with
`workflow plan-spec`, then execute with `workflow submit`. Use the same
overrides throughout. Add `--runtime` when the graph needs parallel waves or
branches that depend on actual results. See [the run lifecycle](../run-lifecycle.md)
for status, logs, restart behavior, and output verification.

## Use individual stages from Python

For an artifact-producing hosted call, use the SDK wrapper:

```python
from npa.sdk.workbench import token_factory

token_factory.generate(
    input_path="./triage-prompts.jsonl",
    output_path="./triage/generations.jsonl",
    model="<available-text-model>",
    output="json",
)
```

Prepare `triage-prompts.jsonl` from the relevant textual reports first. The
wrapper saves the artifact, prints status, and returns `None`. The lower-level
`npa.workbench.token_factory.generate_text` returns an in-memory dataclass;
call `write_generations` explicitly to persist it. See the
[integration guide](token-factory.md#python-integration) for model selection,
S3 environment requirements, and related APIs.

Existing serverless training runners can triage or rank completed run prefixes;
see [the cookbook](cookbooks/tokenfactory-compute-combos.md#serverless-runner-alternatives).
For new pipelines, keep stage order and artifact dependencies in the NPA spec.
