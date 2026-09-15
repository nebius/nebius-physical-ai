# Workbench reference assets

[Workflow catalog](../../../workflows/README.md) · [Workbench docs](../../../docs/workbench/README.md)

This directory contains configuration assets, native Ray development examples,
and Sim2Real operator notes. Start in the root [workflow catalog](../../../workflows/README.md)
when you want to compose and run a pipeline with `npa workbench workflow`.

## Choose an example

| Goal | Start here | Result |
| --- | --- | --- |
| Compose Workbench tools | [Declarative workflows](../../../workflows/README.md) | A planned state graph and run-scoped artifacts |
| Run the 14-stage robot pipeline | [Sim2Real runbook](../../../docs/workbench/guides/sim2real-workflow.md) | Per-stage evidence, evaluation, and Rerun/MCAP outputs |
| Edit a GPU application and resubmit Python | [Ray CLIP](ray-clip-development/README.md) | Image embeddings, retrieval results, and source-change evidence |
| Train a small distributed model | [Ray Train](ray-train-synthetic/README.md) | Synthetic-regression checkpoints and rank metrics on two B200 hosts |
| Try hyperparameter search and checkpoint recovery | [Ray Tune](ray-tune-synthetic/README.md) | Three CPU trials, the selected optimum, and verified JSON results |

## Preview a workflow locally

From the repository root, after [installing NPA](../../../docs/install.md):

```bash
npa workbench workflow validate-spec workflows/testing/cosmos3-generate.yaml
npa workbench workflow plan-spec workflows/testing/cosmos3-generate.yaml --run-id demo
```

Expect a valid spec and one `generate` stage. These commands require no cloud
resources. To execute, complete the selected workload's input, credential,
image, and resource setup, then follow its submission and cleanup commands.
The [workflow guide](../../../docs/workbench/npa-workflow-guide.md) explains
`prepare-run`, `submit --runtime`, monitoring, and resume.

## Directory map

| Directory | Purpose |
| --- | --- |
| `configs/` | Component and benchmark configuration |
| [`sim2real/`](sim2real/README.md) | Operator notes and finite legacy compatibility assets |
| `schemas/` | Parameter, artifact, naming, and runtime conventions |
| `steps/`, `templates/` | Legacy compatibility placeholders |
| `ray-*/` | Standalone native Ray application examples and platform setup |

## Raw SkyPilot YAML

NPA renders declarative workflows to SkyPilot when submitting. The workflow
command also accepts customer-owned raw SkyPilot tasks. Shipped raw tasks live
in specific example directories: [burst](../../src/npa/burst/examples/README.md),
[BYOF profiles](../../src/npa/workflows/byof/profiles/README.md), and
[NuRec single-pod execution](../../src/npa/workbench/nurec/examples/README.md).
The retired raw workflow catalog must not be restored.

Use the [authoring skill](../../../skills/workflows/author-npa-workflow/SKILL.md)
for specification edits and the
[pipeline design skill](../../../skills/workflows/generate-npa-workflow/SKILL.md)
for new compositions. Stop active jobs and preserve outputs before removing
owned infrastructure; see [teardown](../../../docs/teardown.md).
