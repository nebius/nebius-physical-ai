# NuRec single-pod SkyPilot example

[Reconstruction guide](../../../../../../docs/workbench/guides/neural-reconstruction.md) · [Workflow catalog](../../../../../../workflows/README.md)

This directory holds the intentionally retained **single-pod** NuRec/NRE SkyPilot
task from #234. It is an example for running the NuRec tool end to end inside one
pod, not a workflow authoring catalog.

The supported workflow authoring surface is the declarative `npa.workflow/v0.0.1`
spec:

```text
workflows/main/nurec-reconstruct.yaml
```

The spec runs each state in its own pod and hands artifacts over through S3. This
example shares `/tmp` across all NuRec stages in one pod, which is the distinct
behavior #234 validated and documented.

| File | What it exercises |
| --- | --- |
| `nurec-reconstruct.yaml` | NuRec/NRE check, fetch, reconstruct, render, visualize, and finalize in one SkyPilot task. |

## Rules

- **One task per file.** A multi-stage pipeline belongs in
  `workflows/` as an `npa.workflow/v0.0.1` spec.
- **Keep `${VAR}` placeholders.** Concrete registry ids, bucket names, run ids,
  access keys, and tokens must never be committed here.
- This file must never move back under the retired raw SkyPilot workflow catalog.
- `npa/tests/guardrails/test_nurec_examples.py` pins these rules.

## Before submitting

Complete the reconstruction guide's project, storage, model/container access,
and RT-core GPU setup. The example selects RTX PRO 6000 and needs a matching
Kubernetes context with enough CPU, memory, and scratch storage for NRE.
Prepare the NGC image-pull Secret in the task namespace and export required
runtime credentials privately. The template's `NPA_SRC_S3_URI` must name a
verified staged source tree; replace every other placeholder consistently too.

For a first reconstruction, use the guide's canonical declarative workflow.
This retained example is for the specific shared-scratch, single-pod behavior
validated in #234; it is not a self-contained fresh-account quickstart.

## Submitting

```bash
npa workbench workflow submit npa/src/npa/workbench/nurec/examples/nurec-reconstruct.yaml \
  --run-id "neural-reconstruction-<scene>-<yyyymmdd>t<hhmmss>z" \
  --var NPA_NUREC_IMAGE=nvcr.io/nvidia/nre/nre-ga:26.04 \
  --var NPA_NUREC_RUN_ID="<run-id>" \
  --var NPA_NUREC_RUN_URI="s3://<bucket>/<prefix>/neural-reconstruction/<run-id>" \
  --var NPA_SRC_S3_URI="s3://<bucket>/npa-src/<tag>" \
  --infra "k8s/<rt-core-context>"
```

Use the same run ID for `--run-id`, `NPA_NUREC_RUN_ID`, and the output prefix.
Inspect all six stages and the final S3 outputs: reconstruction USDZ, novel-view
renders, `reports/sim2real.rrd`, and `reports/final.json`. The pod's `/tmp` data
is temporary; preserve outputs before following the
[cleanup procedure](../../../../../../docs/teardown.md).
