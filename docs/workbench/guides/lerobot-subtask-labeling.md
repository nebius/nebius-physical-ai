# Label LeRobot subtasks in FiftyOne

[Guides](README.md) · [LeRobot training](reachy2-lerobot-policy.md) · [FiftyOne skill](../../../skills/tools/fiftyone/SKILL.md)

Use this workflow when a policy needs an episode-level task for planning and a
frame-aligned subtask for low-level control. LeRobot remains the training data
contract; FiftyOne 1.22 supplies synchronized camera/state/action playback and
the human timeline editor. No NVIDIA model or service is required.

## 1. Load the source dataset

Deploy or update the FiftyOne workbench, then load a LeRobot v3 dataset:

```bash
npa workbench fiftyone load-dataset \
  --name robot-subtasks \
  --input-path 's3://<bucket>/<source-dataset>/' \
  --format lerobot

npa workbench fiftyone open
```

The source is loaded as one multimodal sample per episode. Existing
`subtask_index` metadata is restored as timeline tags when present.

## 2. Review and label

Open an episode, scrub the synchronized streams, then Shift-drag each interval
on the timeline. Name every subtask tag with this exact namespace:

```text
subtask:approach
subtask:align
subtask:grasp
subtask:place
```

Intervals are half-open (`[start, end)`). They may touch, but must not overlap,
and every frame in every exported episode must be covered. Other temporal tags
are left untouched and are not training labels. The episode's existing LeRobot
task instruction remains the high-level task.

## 3. Export a derived LeRobot dataset

```bash
npa workbench fiftyone export-lerobot-subtasks \
  --dataset-name robot-subtasks \
  --output-path 's3://<bucket>/<derived-dataset>/' \
  --output-format json
```

The output prefix must be empty. The command exports a self-contained copy,
validates coverage, and writes:

- `subtask_index` on every frame row;
- `meta/subtasks.parquet` as the label catalog;
- `meta/lerobot_annotations.json` for resumable annotation state; and
- the `subtask_index` feature declaration in `meta/info.json`.

The original dataset is never rewritten. Train against the derived prefix only
after the report shows `unlabeled_frame_count: 0`.

## 4. Prove the reviewed labels with YAML

Use [`lerobot-subtask-proof.yaml`](../../../workflows/testing/lerobot-subtask-proof.yaml)
for the repeatable check after interactive review. It
reads the derived LeRobot Parquet files, requires every frame to resolve through
`meta/subtasks.parquet`, and publishes one row-level proof without rewriting the
dataset.

```bash
npa workbench workflow validate-spec \
  workflows/testing/lerobot-subtask-proof.yaml --json

npa workbench workflow plan-spec \
  workflows/testing/lerobot-subtask-proof.yaml \
  --run-id subtask-proof-preview \
  --var 'bucket=<configured-bucket>' \
  --var 'reviewed_dataset_uri=s3://<bucket>/<derived-dataset>/' \
  --var 'proof_uri=s3://<bucket>/<proof-prefix>/subtask-proof.json' \
  --json
```

Submit the same spec and overrides after the plan is correct:

```bash
npa workbench workflow submit workflows/testing/lerobot-subtask-proof.yaml \
  --run-id '<new-run-id>' \
  --var 'bucket=<configured-bucket>' \
  --var 'reviewed_dataset_uri=s3://<bucket>/<derived-dataset>/' \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

| Setting | Meaning |
| --- | --- |
| `reviewed_dataset_uri` | Required derived LeRobot v3 dataset prefix, after export. |
| `expected_subtask` | Label to prove; defaults to `grasp`. Override for your vocabulary. |
| `proof_uri` | Exact output object, outside the dataset. Defaults to the run's `proof/subtask-proof.json`. |
| `source_overlay` | Defaults to `true` so the worker runs the submitted checkout. |

The proof includes
the episode, frame, timestamp, resolved label, integer `subtask_index`, source
Parquet and label-catalog SHA-256, row SHA-256, and dataset-wide coverage counts.
The worker downloads metadata and frame Parquet only; videos are not needed for
this check. Missing frames, unlabeled rows, unknown indexes, and an absent
requested label fail the workflow before it publishes a proof.

## Reproducible test evidence

The local test executes the YAML through the workflow interpreter against a
synthetic LeRobot Parquet fixture after applying the actual subtask exporter:

| Episode | Frame interval | Subtask | `subtask_index` |
| --- | --- | --- | --- |
| 0 | `[0, 2)` | approach | 1 |
| 0 | `[2, 4)` | grasp | 2 |
| 1 | `[0, 1)` | align | 0 |
| 1 | `[1, 4)` | place | 3 |

It proves `grasp` at episode `0`, frame `2`, timestamp `0.2`, with `8/8` frames
labeled across four segments. Tests independently recompute the evidence hashes
and confirm that verification preserves the source bytes. This validates the
data contract; the fixture is synthetic and does not establish real-world label
accuracy or policy improvement.

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_lerobot_subtask_proof.py -q
```

The CPU live-submit matrix seeds a separate four-frame fixture and reads back
the published S3 proof, data Parquet, and catalog to verify the actual row and
digests. Current cloud prerequisites and execution status are recorded in the
[readiness record](../../../workflows/testing/lerobot-subtask-proof.readiness.json).

## Scope

This path is manual and CPU-only. A VLM can propose labels upstream, but it is
optional and proposals still need review before export. No NVIDIA labeling tool
is required. The source candidate is not a newly published FiftyOne image until
its exact image bytes pass the normal build, security, functional, and promotion
gates.

Training must explicitly consume `subtask_index` or its decoded instruction in
the policy or sampler. Exporting labels alone does not make a standard LeRobot
policy use hierarchical conditioning.
