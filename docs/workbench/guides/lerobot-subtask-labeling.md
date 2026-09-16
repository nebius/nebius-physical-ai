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
  --input-path s3://<bucket>/<source-dataset>/ \
  --format lerobot

npa workbench fiftyone open
```

The source is loaded as one multimodal sample per episode. Existing
`subtask_index` metadata is restored as timeline tags when present.

## 2. Review and label

Open an episode, scrub the synchronized streams, then Shift-drag each interval
on the timeline. Name every subtask tag with this exact namespace:

```text
subtask:approach object
subtask:align gripper
subtask:close gripper
subtask:place object
```

Intervals are half-open (`[start, end)`). They may touch, but must not overlap,
and every frame in every exported episode must be covered. Other temporal tags
are left untouched and are not training labels. The episode's existing LeRobot
task instruction remains the high-level task.

## 3. Export a derived LeRobot dataset

```bash
npa workbench fiftyone export-lerobot-subtasks \
  --dataset-name robot-subtasks \
  --output-path s3://<bucket>/<derived-dataset>/ \
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

## Scope

This path is manual and CPU-only. A VLM can propose labels upstream, but it is
optional and proposals still need review before export. The source candidate is
not a newly published FiftyOne image until its exact image bytes pass the normal
build, security, functional, and promotion gates.
