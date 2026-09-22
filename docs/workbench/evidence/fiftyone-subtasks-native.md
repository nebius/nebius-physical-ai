# Native FiftyOne subtask export validation

Executed locally on macOS with FiftyOne **1.22.0**, its real MongoDB service,
Python 3.12, and the production Workbench import/export code. No GPU or new cloud
infrastructure was provisioned.

Source: `lerobot/svla_so100_pickplace`, revision
`728583b5eaf9e739a7f119e2def466fa1d552402`, real SO100 camera/action data.
The test supplies four programmatic labels to distinguish both episodes and both
halves of each episode; these are identity assertions, not semantic annotations.

| Check | Observed result |
| --- | --- |
| Native import order | Episodes `7, 2` |
| Deterministic derived mapping | Source `2 → 0`, source `7 → 1` |
| Frames checked against original observations/actions | 771 / 771 |
| Native temporal tags materialized | 4 |
| Unlabeled frames | 0 |
| Native re-import and restored temporal tags | 4 / 4 |
| Integration test | 1 passed |

The first native run exposed an actual upstream-format mismatch: this dataset's
task names live in its Pandas index, while FiftyOne requires a `task` column.
Workbench now normalizes that metadata in a separate staging copy during import.
Source task metadata, camera files, and frame tables are not modified.

## Episode identity contract

[FiftyOne 1.22's exporter](https://github.com/voxel51/fiftyone/blob/v1.22.0/fiftyone/utils/lerobot_export.py)
builds export specifications in collection order and rewrites `episode_index`
contiguously. Mapping tags directly to original episode indexes would mis-handle
subsets. Workbench sorts by validated source episode identity, freezes the sample
IDs in an ordered view, and uses that same order for both labels and native
export. The report exposes the source-to-derived mapping. Regression tests cover
shuffled/noncontiguous episodes and invalid/duplicate source identities.

## Scope and reproduction

Run [the committed opt-in test](../../../npa/tests/e2e/test_fiftyone_subtasks_native.py)
using the environment and command in the
[labeling guide](../guides/lerobot-subtask-labeling.md#reproducible-test-evidence).
It executes `export_fiftyone_subtasks` against a real persistent FiftyOne dataset,
then independently compares source frame columns and resolved label sequences.

This is **native API/DB integration evidence**, not a browser annotation session
or a live cloud S3 upload. Hermetic tests execute `export_fiftyone_subtasks_to_s3`
and the generated remote script using the real shared StorageClient with mocked
S3 transport, including adaptive retry configuration, empty-prefix refusal,
symlink refusal, URI validation, and operation without an installed NPA package.

Earlier [MP4/RRD evidence](lerobot-video-subtasks/README.md) remains unchanged and
is bound to its historical recipe, not relabeled as output of this new run.
