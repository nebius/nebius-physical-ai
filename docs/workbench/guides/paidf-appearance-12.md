# Generate twelve appearance variants

The [twelve-profile recipe](../examples/paidf-appearance-12.yaml) supplies a
complete, opt-in configuration for `workflows/main/paidf-cosmos3.yaml`. It
generates twelve separate videos from one source using oak, walnut, aluminum,
stainless steel, granite, terracotta, rubber, cork, ceramic, green laminate,
warm side lighting and cool twilight lighting. Existing workflow defaults remain
unchanged.

The profiles edit an **existing stationary work surface** while preserving
robot parts, task objects, geometry, contacts and timing. Select a source where
that surface is visible. These are appearance instructions, not guarantees that
the model preserves every task detail. For a scene without a visible work
surface, define suitable profiles instead of asking the model to invent one.

## Apply the recipe

Complete the [operator setup and preflight](../../../workflows/guides/paidf-cosmos3.md)
first. Create a private copy of the canonical workflow and merge only the
recipe's configuration. The recipe itself is not a standalone workflow.

```bash
umask 077
PRIVATE_RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paidf-appearance-12.XXXXXX")"
PRIVATE_SPEC="$PRIVATE_RUN_DIR/paidf-cosmos3.yaml"
npa/.venv/bin/python - "$PRIVATE_SPEC" <<'PY'
import sys
from pathlib import Path
import yaml

workflow = yaml.safe_load(Path("workflows/main/paidf-cosmos3.yaml").read_text())
recipe = yaml.safe_load(
    Path("docs/workbench/examples/paidf-appearance-12.yaml").read_text()
)
workflow["config"].update(recipe["config"])
Path(sys.argv[1]).write_text(yaml.safe_dump(workflow, sort_keys=False))
PY
npa/.venv/bin/npa workbench workflow validate-spec "$PRIVATE_SPEC" --json
```

Use `PRIVATE_SPEC` as `SPEC` in the setup guide's reserve, plan and submit
commands. Include `--max-wait-seconds 0` in the submit command so the runtime
waits for the complete generation stage. The CLI otherwise defaults to a
one-hour deadline and cancels the running stage when it expires; twelve
sequential variants can take longer. This is a submit option, not a YAML
configuration key.

Supply the real project, bucket and explicit `--input-video` or
LeRobot episode/camera selection there. Keep source-specific descriptions,
storage locations, media and run evidence outside the checkout. The recipe
already includes generic preservation and caption instructions; any task-specific
clarifications belong in the private copy. Existing `--var` overrides take
precedence over the copied configuration.

The setup guide's canonical workflow also accepts these values directly as
`--var key=value` arguments. Preserve `appearance_profiles_json` as one JSON
string, not a YAML sequence. Each profile has exactly four fields:
`lighting`, `background`, `color_grade` and `surface_finish`.

## Count, concurrency and outputs

`variant_count=12` selects all twelve profiles once. They are shuffled using
`augmentation_seed=30`, so output order differs from file order. Read each
variant's `metadata.json` to identify its profile. Raising the count above
twelve repeats profiles with new generation seeds; it does not define new looks.
Twelve unique profiles define requested looks; verify the rendered material in
each output, since the model can miss a material instruction.

`variant_parallelism=1` runs the variants sequentially on one GPU. Increasing
that setting requires a generation job with multiple visible GPUs. It does not
allocate GPUs or spread variants across independent single-GPU nodes.

Each generated video is published separately beneath the run prefix:

```text
cosmos_augmented/variant-0000/augmented_video.mp4
...
cosmos_augmented/variant-0011/augmented_video.mp4
```

The generation manifest and progress report account for completed and failed
variants. Retain videos and evidence from rejected attempts. A final workflow
rejection does not turn generated candidates into accepted dataset entries.

## Review the evidence

The recipe requires timestamp alignment, grade threshold 0.75 and appearance
attribute threshold 1.0. Temporal and appearance-fidelity diagnostics retain the
canonical advisory policy; show their results alongside the gate decision.
Every generated clip must pass the gate. A high average score does not override
a failed profile. Increasing the count adds more clips that must pass.
An accepted gate does not certify motion, contacts or training suitability.

New Cosmos 3 runs record the scene rectangle in `source_content_region` during
reference preparation. The pipeline also inspects every source frame for
persistent paired black bands at opposite edges, including bars embedded in
the source. Detection requires near-black bands, roughly symmetric widths and
a clear brightness step into the scene. It abstains on ambiguous, one-sided,
changing or insufficiently observed borders. Known normalization padding stays
protected even when embedded-bar detection abstains. Interior dark objects are
not padding. No additional YAML setting is needed.

Before publication, the pipeline restores those borders from the source and
encodes a lossless RGB H.264 MP4. Every decoded generated scene pixel and source
border pixel must match exactly, with unchanged frame count and timing. These
MP4s are larger than lossy model outputs. `raw_model_video.mp4` and
`raw_model_metadata.json` preserve the unmodified model evidence; the published
`augmented_video.mp4`, extracted frames and alignment hashes describe the
corrected output. `metadata.json.padding_preservation` records both video
hashes, scene/border pixel hashes and the verified policy. This fixes border
artifacts; it does not repair defects within the generated scene.

The evaluator verifies the source hash and complete
video alignment, then uses lossless scene crops for all four checks. Artificial
letterbox or pillarbox pixels cannot lower the scene score or satisfy a
requested appearance change. Real black objects remain in the scene. Explicit
metric regions keep their full-canvas coordinates and are intersected with the
scene; a padding-only region is an error.

Each clip's `spatial_evidence` reports the full-video and crop hashes, rectangle,
excluded pixel fraction and a separate padding diagnostic. The diagnostic
reports mean RGB error and the fraction of padding pixels differing from the
source by more than 8 RGB levels in any channel. It is advisory, has no quality
pass threshold, and does not change the scene score. Successfully preserved
borders have zero RGB error against the reference. The raw output remains
available to inspect what the model originally generated.

Older variants without region provenance retain full-frame scoring. Start a
fresh run to detect and preserve borders automatically; existing stored videos
are not rewritten. Invalid or stale provenance fails instead of silently
selecting another crop.

Compare source/output frames at contacts, generation-window joins and the final
state. Check that the requested material or lighting is visibly present inside
the original camera area, and inspect added padding separately. Preserve raw
outputs when making any presentation crop. Use the
[realism review procedure](paidf-realistic-augmentation.md#review-and-tune-the-actual-outputs)
and [public LeRobot comparisons](paidf-lerobot-realism.md) to assess limitations.

Validate the unchanged recipe on independent sources before making broader
claims. Source-specific prompts, output media and exact infrastructure evidence
must remain in private storage for private recordings.

The retained-run audit checks workflow completion, all twelve profiles,
complete video decoding, source/output alignment, native control receipts and
evaluator accounting. It verifies that the evaluator's video hashes match the
retained outputs, so an earlier pass cannot stand in for the final results.
Create a private JSON array with `run_uri` and `expected_frames` for each proof,
then run:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_PROJECT="$PROJECT_ALIAS" \
NPA_PAIDF_APPEARANCE_CASES="$PRIVATE_RUN_DIR/proof-cases.json" \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_paidf_appearance_recipe_live.py -q
```

This audit makes read-only storage calls and downloads actual videos. It does
not generate more candidates or reinterpret a rejection as acceptance.
The canonical workflow deliberately finishes a rejected batch with a failed
`reject-quality` terminal stage after preserving review evidence. The audit
requires that exact rejection path, a valid rejected disposition and successful
preceding stages; other workflow failures do not qualify.

To audit padding-aware scoring on retained outputs, use the same private case
file with `NPA_PAIDF_PADDING_CASES`, set `NPA_PAIDF_PADDING_EVIDENCE_DIR` to a
private local directory, and run
`npa/tests/e2e/test_paidf_padding_evaluation_live.py`. This audit downloads the
videos and makes real Token Factory calls. It requires the same hallucination
engine as the retained grades; for production Cosmos Evaluator runs, use its
pinned NVIDIA checkout and dependencies (`NPA_COSMOS_EVALUATOR_SRC`). Legacy
rectangles are reconstructed only from the retained normalization recipe and
the hash-verified original and prepared videos. Reports are written locally;
retained videos and original grades are not overwritten. The audit also measures
a fresh full-frame hallucination baseline with the same engine to isolate the
effect of padding removal on that metric.

To test source detection and border preservation on retained real outputs, use
`NPA_PAIDF_PADDING_CASES` with the same private case file and set
`NPA_PAIDF_PADDING_PRESERVATION_DIR` to a new private local directory, then run
`npa/tests/e2e/test_paidf_padding_preservation_live.py` with the integration and
project variables above. This makes read-only storage calls, detects borders
without relying on old preparation metadata, and checks every decoded scene
and border pixel in each corrected video. It neither generates new model
outputs nor reruns the VLM.

## Public reference measurement

The unchanged recipe was run on episode 0, camera
`observation.images.cam_high`, seconds 0–8 of
[`lerobot/aloha_static_cups_open`](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/tree/d793c969cf716001dcca18a0842c3d7e9de9e41b).
The prepared reference and each output contain 192 frames at 24 fps. Both
generation passes produced all twelve separate videos, with full decoding and
timestamp alignment verified. Results recorded on October 7–8, 2026 using the
original full-frame evaluator:

| Generation pass | Guidance / steps | Mean score | Clips passing gate | Temporal advisory passes | Appearance advisory passes | Batch decision |
|---|---|---|---|---|---|---|
| Initial | 5.0 / 35 | 0.722930 | 2/12 | 0/12 | 3/12 | Rejected |
| Canonical refinement | 4.5 / 39 | 0.708905 | 2/12 | 0/12 | 3/12 | Rejected |

This demonstrates twelve-profile execution, not an accepted training dataset.
Refinement did not improve this batch's measured result. Matched-frame review
found visible lighting and material variation, but also incomplete surface
coverage, artificial material boundaries, color spill onto robot parts and
changes to small task details. More profiles increase requested diversity;
they do not guarantee realism, preservation or a higher acceptance rate.
