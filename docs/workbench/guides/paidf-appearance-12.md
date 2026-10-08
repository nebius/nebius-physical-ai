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

## Public reference measurement

The unchanged recipe was run on episode 0, camera
`observation.images.cam_high`, seconds 0–8 of
[`lerobot/aloha_static_cups_open`](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/tree/d793c969cf716001dcca18a0842c3d7e9de9e41b).
The prepared reference and each output contain 192 frames at 24 fps. Both
generation passes produced all twelve separate videos, with full decoding and
timestamp alignment verified. Results recorded on October 7–8, 2026:

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
