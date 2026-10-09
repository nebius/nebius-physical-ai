# Image Attribute Augmentation getting started

[Shared setup and YAML decision](README.md) · [NPA workflow](paidf-image-attribute-augmentation.yaml) · [Upstream guide](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/docs/image-attribute-augmentation/getting-started.md)

IAA creates person-image variants with sampled clothing, colors, footwear, and
accessories. It runs Qwen Image Edit, the real PAIDF augmentation verifier,
upstream pane postprocessing, and Person Attribute Search before assembling and
reopening the final dataset. There are nine states, including provenance and
terminal validation.

## Prepare input

Stage an authorized person-image dataset to an S3 prefix, retaining the upstream
one-folder-per-person layout for source traceability:

```text
person-images/
  person-0001/front.jpg
  person-0001/side.jpg
  person-0002/front.png
```

The current NPA adapter enumerates the prefix recursively and treats each image
as an independent input. It does not reproduce upstream's person-folder limit,
multiple-view concatenation, or person-ID grouping. The prepared manifest retains
each `source_uri` and source hash; use those to recover the source identity.
Images must decode and be at least 64 pixels in each dimension. Preparation
normalizes them to RGB JPEG and preserves both source and prepared hashes.
There is no implicit sample dataset. Stage a selected subset explicitly when
evaluating a subset; Airflow `max_imgs` is not an NPA override.

## Configure, plan, and run

Complete [shared setup](README.md#shared-npa-setup). Set the following variables
in your private shell to the selected project, context, output bucket, input
prefix, and exact scanned private digests. None is inferred from a prior run.

```bash
SPEC=workflows/cosmos-data-factory/paidf-image-attribute-augmentation.yaml
# Set PROJECT, KUBE_CONTEXT, BUCKET, INPUT_URI, IAA_IMAGE, and ATTRIBUTE_SEARCH_IMAGE.
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"
IAA_OVERRIDES=(
  --var "bucket=$BUCKET" --var "input_uri=$INPUT_URI"
  --var "generation_image=$IAA_IMAGE"
  --var "attribute_search_image=$ATTRIBUTE_SEARCH_IMAGE"
)

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" "${IAA_OVERRIDES[@]}" \
  --run-id "$RUN_ID" --check-render --json
npa workbench health preflight --project "$PROJECT" --checks s3,token_factory,hf,ngc,nebius
npa workbench health access --capability paidf-iaa,paidf-label-attribute-search
npa workbench workflow preflight-images "$SPEC" "${IAA_OVERRIDES[@]}" \
  --project "$PROJECT" --infra "k8s/$KUBE_CONTEXT"
npa workbench workflow submit "$SPEC" "${IAA_OVERRIDES[@]}" \
  --project "$PROJECT" --infra "k8s/$KUBE_CONTEXT" --run-id "$RUN_ID" --runtime \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env HF_TOKEN --secret-env NGC_API_KEY
```

Use Bash for the array. Stop at a failed check before running the next command.
Add `--var num_augmentations=<count>` or `--var seed=<integer>` consistently to
the array for requested sampling changes. Attribute distributions currently live
in the reviewed adapter; upstream `cosmos.variable_distribution` JSON is not a
drop-in NPA config. Changing those distributions requires a reviewed adapter
change, rather than an ignored YAML key. Generation pins remain Qwen Image Edit
2511 at `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`.

## Find and assess outputs

Under `paidf-image-attribute-augmentation/<run-id>/`, inspect:

- `prepared/manifest.json` and `configs/manifest.json`: source mapping and sampled attributes.
- `cosmos/result.json`, `cosmos/validation.json`, and `postprocessing/result.json`:
  generation, verification, and accepted pane outputs.
- `auto_labeling/person-attribute-search.json`: label producer and sidecar lineage.
- `augmented_dataset/dataset.json`: NPA's canonical final manifest, with accepted
  samples and rejected/skipped accounting. This differs from upstream's
  `augmented_data.json` filename.
- `reports/upstream.json` and `reports/terminal-validation.json`: exact source
  contract and independently reopened final handoffs.

Follow [monitoring and outputs](README.md#monitoring-and-outputs). Successful
protocol validation does not prove person identity, pose, or requested attribute
quality; review actual images before using them for re-identification training.
