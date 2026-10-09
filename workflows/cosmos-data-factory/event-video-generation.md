# Event Video Generation getting started

[Shared setup and YAML decision](README.md) · [NPA workflow](paidf-event-video-generation.yaml) · [Upstream guide](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/docs/event-video-generation/getting-started.md)

EVG generates event videos from seed images with Cosmos3 Super Image2Video, then
runs real detection/tracking, captioning, anomaly Visual QA, person Visual QA,
and Person Attribute Search. Dataset assembly and final reopening complete the
twelve-state graph. Examples include falling, fighting, fire/smoke, and shoplifting;
the sampled event and environment are recorded in the config manifest.

## Prepare input

Stage an authorized flat image directory to an S3 prefix, for example:

```text
event-seeds/
  warehouse-entrance.jpg
  loading-dock.png
```

Use a prefix even for one seed; the current S3 adapter enumerates prefixes rather
than accepting upstream's direct single-object or HTTP input form. It reads every
supported image recursively, verifies decoding and a minimum 64-by-64 size,
normalizes to RGB JPEG, and records source/prepared hashes. There is no implicit
demo or `max_images` selector: stage the exact requested subset. Upstream's flat
directory convention is recommended for an unambiguous source-to-video mapping.

## Configure, plan, and run

Complete [shared setup](README.md#shared-npa-setup). Set the listed shell variables
to authorized project/storage/context values and exact scanned private digests.
The generation profile needs two B200s on one node for CFG parallelism; subsequent
GPU labeling states are serial. Verify hosted model availability for your account.

```bash
SPEC=workflows/cosmos-data-factory/paidf-event-video-generation.yaml
# Set PROJECT, KUBE_CONTEXT, BUCKET, INPUT_URI, EVG_IMAGE, DETECTION_IMAGE,
# CAPTIONING_IMAGE, VISUAL_QA_IMAGE, and ATTRIBUTE_SEARCH_IMAGE.
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"
EVG_OVERRIDES=(
  --var "bucket=$BUCKET" --var "input_uri=$INPUT_URI"
  --var "generation_image=$EVG_IMAGE" --var "detection_image=$DETECTION_IMAGE"
  --var "captioning_image=$CAPTIONING_IMAGE" --var "visual_qa_image=$VISUAL_QA_IMAGE"
  --var "attribute_search_image=$ATTRIBUTE_SEARCH_IMAGE"
)

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" "${EVG_OVERRIDES[@]}" \
  --run-id "$RUN_ID" --check-render --json
npa workbench health preflight --project "$PROJECT" --checks s3,token_factory,hf,ngc,nebius
npa workbench health access \
  --capability paidf-evg,paidf-label-detection,paidf-label-captioning,paidf-label-visual-qa,paidf-label-attribute-search
npa workbench workflow preflight-images "$SPEC" "${EVG_OVERRIDES[@]}" \
  --project "$PROJECT" --infra "k8s/$KUBE_CONTEXT"
npa workbench workflow submit "$SPEC" "${EVG_OVERRIDES[@]}" \
  --project "$PROJECT" --infra "k8s/$KUBE_CONTEXT" --run-id "$RUN_ID" --runtime \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env HF_TOKEN --secret-env NGC_API_KEY
```

Use Bash for the array. Stop at a failed check before continuing. Add requested
`num_augmentations`/`seed` overrides consistently. Event/environment distributions
come from the reviewed adapter; upstream `cosmos.variable_distribution`,
service-mode JSON, and Airflow payloads are not NPA overrides. Cosmos3 Super
Image2Video remains pinned at `4f847566f3d3388fbf0ac07b99dd1a6432db9ecd`.

The NPA path enables request guardrails, unlike upstream's disabled template.
Guardrail/model snapshots and exact source adaptations are verified offline.
Visual QA uses published ten-image sampling controls to fit the selected hosted
VLM, replacing upstream's 16 anomaly frames and 12 person crops. These are
documented protocol adaptations, not an unchanged reproduction of the DAG.

## Find and assess outputs

Under `paidf-event-video-generation/<run-id>/`, inspect:

- `prepared/manifest.json` and `configs/manifest.json`: sources and sampled events/environments.
- `cosmos/result.json` and `cosmos/validation.json`: generated videos and their validation.
- `auto_labeling/`: detection, captioning, both Visual-QA passes, and attribute-search
  result documents plus the required native sidecars referenced by them.
- `anomaly_dataset/dataset.json`: final accepted scenes with video/annotation
  paths and skipped/rejected accounting.
- `reports/upstream.json` and `reports/terminal-validation.json`: source/model
  contract and reopened final handoff evidence.

Follow [monitoring and outputs](README.md#monitoring-and-outputs). Independently
decode generated MP4s and inspect whether the intended event occurred. QA answer
coverage is separate from terminal protocol success; trackless scenes may omit
only the supported track-dependent PAS outputs. Existing live evidence records
partial person-QA answers and visual limitations rather than claiming perfect labels.
