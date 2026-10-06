# Reproduce the twelve-scenario ALOHA cup fanout

Run [the single-pass review workflow](../../../workflows/testing/paidf-aloha-cups-fanout.yaml)
on Nebius RTX PRO 6000 to generate twelve full-episode appearance candidates
from a fresh pinned LeRobot download. It recreates the source selection,
profiles and sampling controls of the [measured eight-second fanout](paidf-lerobot-realism.md#measured-twelve-scenario-fanout).
The earlier strict gate rejected all twelve candidates. A new review must retain
its actual measurements and failures; comparable appearance does not establish
training suitability or faithful contacts.

The graph is `prepare-input → generate-configs → annotate-original →
generate-variants → evaluate → quality-disposition → visualize-quality-evidence`.
Every input, caption, configuration, candidate, evaluation and recording belongs
to the new run. The terminal recording is `reports/quality-evidence.rrd`.
A successful workflow means the review artifacts were produced. Read
`grade/quality_disposition.json` to determine acceptance; this graph performs no
refinement, accepted-data curation or promotion.

## Prepare the operator environment

Use your own current `main` checkout and its Python 3.12 development environment.
Complete the [PAIDF setup and GPU health checks](../../../workflows/guides/paidf-cosmos3.md#s--one-time-setup)
on a Linux operator host. Select the configured project, storage destination,
kubeconfig, supported vision model and discovered GPU name outside Git:

```bash
export PROJECT_ALIAS='<configured-project-alias>'
export BUCKET='<verified-artifact-bucket>'
export KUBE_CONTEXT='<verified-gpu-context>'
export KUBECONFIG='<private-kubeconfig-path>'
export CAPTION_MODEL='<available-vision-model-id>'
export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
export NPA_WORKFLOW_GPU_ACCELERATOR='<discovered-rtx-pro-6000-name>:1'
SPEC=workflows/testing/paidf-aloha-cups-fanout.yaml
RUN_ID="$(npa/.venv/bin/npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS")"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/workflow-runs/$RUN_ID/skypilot"
```

Stop if a command fails. Preserve this isolated SkyPilot directory for submit,
status, cancellation and controller cleanup. Verify storage and Token Factory
credentials and exact Cosmos guardrail access before allocating GPU time;
[the setup guide](../../../workflows/guides/paidf-cosmos3.md) gives those commands.
Public image selection uses the governed immutable Cosmos3/Evaluator candidates
and Rerun release with the submitted checkout's adapters through
`source_overlay: true`. Keep image preflight enabled.

## Download and verify the source

Select `lerobot/aloha_static_cups_open` at revision
`d793c969cf716001dcca18a0842c3d7e9de9e41b`, episode `0`, camera
`observation.images.cam_high`. The [source manifest](../examples/paidf-lerobot-realism-sources.json)
pins all four files and their SHA-256 hashes. The dataset card declares MIT;
retain the README with its unchanged source metadata and camera video.
This is video augmentation: action/state tables are not consumed or reconstructed.

```bash
LEROBOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paidf-aloha-source.XXXXXX")" || exit 1
DATASET_BASE='https://huggingface.co/datasets/lerobot/aloha_static_cups_open/resolve/d793c969cf716001dcca18a0842c3d7e9de9e41b'
for file in README.md meta/info.json meta/episodes/chunk-000/file-000.parquet \
  videos/observation.images.cam_high/chunk-000/file-000.mp4; do
  mkdir -p "$(dirname "$LEROBOT_DIR/$file")"
  curl --fail --location "$DATASET_BASE/$file" --output "$LEROBOT_DIR/$file" || exit 1
done
npa/.venv/bin/python - "$LEROBOT_DIR" <<'PY' || exit 1
import hashlib
import json
import sys
from pathlib import Path
from npa.workflows.paidf_cosmos3 import _select_lerobot_video

manifest = Path("docs/workbench/examples/paidf-lerobot-realism-sources.json")
selection = json.loads(manifest.read_text())["selections"][0]
root = Path(sys.argv[1])
for name, expected in selection["downloaded_sha256"].items():
    with (root / name).open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"Checksum mismatch: {name}; stop before uploading")
video, interval, camera = _select_lerobot_video(root, 0, selection["camera"])
assert camera == "observation.images.cam_high"
assert interval == (0.0, 8.0)
assert video.is_file()
print("Verified four pinned files and the exact eight-second camera selection")
PY
```

Inspect representative source frames before submission to confirm that the
selected high camera depicts the intended cup-opening task. Metadata and checksums
prove selection and integrity, not depicted objects or contact fidelity.

## Stage fresh data and submit

Upload only the freshly verified directory to a run-scoped dataset prefix through
the existing storage client. The configured project supplies credentials and the
exact endpoint. An occupied destination fails instead of replacing a prior input.
Save the timestamp with private run evidence before this upload and submission:

```bash
LEROBOT_URI="s3://$BUCKET/datasets/paidf-aloha-cups-fanout/$RUN_ID/"
export NPA_PAIDF_ALOHA_FRESH_AFTER="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
npa/.venv/bin/python - "$PROJECT_ALIAS" "$LEROBOT_DIR" "$LEROBOT_URI" <<'PY' || exit 1
import sys
from npa.clients.project_credentials import storage_client_for_project

storage = storage_client_for_project(sys.argv[1], allow_host_creds=True)
storage.upload_directory(sys.argv[2], sys.argv[3], require_empty=True)
print("Uploaded the verified source into an empty run-scoped dataset prefix")
PY
npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" --json || exit 1
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" --check-render \
  --var bucket="$BUCKET" --var lerobot_dataset_uri="$LEROBOT_URI" \
  --var caption_model="$CAPTION_MODEL" --json || exit 1
npa/.venv/bin/npa workbench workflow preflight-images "$SPEC" \
  --project "$PROJECT_ALIAS" || exit 1
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" --project "$PROJECT_ALIAS" \
  --var bucket="$BUCKET" --var lerobot_dataset_uri="$LEROBOT_URI" \
  --var caption_model="$CAPTION_MODEL" \
  --runtime --infra "k8s/$KUBE_CONTEXT" --durable-s3 \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

The distinct workflow name uses generic configuration overrides, so the canonical
starter's automatic source selector cannot substitute another video. Do not pass
`--lerobot-uri` or `--input-video`: these submission flags belong to the canonical
PAIDF workflows. Automatic source staging supplies current NPA adapters to every
worker. Do not reuse an older prepared source, caption report or generated media.

Preparation retains the full eight-second selected episode and resamples it to
192 frames at 24 fps, letterboxed to 832×480. The twelve profiles are embedded in
the YAML and guarded against the shared profile example. `augmentation_seed=30`
fixes their shuffled order; diffusion seeds are `17` through `28`. Model controls
remain 35 steps, guidance 5, edge guidance 1, low Canny thresholds, native RGB hint
weight 0.25, zero first-window source RGB frames, disabled CFG normalization and
zero source-pixel blending. Guardrails and decoded timestamp alignment remain on.

The default uses one visible RTX PRO 6000 and `variant_parallelism=1`. The original
fanout used two GPUs; serial generation changes throughput, not profiles or
sampling settings. For two visible GPUs, select an accelerator count of two and
add `--var variant_parallelism=2` consistently to planning and submission.
Inspect the manifest's actual effective concurrency.

Keep grade threshold 0.75, attribute threshold 1.0, temporal consistency required
at 0.8, and appearance fidelity advisory. Newly generated hosted captions and
available model/runtime versions can change results; this is a fresh comparable
experiment, not a promise of byte-identical videos or identical scores.
The historical fanout's exact base prompt and captions were not preserved in
the public comparison report. This recipe records its explicit preservation
prompt and freshly generated captions for the new experiment. Different FFmpeg
builds can also produce different prepared-video hashes from identical source
files; synchronize against this run's actual `input/source.mp4` and timeline.

## Review and retain evidence

Retain input provenance/timeline, sampled configurations, source captions,
augmentation manifest/progress, every original-resolution generated MP4 and its
native controls, `grade/cosmos_evaluator.json`, `grade/quality_disposition.json`,
`grade/decision.json`, and `reports/quality-evidence.rrd`. Fully decode all twelve
MP4s, verify their hashes and 192-frame timelines, and decode the Rerun recording
to check run identity, embedded media and truthful candidate dispositions.
Inspect the first/last frames, grasp/release contacts and generation joins at
frames 88 and 176. Preserve the actual reviewed indices and remaining uncertainty.

Create the synchronized overview using only resized decoded source/output frames
and static labels, following the [comparison procedure](paidf-realistic-augmentation.md#make-a-synchronized-comparison).
Retain complete source and candidates alongside it. Evaluate the clear cup and
lid, gripper details, contacts, trajectory and final state separately from tabletop
appearance. The temporal score is an image-motion diagnostic, not physical proof.

The live-submit matrix registers this full twelve-candidate review without
reducing variant count or overriding public images. After fresh source staging,
select it with `NPA_E2E_NPA_WORKFLOW_SUBMIT_SPECS=paidf-aloha-cups-fanout.yaml` and
set `NPA_E2E_PAIDF_ALOHA_DATASET_URI` to that verified dataset directory. It is
excluded from automatic rotations that cannot prepare this exact public input.
The [readiness record](../../../workflows/testing/paidf-aloha-cups-fanout.readiness.json)
distinguishes local graph/render checks from completed live evidence.

Follow [the setup guide's owned cleanup](../../../workflows/guides/paidf-cosmos3.md#r7-finish-owned-cleanup)
after retaining artifacts: reconcile/cancel the exact run before removing its
owned controller and isolated API, then destroy only infrastructure provisioned
for that run. Keep original reports and private operational receipts outside Git.
