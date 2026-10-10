# Headless manufacturing experiment: pallet detection

Run [`marble-manufacturing-pallet-detection.yaml`](../../workflows/testing/marble-manufacturing-pallet-detection.yaml)
to measure whether generated factory backgrounds improve pallet detection on
real images. The workflow is experimental and lives under `workflows/testing`.
It trains a supervised detector. For robot RL and imitation learning, see the
[additional scene/task requirements](marble.md#which-workloads-benefit).

**World Labs hosts Marble generation behind its API. Nebius GPUs render its
exported splats and train/evaluate the downstream Faster R-CNN detectors.**
No Marble model weights run on Nebius. HTML is an optional output; every stage
executes without a browser, interactive login, display, or deployed service.

| Stage | Execution | Result |
| --- | --- | --- |
| Preflight | CPU | Require the API key, decode real images, validate labels, reject train/test leakage, snapshot inputs |
| Generate | CPU API client → hosted World Labs | New industrial receiving-bay world, SPZ and collider, operation journal |
| Render | Nebius RTX PRO 6000, gsplat CUDA | Factory-background RGB views and camera poses |
| Benchmark | Nebius RTX PRO 6000, PyTorch/torchvision CUDA | Real-only and augmented Faster R-CNN models; evaluation on identical held-out real images |
| Report | CPU | Signed mAP/AP50/AP75 differences, checkpoints, provenance, JSON and static HTML |

The two GPU stages run sequentially and request one GPU each. Baseline training
repeats real images to match the augmented arm's row count. Both arms use the
same seed, epochs, batch size, learning rate, pretrained COCO initialization,
and optimizer-update count. This controls for simply training longer; CUDA
execution is not guaranteed bit-for-bit deterministic.

## Headless run

Create a World API key and fund API credits using the
[World Labs platform](https://platform.worldlabs.ai/api-keys).
Authentication uses the provider's `WLT-Api-Key` header, as documented in the
[API quickstart](https://docs.worldlabs.ai/api). A Marble web subscription
alone does not fund API generation.

Complete the account and storage setup in the
[Marble-to-Nebius guide](guides/marble-to-nebius.md#1-prepare-the-two-accounts-and-storage),
which sets the project, context, bucket, and API key. Then submit with a fresh
run ID and the URI of your real dataset manifest:

```bash
export NPA_MARBLE_RUN_ID="marble-pallets-$(date -u +%Y%m%dt%H%M%sz)"
export NPA_MARBLE_DATASET_URI="s3://$NPA_MARBLE_BUCKET/datasets/pallets/manifest.json"

npa workbench workflow submit \
  workflows/testing/marble-manufacturing-pallet-detection.yaml \
  --project "$NPA_MARBLE_PROJECT" --infra "k8s/$NPA_MARBLE_CONTEXT" \
  --run-id "$NPA_MARBLE_RUN_ID" --stage-src --runtime \
  --var "bucket=$NPA_MARBLE_BUCKET" \
  --var "dataset_uri=$NPA_MARBLE_DATASET_URI" \
  --secret-env WLT_API_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --max-wait-seconds 0 --image-bootstrap-timeout-seconds 0 \
  --output-format json
```

Native submit automatically requires and forwards `WLT_API_KEY` for this
workflow; the explicit flag above documents the dependency. The token can also
be stored as `tokens.WLT_API_KEY` in the private NPA credentials file. Missing
tokens fail before provisioning. The worker checks the token again and validates
the dataset before requesting paid generation. Invalid provider credentials or
insufficient API credits fail the generation stage before GPU consumers launch.
No secret is included in the workflow YAML or HTML.

The generation state fixes `world_source=generate`. The benchmark also rejects
imported/sample worlds. There is no fallback to the Hobbit sample. The existing
operation journal resumes accepted generation without repeating its POST;
uncertain acceptance requires provider reconciliation before retrying.

## Required real dataset

Upload a JSON manifest and its image objects to operator-owned S3. Supply
complete pallet annotations in pixel XYXY coordinates for every real image;
negative images may have an empty `boxes` list. Both partitions need at least
one positive image. `group` identifies a capture session/sequence and must not
cross train and evaluation partitions. Keep the evaluation partition held out
when preparing data and choosing parameters.

```json
{
  "train": [
    {"id": "train-001", "group": "line-a-session-1",
     "uri": "s3://example-bucket/pallets/train-001.png",
     "boxes": [[80, 100, 310, 280]]}
  ],
  "evaluation": [
    {"id": "test-001", "group": "line-a-session-2",
     "uri": "s3://example-bucket/pallets/test-001.png",
     "boxes": [[65, 120, 290, 310]]}
  ],
  "cutouts": [
    {"uri": "s3://example-bucket/pallets/cutout-001.png",
     "source_id": "train-001"}
  ]
}
```

This is a format example, not a supplied dataset. Use representative real
training and evaluation collections. Cutouts must be RGBA PNG images with
visible and transparent pixels, derived only from referenced training images.
The preflight rejects duplicate IDs, shared capture groups, identical decoded
pixels across train/test, out-of-image boxes, empty cutouts, and test-derived
cutout references. It cannot establish whether a declared source lineage is
truthful or detect every near-duplicate; dataset preparation remains essential.

## Outputs and interpretation

Under `s3://<your-bucket>/runs/<run-id>/marble-manufacturing/`:

- `dataset/`: run-scoped snapshots, pixel/source hashes, and labels.
- `world/`: World API operation and downloaded world assets.
- `captures/`: actual CUDA renders and timing/device evidence.
- `benchmark/baseline/` and `benchmark/augmented/`: trained checkpoints and
  detailed training/evaluation records.
- `benchmark/comparison.json`: signed AP differences, equal update counts,
  evaluation image hashes, actual GPU identity, and checkpoint hashes.
- `report/comparison.json` and `report/index.html`: the headless result and
  optional human-readable report; `report/synthetic/` contains the composites.

The synthetic images are **2D alpha composites**, with exact boxes around the
inserted pallets. They do not simulate pallet contact, metric placement,
lighting interaction, or physical occlusion. Although the prompt requests empty
backgrounds, Marble can generate unintended objects; labels cover only inserted
pallets. This can reduce quality, which is why the real held-out comparison is
required. A positive single-seed AP change is an experimental observation,
not production acceptance. Negative/zero changes remain `no_improvement`.

No production model is promoted automatically. Run multiple seeds and evaluate
across independent factory conditions before drawing a deployment conclusion.
The manufacturing workflow has not yet completed a live run: the funded API
key and the operator's real labeled dataset are still required. Read the
[readiness record](../../workflows/testing/marble-manufacturing-pallet-detection.readiness.json)
for the exact validation scope.
