---
name: fiftyone
description: Use when deploying, launching, loading data into, or reviewing the FiftyOne workbench dataset curation and visualization tool.
---

# FiftyOne

FiftyOne is the dataset curation and visualization tool. It is CPU-only and does not require a GPU.

## Interfaces

API:

- `POST /load-dataset`
- `GET /status`
- `GET /system-info`

CLI:

```bash
npa workbench fiftyone deploy
npa workbench fiftyone launch
npa workbench fiftyone load-dataset
npa workbench fiftyone curate-augmented   # real FiftyOne Brain curation of a paidf run
npa workbench fiftyone status
npa workbench fiftyone system-info
npa workbench fiftyone list
```

`npa workbench fiftyone open` wraps `kubectl port-forward`; callers should not need raw `kubectl`.

## Deployment And Access

The deploy `--public-ip` flag creates a LoadBalancer Service for external access, intended for partner demos. `npa workbench fiftyone status` shows the Public URL when deployed with `--public-ip`.

Stock FiftyOne App has no `/health` endpoint: `GET /` returns 200 and `GET /health` returns 307.

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

BYOVM deploys record `endpoint_strategy: public` or `endpoint_strategy:
ssh_fallback` in `~/.npa/config.yaml`. Live `status`, `launch`, and
`load-dataset` commands honor that strategy and self-heal blocked public
endpoints through a transient SSH-local route.

## Real Curation (Brain)

The image ships FiftyOne Brain (uniqueness / similarity / visualization) so it can
do *real* curation, not just viewing. `npa workbench fiftyone curate-augmented
--augment-uri <cosmos_augmented/> --report-uri <curation/report.json>` runs the
Physical AI Data Factory curation in-container: it builds a real `fiftyone.Dataset`
from the augmented scenario variants, computes a GPU-free per-variant embedding
(downsampled RGB + color histogram, via Pillow/numpy), and runs
`fob.compute_uniqueness` + `fob.compute_similarity(...).find_duplicates()` +
`fob.compute_visualization(method="pca")`. The report records `curation_engine:
fiftyone-brain`, per-variant `uniqueness`, near-duplicate clusters, and which
variants were kept vs dropped. Outside the image (no FiftyOne) it degrades to a
report-only counts path. The container functional smoke
(`docker/workbench/fiftyone/smoke_functional.py`) exercises this Brain path.

The `npa-fiftyone` image bundles `mongod` (the prebuilt `fiftyone_db` wheel ships
no mongod for trixie) into `fiftyone/db/bin/` so FiftyOne launches its own
metadata DB with no external MongoDB — required for any Brain method. To run
curation against an *un-rebuilt* image, supply an external DB instead
(`-e FIFTYONE_DATABASE_URI=mongodb://<host>:27017`).

## Data Patterns

FiftyOne Brain uses `fob.compute_visualization` for CLIP UMAP embeddings. Brain
methods also accept precomputed `embeddings=` (no model / GPU) — that is how the
paidf `curate-augmented` path runs uniqueness/similarity/visualization CPU-only.

FiftyOne supports custom field schemas. Do not assume generic auto-extracted fields are required.

BDD100K demo dataset: `bdd100k-real-data-demo`, live at the public IP.
