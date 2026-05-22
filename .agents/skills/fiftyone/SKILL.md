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
npa workbench fiftyone status
npa workbench fiftyone system-info
npa workbench fiftyone list
```

`npa workbench fiftyone open` wraps `kubectl port-forward`; callers should not need raw `kubectl`.

## Deployment And Access

The deploy `--public-ip` flag creates a LoadBalancer Service for external access, intended for partner demos. `npa workbench fiftyone status` shows the Public URL when deployed with `--public-ip`.

Stock FiftyOne App has no `/health` endpoint: `GET /` returns 200 and `GET /health` returns 307.

## Data Patterns

FiftyOne Brain uses `fob.compute_visualization` for CLIP UMAP embeddings.

FiftyOne supports custom field schemas. Do not assume generic auto-extracted fields are required.

BDD100K demo dataset: `bdd100k-real-data-demo`, live at the public IP.
