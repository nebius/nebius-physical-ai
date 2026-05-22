---
name: cosmos
description: Use when working on Cosmos world model serving, inference, downloads, backend selection, or rendering limitations.
---

# Cosmos

Cosmos is the world model tool for synthetic data generation and video generation.

It requires a GPU. RT cores are not required for standard serving and inference, unlike Isaac Lab. Cosmos visual generation has the same container EGL/DRI rendering gap as Genesis.

## Interfaces

API:

- `POST /serve`
- `POST /infer`
- `POST /download`
- `GET /status`
- `GET /system-info`
- `GET /list`

CLI:

```bash
npa workbench cosmos deploy
npa workbench cosmos serve
npa workbench cosmos infer
npa workbench cosmos download
npa workbench cosmos status
npa workbench cosmos system-info
npa workbench cosmos list
```

## Backend Selection

Use `--backend` to select one of:

- `basic`
- `nim`
- `triton`

Choose based on workload. For multiple models, use named workbenches or the download-plus-serve swap pattern.

E2E is pending: EGL/DRI blocks rendering, while serving and inference may work independently.
