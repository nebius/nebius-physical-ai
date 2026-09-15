---
name: fiftyone
description: Use when deploying, launching, loading data into, or reviewing the FiftyOne workbench dataset curation and visualization tool.
---

# FiftyOne

The PAIDF worker image remains contract-attested and non-root for SkyPilot
0.12.2, with passwordless sudo, SSH/rsync/service prerequisites, writable paths,
and a forwarding entrypoint. Submit the verified immutable digest; never use a
`runAsUser: 0` workflow override.

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

`npa workbench fiftyone open` maintains a verified SSH tunnel for a VM or
`kubectl port-forward` for Kubernetes; callers should not need raw tunnel commands.

## Deployment And Access

The App is an operator interface with access to service-readable files. Bind it
to `127.0.0.1` and access it through authenticated SSH or Kubernetes port-forward.
`--public-ip`, non-loopback `--address`, and application ingress are rejected.
Kubernetes uses a ClusterIP Service only as a port-forward target; the pod's App
and its exec readiness check use loopback. Keep Kubernetes port-forward RBAC
limited to authorized operators. CORS does not provide authentication.

Stock FiftyOne App has no `/health` endpoint: `GET /` returns 200 and `GET /health` returns 307.

Managed VM `deploy` defaults to in-place updates for existing aliases. Terraform
plans that would destroy or replace critical infrastructure are blocked unless
the operator passes `--replace` and confirms with `--yes` for automation.

VM and BYOVM deploys record `endpoint_strategy: ssh_fallback`. Live status and
GraphQL requests always create a fresh SSH forward with verified host keys,
including for older aliases that recorded a public endpoint. Launch and dataset
loading run over SSH. Use `open` to keep a browser tunnel alive. Unknown or
changed SSH host keys fail closed; provide an independently verified
`NPA_SSH_KNOWN_HOSTS` file for a BYOVM host without a provider pin.

Redeploy old Kubernetes and container deployments to replace public listeners;
native VM launch also migrates its service environment to loopback. Existing
published images are old bytes until an updated image is built and validated.

## Real Curation (Brain)

The image ships FiftyOne Brain (uniqueness / similarity / visualization) so it can
do *real* curation, not just viewing. `npa workbench fiftyone curate-augmented
--augment-uri <cosmos_augmented/> --curator-report-uri
<curation/cosmos_curator.json> --report-uri <curation/report.json>` runs the
Physical AI Data Factory curation in-container: it builds a real `fiftyone.Dataset`
from the augmented scenario variants, computes a GPU-free per-variant embedding
(downsampled RGB + color histogram, via Pillow/numpy), and runs
`fob.compute_uniqueness` + `fob.compute_similarity(...).find_duplicates()` +
`fob.compute_visualization(method="pca")`. The report records `curation_engine:
fiftyone-brain`, per-variant `uniqueness`, near-duplicate clusters, and which
variants were kept vs dropped. A completed real Cosmos Curator report is
required via `--curator-report-uri`; missing/unavailable Curator or FiftyOne
Brain is always a hard failure. The shipped PAIDF workflow also passes
`--require-fiftyone` as an explicit compatibility assertion through the
`workbench.fiftyone.curate_augmented` toolRef, so a claimed PAIDF FiftyOne review
always means both real stages ran. The container functional smoke
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

Open the selected BDD100K demo dataset through the same authenticated local route.
