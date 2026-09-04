---
name: encord
description: Use when pushing Nebius object-store media into the Encord curation SaaS (register-in-place, bytes stay in the bucket), curating headlessly with Encord quality-metric filter presets (no human in the app), pulling curated Collections/Datasets/Project labels back to S3, or wiring the encord-push / encord-curate / encord-pull / encord-roundtrip-smoke workflows.
---

# Encord (curation SaaS push/curate/pull)

Encord is a third-party labeling/curation platform. The workbench integration
is a **register-in-place** loop: `push` lists an S3 prefix, registers public
objectUrls with Encord through a cloud integration (bytes never leave the
bucket), and links the items into an Encord dataset; curation is either
**headless** (`curate` — workbench-declared quality filters evaluated
server-side by Encord into a Collection, no human in the app) or manual (a
human curates in the Encord app); `pull` materializes the curated Collection,
Dataset, or a Project's labels back to S3 as media + per-item JSON + a lineage
manifest.

## Three-access pattern

Implementation lives in `npa/src/npa/workbench/encord/`: one module per verb
(`push.py`, `curate.py`, `pull.py`, `verify.py`, `cleanup.py`, `seed_demo.py`,
`system_info.py`), the SaaS seam in `client.py` (auth, title-or-id resolution
returning `ResolvedRef`), exact identity in `identity.py`, checksums in
`integrity.py`, and the S3-only artifact writer in `storage.py`. The CLI
(`npa/src/npa/cli/workbench/encord.py`) and SDK
(`npa/src/npa/sdk/workbench/encord.py`) are thin wrappers over the `run_*`
functions; every CLI verb carries `@json_stdout_contract`, so `--output json`
yields exactly one JSON document on stdout even on failure, and only
`EncordToolError` maps to exit 1 (anything else is a bug and exits 2 through
`app_entry`). There is no service tier: the UI is Encord's own app. The
`encord` PyPI package is an optional extra (`npa[encord]`), lazy-imported;
workflow stages on the default image install it via `TOOL_REF_PIP_EXTRAS`.

Operator walkthrough (account setup through troubleshooting):
`docs/workbench/encord.md`.

## One-time Encord-side setup (operator)

1. In the Encord app create an **S3-compatible cloud integration** (the
   MinIO/OTC pattern) pointing at the Nebius endpoint
   (`https://storage.<region>.nebius.cloud`) with a dedicated key pair that has
   read access to the media bucket. Prefer "strict client-only access" so Encord
   signs URLs client-side instead of copying media server-side.
2. Give the bucket a read policy for that key pair, and if the Encord viewer
   loads media directly in the browser, a CORS rule allowing `*.encord.com`.
3. Note the integration **title** — the tool accepts it directly
   (`--integration nebius-s3`); no ids need to be copied around.

## Auth

Generate a key pair in the Encord app (public keys). Exactly **two** credential
transports exist — a raw multi-line PEM pasted into YAML/env is
truncation-prone (an observed live failure) and is not accepted:

```yaml
# ~/.npa/credentials.yaml — laptops: point at the key file
tokens:
  ENCORD_SSH_KEY_FILE: /Users/<you>/.ssh/encord-private-key.ed25519
```

For pods/workflow submits, the base64 form is the multi-line-safe transport
(`base64 < key.pem | tr -d '\n'`), forwarded by name only:

```bash
--secret-env ENCORD_SSH_KEY_B64
```

`ENCORD_DOMAIN` selects a non-default (e.g. US) API domain. Verify before
spending time:

```bash
npa workbench health preflight --checks encord   # live auth probe
npa workbench encord system-info                 # no API call: SDK pin, domain, which transport is set (names only)
```

## Interfaces

```bash
# Register a prefix in place and link a dataset for annotation.
npa workbench encord push \
  --input-path s3://<bucket>/raw-media/ \
  --integration nebius-s3 \
  --folder my-batch --dataset my-batch \
  --output-path s3://<bucket>/encord/push/

# Curate headlessly: Encord evaluates the filters server-side into a
# Collection (no human in the app). Zero selected items fails closed.
npa workbench encord curate \
  --folder my-batch \
  --filter brightness:0.2:0.8 --filter width:640:4096 \
  --collection my-batch-keepers \
  --output-path s3://<bucket>/encord/curate/

# Or after curating in the Encord app: materialize the keeper Collection.
npa workbench encord pull \
  --source collection --source-id <collection-uuid-or-name> \
  --output-path s3://<bucket>/encord/pull/

# Or pull a whole dataset / a project's labels (labels export iff project).
npa workbench encord pull --source dataset --source-id my-batch ...
npa workbench encord pull --source project --source-id <project-hash> ...
```

Push has two transfer modes: `--transfer register` (default — bytes stay in the
bucket; Encord references objectUrls through the integration) and
`--transfer upload` (bytes are copied into Encord-hosted storage; no
integration needed). `seed-demo` and the `workbench.encord.seed_demo` toolRef
share that register default; `encord-cosmos3-augment.yaml` is the one shipped
spec that opts into upload, for its public demo clip. Pull is mode-agnostic:
registered items come back as zero-egress server-side copies, uploaded items
stream back through Encord signed URLs (a same-bucket copy that fails falls
back to download and records `copy_error` in the manifest row).

Titles resolve wherever they are unique; UUID/hash-shaped values must exist.
`push --folder/--dataset` and `curate --collection` titles are created when
absent; `curate --folder` and `pull --source-id` never create. `curate` refuses
a Collection that already holds items (Encord's evaluation only adds, so a
stale selection could not be told apart from this run's); register-mode
`push`/`seed-demo` refuse an empty `--integration` before any I/O.

## Headless curation (`curate`)

`--filter metric:min:max` (repeatable; comma-separable in one value, which is
the workflow-template form) maps onto a run-scoped Encord **filter preset**
(`npa-curate-<run-id>`, deleted best-effort once the selection lands — the
receipt's `filter_preset_json` is the reproducibility record) using the filter
shape pinned by live spike:
`{"include", "values": [min, max], "domain": "data", "metric": "<id>",
"type": "metric"}`. The tool creates/reuses the Collection and calls
`add_preset_items` — Encord evaluates the selection server-side (asynchronous;
the tool polls until stable, `--poll-seconds` default 300).

Metric allowlist (unknown names fail closed **before** any Encord call — an
unpinned shape hangs Encord's evaluation request): intrinsic `width`,
`height`, `area`, `aspect-ratio` evaluate on any folder immediately; computed
`brightness`, `sharpness`, `file-size` match nothing until quality metrics
have been computed for the folder — a one-time action in the Encord app with
no public API. Zero selected items fails closed (exit 1) with a diagnostic
that names this cause when a computed metric is in play.

## Supported formats in NPA

The NPA tool supports individual `.mp4`, `.png`, `.jpg`, and `.jpeg` objects.
Its Cosmos augmentation workflow accepts video only. Encord itself supports
additional modalities (including MCAP and point-cloud scenes), but NPA does not
yet construct the required scene/stream payload: `.mcap` under `--media mcap`
or `--media all` is recorded as `experimental_error` and the push fails closed.
ROS bags, point clouds, and other sensor formats are unsupported rather than
generic file transports. Pull also fails per item for composite image groups and
DICOM series because they lack one signed media URL.

## Data contract

- **Exact identity** (adopted from PR #363): every pushed item is registered
  with namespaced `npa.source_uri` clientMetadata, and receipt lineage resolves
  through that metadata or the item's normalized objectUrl — display names,
  and in particular basenames, are never identity. Conflicting signals fail
  the item closed (`identity signals conflict`).
- Push receipt: `push_receipt.json` (`npa.encord.push_receipt.v1`) — per-item
  source_uri/objectUrl/uuid/status plus source size + checksum (single-part S3
  ETag as md5 in register mode; sha256 of the bytes in upload mode), unit
  counts, folder/dataset lineage. A **write-ahead** copy with `status: planned`
  lands before the first Encord mutation, then the final receipt overwrites it
  (crashes after Encord was mutated land in the artifact's `error` field); any
  unit error fails the command closed (exit 1).
- Roundtrip report: `roundtrip_report.json` (`npa.encord.roundtrip_report.v1`)
  from `encord verify --receipt-uri ... --manifest-uri ...` — joins receipt to
  manifest by Encord uuid and fails closed on missing/unexpected items, size
  mismatches, or checksum mismatches (kinds that cannot be compared, e.g. a
  multipart ETag, count as `checksum_unavailable`, not failures). It also fails
  closed when the receipt's `status` is not `done` (a write-ahead `planned`
  copy or a failed push), when zero items are attributable (`defects` names
  the reason, so 0/0 never reads as `passed`), and on matched items with
  neither a comparable checksum nor a size on both sides (`unverifiable`). This is the machine-checkable
  checksum evidence; the roundtrip smoke ends with it.
- Curate receipt: `curate_receipt.json` (`npa.encord.curate_receipt.v1`) — the
  parsed filters, the exact preset JSON sent to Encord, preset + Collection
  uuids/lineage, `items_total` / `items_selected`, and `preset_deleted` (the
  transient `npa-curate-<run-id>` preset is deleted in a `finally`, even after a
  crash; an ad-hoc run without a run id gets a timestamped, random-suffixed
  title). Same write-ahead + written-before-failure contract as push; zero
  selected items fails closed.
- Pull output under `--output-path`: `media/<item_uuid>__<name>`,
  `items/<item_uuid>.json`, `labels/<label_hash>.json` (project source only),
  and `manifest.json` (`npa.encord.pull_manifest.v1`) with copy/download/failed
  counts. Media registered from this bucket returns as zero-egress
  **server-side copies**; anything else streams through the Encord signed URL.
- Re-push is idempotent in **both** modes as a caller-side invariant: push
  resolves exact identity against the folder first and transfers only what is
  absent, so a retried stage re-registers nothing and copies no duplicate
  bytes. Re-pull overwrites. Run-scoped Encord state is torn down with
  `npa workbench encord cleanup --title-prefix <npa-...->` (datasets are
  reported, not deleted — the SDK has no dataset deletion).

## Workflows

- `npa/workflows/workbench/npa-workflows/encord-push.yaml` — production push;
  terminal after the receipt (curation is human-in-the-loop).
- `npa/workflows/workbench/npa-workflows/encord-pull.yaml` — production pull,
  run after curation with the Collection uuid (or dataset/project reference).
- `npa/workflows/workbench/npa-workflows/encord-cosmos3-augment.yaml` — the
  curation-to-augmentation loop in one run: pull an Encord video, generate two
  distinct real Cosmos 3 video2video variants on the GPU profile, and push all
  results back into Encord as `npa-aug-<run-id>`. Runs out of the box: by
  default a `workbench.encord.seed_demo` stage stages the packaged pinned
  starter clip (CC-BY-4.0, SHA-256-verified) into a run-scoped
  `npa-demo-src-<run-id>` dataset. This spec alone sets
  `encord_transfer: upload` (public demo bytes; no cloud integration needed).
  For real data override `--var encord_source_id=<your curated id>` (the seed
  stage no-ops) **and** `encord_transfer=register` +
  `encord_integration=<title>`; `encord_item_index` selects which pulled video
  to augment.
- `npa/workflows/workbench/npa-workflows/encord-cosmos3-groot-finetune.yaml` —
  the fully unattended loop: push a LeRobot camera stream (register mode, the
  tool default: pass `--var encord_integration=<title>`), curate it
  headlessly into `npa-groot-curated-<run-id>` (default filter is the
  intrinsic `width:32:16384`; override `encord_curate_filters` for
  brightness/sharpness on metric-computed folders), pull the curated
  Collection, augment with Cosmos 3, materialize, and fine-tune GR00T.
- `npa/workflows/workbench/npa-workflows/encord-roundtrip-smoke.yaml` — the
  live e2e test: push fixture media into a fresh `npa-e2e-<run-id>` folder +
  dataset, curate a run-scoped Collection headlessly, then pull both the
  dataset (by title) and the Collection in the same run. Submit with
  `--secret-env ENCORD_SSH_KEY_B64 --secret-env AWS_ACCESS_KEY_ID
  --secret-env AWS_SECRET_ACCESS_KEY`. Clean up `npa-e2e-*` folders/datasets/
  collections in Encord afterwards (curate deletes its own presets).

toolRefs: workbench.encord.push, workbench.encord.curate, workbench.encord.pull,
workbench.encord.verify, workbench.encord.seed_demo (declared outputs of push,
curate, pull, and verify are pinned to the tool's `*_uri_for` helpers by
`npa/tests/guardrails/test_spec_declared_outputs.py`).

## GPU routing

CPU-only, every verb. No container image; stages run on the SkyPilot
default image with the `npa[encord]` extra installed at setup.

## Known issues and boundaries

- **MCAP is experimental and currently fails closed.** The pinned
  `encord==0.1.x` upload format has no cloud-registration category for a raw
  `.mcap` (scenes require per-stream SceneBuilder assets). `--media mcap|all`
  discovers `.mcap` keys and records them in the receipt as
  `experimental_error` without sending a guessed schema; the receipt-visible
  failure is the honest v1 boundary pending a live spike.
- Composite Encord items (image groups, DICOM series) expose no single signed
  URL and are recorded as per-item pull errors.
- Licensing/egress: the `encord` SDK wheel is Apache-2.0 (verified from package
  metadata). Push sends object URLs + metadata only; media bytes leave the
  bucket only on the cross-origin download path during pull. Review your Encord
  agreement for any field-of-use terms on exported labels before training on
  them.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/workbench/test_encord.py npa/tests/cli/test_encord_cli.py npa/tests/workflows/test_encord_workflow.py npa/tests/workflows/test_encord_loop.py npa/tests/workflows/test_encord_groot_loop.py -q
```

Design note and live spike evidence: `docs/workbench/encord-headless-curation.md`.
