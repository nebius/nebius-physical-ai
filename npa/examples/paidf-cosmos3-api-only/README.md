# Hosted Cosmos3: twelve variants from a managed endpoint

This standalone Python example adapts the generation/review subset of
[`workflows/main/paidf-cosmos3.yaml`](../../../workflows/main/paidf-cosmos3.yaml)
to an **existing managed** Cosmos3 HTTPS endpoint. The service owns every media
stage: source validation and border analysis on its CPUs, guarded GPU
generation, restoration of verified source borders, lossless RGB browser
previews, checksums, publication and a finishing report. This client only
discovers the contract, submits twelve logical tasks, polls durable status,
downloads checksum-verified results and assembles an offline HTML gallery from
the server's files. It runs **no FFmpeg, NumPy or other media decoding**, and it
has no CPU/GPU, node, replica, container, placement or concurrency settings.

[`paidf-managed-endpoint.yaml`](paidf-managed-endpoint.yaml) is a standalone
campaign format interpreted by [`run.py`](run.py), **not** an `npa.workflow`
spec. Do not pass it to `npa workbench workflow submit-spec`.

| Canonical stage | This adaptation |
| --- | --- |
| Source preparation | Existing authorized MP4 in registered S3; the service decodes and analyzes it |
| Captioning | Omitted; explicit appearance prompts in YAML |
| Generation | Twelve guarded Cosmos3-Super full-source edge augmentations |
| Cosmos Evaluator | Service `technical_report` and finishing report; no semantic evaluator |
| Cosmos Curator | Omitted; manual sampled review with explicit accepted/rejected results |
| FiftyOne / Voxel51 | Omitted; self-contained HTML gallery |
| Rerun | Replaced by the gallery of service lossless RGB previews |
| Automatic refinement | Omitted |

This is a scoped adaptation, not the full canonical Curator/evaluator/refinement
graph.

## Requirements

The endpoint must advertise managed finishing: authenticated `GET /v1/models`
must report `managed_finishing.status: "available"` for `nvidia/Cosmos3-Super`
with `request_field: "output_policy"`. Otherwise `admit` stops **before** any
generation request. There is no client-side media fallback.

Use Python 3.12+, Git and this checkout (`npa/.venv`, or `python3 -m venv
npa/.venv` and `npa/.venv/bin/python -m pip install -e npa`). The runner loads
the repository's [portable HTTPS client](../../../skills/tools/nebius-world-factory/SKILL.md).
It needs no private hosting repository, model weights, Docker, Kubernetes,
SkyPilot, FiftyOne, local GPU, FFmpeg or NumPy.

Obtain the service origin and scoped key privately. Follow the client guide to
register existing input and server-verified Nebius output storage. Retain the
source's license/provenance and permission privately. The shipped prompts assume
a robot manipulation source; edit prompts for another source **before admission**.
Discovery and server validation must qualify the requested route/model/settings.
The client verifies TLS and refuses redirects. `WMH_CA_BUNDLE` is optional and
only selects an operator-supplied trusted CA; hostname verification stays enabled.

Create a mode-0700 directory **outside any Git checkout**. Put a mode-0600
`private-config.json` there with exactly these string fields, replacing the
placeholders locally. `wmh_api_key_file` must be an absolute path to an owned,
regular mode-0600 file containing the key; never put key bytes in this JSON.

```json
{
  "wmh_base_url": "<supplied HTTPS origin>",
  "wmh_api_key_file": "<absolute path to private key file>",
  "input_connection_id": "<registered input connection>",
  "output_connection_id": "<verified output connection>",
  "source_uri": "<authorized S3 source URI>",
  "output_prefix": "<S3 prefix inside the registered output scope>"
}
```

The YAML's four `${PAIDF_...}` placeholders resolve from this file, not ambient
environment variables. `--private-config` is required for `admit` and `resume`.
`--work-dir` is required for every phase except offline `validate`; it holds
private requests, durable job states, downloaded results and the final HTML.
`--workflow` optionally selects another YAML (default: adjacent
`paidf-managed-endpoint.yaml`). Keep that YAML and all bindings unchanged after
admission. The key file can be rotated to another key for the same owner.

## Output policy

```yaml
output_policy:
  border_preservation: restore_detected_source_borders
  browser_preview: lossless_rgb_webm
```

These are data/quality choices only. With `restore_detected_source_borders`
the service restores persistent paired black source bands exactly while keeping
every generated scene pixel, and verifies both after lossless encoding. It
abstains on one-sided, changing or dark scene edges and then publishes the raw
model output unchanged. `lossless_rgb_webm` adds VP9 lossless RGB previews whose
every frame the service decodes and compares with the finished RGB frames; the
RGB H.264 file is not used for browser playback because Chrome renders it with
wrong colors.

## Execute and recover

From the repository root, set `PAIDF_WORK_DIR` to the private directory above:

```bash
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py validate
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py admit --work-dir "$PAIDF_WORK_DIR" --private-config "$PAIDF_WORK_DIR/private-config.json"
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py resume --work-dir "$PAIDF_WORK_DIR" --private-config "$PAIDF_WORK_DIR/private-config.json"
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py collect --work-dir "$PAIDF_WORK_DIR"
```

`admit` checks the managed capability and qualifies every request, then submits
twelve distinct requests concurrently (`client_admission_parallelism`). It
persists each idempotency key before its POST. If admission is interrupted or
ambiguous, repeat **the same `admit` command with the same directory and YAML**.
Do not create a replacement work directory to retry. `resume` only polls and
downloads retained jobs, verifying every artifact's SHA-256; it never submits
generation and waits without an overall deadline. A job in stage `finishing`
is service CPU work after its GPU result was saved. A `publication_retryable`
job needs the service's publication-only retry, never a new generation.

`collect` performs receipt checks only: guarded configuration, prompt and seed
receipts, the applied output policy, one generation dispatch per job, artifact
hashes against the manifest, and finishing-report lineage (`source`,
`model-raw`, `generated`, previews), timeline and border outcome. It writes
`technical.json` with client admission parallelism and, separately, the
observed service model-request concurrency computed from service-reported
generation windows. These checks do not establish task fidelity, semantic
quality or training readiness.

Run on an existing VDI inside `tmux` if polling should survive a closed laptop.
Hosted jobs continue independently after admission.

## Review and standalone HTML

Inspect the downloaded `generated-preview.webm` files (or `generated.mp4`) with
any player. Write a mode-0600 `quality-review.json` with exactly one record for
each YAML variant:

```json
{
  "variants": [{
    "variant_id": "variant-01-warm-daylight",
    "verdict": "accepted",
    "criteria": {
      "motion_and_timing_plausible": true,
      "robot_object_identity_intact": true,
      "contacts_physically_plausible": true,
      "requested_appearance_visible": true
    },
    "notes": "Describe the inspected frames/times and observed evidence."
  }]
}
```

Use `rejected` and the observed false criteria for failed variants. Acceptance
requires all four booleans to be true. Keep rejections in the gallery.

```bash
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py report --work-dir "$PAIDF_WORK_DIR" --quality-review "$PAIDF_WORK_DIR/quality-review.json"
```

`report` re-verifies receipts and hashes, then writes
`$PAIDF_WORK_DIR/paidf-managed-endpoint-proof.html`, embedding the service's
source preview and twelve generated previews as WebM data URLs with each
preview's service-recorded frame-zero RGB reference samples. It needs no server
or external assets and escapes prompts/notes. Treat it as private media. Browser
playback and color checks are separate from HTML generation.

## Tests

```bash
npa/.venv/bin/python -m pytest npa/tests/clients/test_nebius_stream_client.py npa/tests/workflows/test_paidf_managed_endpoint_example.py -q
```

The offline tests use a fake endpoint and fail if the client starts FFmpeg or
ffprobe.
