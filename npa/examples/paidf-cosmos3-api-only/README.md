# Hosted Cosmos3: twelve variants without a cluster

This standalone Python example adapts the generation/review subset of
[`workflows/main/paidf-cosmos3.yaml`](../../../workflows/main/paidf-cosmos3.yaml).
It uses an **existing** Cosmos3 HTTPS API, existing registered S3 storage and
CPU processing on a Mac or an existing VDI. It provisions no infrastructure.
Hosted GPU inference still runs on the service provider's infrastructure and
may incur charges. The service controls GPU concurrency; twelve concurrent
client admissions do not imply twelve concurrent GPU executions.

[`campaign.yaml`](campaign.yaml) is a standalone campaign format interpreted by
[`run.py`](run.py), **not** an `npa.workflow` spec. Do not pass it to
`npa workbench workflow submit-spec`. This is a separate example, not a change
to the canonical PAIDF workflow or a native NPA CLI command.

| Canonical stage | This adaptation |
| --- | --- |
| Source preparation | Already prepared, authorized robot manipulation MP4 in registered S3 |
| Captioning | Omitted; explicit appearance prompts in YAML |
| Generation | Twelve guarded Cosmos3-Super full-source edge augmentations |
| Cosmos Evaluator | Service `technical_report`, SHA-256, ffprobe and full FFmpeg decode; no semantic evaluator |
| Cosmos Curator | Omitted; manual sampled review with explicit accepted/rejected results |
| FiftyOne / Voxel51 | Omitted; self-contained HTML gallery |
| Rerun | Replaced by the HTML gallery |
| Automatic refinement | Omitted |

## Setup

Use Python 3.12+, an installed NPA checkout, Git, `ffmpeg` and `ffprobe` on PATH.
The FFmpeg build must include `libx264rgb` and `libvpx-vp9` encoders.
NPA supplies the CPU border-detection/preservation code and NumPy. The runner loads
the repository's [portable HTTPS client](../../../skills/tools/nebius-world-factory/SKILL.md),
so keep it in this checkout. It requires neither a private hosting repository
nor model weights, Docker, Kubernetes, SkyPilot, FiftyOne or a local GPU.
Use the existing contributor environment (`npa/.venv`), or install this checkout
with `python3 -m venv npa/.venv` and
`npa/.venv/bin/python -m pip install -e npa` from the repository root.

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
private requests, durable job states, downloaded media and the final HTML.
`--workflow` optionally selects another YAML (default: adjacent `campaign.yaml`).
Keep that YAML and all bindings unchanged after admission. The key file can be
rotated to another key for the same owner; the portable client checks ownership.

## Execute and recover

From the repository root, set `PAIDF_WORK_DIR` to the private directory above:

```bash
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py validate
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py admit --work-dir "$PAIDF_WORK_DIR" --private-config "$PAIDF_WORK_DIR/private-config.json"
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py resume --work-dir "$PAIDF_WORK_DIR" --private-config "$PAIDF_WORK_DIR/private-config.json"
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py collect --work-dir "$PAIDF_WORK_DIR"
```

`admit` performs discovery/preflight for every request, then submits twelve
distinct requests concurrently. It persists each idempotency key before its
generation POST. If admission is interrupted or ambiguous, repeat **the same
`admit` command with the same directory and YAML** to reconcile those identities.
Do not create a replacement work directory to retry. `resume` only polls and
downloads retained jobs; it never submits generation. It waits for active jobs
without an overall deadline. After a transport error, rerun the same command.
Terminal job failures require inspection of private job records, not replacement
generation. Stop local polling without cancelling service work at any time.

`collect` verifies canonical media against the service manifest, guarded
configuration, prompt and seed receipts, complete local decode, frame count,
frame rate and dimensions. It also detects persistent paired source padding
across every source frame and restores only those border pixels from the source,
using NPA's existing lossless RGB H.264 preservation path, followed by lossless
VP9 RGB WebM for correct browser color rendering. Every generated scene
pixel remains unchanged; source borders and scene pixels are verified after
encoding. Raw `downloads/<variant-id>/generated.mp4` files remain untouched.
Corrected browser videos and hash-bound receipts are written to
`publication/<variant-id>/{augmented.webm,padding-webm.json}`. The intermediate
RGB H.264 MP4 and its original padding receipt remain alongside them. The WebM
export is checked against the same complete scene/border RGB hashes and source
timeline before publication; this is a lossless conversion, not a lossy preview.
Subsequent collection
verifies that cached publications still match source, raw video and published
hashes; changed publications fail rather than silently reusing old evidence.
Full-frame sources are not reencoded. Ambiguous/one-sided/changing dark edges
are not assumed to be padding. No YAML switch or additional API call is needed.

Collection writes `technical.json`, including raw and published hashes, source
content bounds and padding receipts. Lossless corrected videos and the resulting
HTML are substantially larger than lossy model outputs. These checks do not
establish task fidelity, semantic quality or training readiness.

To keep local polling alive while your laptop is closed, run it on an **existing
VDI** inside `tmux` (start the session on that VDI, not on the laptop):

```bash
tmux new-session -s paidf-api-only
# Run the commands above in this session; detach with Ctrl-b then d.
```

The hosted jobs continue independently after admission even if the client
disconnects. Reattach with `tmux attach-session -t paidf-api-only`. This example
does not create a VDI, install a controller service or run an agent loop.

## Review and standalone HTML

Inspect the source and all twelve published videos identified by
`technical.json`'s `publication_path`. For detected padding these are the corrected
`publication/<variant-id>/augmented.webm` files; raw model outputs remain separately
available for provenance. The report always embeds the published videos.
Write a mode-0600 `quality-review.json` in the private directory, with exactly
one record for each YAML variant. Example record shape:

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
requires all four booleans to be true. Review notes are a manual sampled
assessment; they do not represent automatic evaluator scores. Keep rejections
in the gallery.

```bash
npa/.venv/bin/python npa/examples/paidf-cosmos3-api-only/run.py report --work-dir "$PAIDF_WORK_DIR" --quality-review "$PAIDF_WORK_DIR/quality-review.json"
```

`report` rechecks the current media and receipts, then writes
`$PAIDF_WORK_DIR/paidf-cosmos3-proof.html`. Open that file directly in a browser.
It embeds the source and twelve outputs as MP4/WebM data URLs, requires no server or
external assets, and escapes prompts/notes. Treat this file as private media;
do not commit it or the work directory. Browser playback is a separate manual
check, not inferred from HTML generation.

## Evidence and tests

A privately retained execution of the predecessor runner delivered twelve
distinct 832×480, 336-frame videos, all checksum-verified and fully decoded.
Sampled review accepted eleven and rejected one for a spurious object.
Observed service running concurrency was one. Its offline HTML played all
thirteen embedded videos in an isolated local browser. The packaged `report`
command was also run against those original media bytes and service receipts:
all thirteen embedded videos played at 832×480 in an isolated local browser
with zero HTTP requests. No new generation was submitted for that report check.
This evidence applies to that campaign; it is not a fresh live admission through
this packaged entrypoint or a claim that the complete canonical PAIDF graph ran
without infrastructure.

Run the portable client and example's offline regression tests with:

```bash
npa/.venv/bin/python -m pytest npa/tests/clients/test_nebius_stream_client.py npa/tests/workflows/test_paidf_api_only_example.py -q
```
