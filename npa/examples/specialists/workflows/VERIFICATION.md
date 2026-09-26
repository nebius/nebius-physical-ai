# Verifying real NuRec workflow output

[`nurec_verify.py`](nurec_verify.py) is an independent artifact reader for the
canonical six-stage NuRec workflow: `check`, `fetch`, `reconstruct`, `render`,
`visualize`, `finalize`. It reads a fresh downloaded run tree and receipts
collected by the operator's trusted adapter. It does not submit workflows,
contact a model, download artifacts or execute the serialized model checkpoint.

Keep its input evidence outside the agents' writable scopes. A successful
verification establishes that the retained artifacts and receipts satisfy the
checks below. It does not establish that one agent architecture is better than
another; that requires the matched experiment and its separately declared
quality, timing and cost criteria.

## Dependencies

Use the repository's `npa/.venv/bin/python` with the normal NPA dependencies.
Those include PyYAML, Pillow, PyArrow and the pinned Rerun SDK. The USD reader
also needs `usd-core`; video decoding needs `ffmpeg` on `PATH` or the
`imageio-ffmpeg` fallback:

```bash
uv pip install --python npa/.venv/bin/python \
  'usd-core==26.8' 'imageio-ffmpeg==0.6.0'
```

The local decoder tests used USD 26.8, Rerun SDK 0.38.1, Pillow 12.3.0,
PyYAML 6.0.3 and imageio-ffmpeg 0.6.0. These describe the verification
environment, not a completed GPU experiment. Record decoder versions with a
trial. Rerun frame verification reproduces the writer's image resize and JPEG
encoding, so retain compatible Pillow/encoder versions on both sides.
Verification runs on CPU and does not need the proprietary NRE container.

## Python interface

The example module is imported by an operator adapter; it is not an installed
Workbench CLI command or SDK entrypoint. From an adjacent adapter:

```python
import json
from pathlib import Path

from nurec_verify import verify_run

result = verify_run(
    Path(downloaded_run_directory),
    scene=protocol["scene"],
    variant=protocol["variant"],
    novel_offsets=tuple(protocol["translation_meters"]),
    terminal_evidence=Path(terminal_receipt_path),
    render_evidence=Path(render_receipt_path),
)
print(json.dumps(result))
raise SystemExit(0 if result["passed"] else 1)
```

`novel_offsets` specifies exactly three nonzero-in-combination translation
components in meters. This verifier expects zero rig rotation. Although the
Python signature permits `render_evidence=None`, that produces a failed render
check; novelty is never inferred from filenames.

`legacy_rrd_review_evidence=Path(...)` is an explicit compatibility input for
older pinned viewers that do not embed their sampling and encoding settings.
It defaults to `None`; a recording without those settings still fails by
default. Supplying legacy evidence for a recording that already embeds settings
also fails. This explicit legacy path retains its existing novel-view image,
frame, run and metric comparisons and reports
`image_verification_scope: "legacy_novel_views_only"`. It does not establish
reconstruction modality coverage or the corrected endpoint-sampling contract.
Fresh recordings use the stronger default checks below; supplying producer
evidence cannot downgrade a recording with embedded settings.

The return value includes `passed`, `offsets_verified`, `checks` and `errors`.
Checks include an inventory of relative paths, byte sizes and SHA-256 hashes.
Missing evidence, missing decoder dependencies or failed decoding produce
failed checks instead of a successful partial result. The dependency imports
used to load this module itself still require a working NPA environment.

## Evidence supplied by the trusted adapter

The downloaded tree must contain native `ncore/manifest.json`,
`reconstruction/last.usdz`, `reconstruction/metrics.yaml`, novel-view images and
videos below `novel_views/`, `reports/sim2real.rrd`, and `reports/final.json`.
Download into a fresh directory; the verifier rejects symlinks. Preserve the
original objects rather than constructing substitutes after a failed run.

The terminal receipt is a JSON object with:

- `source: "workbench-live-status"`, `terminal: true`, `status: "succeeded"`,
  the exact `run_id`, and a timezone-aware `observed_at` timestamp.
- A `stages` array containing each of the six required names exactly once,
  each with `status: "succeeded"`.
- `submitted_spec_sha256`, and `raw_status_path` plus `raw_status_sha256`.
  The raw path is resolved relative to the terminal receipt's directory.

The render receipt uses `source: "workbench-render-receipt"`, the same run and
submitted-spec identity, and `receipt` containing the native Workbench render
result. It also provides `raw_receipt_path` and `raw_receipt_sha256`,
`artifact_sha256` for the trained USDZ, and `media_sha256` mapping every
novel-view image/video's run-relative path to its hash. The normalized native
render result must equal the retained raw render JSON exactly.

The adapter must obtain these records from actual Workbench execution, retain
the raw status/log sources, and associate them with the exact immutable
submission. A source label and hash are not authentication. In particular,
terminal status normalization is trusted: the current verifier checks the raw
status hash but does not itself interpret its provider-specific status schema.
The collector must verify that the raw status supports its normalized success,
stage names and run identity. Agents must not author those receipts.

For a legacy viewer, the additional review receipt contains:

- `source: "workbench-pinned-viewer"`, the exact `run_id` and
  `submitted_spec_sha256`, and `rrd_sha256` for the unchanged recording.
- `producer_image` with an immutable `@sha256:` reference, and
  `producer_source_sha256` for the verified image's writer source.
- `raw_producer_path` and `raw_producer_sha256`, binding the retained producer
  evidence relative to the receipt's directory. That JSON document uses
  `source: "verified-viewer-producer"` and repeats the same `producer_image`,
  `producer_source_sha256` and `settings` values.
- `settings` with `schema: "npa.nurec.rrd-review.v1"` and integer
  `max_frames_per_entity`, `max_frame_dim` and `jpeg_quality` values.

The trusted collector must establish these settings from the exact pinned
producer and its declared invocation/environment before comparing the output.
Do not search encoding settings until an image matches. Preserve OCI/source
verification records and distinguish observed environment from anything not
captured. The artifact reader checks the bindings and every expected encoded
image; it does not authenticate a registry or attest the producer from hashes
alone. Its result records `review_settings_source` and the external review
receipt's hash. Preserve any earlier failed verdict when adding compatibility
after a trial.

## What the checks establish

| Evidence | Check |
| --- | --- |
| Workflow | Terminal success for all six stages, timestamp format, submitted-spec digest and unchanged raw status bytes |
| Capture | Requested and observed scene/variant match, with camera and shard counts in the fetch manifest |
| USDZ | Safe unique archive members, uncompressed USDZ entries, CRC checks, a readable USD stage with prims, and a nonempty native checkpoint archive |
| Checkpoint | Parseable pickle opcodes with tensor references and at least one nonempty tensor storage; optional zero-length storage is counted, without unpickling or importing vendor model code |
| Metrics | Finite PSNR, SSIM and LPIPS in valid basic ranges, read from the native metrics document |
| Media | Every novel-view image loads; every video decodes fully, with nonzero frame counts |
| Rerun | Correct application/run identity; source-matched image bytes and original frame indices under complete novel-view and reconstruction modality/camera paths; sampling count and endpoints; embedded metrics matching the native metrics document |
| Render | Native receipt, submitted identity, exact model/media hashes, requested translation, zero rotation, and `--no-replicate-training-views` |
| Final report | Matching run/capability and success flags consistent with the independently decoded artifacts |

## Proof limits and comparison requirements

The checkpoint check is structural. It does not validate Gaussian parameter
semantics or prove that arbitrary model weights are numerically correct. The
native render receipt connects the retained USDZ and output hashes to the
actual render observation; that connection depends on the trusted collector.

The verifier reads NRE's validation metrics rather than independently
recomputing them. It checks validity, not an experiment-specific quality floor
or equivalence margin. Freeze those criteria before running either arm, retain
the raw metric distributions, and apply the same comparison to both arms.

Scene/variant checks do not prove dataset revision, complete source coverage,
training-split membership, sampled training frames, training budget, image
digest, accelerator type or GPU concurrency. Record those independently in the
protocol, effective NRE configuration, submitted plan and execution evidence.
The timestamp check requires a timezone but does not establish freshness; the
collector must bind observations and artifacts to this trial.

For recordings with embedded settings, the reader inventories source images
independently of the writer's grouping and sampling helpers. It preserves the
full relative parent path: for example,
`reconstruction/val/pred_rgb/camera/000043.png` must appear at
`/reconstruction/val/pred_rgb/camera` on frame `43`. Each entity/frame pair has
exactly one image. A positive cap selects `min(source_count, cap)` unique source
frames; caps of at least two retain both endpoints, cap one retains the first,
and nonpositive caps retain every frame. Interior interpolation is unrestricted.
These checks concern semantic frame indices, not physical RRD chunk order.

The result reports `image_verification_scope` and separate verified novel-view
and reconstruction row counts. Image comparison checks the declared resize and
JPEG encoding of each source image, including depth/opacity review images; it
does not establish numerical depth or physical opacity calibration. Legacy
producer-bound recordings retain their previously reviewed sampling behavior
and explicitly narrower novel-view scope.

Complete video decoding proves media integrity, not visual
quality or a pixel-by-pixel equivalence between the MP4 and every source PNG.
Timing, human interventions, cloud allocation, billing and model token costs
are outside this verifier. Count failures and recoveries in the comparison.

Raw records stay private. Structured output avoids absolute artifact paths and
raw provider diagnostics, but relative filenames can still identify customer
data; review and sanitize any report before publishing it.

The offline tests use synthetic fixture artifacts with real USD, image, video
and Rerun decoders, including corrupt-artifact negative cases:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/agent_eval/test_specialist_nurec_verification.py -q
```

Passing those tests validates the reader's checks. It is not evidence that a
NuRec reconstruction or an agent comparison has completed on GPUs.
