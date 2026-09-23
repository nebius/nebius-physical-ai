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

## What the checks establish

| Evidence | Check |
| --- | --- |
| Workflow | Terminal success for all six stages, timestamp format, submitted-spec digest and unchanged raw status bytes |
| Capture | Requested and observed scene/variant match, with camera and shard counts in the fetch manifest |
| USDZ | Safe unique archive members, uncompressed USDZ entries, CRC checks, a readable USD stage with prims, and a nonempty native checkpoint archive |
| Checkpoint | Parseable pickle opcodes with tensor references and nonempty tensor-storage members, without unpickling or importing vendor model code |
| Metrics | Finite PSNR, SSIM and LPIPS in valid basic ranges, read from the native metrics document |
| Media | Every novel-view image loads; every video decodes fully, with nonzero frame counts |
| Rerun | Correct application/run identity, embedded novel-view image bytes matching the run's sampled images, and embedded metrics matching the native metrics document |
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

Rerun verification compares the frames selected by the writer's recorded
sampling settings. Complete video decoding proves media integrity, not visual
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
