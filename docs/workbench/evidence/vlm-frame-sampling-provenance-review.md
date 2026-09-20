# VLM frame-sampling provenance: reviewer-safe evidence

This report covers the frame-sampling provenance change at
`3d702087c`. Raw provider responses, provider request identifiers, source paths,
and operational logs remain in access-controlled evidence. No source image or
video is published here.

## Hardware and workload

| Field | Result |
| --- | --- |
| Change under test | Evidence schema v2 retains source kind, zero-based source index, source count, source video timestamp when known, requested strategy, frame limit, and metadata-completeness flags |
| Client execution | Local CPU |
| Inference execution | Hosted vision API; provider hardware unknown; no local-GPU claim |
| Real workload | One real three-image request to `MiniMaxAI/MiniMax-M3` through `npa workbench vlm-eval run` |
| Provider result | HTTP 200, exact returned model, `finish_reason: stop` |
| Usage | 2,023 prompt tokens, 130 completion tokens, 2,153 total tokens |
| Latency | 3.128406 seconds |
| Independent review | Implementation at `ba77a34f7` is `review-ready`; its patch is byte-identical to rebased commit `3d702087c` |
| Remaining gate | Draft PR and remote exact-head CI |

Smoke checks are reported separately from workload proof. The exact rebased
head passed 132 focused tests with 3 skipped, 114 smoke tests, 3,866 guardrail
tests, and repository lint/format checks.

## Objective controls

All controls used actual decoded image bytes:

| Input | Available frames | Selected source indices | Selected timestamps | Result |
| --- | ---: | --- | --- | --- |
| Ordered image sequence | 5 | `[0, 2, 4]` | not applicable | pass |
| NumPy RGB episode | 6 | `[0, 2, 5]` | not applicable | pass |
| Synthetic 2 fps MP4 | 6 | `[0, 2, 5]` | `[0.0, 1.0, 2.5]` seconds | pass |
| Unknown-count video fallback | unknown | null | null | pass: metadata remains explicitly incomplete |

Failure-path regressions also prove that filename text cannot inject a timestamp,
incomplete timestamp metadata fails closed to null, and extracted frame numbers
remain numerically ordered beyond 999.

## Real hosted evidence

The real request selected source indices `[0, 2, 4]` from five ordered Open3D
views. The request manifest records `coverage_complete: true`, meaning every
submitted frame has source-index metadata; it does not mean all source frames
were submitted.

A producer-side recomputation report found zero mismatches in the selected
normalized frame bytes and metadata, prompt, rubric, request manifest, and raw
response. An independent reviewer separately repeated those checks from the
source inputs and raw response, then verified that the reviewed patch and the
rebased patch are byte-identical. Private evidence identities:

- frozen plan:
  `d97685e83237ac720711650e96ff612ba54054303daaeb8c69c92f0accf5af30`;
- objective-control report:
  `c43e69af9d72f17e02d6d93cdef5504fd6fa8b33b546d0f11c4bf682632b5077`;
- hosted result:
  `033955ad35132eb942b2b768f0a96529ab9580235cb9ab5d27e18d0f2d634e47`;
- producer recomputation report:
  `dcd442965340cb6b1adaa0ef6b2e9422383a08e8f001cd74b53e2e4b66549443`.

The model correctly distinguished two scan-plus-surface views from one
samples-only view. It also read a visible `1.0 m` scale label as `10 m`. That
failure remains in the retained response: the model's score does not establish
scale, geometry accuracy, or visual acceptance.

## Reproduction boundary

The public command surface can be exercised with private inputs and a
credential supplied only through the environment:

```bash
export NEBIUS_TOKEN_FACTORY_KEY="<token-factory-key>"
npa workbench vlm-eval run \
  --input-path <ordered-private-frames> \
  --output-path <private-result.json> \
  --backend api \
  --model MiniMaxAI/MiniMax-M3 \
  --frame-selection keyframes \
  --max-frames 3 \
  --task "<task>" \
  --rubric "<frozen-rubric>" \
  --success-threshold 0.8 \
  --output json
```

This is an exerciser, not a bit-for-bit reproducer. Exact media and raw
responses remain private, and hosted outputs are not deterministic. Sampling
traceability does not establish temporal causality, task completion, geometry,
collision suitability, physical correctness, or robot safety.
