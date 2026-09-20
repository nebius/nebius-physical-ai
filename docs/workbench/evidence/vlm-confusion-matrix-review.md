# VLM benchmark confusion-matrix review

## Result

`vlm-eval benchmark` now emits report schema
`npa_vlm_eval_benchmark_report_v2`. Every ranked
model/rubric/threshold configuration carries an explicit actual-by-predicted
2x2 matrix, false-positive and false-negative rates, and ordered IDs for both
failure classes. Each ID resolves to one complete per-item result because
duplicate dataset IDs are rejected before evaluation.

The tested implementation source is
`a098219c7443a85a28aa6f1126d6290e44ec3c75`.

## Retained hosted evidence

No new provider call was made for this change. The new implementation
recomputed two previously retained, independently reviewed labeled reports:

- MiniCPM three-case report:
  `51f66030caacf9a5f2e135d69254b089fa45818ffe21554d9742fd70455f6c51`
- MiniMax/Gemma six-result report:
  `1f7d37bc71a0a11a29fcc3fac48cd3c6048e0e1906427d0cae76103b3f9e688c`

All configurations used backend `api`, rubric `open3d-identity-v1`, threshold
`0.8`, final-frame selection, and one frame.

| Hosted judge | Labeled cases | TP | TN | FP | FN | FPR | FNR | Visible failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `MiniMaxAI/MiniMax-M3` | 3 | 1 | 2 | 0 | 0 | 0.0 | 0.0 | none observed |
| `google/gemma-3-27b-it` | 3 | 1 | 2 | 0 | 0 | 0.0 | 0.0 | none observed |
| `openbmb/MiniCPM-V-4_5` | 3 | 1 | 1 | 1 | 0 | 0.5 | 0.0 | FP: `mismatched-run-negative` |

The retained MiniCPM false positive remains visible in both its best and ranked
configuration representation. Recomputed counts match every legacy scalar
count. The sanitized derived record is independently reviewed by hash
`fd4bb78f38dd81aeaed6370db0e8aab4943f6f0165779c8fe29c6a2426f5fffd`;
raw responses, rationales, request IDs, request manifests, and local paths are
excluded from this public report.

## Hardware and workload applicability

| Component | Placement | Evidence boundary |
| --- | --- | --- |
| Hosted judges | Nebius Token Factory hosted API; provider hardware unavailable | Existing image requests only; no provider hardware identity was retained |
| B052 recomputation | CPU-only NPA client | No cluster, GPU, or new hosted request |
| Applicable workload | Labeled visual-judge calibration | Reports classification errors; does not evaluate policy physics |

## Compatibility and validation

- New product-generated reports explicitly identify schema v2.
- A direct constructor using the historical field set defaults to v1.
- Historical unversioned JSON remains readable as legacy v1 and can be
  recomputed from its complete item results.
- One canonical ordered classification pass produces all legacy counts, matrix
  cells, rates, and failure IDs.
- Asymmetric regression controls verify every cell mapping, both rate
  denominators, exact failure membership/order, null zero-denominator behavior,
  and all serialized ranked configurations.

## Limitations

- Three labeled cases per model do not estimate an operational error rate.
- These controls test whether an image visibly contains the expected Open3D
  result class, not geometry accuracy, task completion, or physical validity.
- A zero observed error count does not qualify a model or threshold.
- VLM output does not certify physical correctness or robot safety.
