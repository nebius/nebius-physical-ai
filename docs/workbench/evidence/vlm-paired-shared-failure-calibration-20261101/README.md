# Paired-judge shared-failure calibration

This sanitized index supports one narrow policy decision: do not combine
`openbmb/MiniCPM-V-4_5` and `google/gemma-3-27b-it` as the two judges in one
`compare-judges` request.

The retained private record is identified as
`vlm-paired-shared-failure-calibration-20261101`. It contains the frozen inputs,
requests, responses, and per-family outcomes. Those adversarial inputs are not
published here. This repository record exposes the model identities, aggregate
matrix, collection date, legacy response schema, and limitations needed to
audit or retire the deny rule without disclosing reusable attack material.

## Sanitized result

The retained profile matrix used six synthetic in-image-instruction control
families with two repetitions per model (12 calls per model):

| Full model ID | False-pass calls | Families with a false pass |
| --- | ---: | ---: |
| `MiniMaxAI/MiniMax-M3` | 2/12 | 1/6 |
| `openbmb/MiniCPM-V-4_5` | 6/12 | 3/6 |
| `google/gemma-3-27b-it` | 10/12 | 5/6 |

MiniCPM and Gemma overlapped on three of the six families, with six joint
false-pass repetitions. Each pair containing MiniMax had no observed shared
false-pass family in this retained matrix. This is motivation for one exact
pair exclusion, not qualification of the other pairs.

See [`evidence.json`](evidence.json) for the machine-readable scope and counts.

## Limitations

- The retained calls used the legacy `passed`/`reason` schema. The current path
  requires bare `success`/`rationale` JSON and complete `finish_reason=stop`.
- No provider call was replayed for this change.
- Provider hardware and deployment stability were not observed.
- The aggregate does not establish safety, robustness to unseen attacks, broad
  model quality, or suitability for a robotics workload.
- The private record must be consulted before changing or removing the rule;
  this sanitized index is intentionally insufficient to reconstruct the inputs.
