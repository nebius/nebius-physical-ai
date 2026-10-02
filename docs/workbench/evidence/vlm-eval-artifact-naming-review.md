# VLM eval artifact naming: reviewer-safe evidence

This report covers implementation head `2b575e364`. Raw media, provider responses and
request identifiers, source paths, and operational logs remain in
access-controlled evidence. No source image or video is published here.

## Hardware and workload

| Field | Result |
| --- | --- |
| Change under test | New VLM directory/object-prefix writes and shipped workflow declarations use `vlm_eval.json`; historical `vlm_eval_stub.json` bundles remain read-only compatible |
| Live-tested writer | `5fba6d0e8`; later candidate commits change readers, gate validation, tests, and historical wording, not the writer/request path |
| Client execution | Local CPU for frame selection, normalization, hashing, manifest construction, and artifact writing |
| Inference execution | Hosted vision API; provider hardware unknown; no local-GPU claim |
| Real workload | One real three-image request to `MiniMaxAI/MiniMax-M3` through `npa workbench vlm-eval run` |
| Provider result | HTTP 200, exact returned model, request identity retained, `finish_reason: stop` |
| Usage | 2,023 prompt tokens, 130 completion tokens, 2,153 total tokens |
| Latency | 2.962663 seconds |
| Smoke (not workload proof) | 114 onboarding smoke tests passed |
| Independent review | Two implementation rounds, the B108 regression, and the SDK export repair are `review-ready` with no findings |
| Remaining gate | Draft PR and remote exact-current-head CI |

The current implementation passed 24,136 hermetic Python 3.12 tests with 154
skipped and 1 xpassed after installing the same CPU runtime extras and exporting
the same virtualenv path used by CI. It also passed 3,875 repository guardrails,
114 onboarding smoke tests, all 95 data-factory stage tests, the 37 focused
SDK/VLM/gate tests with 1 live-GPU test skipped, repository lint/format checks,
119 skill checks, CLI documentation drift checks, and validation of all seven
changed workflow specs.

## Objective and failure controls

| Control | Result |
| --- | --- |
| Fixture directory write | Creates `vlm_eval.json`; payload says `backend: stub`; no legacy sibling |
| Explicit local `.json` path | Preserved exactly |
| Object-prefix write | Uploads the canonical filename |
| Rollout-set write | Every per-rollout result uses the canonical filename; aggregate report is unchanged |
| Shipped workflow declarations | Nine VLM stages resolve to the producer's canonical result URI |
| Historical bundle | Legacy-only result remains readable |
| Canonical plus legacy | Canonical result wins |
| Malformed canonical plus valid legacy | Fails closed; stale legacy data is not substituted |
| Fail-closed mutation (`break` → `continue`) | The B108 regression fails with `promote_checkpoint`, proving the gate is load-bearing |
| Gate status vocabulary | VLM `passed` and Cosmos `completed` remain producer-specific; one producer cannot promote with the other's status |

No compatibility alias is written. The old filename is a reader fallback only,
so a new real result cannot be mistaken for a fixture from its name.

## Real hosted evidence

The CLI `written_uri`, serialized `result_uri`, and actual file all ended in
`vlm_eval.json`; no `vlm_eval_stub.json` sibling existed. Independent
recomputation matched the selected normalized frame bytes and metadata, prompt,
rubric, request manifest, raw response, returned model, request identity, usage,
and finish reason.

Private evidence identities:

- frozen plan:
  `11a661115b572b6aa8e9ade5275790aeee452abf7054cb3ae4de9d055c00babb`;
- hosted result:
  `56f45e6ac4ecde037ed453a9f751a5b7fa4a1504f24661c8d73d87fbe48c79eb`;
- CLI output:
  `955db0150abd3f910f13648c4313cf9bffce281d4b85bd7ecb6af6386e38c01a`;
- producer recomputation:
  `5a0247c455cc0b7a7141cf303c3697163b84e9338cd44064da8c44d2f9672ddc`.

The model distinguished two scan-plus-surface views from one samples-only view,
but read a visible `1.0 m` label as `10 m`. That failure remains retained. The
score is not scale, geometry, visual-acceptance, physical-correctness, or safety
evidence.

## Independent review history

The first review found a pre-existing VLM/Cosmos gate-status mismatch, stale
legacy fallback after malformed canonical reports, incomplete path-ledger
scope, stale historical wording, and missing write/loop/toolRef regressions.
After those were fixed, a second review found that the first status fix accepted
a union of both producers' vocabularies. Producer-specific validation and
negative controls closed it. Final review reported no findings.

A later mutation review found that the grade gate's malformed-canonical control
used a stale legacy fixture that could not actually promote, so changing the
fail-closed `break` to `continue` survived the suite. The corrected B108 control
uses a valid stale passing VLM result and catches that mutation: correct code
loops back with no authoritative report hash, while the mutation promotes at
`0.95`. Independent review of exact commit `cc5abb9e5` reported no findings.

The expanded suite then found that the two filename strings had been added to
the module's callable-only SDK export list. Commit `2b575e364` keeps both direct
imports available to internal readers while removing them from `__all__`; all
28 exported SDK entries are callable. Independent review reported no findings.

## Reproduction boundary

The public command surface can be exercised with private input and a credential
supplied only through the environment:

```bash
export NEBIUS_TOKEN_FACTORY_KEY="<token-factory-key>"
npa workbench vlm-eval run \
  --input-path <private-image-directory> \
  --output-path <private-result-directory> \
  --backend api \
  --model MiniMaxAI/MiniMax-M3 \
  --frame-selection keyframes \
  --max-frames 3 \
  --task "<task>" \
  --rubric "<frozen-rubric>" \
  --success-threshold 0.8 \
  --output json
```

This is an exerciser, not a bit-for-bit reproducer. Hosted outputs are not
deterministic, and exact source media remains private.
