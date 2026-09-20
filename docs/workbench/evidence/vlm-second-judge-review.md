# Paired VLM disagreement review

This report tracks the audit-only `vlm-eval compare-judges` capability. It is
not a model-qualification record and must not be used as a robot-safety,
physical-correctness, or critical-defect certificate.

## Hardware and workload applicability

| Item | Evidence |
| --- | --- |
| Source commit | `960ba079e39f694208ab51a78010e1d31c7abe6a` |
| Container image | Not applicable; this change uses the installed NPA client |
| CPU | Real frame normalization, request construction, parsing, and report validation |
| GPU | Not applicable; no local model is served |
| Hosted API | Six real image attempts: three fixed primary/secondary pairs whose payloads differed only in `model` |
| Smoke/unit result | 229 affected checks passed and 2 skipped on Python 3.12 |
| Guardrails | 3,881 passed |
| Full workload result | All six responses were complete; both judges matched all three frozen labels; observed escalation was 0/3 |
| Objective controls | Same-model pre-transport rejection, exact request equivalence, frame mismatch rejection, provider-error retention, fail-closed disagreement |
| Visual calibration | Three independently labeled controls; n=3 is not an operational-rate estimate |
| Independent review | Two changes-required source/harness rounds were resolved; a separate pixel/evidence reviewer accepted the audit-only mechanism |

## Contract

The command selects and normalizes frames once, creates one prompt, and sends
two request objects whose canonical JSON differs only in `model`. The distinct
`vlm_judge_disagreement.json` artifact retains both complete outcomes or typed
errors. It never emits a mean score.

Disagreement or either judge error produces `passed=false` and
`escalation_required=true`. Agreement can describe the two returned verdicts,
but the report remains `deployment_status=audit_only`.

## Hosted control outcome

| Control | Frozen label | MiniMax-M3 | MiniCPM-V-4_5 | Result |
| --- | --- | --- | --- | --- |
| Blank reconstruction evidence | fail | `0.0`, fail, label match | `0.0`, fail, label match | agreement; no escalation |
| Upright bottle and bowl | pass | `1.0`, pass, label match | `1.0`, pass, label match | agreement; no escalation |
| Offset reconstruction fragment | fail | `0.2`, fail, label match | `0.0`, fail, label match | agreement; no escalation |

All six provider responses had HTTP 200, `finish_reason=stop`, usage, unique
request identity, exact returned-model identity, and retained raw-response
hashes. Independent review recomputed request-file, canonical-request,
report-transport, frame, prompt, rubric, manifest, response, state, and summary
hashes without mismatch. Forced offline disagreement and provider-error
controls separately proved that those paths fail closed and require escalation.

The predeclared directional expectation was at least 2/3 escalations; the
observed 0/3 is a descriptive surprise for these controls only. It is not an
operational disagreement-rate estimate or evidence that either model is
qualified. In particular, prior instruction-resistance, system-role, and
frame-citation controls still disqualify MiniCPM-V-4_5 as an acceptance judge.

## Frozen controls

Three one-frame Physical AI controls were labeled before provider calls:

- blank reconstruction evidence:
  `5dbe4fd672a7f1c5b30aa0a6be882f3c8450f47cfc9f24d7018d0f14f3d77162`;
- upright green bottle and blue bowl:
  `8ddbf811d9c99717e199dac0710d51d7884195802aa7ac0bc3efcd1df3b11356`;
- visibly offset reconstruction fragment:
  `7114567bf3b82e7ac2b21a4a7b55890088d42b61bd939785ecd6fccb218e1850`.

All labels are high-confidence visible-appearance answers. They do not assert
hidden simulator state or physical geometry. Original pixels are not published
here because this review does not establish redistribution permission.

## Limitations

- Both default judges remain unqualified acceptance judges.
- Prior controls found MiniCPM-V-4_5 unreliable; its divergence can confound
  disagreement.
- Fixed call order has no order-effect control.
- Shared request fields omit model-specific tuning and may reduce individual
  model performance.
- Instructions embedded in pixels remain an unresolved input-integrity risk.
- Visual agreement cannot prove a critical defect absent, physical correctness,
  or robot safety.
