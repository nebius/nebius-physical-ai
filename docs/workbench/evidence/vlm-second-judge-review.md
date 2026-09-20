# Paired VLM disagreement review

This report tracks the audit-only `vlm-eval compare-judges` capability. It is
not a model-qualification record and must not be used as a robot-safety,
physical-correctness, or critical-defect certificate.

## Hardware and workload applicability

| Item | Evidence |
| --- | --- |
| Source commit | Pending the reviewed source freeze |
| Container image | Not applicable; this change uses the installed NPA client |
| CPU | Real frame normalization, request construction, parsing, and report validation |
| GPU | Not applicable; no local model is served |
| Hosted API | Required: two real image requests over identical payloads except `model` |
| Smoke/unit result | 254 focused checks passed on Python 3.12 |
| Full workload result | Pending independent source/harness approval |
| Objective controls | Same-model pre-transport rejection, exact request equivalence, frame mismatch rejection, provider-error retention, fail-closed disagreement |
| Visual calibration | Three independently labeled controls; n=3 is not an operational-rate estimate |
| Independent review | Initial source review found actionable issues; corrected candidate re-review pending |

## Contract

The command selects and normalizes frames once, creates one prompt, and sends
two request objects whose canonical JSON differs only in `model`. The distinct
`vlm_judge_disagreement.json` artifact retains both complete outcomes or typed
errors. It never emits a mean score.

Disagreement or either judge error produces `passed=false` and
`escalation_required=true`. Agreement can describe the two returned verdicts,
but the report remains `deployment_status=audit_only`.

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
