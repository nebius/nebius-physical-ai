# Cosmos Evaluator report inspection

[Workbench docs](README.md) · [Manual operations](guides/manual-workflow-operations.md) · [Dataset batches](guides/paidf-dataset-batches.md)

Read an existing `npa.cosmos_evaluator.report.v1` artifact to see each variant's
score, reported pass/fail and attribute, hallucination, temporal and appearance
diagnostics. Inspection does not run inference or change the report.

Start with the shared [Insights report command](insights-reports.md), which also
supports dataset validation reports. This page explains Cosmos-specific evidence.

## Inspect a report

Choose the exact local file or S3 object listed in your run's artifacts:

```bash
npa workbench insights report \
  --input-path ./cosmos_evaluator.json

npa workbench insights report \
  --input-path 's3://<bucket>/<exact-report-key>' \
  --output-format json
```

The default is a readable table. Use `--output-format json` for scripts.
Both formats omit source URIs, prompts and arbitrary metadata. The command
reads only the selected report; it does not scan a prefix or select the latest.
The shared response puts Cosmos-specific details under `summary`; the gate and
evidence state are also available at the top level. The original
`cosmos-evaluator report` command retains its original Cosmos-only response.

## Interpret the result

Each diagnostic states its role in the reported gate:

| Role | Meaning |
| --- | --- |
| `required` | Contributed to the variant's pass/fail decision. |
| `advisory` | Did not contribute to that decision. |
| `unverified` | The artifact lacks facts needed to determine the role. |

Check `evaluation_state` separately from the reported pass/fail:

| State | Meaning |
| --- | --- |
| `graded` | Required diagnostic evidence and enforcement facts are complete; check the reported gate to see whether quality passed. |
| `incomplete` | Required evidence or enforcement facts are missing; inspect the affected diagnostics. |
| `ungraded` | The report has no variants to grade. |

A missing diagnostic is `not_evaluated`; unavailable, skipped and error states
remain explicit. Missing evidence is never averaged into a new passing score.

## Exit status and limits

A valid report with failed quality still exits zero: the read succeeded.
Scripts must check `reported_gate.passed` and the evidence state. An unreadable,
malformed or contradictory report exits nonzero, including a pass contradicted
by required diagnostics, duplicate IDs or inconsistent report counts.

The score is a gate score, not calibrated confidence. This reader does not
establish semantic material preservation or multiview consistency; multiview
assessment remains `not_evaluated` even when the input claims it.

## Python SDK

```python
from npa.sdk.workbench import insights

result = insights.report(
    input_path="s3://<bucket>/<exact-report-key>"
)
summary = result["summary"]
```

The SDK returns the same diagnostic projection and reads only the selected object.
