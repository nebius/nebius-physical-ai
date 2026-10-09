# Cosmos Evaluator report inspection

`npa workbench cosmos-evaluator report` reads one existing
`npa.cosmos_evaluator.report.v1` JSON artifact. It does not run inference,
submit work, mutate the artifact, list an object prefix, or select a latest
report.

Use an exact local file or exact S3 object:

```bash
npa workbench cosmos-evaluator report \
  --input-path ./cosmos_evaluator.json

npa workbench cosmos-evaluator report \
  --input-path s3://<bucket>/<exact-report-key> \
  --output-format json
```

The default table lists the reported gate score and disposition for each
variant, followed by attribute, hallucination, temporal, and appearance
diagnostic state. JSON emits the same bounded projection. It deliberately
excludes source URIs, prompts, and arbitrary report metadata.

The reported score is an evaluator gate score, not calibrated confidence. This
reader does not establish semantic material validation or multi-view validation;
multi-view assessment is reported as `not_evaluated` when present in an input
artifact.

## Diagnostic states and gate semantics

`required` diagnostics contributed to the evaluator gate for that variant;
`advisory` diagnostics did not. `unverified` means an older artifact lacks the
enforcement facts needed to classify the diagnostic. A missing companion field
is shown as `not_evaluated`; a report is only fully graded when its required and
enforcement evidence is complete.

The inspector preserves a producer's reported quality outcome. A completed
failed-quality report is a successful read and exits zero. Malformed or
contradictory artifacts, including non-finite scores, non-boolean dispositions,
duplicate variant IDs, count mismatches, or a pass contradicted by required
diagnostics, fail clearly. Missing evidence is never averaged into a new passing
score.

## SDK

```python
from npa.sdk.workbench import cosmos_evaluator

summary = cosmos_evaluator.report(
    input_path="s3://<bucket>/<exact-report-key>"
)
```

The SDK reads the exact object with the scoped storage client. It does not scan
the bucket and does not infer a report location.
