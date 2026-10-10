# Inspect quality reports

[Workbench docs](README.md) · [Manual operations](guides/manual-workflow-operations.md)

Use one command to read an existing report from a supported solution. Its schema
selects the reader; inspection never runs evaluation or changes the artifact.

```bash
npa workbench insights report --input-path ./selected-report.json

npa workbench insights report \
  --input-path 's3://<bucket>/<exact-report-key>' --output-format json
```

Choose the exact object listed in your run's artifacts. The reader does not list
a prefix, follow links in the report or choose a "latest" file. Text is the
default; use JSON for scripts. Both formats exclude source locations, prompts,
dataset identities, free-form failure text and arbitrary metadata.

## Supported reports

| Report schema | What inspection shows |
| --- | --- |
| `npa.cosmos_evaluator.report.v1` | Producer's score, gate and per-variant attribute, hallucination, temporal and appearance diagnostics. See [Cosmos evidence and gate roles](cosmos-evaluator-report.md). |
| `npa.dataset.validation_report.v1` | Producer's gate, completeness/corruption thresholds, record counts, quality rates and failed-check count. Requires the evidence emitted by the current dataset validator. |

Other formats are rejected explicitly. Cross-tool metrics aggregation remains
available through `insights ingest-run`, `query`, `compare` and `dashboard`;
their supported ingestion formats are a separate contract. This inspection
command does not add records to an Insights store.

## Interpret the result

JSON returns a `npa.insights.report.v1` envelope:

| Field | Meaning |
| --- | --- |
| `source_schema`, `tool` | Identifies the supported producer and format. |
| `reported_gate` | Preserves the producer's decision and available score/thresholds. |
| `evaluation_state`, `evidence_complete` | States whether required evidence is complete; check independently from pass/fail. |
| `summary` | Validated diagnostics specific to that report format. |

Scores retain their original meaning. Dataset corruption rates and evaluator
scores are different measurements; inspection does not normalize them into a
shared confidence score. Missing evidence never becomes a passing score.

A valid failed-quality report still exits zero because inspection succeeded.
Scripts must check `reported_gate.passed` and `evidence_complete`. Unreadable,
unsupported, malformed or contradictory reports exit nonzero. Cosmos reports
can expose incomplete evidence; dataset reports missing required evidence are
rejected.

## SDK and service

```python
from npa.sdk.workbench import insights

result = insights.report(input_path="s3://<bucket>/<exact-report-key>")
```

Use `--service --endpoint '<insights-service-url>'` to inspect through the
deployed Insights service, or pass `service=True, endpoint=...` in the SDK.
The service exposes `POST /report` with `{"input_path": "..."}` and uses its
existing bearer-token authentication and configured local/S3 storage roots.
Local paths refer to the service host. Set `INSIGHTS_TOKEN` for authentication;
`--token-env` selects a different token variable. Embedded CLI/SDK reads use
the operator's local storage credentials.

## Cosmos compatibility

`npa workbench cosmos-evaluator report` and `cosmos_evaluator.report(...)`
remain available with their original Cosmos-only projection. The shared reader
uses the same Cosmos validator; its `summary` contains that projection. Use
`insights report` in new scripts that handle reports from several solutions.
