# Audit-only rich visual reviews

`npa workbench vlm-eval review-visual` writes a private `vlm_visual_review.json`
with five separate dimensions: visible task evidence, artifact fidelity,
reviewability, subjective impressiveness, and a downstream-usefulness hypothesis.
The record cannot change the normalized completion score or pipeline gate.

Use an exact hosted vision model that your Token Factory key can reach. Verify
the key with `npa workbench token-factory verify` and inspect
`npa workbench token-factory models` before running. Credentials come from the
named environment variable or the existing private credential resolver.

```bash
npa workbench vlm-eval review-visual \
  --input-path 's3://<bucket>/rollout/' \
  --output-path 's3://<bucket>/private-review/' \
  --model MiniMaxAI/MiniMax-M3 \
  --task 'Describe the visible cube placement and evidence limits.' \
  --api-key-env NEBIUS_TOKEN_FACTORY_KEY \
  --frame-selection sequence --max-frames 4 \
  --output-format json
```

The SDK `npa.sdk.workbench.vlm_eval.review_visual` accepts the same options and
also supports local development inputs and outputs. A local output prefix must
be private; files are created with mode `0600` and directories with mode `0700`.
An exact output filename must be `vlm_visual_review.json`. CLI stdout contains
only the schema, status, escalation flag, attempt count, and model.

`--baseline-path` enables comparison. Both sources are sampled independently,
sent once in each neutral A/B order, and mapped back only in the private report.
Private paths, source roles, objective references, and matched-view metadata do
not enter provider-facing text. Disagreement, low confidence, and malformed
responses remain visible and require escalation. No averaging creates agreement.

`--objective-evidence-path` accepts private JSON reference locators;
`--matched-view-map-path` accepts private view metadata. Both remain unverified
producer metadata. Usefulness always remains `hypothesis_only` and requires a
future measured consumer test. Frame citations prove membership in the submitted
sample, not the truth of a model statement.

The evidence journal records the request and transport-start marker before
inference, and the raw response before parsing. A consumed output identity is
never replayed after a crash. Retain its failure evidence and choose a new output
identity only for a separately justified protocol; do not retry for a preferred
model answer. Evidence-retention failure stops execution before another call.

The historical [hosted negative result](evidence/vlm-rich-visual-review.md)
remains applicable to its original source and protocol. Its six attempts did
not yield a contract-valid review. Rich judgments do not qualify a model,
establish task completion, prove physical correctness or usefulness, or certify
robot safety. Instructions embedded in images remain an input-integrity risk.
# Exact failure-byte retention

The private journal writes `response-bytes-<ordinal>.json` before reading the
HTTP response's decoded text or parsing JSON. It records reversible base64,
the SHA256 and size of exact `response.content`, HTTP status, request identity
when returned, and observed latency. Invalid UTF-8 and declared non-UTF8 error
bodies therefore remain distinguishable even when decoded text is lossy.
Byte-journal write failure stops the attempt without replaying it. The existing
decoded-text response journal and parser interface remain compatible.

Earlier observations that retained decoded text only are not retroactively
wire-byte evidence. Keep their original artifacts and label their hashes as
decoded-response-text hashes.
