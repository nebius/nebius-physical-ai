# Scheduled Token Factory migration contracts

`.github/workflows/token-factory-live.yml` runs all three migration suites with
real provider credentials at **06:17 UTC daily**, on manual dispatch, and on the
explicitly reviewed migration continuation branch. GitHub activates the cron
only when this workflow reaches `main`; a branch push exercises the same job
before merge. This is separate from secret-free ordinary/fork PR CI.

The `token-factory-live` GitHub environment must restrict deployments to `main`
and the explicitly reviewed continuation branch in the workflow. Store the
authorized test key as the environment secret `NEBIUS_TOKEN_FACTORY_KEY`.
Neither `pull_request` nor `pull_request_target` can trigger this workflow, and
the job independently checks its repository and branch. Preserve existing
environment approvals and branch protections; a blocked job is not successful
verification. The checkout does not persist GitHub credentials.

Optional environment configuration:

| Setting | Purpose |
| --- | --- |
| Secret `NEBIUS_TOKEN_FACTORY_BASE_URL` | Authorized endpoint override; empty uses the public endpoint |
| Variable `NPA_TF_RECHECK_SCOPE` | Scope label, retained only as a hash in the published receipt |
| Variable `NPA_TF_RECHECK_REQUIRED_MODELS` | Comma-separated additional models that must pass actual inference |
| Variable `NPA_TF_RECHECK_JSON_BASELINE` | Reviewed structured-output expectation: `malformed_json` (initial), `schema_invalid`, or `healthy` |

The migration's configured text, reasoning and vision defaults always remain
required. Catalog membership alone cannot pass: actual requests must return
the expected model identity, a complete visible answer, a request identity and
positive token accounting. Additional model checks do not replace or silently
reroute any default. Existing explicit model, endpoint and SDK argument
precedence is unchanged. Results cover only the configured account and endpoint;
repeat separately for other authorized scopes without claiming global coverage.

The contract probes send both enabled and disabled thinking controls for
Lightning (`enable_thinking`) and MiniMax (`thinking_mode`). Disabled requests
must have no reasoning trace/tokens; enabled requests must expose reasoning.
All modes must answer the synthetic question correctly with complete output.
The three live suites additionally exercise synthetic inventory generation,
image captioning, reasoning, visual judging, hosted temporal evaluation and the
rendered agent HTTP backend. The HTTP suite is independent of deployed browser
UI proof and does not claim a deployed UI.

MiniMax `json_object` and `json_schema` are checked against the original response
bytes with strict JSON parsing and exact synthetic schema/value checks. A
healthy response changes the recorded behavior and fails an initial
`malformed_json` baseline, making a vendor fix observable. Review the provider
change and the workaround before explicitly updating that baseline. Prompted
JSON must remain healthy on every run. The checks never remove malformed
prefixes, repair scores, or automatically rewrite an expectation.

The entrypoint is reproducible in protected operator automation as well:

```bash
npa/.venv/bin/python npa/scripts/token_factory_live_recheck.py \
  --evidence-dir "$NPA_PRIVATE_EVIDENCE_DIR"
```

Load the authorized key into the process environment without printing it. The
runner requires an explicit environment key; it fails before invoking pytest
if that key is absent. Direct pytest jobs can enforce the same prerequisite
with `--require-token-factory-live`. Ordinary developer invocations retain their
credential-free skip behavior. The runner requires all three suites, nonzero
collection, equality of collected/executed/passed counts, and zero failures,
skips or collection errors. A skipped/xfail test cannot create green proof.

Every invocation writes an exclusive `0600` `receipt.json` with source commit
and file hashes, execution location, scope-label and endpoint hashes, test
outcomes/counts, provider contract observations, model identity hashes, usage,
and visible-output hashes. Raw provider text, pytest tracebacks, credentials,
private endpoints and image bytes are excluded from the published receipt.
Custom model IDs are hashed; only the two canonical public migration model names
are retained literally. The built-in confidentiality guard checks the receipt
before it is written.
The hosted job also runs the entrypoint with an explicitly empty key and
requires a failed receipt with zero collected/executed tests before the
credentialed run. GitHub uploads only those two sanitized receipts and puts
the credentialed result in the job summary even on failure. A local invocation is labeled `local-manual`; an operator scheduler
can set `NPA_TF_RECHECK_EXECUTION=operator-automation`. Neither is represented as
hosted CI. Use a new evidence directory for every invocation; finalized receipts
cannot be overwritten.

The model availability gate, default drift assertion, and actual served-model
reporting build on the relevant ideas raised by Jonathan Lwowski in
[PR #389](https://github.com/nebius/nebius-physical-ai/pull/389). This recheck
implementation was written for this migration; no code was copied from that PR.
