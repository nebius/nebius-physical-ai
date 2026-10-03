# Real hosted VLM audit contracts

The shared runner selects exactly one supported audit kind per invocation.
Generated controls require an explicit selector, matching schema, exact ordered
control names and all three executed cases. Mixed or unknown families fail closed.

## Blinded preference controls

`compare-preference` is an API-only, audit-only capability. It compares two
images in both orders; it does not qualify a deployment or estimate a success
rate. Core behavior lives in `npa.workbench.vlm_eval`, shared by CLI and SDK.

### Scheduled execution

The protected `token-factory-live` nightly workflow keeps its existing hosted
suites and separately invokes `npa/scripts/vlm_audit_live_recheck.py` with
`--generated-controls --audit-kind preference`. The explicit selector is required
for generated controls; unknown or omitted generated kinds fail before inference.
Select `paired` in a separate invocation for paired-judge controls. A directory label never selects a lane.
The same protected Token Factory key is required. No
GPU, project, served-model endpoint, vendor weights, or new secret is needed.
The runner verifies credential presence and key-scoped model availability before
inference. Hosted use remains subject to the operator's provider/model terms;
model listing alone is not inference evidence or a license entitlement.

Three frozen visual controls exercise both preference directions and identical
images. Each control makes two real hosted requests with swapped image order,
requiring exact served-model identity, complete JSON, nonempty observations,
provider usage, request IDs, and exact retained request/response hashes. Pixel
hashes and expectations are fixed before inference. Confidence below high,
unresolved output, order disagreement, or errors require escalation and cannot
satisfy these frozen high-confidence controls. A failed control retains its
actual private outcomes rather than retrying until it agrees.

The lane rejects zero tests, skipped tests, xfail/XPASS, partial selection,
deselection, or a mismatch between configured, collected, executed, and passed
counts. Ambient pytest flags and CI shard selectors are removed. Synthetic
images are inputs to real inference, not substitutes for provider responses.

Passing test counts alone are insufficient: every preconfigured artifact must
exist, contain valid retained outcomes, and meet its frozen expectations.
Configuration changes during execution fail the lane. Missing or malformed
artifacts retain their original control indices in sanitized failure rows;
extra files cannot substitute for them. Typed provider errors remain visible
as private evidence and bounded diagnostics, never successful inference proof.

Raw requests, images, outputs, and pytest logs stay in an owner-only directory
outside Git. Scheduled raw evidence stays on the ephemeral runner; only the
allowlisted, confidentiality-scanned `receipt.json` reaches Actions artifacts
or summaries. Local operator runs preserve complete raw evidence for review.
There is no job-owned cloud compute to clean up.

### Operator execution

Use the checkout's isolated Python environment with `npa[dev,adapter]`. Provide
`NEBIUS_TOKEN_FACTORY_KEY` through an owner-only credential store/environment,
never through command arguments, workflow configuration, or committed files.
Verify service credentials with `npa workbench health preflight` first.

```bash
umask 077
npa/.venv/bin/python npa/scripts/vlm_audit_live_recheck.py \
  --generated-controls --audit-kind preference --evidence-dir "<new-private-directory-outside-checkout>"
```

Alternatively set `NPA_VLM_AUDIT_LIVE_CONFIG` to an owner-only JSON file. Its
`cases.blinded-preference` contains a `request` matching
`VlmPreferenceComparisonRequest` and `expectations`, or a nonempty `controls`
mapping whose values each contain those objects. Each expectation must freeze
two `mapped_preferences` values (`baseline`, `candidate`, `tie`, or `unresolved`)
and a boolean `escalation_required`. Other dotted report fields may also be
asserted. The configured `api_key_env` must name an explicitly populated
environment variable. The runner replaces output paths with unique private
locations without altering the source config. Run without `--generated-controls`.

The executable test is `npa/tests/e2e/test_vlm_audits_live.py`; direct invocation
requires both `NPA_INTEGRATION_E2E=1` and `NPA_VLM_AUDIT_LIVE_CONFIG`. Its ordinary
unconfigured skip is not live proof. Use the runner for fail-closed designated
validation. The separate served-model provenance config is not consumed here.

Scheduled wiring is a runnable contract, not a claim that this unmerged branch
has already run on the nightly schedule. Retain the exact-SHA operator receipt
and inspect future scheduled receipts independently.

## Paired judge controls

`npa/scripts/vlm_audit_live_recheck.py` is the executable hosted audit lane for
`npa/tests/e2e/test_vlm_audits_live.py`. It runs both real hosted judges over the
same normalized frames and verifies their retained provider responses, model
identities, positive usage, response hashes, shared request evidence, and frozen
score-gate expectations. It uses no recorded response or score override.

The nightly Token Factory workflow runs this lane separately from its existing
three hosted migration suites, using `--generated-controls --audit-kind paired`. The GPU served-model
provenance test has its own contract and is not covered by this hosted lane:
`NPA_VLM_PROVENANCE_LIVE_CONFIG`, described in the
[VLM runbook](../workbench/cookbooks/vlm-eval-loop-runbook.md).

### Scheduled visual contract controls

The existing protected `token-factory-live` environment supplies only its Token
Factory key to the paired step. No GPU, Nebius admin, SSH, S3, gated payload, or
additional private configuration is needed. The workflow's existing branch
policy is unchanged. The runner creates private deterministic PNGs and a private
configuration locally: a red square inside a green outline (pass), outside the
outline (fail), and a blank image (fail). All three share one frozen task and
rubric. Pixel hashes and labels are regression-tested independently of inference.

Both `MiniMaxAI/MiniMax-M3` and `google/gemma-3-27b-it` receive the same normalized
image and request with `temperature=0`; their model IDs alone differ. Kimi's
generation contract is incompatible with this pair; this lane does not change
single-judge model support. An authenticated model-catalog check precedes inference but cannot
produce a passing live receipt. Each of the three tests must obtain successful,
exact-model responses from both judges and satisfy its frozen expectations.
Changing labels to match a response is not permitted. These are visual contract
controls, not robot-performance measurements or proof of deterministic models.

This is a **common-body experiment**, not a comparison of each model's tuned
single-judge profile. Both bodies contain only `model`, `temperature`, and
`messages`; paired judging deliberately omits scalar-profile
`chat_template_kwargs`, `reasoning_effort`, and `response_format`. The shared
profile gate checks compatibility with the common temperature request, not
equality of optional scalar defaults. In particular, this does not claim
MiniMax thinking is disabled or Gemma JSON mode is enabled. Use individual
evaluation for those model-specific settings. A formatting failure is retained
as a judge error and fails the frozen acceptance controls; the scheduled step
does not ignore it. Passing controls establishes the observed pair's contract
on those requests, not future response determinism or general model ranking.

The workflow first verifies that the same generated lane fails with an empty
key. It then executes the credentialed lane even if a preceding hosted check
failed. Only sanitized receipts are uploaded. Raw images, configuration, logs,
and complete provider responses stay in the private runner directory and expire
with the hosted runner; no durable raw-evidence archive is implied. Receipts
retain allowlisted per-control verdicts, scores, error types and response/artifact
digests. For durable raw evidence, run locally in access-controlled storage:

```bash
npa/.venv/bin/python npa/scripts/vlm_audit_live_recheck.py \
  --generated-controls --audit-kind paired --evidence-dir "$NPA_PRIVATE_EVIDENCE_DIR"
```

Load `NEBIUS_TOKEN_FACTORY_KEY` without echoing it. Generated controls pin the
public Token Factory endpoint and ignore operator endpoint/config overrides.
Workflow registration alone is not evidence that a scheduled run has passed.

### Prepare the private inputs

Install this checkout's `npa[dev,adapter]` in `npa/.venv` on Python 3.12. Select
two distinct exact hosted model IDs and a stable image, video, array, or rollout
directory whose visible control you have labeled before inference. Blank and
unrelated controls should use the same task and rubric as positive controls.
Use separate invocations and new evidence directories for each control. A
synthetic image can test a visible property, but cannot establish robotics
performance. A fixture score or model listing is never inference evidence.

Keep the configuration outside Git and owner-readable only (`0600`). Set
`NPA_VLM_AUDIT_LIVE_CONFIG` to its path. Its JSON shape is:

```json
{
  "cases": {
    "paired-judges": {
      "request": {
        "input_path": "<private-control-image-or-s3-uri>",
        "primary_model": "<exact-first-model-id>",
        "secondary_model": "<exact-second-model-id>",
        "api_key_env": "NEBIUS_TOKEN_FACTORY_KEY",
        "task": "Describe the visible scene, then judge whether the target object is present.",
        "rubric": "Score 1 only when the target is visibly present; otherwise score 0.",
        "success_threshold": 0.8,
        "frame_selection": "keyframes",
        "max_frames": 4
      },
      "expectations": {
        "primary.result.passed": false,
        "secondary.result.passed": false
      }
    }
  }
}
```

Configurations must contain exactly one supported case family. The optional
top-level `audit_kind` is `paired` for this lane and must match `paired-judges`;
unknown or mixed families fail before execution. Generated configurations require
`audit_kind: paired`, `control_schema: npa.paired_visual_controls.v1`, and exactly
the ordered `inside`, `outside`, `blank` controls. Collection selects this
validated family even when other audit operations are installed; it never
collects an unrelated lane or ignores an extra configured case.

Replace the task, rubric, and expectations with the frozen control definition;
do not change them to make a response pass. The runner assigns `output_path`
inside its new evidence directory, leaving the source configuration unchanged.
An optional `endpoint_url` selects the authorized hosted endpoint; absent uses
the product's configured endpoint. Supply the credential through the request's
`api_key_env` (default `VLM_EVAL_API_KEY`), never the JSON. The lane requires that
environment key even though ordinary product calls can use saved credentials.
Load it from the existing owner-only store without echoing it.

For Token Factory, prove the selected credential/endpoint with
`npa workbench health preflight --checks token_factory --json` before running.
That read-only probe is a prerequisite, not live inference evidence. It uses
`NEBIUS_TOKEN_FACTORY_KEY` and `NEBIUS_TOKEN_FACTORY_BASE_URL`; keep these aligned
with the audit's explicit key/endpoint. Other OpenAI-compatible hosted providers
need their own authenticated readiness probe. The audit performs no model
download and no cloud provisioning. Any gated fixture must already have its
exact access verified through the applicable provider mechanism.

Local fixtures need no S3 credentials. For S3 input, use a task-scoped
`NPA_CONFIG_DIR` or explicit `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_ENDPOINT_URL`, verify access to the exact input prefix, and run
`npa workbench health preflight --checks s3 --json` with the same storage
selection. The product materializes inputs privately and removes temporary
downloads when the call exits. The lane does not modify or delete source data.

### Execute and retain evidence

Choose a new private directory outside the checkout for every invocation:

```bash
npa/.venv/bin/python npa/scripts/vlm_audit_live_recheck.py \
  --evidence-dir "$NPA_PRIVATE_EVIDENCE_DIR"
```

The runner enables `NPA_INTEGRATION_E2E=1`, removes inherited pytest selection,
sharding, and dry-run settings, and invokes the entire audit test file in a
fresh interpreter. Missing configuration/credentials/expectations, collection
errors, zero tests, failed checks, xfail, xpass, skipped tests, and deselection
cannot pass. Executed/passed counts must equal the configured control count.
Counts alone cannot pass: every configured artifact must remain a private regular
file under its original control index, match its recorded byte hash and frozen
expectations, and contain valid typed outcomes and a consistent known status.
Descriptor-based no-follow reads reject symlinked parents and non-regular files
(including FIFOs) without reading or blocking on them. Prepared expectations are
frozen before execution; changing the private config afterward cannot alter them.
The tests require real successful responses from both judges; disagreement is
preserved as an audit outcome and can satisfy a predeclared control expectation.
Provider errors retain their private product artifact but fail this live gate.

`receipt.json` records source hashes, commit SHA, configuration hash, pytest
exit status, the explicit `paired` audit kind, counts, and sanitized outcomes.
Generated audits reject missing or unsupported kinds before fixture creation or
provider access; another audit lane cannot silently substitute for this one.
Outcome indices follow control order
(inside, outside, blank for generated controls). Counts come directly from pytest
reports, including collection, setup and teardown errors; no XML is parsed.
`pytest.log`, `execution.json`, the
prepared configuration, and reports under `paired-judges/` remain in the `0700` evidence
directory. Raw logs and artifacts can contain private task/provider data: keep
the complete directory access-controlled and publish only the sanitized receipt.
A failed setup can have no product artifact; the receipt remains failed.
The output directory cannot be reused, and paired product publication itself
rejects replacement, including concurrent local writes and S3 conditional-write
collisions. Preserve evidence after success or failure. There is no job-owned
GPU compute to clean up in this hosted lane.

The operator-configured mode can also use `"controls": {"<label>": {"request":
{...}, "expectations": {...}}}` inside `paired-judges` for multiple controls in
one run. Each receives a distinct private output directory. An optional
`input_sha256` binds a local fixture's bytes before inference.

An operator scheduler must install the real private configuration and credential
loading before using configured mode. This lane makes no claim about
judge qualification, physical correctness, safety, or future model behavior.
