# Configured paired VLM audit checks

`npa/scripts/vlm_audit_live_recheck.py` is the executable hosted audit lane for
`npa/tests/e2e/test_vlm_audits_live.py`. It runs both real hosted judges over the
same normalized frames and verifies their retained provider responses, model
identities, positive usage, response hashes, shared request evidence, and frozen
score-gate expectations. It uses no recorded response or score override.

The nightly Token Factory workflow runs this lane separately from its existing
three hosted migration suites, using `--generated-controls`. The GPU served-model
provenance test has its own contract and is not covered by this hosted lane:
`NPA_VLM_PROVENANCE_LIVE_CONFIG`, described in the
[VLM runbook](../workbench/cookbooks/vlm-eval-loop-runbook.md).

## Scheduled visual contract controls

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

The workflow first verifies that the same generated lane fails with an empty
key. It then executes the credentialed lane even if a preceding hosted check
failed. Only sanitized receipts are uploaded. Raw images, configuration, logs,
and complete provider responses stay in the private runner directory and expire
with the hosted runner; no durable raw-evidence archive is implied. Receipts
retain allowlisted per-control verdicts, scores, error types and response/artifact
digests. For durable raw evidence, run locally in access-controlled storage:

```bash
npa/.venv/bin/python npa/scripts/vlm_audit_live_recheck.py \
  --generated-controls --evidence-dir "$NPA_PRIVATE_EVIDENCE_DIR"
```

Load `NEBIUS_TOKEN_FACTORY_KEY` without echoing it. Generated controls pin the
public Token Factory endpoint and ignore operator endpoint/config overrides.
Workflow registration alone is not evidence that a scheduled run has passed.

## Prepare the private inputs

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

## Execute and retain evidence

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
The tests require real successful responses from both judges; disagreement is
preserved as an audit outcome and can satisfy a predeclared control expectation.
Provider errors retain their private product artifact but fail this live gate.

`receipt.json` records source hashes, commit SHA, configuration hash, pytest
exit status, counts, and sanitized outcomes. Outcome indices follow control order
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
