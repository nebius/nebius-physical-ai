---
name: nebius-stream-beta
description: Use Nebius Stream beta through a supplied Cosmos3 HTTPS API independently to register customer S3 storage, discover Nano or Super capabilities, run paired media-generation iterations, manage source inventories and batches, recover verified artifacts, and export owner-scoped usage. Explain native LeRobot import limits without requiring a private SDK, repository, Nebius CLI, or SSH.
---

# Nebius Stream beta

Use this portable customer client with an operator-supplied HTTPS API origin.
Get the endpoint and a customer-scoped inference key from the beta operator
through their approved private channel; this public skill does not grant access. The
customer's “stream key” is an inference API key, not an RTMP key or a promise
of streaming inference. The package contains no service locator and requires
no private repository, Nebius CLI, SSH, GPU access or model download. Python
3.10+ on macOS/Linux is enough once the supplied API is independently reachable
and source media is staged. Do not claim public readiness from package presence
or internal validation when external ingress has not been verified.

Use the bundled [client](scripts/cosmos_client.py). It verifies TLS, refuses
credential-bearing redirects, reads private files, retains submission identity,
and checks downloaded artifacts. Do not replace these safeguards with curl
commands containing keys, disabled TLS verification, browser localStorage,
chat-pasted credentials, or request-body logging.

## Private setup

The customer supplies `WMH_BASE_URL` (one actual HTTPS origin),
`WMH_API_KEY_FILE` (a private text file with the inference key), and
`WMH_S3_CREDENTIAL_FILE` (a private JSON file containing only
`aws_access_key_id`, `aws_secret_access_key`, and optional `aws_session_token`).
Supply the endpoint and file paths through the environment; never put the keys
in environment variables or command arguments. Files must be owned by the
current user with mode 0600, in a mode 0700 working directory outside Git.
The API key needs `models:read`, `storage:manage`, `jobs:read`,
`artifacts:read`, and the selected generation route's scope (`videos:write`,
`images:write`, or `augmentations:write`) with its explicit model grant.

If the customer has not made credential files, set the two destination file
paths and run `python3 scripts/cosmos_client.py init-secrets` in their terminal;
it asks for hidden input and writes private files. Use `init-storage-secret`
with an unused S3 file path for another provider or an existing API key file.
An agent must not collect
these secrets through chat. For an operator-provided private CA only, set
`WMH_CA_BUNDLE` to that trusted CA file; hostname verification remains enabled.
For ordinary customer access, use the supplied publicly trusted HTTPS origin.

Run from this skill's directory. Follow the private setup in the
[iteration guide](references/iteration.md) to set `WMH_WORK_DIR`, register
input storage, and create a full-source augmentation request. Then run:

```bash
python3 scripts/cosmos_client.py discover --output "$WMH_WORK_DIR/contract.json"
python3 scripts/cosmos_client.py register --config "$WMH_WORK_DIR/output-storage.json"
python3 scripts/cosmos_client.py submit --route /v1/augmentations --request "$WMH_WORK_DIR/request.json" --state "$WMH_WORK_DIR/run.json"
python3 scripts/cosmos_client.py resume --state "$WMH_WORK_DIR/run.json" --output-dir "$WMH_WORK_DIR/results"
```

Read [the request guide](references/requests.md) for storage registration and
credential injection. Its [full-source video-to-video example](references/requests.md#full-source-video-to-video)
contains the complete Super request JSON, Nano model selection and input
requirements, HTTP headers, and explanations of all five sampling settings.
Keep full-source augmentation separate from the fixed video profiles in that
guide. Read
[the independent iteration guide](references/iteration.md) for complete
customer setup, staged MP4 and LeRobot-derivative boundaries, baseline/tuned
requests, source inventories, batch recovery and private usage exports. Replace placeholders
locally and make configuration/request files mode 0600. Register input storage
separately when it uses another connection or credential. The helper reads the
current S3 credential file for each registration. Media must already exist in
the registered input bucket; the helper does not upload local media.

## Discover before generating

Inspect the saved authenticated `/v1/models` and `/v1/openapi.json` contract.
Choose Super or Nano explicitly; use a profile only when its running capability
is exposed, implemented and qualified. The helper refuses unsupported profiles.
Do not infer speech-to-text, text-to-speech, arbitrary reference audio, arbitrary
shapes, full-source preservation, or action fidelity from a model name. An
example is a schema illustration, not proof that a deployment serves it.

Generated output requires a server-verified Nebius connection. A customer
provider label or S3-compatible input does not establish output eligibility.
The server verifies canonical endpoint identity and actual bucket region with
the supplied credentials, then encrypts credentials under its operator key.
It never returns raw credentials. Source inputs may use other approved public
S3-compatible services. Private-network S3 endpoints are unsupported by this
self-service path. Customer files and generated artifacts stay private.

## Source selection, iteration and usage

For source inventories, use `submit --route /v1/source-inventories`,
`inventory-resume` and `inventory-objects`. Inventory discovery is durable
and paginated; inspect rejected and failed objects before admission. For a
Super full-source inventory batch, run `batch-preflight`, then
`submit --route /v1/augmentation-batches` and `resume`. Batch operations also
need `batches:write`. The distinct `/v1/batches` schema supports explicit Nano
T2V items. Native model batching is not implied by either API batch form.

For a controlled baseline/tuned comparison, retain source identity, seed,
prompt, model and control; change one supported generation setting and use a
new request/state/output prefix. Inspect the actual technical report and
artifacts before deciding which result to use. The only exposed evaluation
mode is report-only `technical_report` with `complete_decode` and
`timeline_coverage`; it does not score semantics or optimize parameters.
Native LeRobot import/export is unavailable. Dataset-derived camera video is
not equivalent to a dataset with action/state/timestamp alignment; preparation
and staging occur before inference API use. Never silently substitute extracts
when native dataset semantics are required.

Use `usage-summary` or crash-resumable `usage-export` for owner-scoped evidence.
Add `--include-synthetic` for demo/benchmark/internal-test campaigns; the
default includes only attributed production traffic. Preserve event identities,
the fixed watermark and distinct request/admission/delivery/poll/artifact
counts. A synthetic customer type is not proof of real company adoption.
Read the iteration guide for UTC filtering, pagination and incremental export.

## Recovery and credential lifecycle

Keep the exact request and run state. After an ambiguous submit, run `submit`
again with that same request and state; it reuses the stored idempotency key.
Once a job ID is retained, use `resume`. Interrupting local polling does not
cancel server work. An HTTP error never authorizes a new identity or another
model. Failed/publication-failed/reconciliation states need inspection through
the supplied API contract; publication-only retry must not rerun generation.
Read-only GETs receiving HTTP 429 retry after `Retry-After` plus small jitter;
generation POSTs are never automatically retried. A download error can be
recovered by rerunning `resume`; reject hash mismatches.

Registration returns 409 if the connection already exists (including after a
lost successful response). Inspect it using `connection --id CONNECTION_ID --output PRIVATE_FILE`
before changing it. Rotate only with the customer's explicit intent:
`python3 scripts/cosmos_client.py rotate --connection CONNECTION_ID` reads the
replacement S3 file. Revoke with `revoke --connection CONNECTION_ID`. Rotation
changes the immutable connection identity and invalidates previously admitted
jobs using it; let active jobs finish first when continuity is required.
Revocation affects subsequent storage operations; it cannot recall bytes or
interrupt an already-open transfer. It does not revoke the underlying key at
Nebius; do that separately when the customer intends to disable it everywhere.

Report source validation, live API validation, technical media integrity and
semantic quality separately. This package is prepared for distribution; its
presence does not establish that any particular endpoint is externally
reachable, or authorize publishing a private repository, endpoint or evidence.
