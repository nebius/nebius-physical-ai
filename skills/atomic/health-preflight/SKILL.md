---
name: health-preflight
description: Use before any deploy, image build, provisioning, or GPU submit to prove service credentials with `npa workbench health preflight`, request `--checks nebius` before provisioning, and verify gated-model access with `npa workbench health access`.
---

# Health preflight (the doctor before you spend)

There is no `npa doctor`. The readiness surface is `npa workbench health`, and it
has exactly two public commands. Run them before anything that costs GPU time or
provisions cloud resources.

```bash
npa workbench health preflight --json
npa workbench health preflight --checks nebius --json
npa workbench health access --json
```

All three exit non-zero on any FAIL. That is the point: they are gates, not reports.
Use `--warn-only` only when you deliberately want the report without the gate
(for example while collecting a status snapshot), never to make a red run green.

## What each command answers

`preflight` answers *"do I hold the credentials nearly every workbench tool
needs?"* — one PASS/WARN/FAIL/SKIP row per check over Hugging Face, NGC, S3, and
Token Factory. Request the optional `nebius` check before provisioning:

```bash
npa workbench health preflight --checks all
npa workbench health preflight --checks nebius --json
npa workbench health preflight --checks s3,token_factory
npa workbench health preflight --checks all --offline  # presence only; Nebius is SKIP
```

Valid `--checks` values are `all`, `hf`, `ngc`, `s3`, `token_factory`,
`nebius`. The default remains `hf,ngc,s3,token_factory`, so hosted-inference
work does not require a Nebius Cloud profile. Explicit `all` includes `nebius`.
Empty selections and unknown check names are errors, including unknown names
combined with `all`. Repeated names run once.

Online `nebius` runs the selected Nebius CLI profile through `iam whoami` and
`iam get-access-token` with browser launch and update checks disabled. It
discards both commands' output and removes ambient `NEBIUS_IAM_TOKEN` and
`NEBIUS_IAM_TOKEN_FILE` values so a stale token cannot mask profile readiness.
PASS requires both calls to succeed. In offline mode it returns SKIP because a
local profile file is not proof of usable authentication.

Profile selection uses `NPA_NEBIUS_PROFILE`, then `NEBIUS_PROFILE`, then the
CLI's active/default profile. The project selector does not choose an auth
profile. PASS proves authentication; permission to provision a particular
resource still depends on the selected identity's access to that project.

Online `hf` authenticates against Hugging Face `whoami-v2`; public repository
metadata is not sufficient. Online `ngc` performs a registry token exchange;
that proves the key, not entitlement to every NGC artifact. `access` performs
the capability-specific repository/artifact probe.

`access` answers the different and more specific question *"is my token actually
entitled to fetch bytes from the gated assets this capability pulls?"* It probes
a representative payload path at the exact catalog revision with HEAD or a
one-byte Range request. Metadata access is not proof. Holding an HF token is not
the same as having account entitlement, and `Ready` is technical fetch evidence,
not proof of legal acceptance.

```bash
npa workbench health access --capability all
npa workbench health access --capability cosmos3,groot
```

Capabilities: `all`, `cosmos`, `cosmos3`, `cosmos3-serving`, `groot`, `lerobot`,
`nurec`, `paidf`, `sim2real`, `sonic`, `token_factory`, `vlm_eval`.

For anything still gated, `access` prints the exact "Agree and access
repository" URL. **Hugging Face gated licenses must be accepted interactively on
the model page** — there is no API that accepts them for you, so no amount of
retrying or re-tokenizing will clear a gate. Open the printed URL, accept, then
re-run. `scripts/accept-model-access.sh` collects the URLs for a batch.

## Ordering: preflight before provisioning, not after

The recurring cold-start failure this prevents is a mid-run stop after you have
already paid for a cluster. Run the checks in this order:

1. `npa configure --show` — confirm the project stanza, bucket, and endpoint you
   think you are using are the ones on disk.
2. `npa workbench health preflight` — credentials exist and authenticate.
3. `npa workbench health preflight --checks nebius`: the selected Nebius CLI
   profile can resolve identity and mint an IAM token before provisioning.
4. `npa workbench health access --capability <the one you will run>`: gated
   model entitlements for that capability only; `all` is slower and reports
   failures you do not care about today.
5. Only then `npa provision-if-absent`, `npa cluster up`, or
   `npa workbench workflow submit`.

Image pullability is a *separate* gate that health does not cover; see
`npa workbench workflow preflight-images <spec.yaml>` in
`skills/atomic/submit-workflow/SKILL.md`. A green health report with an
unpullable image still hangs in `ImagePullBackOff`.

**A green `s3` row does not mean the submit can write.** The `s3` check lists
the configured project bucket; `submit` runs a stricter one that puts a unique
object into the bucket the *workflow* resolves, which is a different bucket
whenever `NPA_S3_BUCKET` or `AWS_ENDPOINT_URL` is set in the environment. The
two disagreeing looks like a passing preflight followed by:

```
Error: Cannot submit <spec>.yaml: missing prerequisites:
  - writable S3 for this workflow (S3 write permission was denied.)
```

Believe the submit, not the preflight row: it is the check closer to the write
you are about to do. Reconcile the endpoint and bucket the workflow sees with
the ones `npa configure --show` reports before reaching for the printed
`npa provision-if-absent --project <alias> --skip-k8s` fix.

## Persisting credentials you already hold

If the tokens are in your environment but not in `~/.npa/credentials.yaml`,
persist them instead of re-exporting them in every shell:

```bash
npa workbench health access --save-env-credentials
npa configure --save-env-credentials
```

Both perform an atomic `0600` write and never print the values. This is the fix
for "it worked in my shell but the submit could not resolve the secret": submit
resolves each requested secret from the explicit process environment first, then
the selected project's configured NPA credentials.

## Hidden sim2real check

`npa workbench health sim2real` exists but is hidden, and is specific to the
14-stage Sim2Real graph rather than general readiness. It adds cluster-shaped
checks (`config`, `coherence`, `s3`, `registry`, `tokens`, `cluster`) including
schedulable GPU count and kube-context pinning:

```bash
npa workbench health sim2real --checks all --json
```

Use it from `skills/workflows/sim2real-operate/SKILL.md`, not as the generic
preflight. Its cluster check counts schedulable `nvidia.com/gpu` but does not yet
match the *requested GPU product* against available node products, so it can pass
on a cluster that has GPUs of the wrong kind for your spec. Confirm the product
with `npa workbench workflow gpus --cluster <name>`.

## Gotchas

- **`--offline` proves presence, not validity.** It skips live provider probes.
  Existing credentials may pass presence checks, while `nebius` returns SKIP
  because profile configuration is not authentication proof. Use offline only
  where there is genuinely no egress.
- **A SKIP is not a PASS.** Checks skip when the corresponding capability is not
  configured. If you are about to run a Cosmos stage and the NGC row says SKIP,
  you have not verified anything about NGC.
- **`preflight` does not check Kubernetes or SkyPilot.** Cluster readiness is
  `npa skypilot verify --cluster <exact-context>` and `npa cluster status`;
  registry pullability is `workflow preflight-images`. Three separate gates,
  three separate commands.
- **Stale `NEBIUS_IAM_TOKEN` defeats provider calls even when health is green.**
  The Nebius provider prefers an ambient (often expired) token over the fresh CLI
  token. `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN` before provisioning or
  submitting.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

With an authenticated operator profile, exercise the real CLI's successful,
missing-profile, and offline paths (including stale ambient token scrubbing):

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_nebius_auth_preflight.py -q
```
