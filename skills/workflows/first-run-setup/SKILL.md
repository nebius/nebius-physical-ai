---
name: first-run-setup
description: Use on a fresh machine or a new Nebius project to get from zero to a first verified result — an ordered, gated path through install, configure, credential preflight, cheapest-proof workload, then cluster provisioning, with an explicit stop condition at every step.
---

# First run: zero to a verified result

The failure mode this skill exists to prevent is spending an hour provisioning a
GPU cluster and discovering at stage three that a token was never accepted. Each
step below has a **gate**: a command whose success is the precondition for the
next step. Do not skip ahead because a later step looks more interesting.

Escalate through the cheapest tier that can prove the thing you need. Most
first-run questions are answered before any GPU is involved.

## Step 1 — Install and prove the CLI runs

```bash
python3 -m venv npa/.venv
npa/.venv/bin/pip install -e "npa[dev]"
npa/.venv/bin/npa --version
```

**Gate:** `npa --version` and `npa --help` both succeed. They must work with no
Nebius, Hugging Face, NGC, Kubernetes, or S3 credentials at all. If they do not,
the problem is the install, not your cloud setup.

Python 3.10+ is required. Terraform 1.x must be on `PATH` for anything managed
(`npa cluster up`, `npa agent fresh-setup`); `pip install -e npa` does not install
it. On Windows, work inside WSL2 — the cloud paths assume POSIX.

Use `npa/.venv/bin/python` and `npa/.venv/bin/npa` explicitly for repo validation
rather than a bare `python` or `npa` from `PATH`.

## Step 2 — Configure Nebius

```bash
npa configure --show          # inspect the expected layout, writes nothing
npa configure                 # interactive: creates/reuses the CLI profile
```

`configure` writes `~/.npa/config.yaml` (machine-managed, non-secret) and
`~/.npa/credentials.yaml` (secrets, `0600`). Interactive setup offers S3 bucket
and access-key provisioning by default. `--no-provision` is a provider-free
project/token setup: it neither probes nor adopts saved storage. Enter an exact
existing bucket name only when you intend to reuse it; pressing Enter generates
a fresh name with a UTC timestamp and random suffix.

For unattended setup, avoid the prompts entirely:

```bash
npa configure --no-interactive --no-provision --save-env-credentials \
  --tenant-id <id> --project-id <id> --region <region> --project-alias <alias>
```

That command imports supported environment credentials and saves the project;
it does not contact Nebius, Hugging Face, or NGC and does not select storage.
Add explicit `--provision` when unattended setup should create or reuse writable
project storage.

**Gate:** `npa configure --show` reports the intended project stanza. When
storage was explicitly provisioned, it also reports the exact bucket and
endpoint. A silent exit or an unexpected stanza means configure did not write
what you think it did — resolve it here, because every later command resolves
credentials through this file. Hugging Face and NGC status in provisioning or
credential-import summaries is informative and never blocks the local save;
`--no-provision` reports those probes as skipped. Step 3 is the enforcing access
gate.

Do not hardcode project IDs, tenant IDs, private registry IDs, or bucket names
anywhere in the repo. The project values belong only in `~/.npa/`; `NPA_REGISTRY`
is a private build/BYOF destination, while runtime selection requires a complete
image reference or an explicit workflow `--registry`.

## Step 3 — Prove credentials before spending anything

```bash
npa workbench health preflight --json
npa workbench health access --capability <the-one-you-will-run> --json
```

**Gate:** both exit zero. `preflight` covers Hugging Face, NGC, S3, and Token
Factory presence and authentication; `access` proves your token is actually
entitled to the gated models a given capability pulls. Gated Hugging Face
licenses can only be accepted interactively on the model page — `access` prints
the exact URL, and nothing else will clear the gate. Full detail in
`skills/atomic/health-preflight/SKILL.md`.

## Step 4 — Get a real result with no GPU and no cluster

Before provisioning, prove the toolchain end to end on the cheapest tier that
produces a genuine artifact. Token Factory is hosted inference: no cluster, no
GPU, real model output.

```bash
npa workbench token-factory verify
npa workbench token-factory caption \
  --input-path ./some-images/ --output-path ./captions.json --max-images 4
```

**Gate:** a real captions artifact exists. You have now proven credentials, S3 or
local IO, and model access without provisioning anything. See
`skills/tools/token-factory/SKILL.md`.

If your goal genuinely needs a container on a GPU but not a cluster, the next
cheapest tier is a single serverless job:

```bash
npa workbench golden-eval list
npa workbench golden-eval run <tool>              # dry run: prints the command
npa workbench golden-eval run <tool> --serverless # one GPU, PASS/FAIL
```

See `skills/tools/golden-eval/SKILL.md`.

## Step 5 — Validate the workflow spec offline

Still free, and it catches most authoring mistakes:

```bash
npa workbench workflow validate-spec <spec.yaml>
npa workbench workflow plan-spec <spec.yaml>
```

**Gate:** the spec validates and the plan shows the stages you expect.

## Step 6 — Provision only what the spec needs

```bash
unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN   # stale ambient tokens break providers
npa provision-if-absent --project <alias> --dry-run --output-format json
npa provision-if-absent --project <alias>
npa skypilot bootstrap
npa skypilot verify --cluster <name> --output-format json
```

`provision-if-absent` is additive only — it never tears down or replaces
resources — so the dry run is a genuine preview. `skypilot bootstrap` pins a
Kubernetes client the controller can use; a newer client makes every `pod_config`
fail validation and the controller retries forever, which presents as a hung
submit rather than an error.

**Gate:** `skypilot verify` passes against the exact context you will submit to.

## Step 7 — Learn what this cluster calls its GPUs

```bash
npa workbench workflow gpus --cluster <name> --json
```

Kubernetes names accelerators after node labels, so the same card can be
`RTXPRO6000` in a spec and something much longer on the cluster. Note the printed
**requestable quantity per node**: SkyPilot places all GPUs of one task on a
single node, so `NAME:2` never schedules on 1-GPU nodes regardless of node count.

## Step 8 — Prove the images are pullable

```bash
npa workbench workflow preflight-images <spec.yaml> --project <alias> --json
```

**Gate:** every image reports `ok`. Supported images resolve from the anonymous
GHCR mirror by default. If you explicitly select a custom/private registry,
`not_found` means the image was never pushed there. A `403` does not fail a job —
Kubernetes retries pulls forever — so an unpullable image silently burns cluster
time in `ImagePullBackOff`. `submit` runs this check before provisioning by
default.

## Step 9 — Submit

```bash
npa workbench workflow submit <spec.yaml> --project <alias> --plan-only
npa workbench workflow submit <spec.yaml> --project <alias> \
  --var bucket=<your-bucket> --secret-env NEBIUS_TOKEN_FACTORY_KEY
```

Reference specs default to `bucket: example-bucket`; override real values through
`--var`. Pass secrets with `--secret-env`, never in the YAML. CPU tool steps and
`run.shell` states have no heavy image and install npa from a source tarball, so
persist that location once:

```bash
npa configure --src-s3-uri s3://<bucket>/<prefix>/npa
```

Then monitor and, on failure, triage with
`skills/atomic/debug-failed-run/SKILL.md`.

## Step 10 — Stop the spend

A first run is not finished until the resources are gone or deliberately kept.
Cancel runs before destroying anything that hosts them; see
`skills/atomic/teardown-and-cost/SKILL.md` for the full ordering and the orphan
audit.

## Common cold-start stumbles

- **`sim2real` and `sim-to-real` are different things.** `sim2real` is the staged
  14-stage VLM-to-RL loop; `sim-to-real` is the older H100 pipeline. The spelling
  is the disambiguator.
- **A Token Factory key is not a Nebius IAM token.** It starts with `v1.` and
  lives in `NEBIUS_TOKEN_FACTORY_KEY`.
- **Isaac Lab needs an RT-core GPU** (L40S or RTX PRO 6000), not H100/H200. See
  `skills/atomic/gpu-selection/SKILL.md`.
- **Scope `status` explicitly where the tool supports it.** For deployed-service
  tools that accept `-p <project>` / `-n <name>` (SONIC among them), a bare
  `status` can resolve a stale endpoint. Flags differ per tool — `detection-training
  status` wants `--run-id`, and `token-factory status` takes neither — so check
  `--help` rather than assuming a common shape.
- **`--offline` on health checks proves presence, not validity.** An expired token
  passes offline and fails on first pull.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
