---
name: submit-workflow
description: Use when submitting, validating, or debugging NPA SkyPilot workflow YAMLs and workflow runner paths.
---

# Submit Workflow

## When To Use

Use this skill for workflow launch, YAML validation, runner scripts, and
SkyPilot submission behavior.

## Procedure

1. Read `skills/tools/skypilot-workflows/SKILL.md` for SkyPilot version and
   cleanup constraints.
2. Prefer `npa.workflow/v0.0.1` specs under
   `npa/workflows/workbench/npa-workflows/`. Parse / `validate-spec` locally
   before launch.
3. Use `NPA_SKYPILOT_BIN` or `npa skypilot status --bin-path`; do not assume
   `sky` from `PATH`.
4. Submit through `npa workbench workflow submit` (accepts npa.workflow specs
   and legacy SkyPilot YAML) or the shared workflow submission helper.
5. Keep cleanup best-effort and avoid tearing down a shared controller unless
   the operator explicitly requests it.

## Three-Tier Contract

- CLI: `npa workbench workflow --help` and tool-specific `workflow` commands.
- SDK: use shared workflow submission helpers rather than shelling out from
  application logic.
- YAML: prefer `npa.workflow/v0.0.1` specs under
  `npa/workflows/workbench/npa-workflows/` for authoring. `npa workbench
  workflow submit` accepts those specs (plans → renders → SkyPilot) and still
  accepts raw SkyPilot YAML under `npa/src/npa/workflows/skypilot/` for
  operator/runtime and SkyPilot-only exceptions (parallel, burst, runbook).

## Live submit prerequisites (real cluster)

A real `npa workbench workflow submit` (not `--plan-only`) needs, on top of a
healthy `sky check kubernetes`:

- **Secrets via `--secret-env`** (never in the YAML): `NEBIUS_TOKEN_FACTORY_KEY`
  for Token Factory stages, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for S3,
  `HF_TOKEN` / `NGC_API_KEY` for gated model pulls. `load_credentials` does NOT
  export AWS creds — read them from `~/.npa/credentials.yaml` (`storage.*`) and
  export before submit.
- **`NPA_SRC_S3_URI` (or `--image`)** for CPU tool steps and `run.shell` states —
  they have no heavy workbench image and install npa from that source tarball,
  else render fails with "planned step has no workbench image and NPA_SRC_S3_URI
  is unset".
- **`--assume-decision promote_checkpoint`** for specs with a dynamic gate/loop.
- **`--var key=value`** to override `config` (e.g. `--var bucket=<real-bucket>`;
  the reference specs default to `bucket: example-bucket`).

## Gotchas

- SkyPilot `envs` does not support self-referencing interpolation. The
  npa.workflow renderer resolves images and config before submit so rendered
  YAML has no `${VAR}` placeholders.
- `sky jobs launch` does not provide a reliable dry-run path in the pinned
  version; use `npa workbench workflow submit --plan-only` for npa.workflow
  specs, or mock submission before live launch.
- Mixed serial and parallel task groups can be fragile; serialize when behavior
  must be deterministic. Parallel sweeps stay SkyPilot-only in v0.0.1.
- **GPU accelerator name is cluster-specific.** Specs use canonical
  `RTXPRO6000:1`, but a cluster may only advertise the raw label (e.g.
  `RTXPRO-6000-BLACKWELL-SERVER-EDITION`). A mismatch fails with
  `FAILED_PRECHECKS` / "cluster does not contain any instances satisfying the
  request" — not a capacity problem. Run `sky gpus list` and resubmit with the
  cluster's exact accelerator name (this is the "retry GPU types" path).
- **Stale `NEBIUS_IAM_TOKEN` breaks sky/terraform.** The Nebius provider prefers
  an ambient (often expired) `NEBIUS_IAM_TOKEN` over the fresh CLI token, giving
  `PermissionDenied` / `Unauthenticated` even though the `nebius` CLI works.
  `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN` before submitting/deploying.

## Teardown

- **Cancel then wait, then tear down.** `sky jobs cancel` only *schedules*
  cancellation; `sky down` on the jobs controller refuses while any managed job is
  non-terminal, so cancelling and immediately tearing down fails with
  `NotSupportedError: In-progress managed jobs found`. `cleanup_all_for_run` and
  `cleanup_jobs_controller` now wait for the queue to drain and retry that specific
  error; if you drive `sky` by hand, poll `sky jobs queue --all` first.
- **A PENDING job may be dead, not slow.** A pod stuck in `ImagePullBackOff` or
  `Unschedulable` is retried by Kubernetes forever, so the job never becomes FAILED.
  `npa workbench workflow status` reports the pod-level reason for a PENDING job.
- **`npa cleanup`** reports what a teardown left behind (local caches, project
  entries, non-terminal managed jobs, the service accounts `npa configure` creates)
  and prints the ordered runbook. `--yes` removes the local caches only; it never
  deletes cloud resources or service accounts.
- **`npa cluster down`** previews the PodDisruptionBudgets that will hold up the
  node drain, so a multi-minute silence is expected rather than alarming.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test invokes workflow help and parses representative workflow YAML.
