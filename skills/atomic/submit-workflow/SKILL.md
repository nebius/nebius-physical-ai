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
- YAML: author shipped workflows as `npa.workflow/v0.0.1` specs under
  `npa/workflows/workbench/npa-workflows/`. `npa workbench workflow submit`
  accepts those specs (plans, renders, then launches SkyPilot) and still accepts
  raw SkyPilot YAML supplied by an operator or by guarded single-task example
  directories.

## Live submit prerequisites (real cluster)

A real `npa workbench workflow submit` (not `--plan-only`) needs, on top of a
successful `npa skypilot verify --cluster <exact-context>`:

- **Secrets via `--secret-env`** (never in the YAML): `NEBIUS_TOKEN_FACTORY_KEY`
  for Token Factory stages, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for S3,
  `HF_TOKEN` / `NGC_API_KEY` for gated model pulls. Submit resolves each requested
  name from the explicit process environment first, then the selected project's
  configured NPA credentials; it fails locally if the value is unavailable.
- **`NPA_SRC_S3_URI` (or `--image`)** for CPU tool steps and `run.shell` states —
  they have no heavy workbench image and install npa from that source tarball,
  else render fails with "planned step has no workbench image and NPA_SRC_S3_URI
  is unset". Persist it once with `npa configure --src-s3-uri s3://bucket/prefix/npa`
  so a new shell resolves it from `~/.npa/config.yaml` instead of failing preflight
  on already-staged objects (`scripts/stage-npa-src.sh` does this for you).
- **`--assume-decision promote_checkpoint`** for specs with a dynamic gate/loop.
- **`--var key=value`** to override `config` (e.g. `--var bucket=<real-bucket>`;
  the reference specs default to `bucket: example-bucket`).

## Gotchas

- **Kubernetes controller launch is transactional.** Immediately before every
  Kubernetes managed-job launch, NPA probes `/readyz` on the exact selected
  context with SkyPilot's `KUBECONFIG` environment. Readiness requires three
  consecutive successes spanning 10 seconds. After any failed/uncertain launch,
  NPA reconciles exact `sky jobs queue --all --output json` evidence: adopt one
  immutable ID, retry only after authoritative absence plus a classified
  transport/API warm-up failure, or fail closed as indeterminate. Never bypass
  this with raw `sky jobs launch`, retry by name, or cancel by name.
- Transaction recovery uses capped exponential jitter and a 180-second recovery
  deadline. This is product behavior, not an operator job/time budget. A
  recovered launch proceeds in the same command; use `--resume-run <same-id>`
  only for crash/restart or a printed indeterminate/deadline recovery action.
- A failed reconciliation with launch sequence zero created no SkyPilot job.
  NPA records a completed no-op rollback instead of leaving a
  `recovery-required` journal that would block unrelated project operations.
  Any failure after a launch may have been issued remains recovery-required.

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
  `RTXPRO-6000-BLACKWELL-SERVER-EDITION`), and the name changes while the NVIDIA
  GPU operator is still labelling nodes (`nebius.com/gpu-name: RTX6000` first,
  `nvidia.com/gpu.product` after). A mismatch fails with `FAILED_PRECHECKS` /
  "cluster does not contain any instances satisfying the request" — not a capacity
  problem. Submit now remaps this automatically; use
  `npa workbench workflow gpus --cluster <name>` to see the names yourself, or
  `--no-resolve-accelerators` to submit the spec's values verbatim.
- **`NAME:N` needs N GPUs on one node.** SkyPilot places all GPUs of a task on a
  single node, so `NAME:2` can never schedule on 2 nodes × 1 GPU no matter how many
  nodes exist. `workflow gpus` prints the requestable quantity per node; submit
  rejects anything above it. Multi-GPU fan-out docs assume N GPUs per pod, which is
  a different cluster shape from "N single-GPU node presets".
- **A workflow's images are not shipped into your registry.** `npa configure` picks
  (or creates) a project registry; nothing mirrors workbench images into it, so a
  spec that pins them needs them built and pushed once per registry. Run
  `npa workbench workflow preflight-images <spec.yaml>` — it reports each image as
  `ok`/`not_found`/`forbidden` and prints the build command for the tag
  `npa/src/npa/deploy/images.py` pins (the guide's tags are pinned to those by
  `tests/guardrails/test_paidf_image_tags_match_code.py`). `submit` runs the same
  check **before `deployIfAbsent`**, so a registry without the images costs no
  cluster time.
- **Multi-tool validation images stay distinct.** Repeat
  `--image-override TOOL_REF=IMAGE` on preflight and submit. Exact tool refs take
  precedence over the optional global `--image`; preflight resolves each selected
  artifact to the digest the renderer uses.
- **A registry `403` stalls rather than fails.** Kubernetes retries image pulls
  forever, so an unpullable image leaves the job in `PENDING`/`ImagePullBackOff`.
  Listing a repository's tags is a *different permission* from pulling it, so a
  `200` on `/v2/<repo>/tags/list` proves nothing. Submit reproduces each planned
  pull with the credentials it injects and refuses to launch on a `403`; run it
  standalone with `npa workbench workflow preflight-images <spec.yaml>`, or skip
  with `--no-preflight-images`.
- **A silent 15-minute submit is usually the kubernetes client.** SkyPilot 0.12.2
  does not cap the client version, and client 36+ makes every `pod_config` fail
  validation, so the managed-jobs controller retries forever. `npa skypilot
  bootstrap` pins a working client and repairs an existing venv; `npa skypilot
  status` reports the installed version. Submit streams SkyPilot output live and
  names this failure when it appears.
- **Stale `NEBIUS_IAM_TOKEN` breaks sky/terraform.** The Nebius provider prefers
  an ambient (often expired) `NEBIUS_IAM_TOKEN` over the fresh CLI token, giving
  `PermissionDenied` / `Unauthenticated` even though the `nebius` CLI works.
  `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN` before submitting/deploying.

## Teardown

- **Cancel then wait, then tear down.** Use `npa workbench workflow cancel
  <run-id> --project <alias> --json`; a planned/staged run that never launched is
  a successful repeat-safe no-op, while a launched run uses NPA's pinned
  SkyPilot runtime and waits for the exact manifest-proven job. Only after all
  workflows are terminal, remove the shared controller with `npa skypilot
  cleanup-controller --yes`. The underlying helpers retry the specific
  in-progress-jobs refusal after the queue drains.
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
