---
name: byof-onboard
description: Use when onboarding an OSS repo via BYOF — containerize on Ubuntu or Isaac Lab, push to Nebius registry, and smoke on live Kubernetes.
---

# BYOF Solution Onboard

Canonical procedure for **bring-your-own-fork** onboarding. The NPA agent `onboard_solution`
intent and `run_byof_repo.py` both follow this skill — do not duplicate long command blocks
in chat replies; point operators here.

## When To Use

- Containerize a public GitHub/GitLab repo and push to the project registry
- Onboard a new workbench solution (toolRef + workflow + live smoke)
- LeIsaac validation (Isaac Lab base + datagen or RL)
- Generic Ubuntu BYOF (any OSS repo, no sim stack required)

For **registry/catalog admission** of an OSS Physical AI solution, also load
`skills/workflows/oss-solution-registry-onboard/SKILL.md`. BYOF proves the repo
can be packaged and run; registry admission additionally requires reading
upstream docs, identifying real capabilities, testing those capability claims,
and collecting live Nebius validation evidence.

## Prerequisites

- `~/.npa/config.yaml` — project alias, registry, `kubernetes` block (`cluster_name`, `gpu_profile`)
- `~/.npa/credentials.yaml` — Nebius IAM (registry push/pull)
- Operator host: Docker, `nebius` CLI, `sky` (for GPU/container smokes)
- Optional: `NPA_NEBIUS_PROFILE=agent-sa` for registry write on shared VMs
- SkyPilot must have Kubernetes enabled for the target context. The
  `solution-smoke` runner runs `sky check kubernetes` automatically before
  submission; if debugging manually, run it with the resolved kubeconfig/context
  before `sky jobs launch`.
  Container/solution smokes use direct `sky launch --down` by default because
  the managed-jobs controller can retain a stale enabled-infra cache for newly
  synced Kubernetes contexts.

Project resolution: `npa.workflows.byof.live.resolve_byof_project()` — never hardcode VM paths.

## Base Image Profiles

| Profile | Flag | Default base | Use when |
| --- | --- | --- | --- |
| `ubuntu` | `--base-profile ubuntu` | `ubuntu:22.04` | Generic OSS repos; containerize + registry smoke |
| `isaac-lab` | `--base-profile isaac-lab` | NPA Isaac Lab image | LeIsaac RL, datagen, Isaac tasks |
| Custom | `--base-image <ref>` | (explicit) | Customer base images; overrides profile |

Override Ubuntu default: `NPA_BYOF_UBUNTU_BASE_IMAGE` or `--base-image ubuntu:24.04`.

## Operator Entrypoint

```bash
npa/.venv/bin/python npa/scripts/run_byof_repo.py \
  --repo-url <repo-url> \
  --repo-ref <ref> \
  --base-profile ubuntu \
  --registry <resolved-from-config> \
  --project <project-alias> \
  --workload container-verify \
  --run-id byof-<stamp> \
  --cleanup
```

Workloads:

| Workload | Base profile | SkyPilot YAML (rtxpro) |
| --- | --- | --- |
| `container-verify` | `ubuntu` or any | `byof-container-smoke-rtxpro.yaml` |
| `solution-smoke` | `ubuntu` or custom | `byof-container-smoke-rtxpro.yaml` with `--smoke-command`, `--solution-name`, `--capability-name`, and `--smoke-artifact-name` |
| `rl-train` | `isaac-lab` | `isaac-lab-rl-train-rtxpro-smoke.yaml` |
| `datagen` | `isaac-lab` | `byof-datagen-rtxpro-smoke.yaml` |

Container layout: OSS repo cloned to `/opt/byof` + `npa_source_metadata.json`.

## Agent Chat Flow (`onboard_solution`)

1. **Contract** — register `workbench.byof.repo` (already in catalog); draft `byof` workflow via chat or:
   ```bash
   npa/.venv/bin/npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/byof.yaml --json
   ```
2. **Containerize** — `run_byof_repo.py` with `--base-profile ubuntu` and `--skip-run` for build-only.
3. **Deploy + test** — `--workload container-verify` (Ubuntu) or `--workload rl-train` / `datagen` (Isaac).
   For registry candidates that have documented upstream commands, use
   `--workload solution-smoke --build-command <install> --smoke-command <smoke>`
   with `--solution-name`, `--capability-name`, and
   `--smoke-artifact-name`. The smoke must create the named artifact under
   `$NPA_SMOKE_OUTPUT_DIR`; import-only checks are not enough.
4. **Registry-ready gate** — if the operator asks to add the OSS project to the
   NPA registry/catalog, follow `oss-solution-registry-onboard`; do not claim
   readiness from build-only or generic import checks.

Agent must return **grounded** markdown with `run_byof_repo.py`, `<repo-url>`, and base-image guidance —
not raw `GET /api/...` paths.

## Validation Repos (live tests)

| Tier | Repo | Profile | Workload |
| --- | --- | --- | --- |
| Ubuntu OSS smoke | `https://github.com/githubtraining/hellogitworld.git` `master` | `ubuntu` | `container-verify` |
| LeIsaac sim | `https://github.com/LightwheelAI/leisaac.git` `main` | `isaac-lab` | `datagen` or `rl-train` |

Override: `NPA_BYOF_REPO_URL`, `NPA_BYOF_REPO_REF`, `NPA_BYOF_BASE_PROFILE`.

## Live Verify

```bash
export NPA_E2E_PROJECT=rtxpro
export NPA_BYOF_LIVE_PIPELINE=1
bash npa/scripts/verify_byof_onboarding_live.sh
```

Ubuntu OSS agent + build + deploy smoke:

```bash
export NPA_E2E_PROJECT=rtxpro
export NPA_BYOF_REPO_URL=https://github.com/githubtraining/hellogitworld.git
export NPA_BYOF_REPO_REF=master
export NPA_BYOF_BASE_PROFILE=ubuntu
export NPA_AGENT_LIVE=1
export NPA_BYOF_LIVE_CONTAINER=1
export NPA_BYOF_LIVE_GPU=1
npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_onboarding_live_e2e.py -q \
  -k "live_agent_oss_repo_onboard or live_byof_ubuntu_oss" --timeout=7200
```

## Source Layout

| Path | Role |
| --- | --- |
| `npa/scripts/run_byof_repo.py` | Build/push + workload dispatch |
| `npa/workflows/byof/live.py` | Project/kubeconfig/YAML resolution |
| `npa/workflows/workbench/npa-workflows/byof.yaml` | Golden workflow spec |
| `npa/src/npa/cli/agent_chat.py` | `onboard_solution` intent |
| `skills/tools/npa-agent/SKILL.md` | Agent VM bootstrap + API reference |

## Gotchas

- Merge does **not** push images — build happens at operator `run_byof_repo.py` time.
- Ubuntu images cannot run LeIsaac datagen; use `isaac-lab` profile for sim workloads.
- GPU smokes may return `FAILED_PRECHECKS` when cluster capacity is tight; container tier is the gate for Ubuntu BYOF.
- A successful BYOF build is not sufficient for registry/catalog admission; test
  the documented upstream capabilities on smoke and live Nebius paths first.
