---
name: robocasa
description: Use to run RoboCasa kitchen-task simulation as a first-class NPA workbench tool — Gymnasium task registration, kitchen asset availability, headless EGL environment reset, and random rollouts with video artifacts, through the npa-robocasa service.
---

# RoboCasa (kitchen-task simulation)

RoboCasa is an Apache-2.0 kitchen-task simulation framework built on robosuite
and MuJoCo. This tool promotes the accepted RoboCasa BYOF candidate into a
first-class workbench tool: a dedicated `npa-robocasa` container with a FastAPI
service, CLI, SDK, and workflow toolRefs that exercise the real upstream
capabilities.

The upstream repo is `robocasa/robocasa` at the pinned `v1.0` tag, with
robosuite at a pinned commit and the exact MuJoCo 3.3.1 / Gymnasium 0.29.1 /
CUDA 12.4 closure from the live-accepted BYOF evidence. Kitchen assets
(textures, fixtures, objects) are NOT baked and download at run time under the
operator's own network access.

## Capabilities

| Capability | What it proves |
| --- | --- |
| `kitchen_task_registration` | Gymnasium `robocasa/PickPlaceCounterToCabinet` is registered |
| `kitchen_asset_availability` | The kitchen assets root exists and is populated |
| `kitchen_egl_env_reset` | A headless `MUJOCO_GL=egl` env creates and resets |
| `kitchen_random_rollout` | A real random rollout runs and writes a video artifact |

## Two execution modes

Direct (the default) runs the work from your CLI invocation. Service mode calls a
deployed Kubernetes endpoint:

```bash
npa workbench robocasa run --capability kitchen_random_rollout \
  --output-path s3://<bucket>/robocasa/<id>/ \
  --iterations 1 --num-envs 1
npa workbench robocasa run --capability kitchen_random_rollout \
  --output-path s3://<bucket>/robocasa/<id>/ --service --endpoint <url>
```

Deploy the service when you want a persistent endpoint several runs share:

```bash
npa workbench robocasa deploy \
  --project <alias> --cluster-name <name> \
  --output-path s3://<bucket>/robocasa/ \
  --gpu-type rtxpro6000 --namespace default \
  --dry-run                       # prints the manifest without applying
npa workbench robocasa deploy --project <alias> --destroy
```

`--gpu-type` is `h100`, `l40s`, `rtx6000`, or `rtxpro6000`. Auth defaults to
`token` (the token comes from the variable named by `--token-env`, default
`ROBOCASA_TOKEN`); `--insecure-no-auth` exists but should not be used. Always
`--dry-run` first and read the manifest.

## Run

```bash
npa workbench robocasa run \
  --capability kitchen_random_rollout \
  --env-id robocasa/PickPlaceCounterToCabinet \
  --output-path s3://<bucket>/robocasa/runs/<id>/ \
  --iterations 1 --num-envs 1 \
  --wait --poll-seconds 30 --timeout-seconds 3600 \
  --output json
```

`--capability` and `--output-path` are required. The legacy `--output-uri`
spelling remains as an alias. `--wait` polls `/status` until
the run completes **and fails if it does not** — without it, the command returns
as soon as the run is accepted.

## Status, system-info, list

```bash
npa workbench robocasa status --run-id <id> --service --endpoint <url>
npa workbench robocasa system-info --service --endpoint <url>
npa workbench robocasa list --service --endpoint <url>
```

`system-info` reports the RoboCasa, robosuite, MuJoCo, and Gymnasium versions,
CUDA availability, and the registered env count.

## In workflows

There are six per-capability toolRefs:
`workbench.robocasa.task_registration`, `.asset_availability`,
`.egl_env_reset`, `.random_rollout`, `.trajectory_export`, and `.policy_eval`.
The first four appear together in `robocasa-smoke.yaml`; the latter two drive
the data-policy workflow's real rollout export and held-out evaluation.

Every run uploads `result.json` and `provenance.json`. Rollout provenance names
the RoboCasa → MuJoCo execution path and hashes each generated MP4, with
machine-readable `rrd: false` and `mcap: false` fields; this tool does not emit
RRD or MCAP recordings.

## Gotchas

- **Without `--wait`, "started" is not "succeeded".** Check `status` before
  reporting a result.
- **Kitchen assets download at run time.** The first run on a fresh image needs
  network access to fetch textures, fixtures, and objects.
- **`--service` needs both a reachable `--endpoint` and the token variable set.**
  A missing token presents as an auth failure from the endpoint, not as a CLI
  validation error.
- **Gymnasium must stay pinned at 0.29.1.** Newer wrappers drop `__getattr__`
  and break `env.sim` video capture.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
