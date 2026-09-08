---
name: robocasa
description: Use to run RoboCasa kitchen-task simulation as a first-class NPA workbench tool — Gymnasium task registration, kitchen asset availability, headless EGL environment reset, and random rollouts with video artifacts, through the npa-robocasa service.
---

# RoboCasa (kitchen-task simulation)

The first-class workbench integration provides a `npa-robocasa` FastAPI service,
CLI, SDK, and workflow toolRefs. Accepted RoboCasa BYOF results do not qualify
this separate service image or a newly built digest.

The image pins RoboCasa source `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`,
robosuite `85abee228d1c43ab1939bce33028099945d453b4`, MuJoCo 3.3.1, Gymnasium
0.29.1, and the CUDA 13.0.2 base digest. Source archives and Python dependencies
are hash checked; the complete NPA package supplies the service's storage and
CLI imports. The cuDNN wheel is filtered before its installation layer exists,
and the saved-image verifier checks every layer against exact runtime bytes.

RoboCasa and robosuite code use MIT licenses with separate MuJoCo Apache-2.0
notices. Source-bundled RoboCasa assets use CC-BY-4.0. Bundled Panda and Sawyer
descriptions retain Apache-2.0 and BSD notices. Large kitchen archives download
at runtime from pinned Hugging Face revisions under the operator's access and
provider-specific terms. Existing caches are reused; provenance reports the
requested revisions without claiming that preexisting cache contents match.

Public development publication requires exact-image licensing, security, and
byte gates; real kitchen trajectory export and decoded video establish only
that demonstrated capability on the tested GPU. Supported promotion requires
the applicable full validation matrix. LeRobot 0.5.1 is an explicit policy-only
subset with documented dependency overrides: do not claim a clean `pip check`,
ACT training, or ACT evaluation from kitchen rollout evidence. See the
[source license](https://github.com/robocasa/robocasa/blob/8f3c96ec8d1bfcd8126cad2bca887da98d30e997/LICENSE),
[pinned robosuite license](https://github.com/ARISE-Initiative/robosuite/blob/85abee228d1c43ab1939bce33028099945d453b4/LICENSE),
[image attribution and dependency boundaries](../../../npa/docker/workbench/robocasa/REDISTRIBUTION.md).

## Capabilities

| Capability | What it proves |
| --- | --- |
| `kitchen_task_registration` | Gymnasium `robocasa/PickPlaceCounterToCabinet` is registered |
| `kitchen_asset_availability` | The kitchen assets root exists and is populated |
| `kitchen_egl_env_reset` | A headless `MUJOCO_GL=egl` env creates and resets |
| `kitchen_random_rollout` | A real random rollout runs and writes a video artifact |
| `kitchen_trajectory_export` | Real camera, joint, and action arrays plus decoded MP4 support dataset conversion |
| `kitchen_policy_eval` | A supplied ACT policy runs in the simulator; requires separate runtime qualification |

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
  network access to fetch the large textures, fixtures, and objects archives.
- **A random rollout is not manipulation success.** Inspect the finite action
  and joint arrays, changing camera frames, decoded MP4, and actual task reward.
- **Video failure fails the capability.** The pinned PyAV encoder must produce
  a real MP4; absent frames or encoding errors cannot become successful exports.
- **`--service` needs both a reachable `--endpoint` and the token variable set.**
  A missing token presents as an auth failure from the endpoint, not as a CLI
  validation error.
- **Gymnasium must stay pinned at 0.29.1.** Newer wrappers drop `__getattr__`
  and break `env.sim` video capture.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
