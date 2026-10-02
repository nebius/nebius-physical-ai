# Start a BEHAVIOR development evaluation

Use Workbench to prepare one task, run the prescribed ten development cases on
a Nebius rendering GPU, and inspect the original metrics and videos. This is
the short entry point; the [campaign guide](behavior-campaign.md) covers durable
case recovery, training panels and policy comparisons.

The `npa workbench workflow challenge` commands prepare the existing
[evaluation workflow](../../workflows/testing/behavior-challenge-eval.yaml).
They support **BEHAVIOR DEV only**. A prepared kit is not a measured result,
GPU readiness proof or competition submission.

## Before you start

Install NPA and configure your project using the
[first-run procedure](../../skills/workflows/first-run-setup/SKILL.md).
Keep the configuration and generated kits outside Git. You need:

- A saved NPA project alias, its storage credentials, an owned S3 prefix and a
  Kubernetes context belonging to that project.
- An operator-prepared simulator image pinned by digest, licensed assets in an
  existing PVC, and a supported rendering GPU such as L40S. Use Workbench GPU
  discovery to select the cluster's actual accelerator name. Policy training
  and serving may use separate GPUs.
- A running policy service configured for the selected task, its exact
  checkpoint archive SHA-256, and a UTF-8 runbook explaining how to start it.
  Include source revision, normalization assets and serving arguments.

The helper does not install OmniGibson, build the image, acquire licensed
payloads, start a policy or accept third-party terms. Follow
[runtime and policy preparation](behavior-challenge.md#prepare-the-licensed-runtime-and-policy)
and [simulator startup qualification](behavior-challenge.md#qualify-a-writable-simulator-startup)
first. The runtime must use the same evaluator revision as the local source.

## 1. Create one setup file

Run from a private working directory. `init` requires a new directory and leaves
existing files untouched:

```bash
npa workbench workflow challenge init --directory ./behavior-setup
git clone --no-checkout https://github.com/StanfordVL/BEHAVIOR-1K ./behavior-setup/upstream
git -C ./behavior-setup/upstream checkout 6cbf70b075816096e9be53958780769f3264d25d
```

Edit `behavior-setup/setup.yaml`. Empty fields are intentional: they must name
your actual resources. Put the policy runbook in this directory as `policy.md`.
Local file paths resolve relative to `setup.yaml`, regardless of your shell's
working directory.

| Field | Meaning / default |
| --- | --- |
| `schema_version`, `challenge` | Keep `npa.challenge-setup/v1` and `behavior`. |
| `project` | Required saved NPA project alias. |
| `infra` | Required `k8s/<your-context>`. |
| `artifact_root` | Required private `s3://<bucket>/<prefix>`; run IDs are appended. |
| `task`, `split` | Official task ID; starts with `turning_on_radio`. Keep `development`. |
| `source.checkout` | Set to `upstream` for the commands above. |
| `source.revision` | Defaults to the supported BEHAVIOR 3.9.3 commit above. |
| `runtime.image` | Required simulator image with full `@sha256:` digest. |
| `runtime.assets_claim` | Required existing asset PVC on the selected cluster. |
| `runtime.accelerator` | Required advertised rendering GPU name in `NAME:1` form. |
| `runtime.upstream_root` | Evaluator checkout inside the image; `/opt/BEHAVIOR-1K`. |
| `runtime.evaluator_python` | Simulator interpreter; `/opt/conda/envs/behavior/bin/python`. |
| `runtime.data_root` | Asset mount inside the worker; `/data/behavior`. Changing it also changes the PVC mount. |
| `policy.host`, `policy.port` | Required hostname reachable from the worker; port defaults to `8000`. |
| `policy.checkpoint_sha256` | Required SHA-256 of the exact served checkpoint archive. |
| `policy.runbook` | Set to `policy.md`, or another readable UTF-8 file. |

Credentials belong in NPA's private project configuration, never in this setup
or the policy runbook. The helper records checkpoint identity as
`operator-declared`: it cannot prove which weights a remote service loaded.

## 2. Check and prepare

```bash
npa workbench workflow challenge check --config ./behavior-setup/setup.yaml
npa workbench workflow challenge prepare \
  --config ./behavior-setup/setup.yaml \
  --directory ./radio-dev-kit \
  --run-id radio-dev-001 \
  --publish
```

`check` is local: it checks configuration, the exact clean evaluator checkout,
the official task registry and a nonempty policy runbook. Success means
`ready-to-prepare`; `gpu_readiness` remains `not-checked`. It does not authenticate
to the project or contact the policy. A failure returns one JSON document with
field-specific issues and exit code 1.

`prepare` validates and expands the shipped workflow, fixes indices 10–19
(instances 311–320), and writes a new owner-only directory:

| File | Purpose |
| --- | --- |
| `recipe.json`, `policy.md` | Exact evaluator inputs. |
| `plan.json` | Prescribed cases and reporting eligibility. |
| `workflow.yaml`, `execution-plan.json` | Runnable workflow and expanded command plan. |
| `manifest.json` | Source revision, image/checkpoint identity, template and input digests. |
| `commands.json`, `RUN.md` | Standard Workbench commands and result/recovery guidance. |

`--publish` verifies the selected project's storage scope and ownership, probes
the exact input prefix, writes only the recipe and runbook, and reads both back.
It never overwrites different existing input bytes. The probe may retain a
small diagnostic object, following the standard execution preflight behavior.
Without `--publish`, preparation makes no remote calls and uploads nothing.
No variant of this command submits a GPU job.

If publication fails, retain the local kit. After repairing access, repeat
`prepare --publish` with a **new local directory and the same run ID and inputs**.
Identical remote inputs are accepted. Do not submit until publication succeeds.
For a new experiment, choose a new run ID before preparing it.

## 3. Run and read the result

Follow the exact commands in `RUN.md`: credential, image and placement checks,
submission preflight, then `workflow submit --runtime`. Image preflight may
launch a capability probe; inspect the generated commands before executing them.
These checks complement the runtime and policy qualification above.
Use the generated `status`, `logs --stage evaluate` and `artifacts` commands to
inspect progress and locate evidence.
The observation commands include the exact durable state URI so they can locate
the run independently of local submission receipts.

The result prefix is `<artifact_root>/<run-id>/results/`. In `summary.json`,
require `complete: true` and `completed: 10` before comparing a full panel.
`evaluated_mean_q` measures task progress; count per-case `success` separately
and inspect the original videos. A partial panel's mean is not a complete
result. DEV results are not reporting scores and this kit creates no submission
ZIP. Earlier measured results and their limits are in the
[campaign status](behavior-campaign.md#scope-and-validation-status).

The basic evaluator does not resume a partial ten-case run. Preserve its
original prefix and worker evidence after failure; do not restart failed cases
under a new identity. Use the campaign path from the outset when you need
durable per-case recovery. It does not automatically import this kit's results.
Use the generated cancellation command before removing an owned controller or
worker; retain shared assets and checkpoints.

## SDK and other challenges

The same implementation is available through
`npa.sdk.workbench.workflow_challenge`: `initialize(Path(...))`,
`check_setup(Path(...))`, and `prepare(config_path, directory, run_id, publish=False)`.
CLI commands emit JSON by default and accept `--output-format json`.

A second challenge needs its own rules, task/split validator, runtime template,
artifact contract and independently reproduced result. Reuse the standard
workflow submission, monitoring and storage paths. This first implementation
does not claim support for another evaluator or provide a generic adapter API.
