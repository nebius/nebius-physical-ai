# Internal SkyPilot task templates

**This directory is not the supported workflow catalog.** The supported,
customer-facing specs are the declarative `npa.workflow` YAMLs
(`apiVersion: npa.workflow/v0.0.1`) under
`npa/workflows/workbench/npa-workflows/`. Author and submit those; SkyPilot is
only the execution engine.

These files are internal, package-owned runtime resources: raw SkyPilot task
YAMLs that the `npa/scripts/run_*.py` wrappers and `npa.workflow` engine render
and launch. They were relocated here (out of `npa/workflows/workbench/`) so the
shown catalog is exclusively `npa.workflow` specs, while SkyPilot-only
capabilities that the engine cannot yet express (parallel sweeps, burst submit,
the trigger watch-loop, and the legacy H100 sim-to-real pipeline/loop) keep a
runnable home.

**Preferred submit path:** `npa workbench workflow submit <npa.workflow.yaml>`
plans the state graph, renders a serial SkyPilot multi-doc YAML, and submits it.
Use the raw YAMLs here only to inspect or operate the underlying SkyPilot task
directly.

**BYOF resource profiles** (the GPU solution-smoke task and the RTX PRO
`imagePullSecrets` global config) that the declarative BYOF specs and the BYOF
runner depend on live under `npa/src/npa/workflows/byof/profiles/`, alongside
this directory.

The supported first path is the Python wrapper or `npa` CLI for each workflow
because wrappers inject secrets, validate image overrides, and clean up owned
clusters.

## Run Pattern

All examples assume SkyPilot 0.12.2.

1. Configure SkyPilot for the target infrastructure and verify the GPU aliases
   used by the YAML are schedulable.

   ```bash
   sky show-gpus --infra kubernetes --all
   ```

2. Copy the YAML to a temporary path and replace only the template values in
   `envs:` and `resources.image_id`. SkyPilot 0.12.2 does not interpolate
   `${VAR}` placeholders inside `envs:`, so do not submit a file that still
   contains placeholders such as `${NPA_S3_BUCKET}` or `docker:${IMAGE}`.
   Avoid blindly running `envsubst` over the whole file because many `run:`
   blocks intentionally contain shell variables.

3. Provide S3-compatible credentials to the pod through SkyPilot secrets,
   Kubernetes secrets referenced by the cluster config, or another supported
   secret mechanism. The YAMLs expect ordinary AWS-compatible variables such as
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, and the
   workflow-specific `s3://...` inputs and outputs listed below.

4. Launch the rendered YAML.

   ```bash
   sky launch -y --infra kubernetes/<context-name> -c <cluster-name> /tmp/rendered.yaml
   ```

   YAMLs that declare `cloud: nebius` can be submitted with the corresponding
   Nebius SkyPilot infra target instead of a Kubernetes context.

5. Collect status and logs, then tear down explicitly. Do not rely on
   autodown for these workflows.

   ```bash
   sky queue <cluster-name>
   sky logs <cluster-name>
   sky down -y <cluster-name>

   while sky status --refresh | grep -q "<cluster-name>"; do
     sleep 10
   done
   ```

## Common Inputs

- `NPA_S3_BUCKET`, `S3_BUCKET`, `S3_PREFIX`, `PIPELINE_ROOT_URI`, and
  workflow-specific `*_URI` values select the S3-compatible input and output
  locations. Use a dedicated run prefix for every launch.
- `AWS_ENDPOINT_URL` or workflow-specific endpoint variables select the S3
  endpoint. Keep the endpoint configurable for BYO S3-compatible storage.
- `HF_TOKEN` is needed only for workflows that fetch a gated or private
  Hugging Face repo, or when your organization requires authenticated
  downloads for public repos.
- `NGC_API_KEY` is needed only where the YAML says NGC is required or when you
  rebuild/pull images that depend on NVIDIA NGC entitlement.
- Private registry images require the cluster image-pull secret configured by
  the operator. The raw YAMLs intentionally use placeholder registry IDs where
  the user must supply their own image.

## Per-YAML Reference

| YAML | Description | Target | S3 I/O | HF rights | NGC entitlement |
| --- | --- | --- | --- | --- | --- |
| `cosmos3-generate.yaml` | Runs a real Cosmos 3 omni-model generation (`text2image`, `image2image`, `text2video`, `image2video`, or `video2video`) in the `npa-cosmos3` image. | Kubernetes `H100:1`, `16+` CPUs, `128+` GB memory. | Reads `NPA_COSMOS3_PROMPT` and optionally `NPA_COSMOS3_INPUT_URI`; writes the artifact plus `generate.json` to `NPA_COSMOS3_OUTPUT_URI` (local-only when empty). | **Required.** The image bakes no weights, so the gated checkpoint (`NPA_COSMOS3_CHECKPOINT`, default `Cosmos3-Nano`) downloads on the node under your own HF license acceptance; pass `HF_TOKEN` as a secret env. | Optional by default; required when `NPA_COSMOS3_REQUIRE_NGC=1` or when rebuilding NGC-derived images. Registry access is needed for `NPA_COSMOS3_IMAGE`. |
| `nurec-reconstruct.yaml` | NuRec/NRE neural reconstruction: fetch a real NCore V4 capture (deriving the `rig -> world` pose edge NRE requires), train 3DGUT Gaussians into a renderable USDZ, render rig-offset novel views, build `reports/sim2real.rrd`, and finalize. Runs inside the NGC NRE container. | Kubernetes `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` (or `L40S:1`) — RT cores are required, never H100/H200. | Reads `NPA_NUREC_DATASET` from Hugging Face; writes `ncore/`, `input/`, `reconstruction/`, `novel_views/`, and `reports/` under `NPA_NUREC_RUN_URI`. | Required for the selected `NPA_NUREC_DATASET`. The default `nvidia/PhysicalAI-NuRec-PPISP` is ungated CC-BY-4.0; the `PhysicalAI-Autonomous-Vehicles*` sets are gated and need their license accepted by the token owner. | Required. `nvcr.io/nvidia/nre/nre` needs an extra entitlement (402 for a standard key); the `-ga` GA repositories are pullable with a standard `NGC_API_KEY`. |

## Standalone Launch Commands

After rendering placeholders into `/tmp/<yaml-name>.yaml`, each YAML can be
launched directly. Use stable, run-specific cluster names so cleanup is
unambiguous.

```bash
sky launch -y --infra kubernetes/<context-name> -c bdd100k-pipeline /tmp/bdd100k-pipeline.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos2-transfer /tmp/cosmos2-transfer.yaml
sky launch -y --infra kubernetes/<context-name> -c dataset-ingest-curate /tmp/dataset-ingest-curate.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos3-t2i /tmp/cosmos3-text-to-image-inference.yaml
npa burst submit-yaml npa/src/npa/burst/examples/isaac-lab-cosmos-sdg-burst-smoke.yaml --name <run-id>
sky launch -y --infra kubernetes/<context-name> -c mjlab-eval /tmp/mjlab-eval.yaml
sky launch -y --infra kubernetes/<context-name> -c retargeting /tmp/retargeting.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-loop /tmp/sim-to-real-loop.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-pipeline /tmp/sim-to-real-pipeline.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-trigger /tmp/sim-to-real-trigger.yaml
sky launch -y --infra kubernetes/<context-name> -c sim2real-actions /tmp/sim2real-actions.yaml
sky launch -y --infra kubernetes/<context-name> -c sonic-locomotion-finetuning /tmp/sonic-locomotion-finetuning.yaml
sky launch -y --infra nebius -c sonic-train-standalone /tmp/sonic-train-standalone.yaml
sky launch -y --infra kubernetes/<context-name> -c vlm-eval-benchmark /tmp/vlm-eval-benchmark.yaml
sky launch -y --infra kubernetes/<context-name> -c vlm-eval /tmp/vlm-eval.yaml
```

## Gated Hugging Face models

Many workflows pass an `HF_TOKEN` so a runtime can download model weights or
datasets from Hugging Face. A token alone is **not** enough for *gated* repos:
you must also open the repo page once while signed in with the same account and
accept its license/usage terms (NVIDIA repos may also require a request form),
or the download fails with `403 Gated`. Public repos need no acceptance and the
token is optional (it only helps avoid anonymous rate limits).

The table below lists, per workflow, the repos you must accept before the run
can fetch weights. Gated repos are marked **(gated — accept license)**.

| Workflow YAML | Hugging Face repos to accept | Notes |
| --- | --- | --- |

The self-contained Sim2Real runbook (`../sim2real/runbook.yaml`) defaults to
dual self-hosted VLM eval: `nvidia/Cosmos-Reason2-8B` and
`nvidia/Cosmos-Reason2-2B`, both **(gated — accept license)**, plus
`nvidia/Cosmos-Transfer2.5-2B` for augment. The public `lerobot/pusht` dataset
needs no HF acceptance. `nvidia/Cosmos3-Super-Reasoner` is **Token Factory only**
(not on Hugging Face); do not use it as `VLM_REASON3_MODEL` for cluster Jobs.

### Gated repos not surfaced by a workflow YAML

Each workbench tool is a containerized service that can be driven by CLI/SDK as
well as the YAMLs above, so the entrypoint does not change which repo is gated.
Two gated repos still aren't visible from the per-workflow table:

- **GR00T has no SkyPilot YAML in this directory** — it is driven only by
  `npa workbench groot` (CLI/SDK). It needs `nvidia/GR00T-N1.7-3B` **and**
  `nvidia/Cosmos-Reason2-2B`, both **gated — accept license**.
- **Driving the Cosmos tool directly** (`npa workbench cosmos ...`) defaults to a
  different repo than the `cosmos3-*` YAMLs above:
  `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` **(gated — accept license)**.

### How to accept a gated repo

1. Sign in to Hugging Face with the account whose token you set as `HF_TOKEN`.
2. Open the repo page (for example `https://huggingface.co/nvidia/GEAR-SONIC`)
   and accept the license / "Agree and access repository" prompt, completing any
   NVIDIA request form.
3. Confirm the token can reach the repo before a long run. `npa workbench cosmos
   check` and `npa workbench groot` validate gated-model access for those tools;
   for other workflows a quick `huggingface-cli download <repo> --revision main`
   smoke check works.

## Cleanup Rules

Raw SkyPilot launches are user-owned. Always keep the cluster name and run
prefix together in your run notes, cancel failed managed jobs explicitly, run
`sky down -y <cluster-name>`, and poll `sky status --refresh` until the cluster
is gone. For Nebius-backed launches, do not rely on autodown.
