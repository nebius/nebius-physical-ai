# Composing Nebius cloud + Token Factory pipelines

This guide shows how to **compose** NPA workbench tools into pipelines that use
both **Nebius AI Cloud compute** (serverless GPU Jobs, Managed Kubernetes, or a
VM) **and** **Nebius Token Factory** hosted inference (zero-GPU text / vision /
reasoning). It covers getting both tokens, the one composition contract, the
reusable building blocks, and two copy-paste recipes. The four shipped combo
workflows are worked examples of exactly this pattern.

## The mental model

Every combo is the same shape:

```
[ Nebius compute stage ]  --writes-->  [ S3 ]  --reads-->  [ Token Factory stage ]
   GPU / VM / k8s                       artifacts             hosted, CPU-only
   (lerobot, genesis, ...)          (videos, logs,        (caption / generate /
                                     checkpoints,          reason / vlm-eval)
                                     scene images)
```

- The **compute stage** does the expensive GPU/sim work and uploads artifacts to
  S3. Use any workbench tool that runs on Nebius (`lerobot train`,
  `lerobot eval`, `genesis`, `isaac-lab`, ...).
- The **Token Factory stage** is **zero-GPU**: it reads those artifacts from S3
  and calls the hosted API (`token-factory caption|generate|reason`, or
  `vlm-eval --backend api`). No GPU server, no vLLM.
- The **glue** is S3 URIs and two credentials. That is the entire contract — get
  those right and any two stages compose.

## Step 1 — get your two tokens

You need **two independent credentials**. They are not the same key.

### A. Nebius AI Cloud token (for compute + storage)

This authorizes serverless Jobs, Kubernetes, VMs, and S3. NPA reads it through
the standard Nebius CLI profile plus object-storage keys in
`~/.npa/credentials.yaml`.

```bash
# 1. Create / refresh a Nebius CLI profile (opens browser SSO the first time).
nebius profile create
nebius iam get-access-token >/dev/null   # proves the profile works

# 2. Bootstrap NPA's config + credentials interactively.
npa configure
```

`npa configure` writes machine config to `~/.npa/config.yaml` and secrets to
`~/.npa/credentials.yaml`. For compute + storage you need, in
`~/.npa/credentials.yaml`:

```yaml
storage:
  AWS_ACCESS_KEY_ID: <nebius-s3-access-key>
  AWS_SECRET_ACCESS_KEY: <nebius-s3-secret-key>
  AWS_ENDPOINT_URL: https://storage.eu-north1.nebius.cloud
```

Your **project ID** (`project-...`) comes from the Nebius console / your profile;
pass it to serverless runs with `--project-id` (never hardcode it in a committed
file). S3 access keys are minted in the Nebius console under your storage
service account.

> Tip: the runner scripts below export these into the environment for you via
> `apply_shared_credential_env`. SkyPilot YAMLs receive them as `--secret`s at
> launch.

### B. Token Factory token (for hosted inference)

This is a separate console and a separate key (`NEBIUS_TOKEN_FACTORY_KEY`). Full
walkthrough: [token-factory.md](./token-factory.md). The 2-minute version:

1. Sign in at <https://tokenfactory.nebius.com/> and make sure the project has
   credit.
2. **API keys → Create API key**, copy it once.
3. Give it to NPA (any one):

```bash
npa configure                       # answer the NEBIUS_TOKEN_FACTORY_KEY prompt, or
export NEBIUS_TOKEN_FACTORY_KEY=nebius_xxx     # env var for CI / one-off shells, or
# put it under tokens: in ~/.npa/credentials.yaml
```

4. Verify it authenticates and see the served catalog:

```bash
npa workbench token-factory verify
npa workbench token-factory models   # confirms e.g. nvidia/Cosmos3-Super-Reasoner
```

### Both tokens, one check

```bash
nebius iam get-access-token >/dev/null && echo "AI Cloud: ok"
npa workbench token-factory verify
```

If both pass, you can run any combo below.

## Step 2 — the building blocks

**Nebius compute stages** (each writes artifacts to an `--output-path`/S3 URI):

| Tool | What it produces |
| --- | --- |
| `npa workbench lerobot train --runtime serverless` | checkpoints + train logs/config |
| `lerobot-eval` (in a GPU SkyPilot stage) | rollout videos + `eval_info.json` |
| `npa workbench genesis ...` / `isaac-lab ...` | sim rollouts / synthetic data |

**Token Factory stages** (each reads an `--input-path`/S3 URI, calls the hosted API):

| Tool | What it does |
| --- | --- |
| `npa workbench token-factory caption` | caption images / frames → `captions.json` |
| `npa workbench token-factory generate` | batch text gen (triage, ranking, synthetic prompts) → `generations.jsonl` |
| `npa workbench token-factory reason` | scene understanding + plan with Cosmos3-Super-Reasoner → `scene_reasoning.json` |
| `npa workbench vlm-eval run --backend api` | score a rollout with a hosted VLM → eval JSON |

**Pure glue helpers** — `npa/src/npa/workflows/token_factory_combos.py` holds
infra-free logic you can reuse so your runner stays unit-testable: bounded
artifact digesting (`summarize_run_artifacts`), prompt builders
(`build_triage_prompt`, `build_ranking_prompt`, `build_sweep_design_prompt`),
URI joining (`join_uri`, `sweep_variant_output_uri`), and Nebius-safe ID/job-name
derivation (`utc_stamp`, `triage_job_name`, `sweep_variants`).

## Step 3 — pick a composition style

There are two idiomatic ways to wire stages together. Use whichever fits.

### Style 1 — Python runner script (best for fan-out, branching, local glue)

A plain script that shells out to `npa` for the GPU stage(s) and calls the
Token Factory tool functions in-process. Good when you need loops (a sweep),
conditional logic, or to download + reshape artifacts between stages. Keep all
*pure* logic in `token_factory_combos.py` and all *I/O* in the runner so the
logic stays testable. Always give the runner a `--render-only` mode that prints
the plan with **no** infrastructure, and a cheap "skip the GPU stage" mode
(e.g. `--from-output-path` / `--rank-existing`) for fast iteration.

Skeleton:

```python
from npa.workflows.token_factory_combos import join_uri, summarize_run_artifacts
from npa.workbench.token_factory import generate_text

# 1. GPU stage on Nebius (serverless) via the CLI.
subprocess.run([sys.executable, "-c", "from npa.cli.main import app_entry; app_entry()",
                "workbench", "lerobot", "train", "--runtime", "serverless",
                "--output-path", out_uri, "--output", "json"], check=True)

# 2. Token Factory stage in-process (zero-GPU).
digest = summarize_run_artifacts(local_artifacts)     # bounded, safe
generate_text(input_path=prompts_jsonl, output_path=join_uri(out_uri, "triage"))
```

Worked examples: `npa/scripts/run_tokenfactory_train_triage.py` (train→triage)
and `npa/scripts/run_tokenfactory_sim_sweep.py` (design→sweep→rank).

### Style 2 — SkyPilot serial YAML (best for a clean, hand-off pipeline on Nebius)

A multi-document YAML with `execution: serial`: one document per stage, each
with its own image and resources. The GPU stage requests `accelerators`; the
Token Factory stage omits them (CPU-only) and uses a small image. Pass `s3://`
URIs through `envs` so each stage reads exactly what the previous one wrote, and
fail fast if `NEBIUS_TOKEN_FACTORY_KEY` is unset.

```yaml
name: my-combo
execution: serial
---
name: gpu-stage
resources: { cloud: kubernetes, accelerators: H100:1 }
envs: { OUT_URI: "s3://<your-bucket-name>/<run-id>/artifacts/" }
run: |
  ... GPU work ...; aws/boto3 upload to ${OUT_URI}
---
name: tokenfactory-stage
resources: { cloud: kubernetes, cpus: 4 }   # no accelerators -> zero-GPU
envs: { OUT_URI: "s3://<your-bucket-name>/<run-id>/artifacts/" }
run: |
  [[ -z "${NEBIUS_TOKEN_FACTORY_KEY:-}" ]] && { echo "NEBIUS_TOKEN_FACTORY_KEY required"; exit 1; }
  npa workbench vlm-eval run --input-path "${OUT_URI}" --backend api ...
```

Launch with both credential sets as secrets:

```bash
sky jobs launch --secret NEBIUS_TOKEN_FACTORY_KEY --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY npa/src/npa/workflows/skypilot/<your>.yaml
```

Worked examples: `tokenfactory-rollout-judge.yaml` (GPU rollout → VLM judge) and
`tokenfactory-scene-to-rollout-judge.yaml` (reason → GPU rollout → VLM judge).

## Step 4 — the shipped combos (study these)

| Workflow | Compute (AI Cloud) | Token Factory | CLI | SDK | YAML |
| --- | --- | --- | --- | --- | --- |
| train-triage | serverless GPU LeRobot train | text triage report | runner | — | `tokenfactory-train-triage.yaml` (k8s) |
| sim-sweep | N serverless GPU trains (fan-out) | text design + ranking | runner | — | — (fan-out: runner only) |
| rollout-judge | k8s GPU rollout | VLM judge | `workflow submit` | `npa.workflow.submit` | `tokenfactory-rollout-judge.yaml` |
| scene-to-rollout-judge | k8s GPU rollout | reason → VLM judge | `workflow submit` | `npa.workflow.submit` | `tokenfactory-scene-to-rollout-judge.yaml` |

How to run each: [cookbooks/tokenfactory-compute-combos.md](./cookbooks/tokenfactory-compute-combos.md).

### CLI / SDK / YAML — all three interfaces

The hosted **building blocks** are exposed in all three interfaces, so any combo
stage is reachable however you drive it:

- **CLI**: `npa workbench token-factory generate|reason|caption`,
  `npa workbench vlm-eval run`, `npa workbench lerobot train`.
- **SDK**: `npa.workbench.token_factory.*`, `npa.workbench.vlm_eval.*`,
  `npa.workbench.lerobot.*` (same callables, no Typer leakage).
- **YAML**: the `tokenfactory-*.yaml` SkyPilot pipelines.

The **YAML combos** are submittable from the SDK too, via `npa.workflow.submit`:

```python
from npa import workflow

workflow.submit(
    "npa/src/npa/workflows/skypilot/tokenfactory-rollout-judge.yaml",
    run_id="rollout-judge",
    var=[
        "NPA_LEROBOT_IMAGE=cr.eu-north1.nebius.cloud/<registry>/npa-lerobot:0.5.1",
        "ROLLOUTS_URI=s3://your-bucket/tokenfactory/run-1/rollouts/",
        "JUDGE_URI=s3://your-bucket/tokenfactory/run-1/vlm-judge/",
    ],
    secret_env=["NEBIUS_TOKEN_FACTORY_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
)
```

> The two **serverless-fan-out** combos (`train-triage`, `sim-sweep`) ship as
> Python runners because they need cross-stage orchestration — await a Job,
> download artifacts, build prompts, fan out N variants — that a single serial
> SkyPilot YAML cannot express. `train-triage` also has an equivalent **k8s
> YAML** form (`tokenfactory-train-triage.yaml`) for the declarative path.

## Checklist for your own combo

- [ ] Both tokens verified (`nebius iam get-access-token`, `token-factory verify`).
- [ ] Compute stage writes to an `s3://` `--output-path` you control.
- [ ] Token Factory stage reads that same URI; no GPU/vLLM in that stage.
- [ ] No hardcoded project/registry/bucket IDs — pass via flag, `--var`, or `--secret`.
- [ ] Pure logic in `token_factory_combos.py` (unit-tested); I/O in the runner/YAML.
- [ ] Runner has `--render-only` and a cheap no-GPU iteration mode.
- [ ] `NEBIUS_TOKEN_FACTORY_KEY` checked at the start of every hosted stage.
