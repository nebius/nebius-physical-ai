# Sim-to-Real VLM→RL Workflow — Step-by-Step User Guide

This is the hands-on guide for the sim-to-real workflow: a closed loop that
generates simulation rollouts, scores them with a vision-language model (VLM),
turns the critique into a reinforcement-learning signal, updates a policy,
evaluates on held-out environments, and emits Rerun observability — all on
Nebius GPUs via one SkyPilot workflow.

It is written so you can (1) run it in minutes, (2) edit it safely with an AI
agent, and (3) plug in your own LeRobot-based training container.

- Workflow file: `npa/workflows/workbench/sim2real/runbook.yaml`
- Reference: `npa/workflows/workbench/sim2real/README.md`
- Submit: `npa workbench workflow submit npa/workflows/workbench/sim2real/runbook.yaml`
- SDK/local debug: `npa.sdk.workbench.sim2real.run()` or `python -m npa.workflows.sim2real_loop full-loop`

---

## 0. What you need (only three credentials)

This workflow needs **exactly three** credential sources — nothing else:

| Credential | Where it goes | Used for |
| --- | --- | --- |
| **Hugging Face token** (`HF_TOKEN`) | `~/.npa/credentials.yaml` → `tokens.HF_TOKEN` | dataset + gated model pulls |
| **NGC API key** | `~/.npa/credentials.yaml` → `ngc.api_key` | NVIDIA container/base pulls |
| **Nebius credentials** | Nebius CLI profile + `~/.npa/credentials.yaml` → `storage.*` (S3 keys) | cluster + S3 |

Non-secret machine config (project/region/registry/bucket) lives in
`~/.npa/config.yaml`. You do **not** need a Token Factory `NEBIUS_API_KEY` for
this workflow — the VLM step runs the genuine Cosmos-Reason image on the
cluster, not a hosted API.

```bash
nebius profile create
npa configure        # interactive; writes ~/.npa/credentials.yaml
```

---

## 1. Prove it locally (no cloud, no GPU)

Before spending GPU, validate the pipeline shape offline:

```bash
npa/.venv/bin/python -m npa.workflows.sim2real_loop full-loop \
  --run-id local-smoke --output-dir /tmp/s2r-smoke \
  --inner-iterations 1 --rollout-count 2 --heldout-env-count 2
```

This runs the loop's CPU stages and writes artifacts under `/tmp/s2r-smoke`
without touching the cluster. Inspect `reports/` and `inner_loop/.../evidence.json`.

---

## 2. Preflight the cluster

The preflight catches the recurring blockers (S3, image pull, tokens, kube
context, GPU count) up front:

```bash
npa workbench health sim2real \
  --s3-bucket <your-bucket> \
  --s3-endpoint <your-s3-endpoint> \
  --k8s-context <your-cluster-context> \
  --k8s-kubeconfig ~/.npa/clusters/<your-cluster>/kubeconfig
```

All checks should report PASS (an optional `config` WARN for unset
`assets_uri`/`scene_spec_uri` is fine — those default to a documented stub).

---

## 3. Run the workflow

Submit the runbook YAML (preferred):

```bash
npa workbench workflow submit \
  npa/workflows/workbench/sim2real/runbook.yaml \
  --run-id pusht-demo \
  --var NPA_SIM2REAL_RUN_ID=pusht-demo \
  --var NPA_SIM2REAL_BUCKET=<your-bucket> \
  --var NPA_SIM2REAL_TRIGGER_DATASET_ID=lerobot/pusht \
  --var AWS_ENDPOINT_URL=<your-s3-endpoint>
```

Or launch with raw SkyPilot (see `npa/workflows/workbench/sim2real/README.md`).

What you get back: a task-success / held-out report, an RL-signal diversity
block, image-digest provenance proving the genuine VLM/eval images ran, and a
Rerun recording (see §6). Outputs land under `s3://<your-bucket>/sim2real/<run-id>/`.

> **Trigger on new data instead of running manually:** drop a LeRobot dataset at
> a watched S3 prefix and let the trigger launch the run:
> `npa workbench workflow trigger watch --s3-bucket <your-bucket> --s3-prefix <trigger-prefix> ...`

---

## 4. Specify a custom LeRobot container

The trainer step is a swappable seam. You can bring a container that builds on
LeRobot in **two ways**, both pure flags — no change to NPA:

### Option A — swap the trainer *image* (runs as a sibling K8s job)

```bash
# Set TRAINER_IMAGE in runbook env / --var when submitting workflow YAML.
export TRAINER_IMAGE=<your-registry>/<your-lerobot-trainer>:<tag>
```

### Option B — swap the trainer *command* (runs in-process in your base image)

Set `BYO_TRAINER_COMMAND` in the runbook env (or pass `--byo-trainer-command` via SDK
for local `sim2real_loop` runs).

**The contract your container/command must honor** (this is all it needs to do):

1. Read the VLM→RL signal batch from the path in `NPA_SIM2REAL_SIGNAL_JSON`
   (schema `npa.sim2real.rl_signal.v1`: per-step `reward`, `advantage`, and an
   action-delta `target`).
2. Run one training update on your LeRobot policy. The documented integration
   point is right after the policy forward pass and before `optimizer.step()`:
   `loss = imitation_loss + signal_loss_weight·corrective_mse − advantage·policy_logit_proxy`.
3. Write the result JSON to the path in `NPA_SIM2REAL_OUTPUT_JSON` with at least:
   `reward_head_after` (float), `policy_output_after` (list of floats),
   `policy_delta_l2` (float); optional `loss_before` / `loss_after`.

If your command fails or writes a non-conforming/empty result, the run **fails
loudly** — it never silently falls back to the reference trainer, so a green run
means your container actually ran. The run records `trainer_source=byo_command`
(or `byo_image`) in the evidence so you can prove it.

> Minimal example of a BYO trainer command:
> ```bash
> --byo-trainer-command 'python - <<PY
> import json, os
> sig = json.load(open(os.environ["NPA_SIM2REAL_SIGNAL_JSON"]))
> # ... your LeRobot update here, using sig["signals"] ...
> json.dump({"reward_head_after": 0.51, "policy_output_after": [0.01, 0.0, 0.0],
>            "policy_delta_l2": 0.004}, open(os.environ["NPA_SIM2REAL_OUTPUT_JSON"], "w"))
> PY'
> ```

You can swap the **signal converter** the same way with `--byo-signal-converter`
(reads `NPA_SIM2REAL_EVALUATION_JSON`, writes an `rl_signal.v1` JSON).

---

## 5. Edit the workflow with an AI agent

The workflow is plain YAML (`runbook.yaml`) plus one shared Python implementation
(`sim2real_loop.py`), which makes it agent-editable. To change it safely with a
Cursor/Claude agent:

1. **Point the agent at the right files and skill.** Ask it to load the
   `workflow-authoring` and `sim-to-real` skills, then edit
   `npa/workflows/workbench/sim2real/runbook.yaml` (knobs/envs) and/or
   `npa/src/npa/workflows/sim2real_loop.py` (behavior).
2. **Describe the change as config when possible.** Most adjustments are env
   values in the runbook `envs:` block — scale (`INNER_ITERATIONS`,
   `ROLLOUT_COUNT`, `HELDOUT_ENV_COUNT`), images (`VLM_IMAGE`, `EVAL_IMAGE`,
   `TRAINER_IMAGE`), backend (`--sim-backend genesis|isaac`), thresholds
   (`SUCCESS_THRESHOLD`), and the BYO seams (`BYO_TRAINER_COMMAND`,
   `BYO_SIGNAL_CONVERTER`, `BYO_RERUN_COMMAND`). Editing a value rarely needs
   code changes.
3. **Tell the agent the guardrails** (so it doesn't break the run):
   - SkyPilot 0.12.2 does **not** interpolate `${VAR}` inside the YAML `envs:`
     block — materialize literals or expand in the `run:` block.
   - Keep credentials out of the YAML; never hardcode project/registry/bucket
     IDs (they are configuration).
   - Reach GPUs via the direct-Kubernetes route, not `sky jobs launch`.
4. **Have the agent validate before you run:**
   ```bash
   npa/.venv/bin/python -m pytest npa/tests/workflows/test_sim2real_loop.py -q
   npa workbench health sim2real --checks config,coherence   # infra-free
   ```
   Then a `--mock`/local smoke (§1), then a small cluster run.

A good prompt: *"Load the sim-to-real and workflow-authoring skills. In
`runbook.yaml`, add a second outer iteration and raise held-out envs to 16, keep
all images and credentials as config, then run the unit tests and the infra-free
preflight and show me the diff."*

---

## 6. Rerun observability

The loop emits a Rerun recording after the run:

```text
s3://<your-bucket>/sim2real/<run-id>/reports/sim2real.rrd
```

It logs rollout camera frames, per-rollout VLM critique + score overlays, the
per-step reward/advantage timeseries, and held-out per-env scores. View it:

```bash
pip install rerun-sdk
rerun /path/to/sim2real.rrd
```

Toggle with `--rerun/--no-rerun` (default on) or `NPA_SIM2REAL_RERUN=0`; swap the
emitter with `--byo-rerun-command`.

---

## 7. Custom simulation assets and robots

- **Objects** (e.g. parts to sort): upload a mesh (`.obj`, `.stl`, `.glb`,
  `.ply`, `.usd`) and point `--assets-uri` / a SceneSpec at it. The mesh loads
  into the sim with content-hash provenance and no silent fallback.
- **Robot embodiment**: use the built-in Franka, or bring your own arm
  (Universal Robots / Flexiv) as an **articulated** description (URDF / MJCF /
  USD — a plain CAD/OBJ mesh has no joints and is only valid for *objects*, not
  the robot). Presets exist for Franka, UR, and Flexiv.
- **Engines**: `--sim-backend genesis` (default, no extra licensing) or
  `--sim-backend isaac` (Isaac Sim/Lab; runs on RT-core GPUs such as RTX PRO
  6000). Both support stock and custom assets.

---

## 8. Anticipated questions

**Do I need a Token Factory `NEBIUS_API_KEY`?** No. This workflow needs only HF,
NGC, and Nebius credentials. The VLM step runs the genuine Cosmos-Reason image on
the cluster.

**Genesis or Isaac?** Genesis is the default and has no service-delivery
licensing concern. Use Isaac (`--sim-backend isaac`) if you need Isaac Sim
parity; it requires RT-core GPUs (L40S / RTX PRO 6000), not H100/H200.

**My held-out scores come back low / 0 — is it broken?** Expected for an
untrained or reference policy on novel geometry. The workflow proves the
*signal* is genuine (diverse, coherent, non-degenerate) and the assets/images
actually loaded — not that an out-of-the-box policy succeeds. Train more
iterations or plug in your own trainer (§4).

**How do I confirm the genuine images really ran?** Check
`component_invocation.image_digests` (resolved pod image digest) and
`signal_diversity.coherent=true` in the run artifacts. Empty digests or a
degenerate signal fail the anti-hollow gate.

**Small metal parts (e.g. nails) — will the sim be accurate?** Thin-body
contact is physically hard in any simulator; provide real dimensions, units, and
mass/material for best fidelity. This is the limiting factor, not asset format.

**My custom trainer ran but the run failed — why?** Your command must write a
conforming JSON to `NPA_SIM2REAL_OUTPUT_JSON` (§4). A missing/empty/invalid
result fails the run on purpose (no silent fallback).

**`sky check` shows 403 / "anonymous" / missing context.** Pin the kube context:
`export KUBECONFIG=~/.npa/clusters/<cluster>/kubeconfig` and pass
`--k8s-context <cluster>`; refresh Nebius credentials if the token expired.
