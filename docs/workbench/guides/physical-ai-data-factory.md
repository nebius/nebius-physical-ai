# NVIDIA Physical AI Data Factory on NPA (no OSMO)

This guide runs the **NVIDIA Physical AI Data Factory** blueprint natively on
Nebius + SkyPilot. It is delivered as a single npa.workflow spec
(`npa/workflows/physical-ai-data-factory.yaml`) that
**composes existing workbench tools** — there is no OSMO orchestrator and no new
"data factory" tool. SkyPilot is the sole orchestrator; every stage hands off
through one S3 run prefix so input, intermediate, and output artifacts are all
viewable in the NPA agent artifact browser.

> **Want the from-zero runbook?** See
> [physical-ai-data-factory-deploy.md](physical-ai-data-factory-deploy.md) for a
> copy-paste deploy guide: install `npa`, set credentials/config, deploy the NPA
> agent, submit this blueprint (including multi-GPU fan-out and real FiftyOne
> curation), and view results in the agent.

## Blueprint mapping

NVIDIA blueprint (OSMO) → NPA stage (toolRef / run):

| NVIDIA stage | NPA state | Tool | Runtime |
| --- | --- | --- | --- |
| Stage 1 Config Generation | `generate-configs` | `run.shell` (sample appearance-only variables) | CPU |
| Stage 2a Understand & Annotate | `annotate-original` | `workbench.token_factory.caption` | Token Factory (zero-GPU) |
| Stage 2b Augment & Multiply | `augment` | `workbench.cosmos2.transfer` | GPU (Cosmos Transfer 2.5) |
| Evaluate & Validate | `grade` loop (`attribute-verify` + `quality-gate`) | `workbench.vlm_eval.run` + `workbench.sim2real.write_decision` | Token Factory + CPU |
| Stage 3 Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Stage 4 Curation | `curate` | `workbench.fiftyone.launch_app` | CPU |
| Finalize | `finalize` | `workbench.sim2real.finalize` | CPU |

**Model roles** (verified available on Nebius Token Factory):

- VLM captioning + attribute verification: `Qwen/Qwen2.5-VL-72B-Instruct`
- Cosmos-family reasoning critic: `nvidia/Cosmos3-Super-Reasoner`
- Prompt / MCQ LLM: `meta-llama/Llama-3.3-70B-Instruct`

Cosmos Transfer 2.5 is **not** a Token Factory model — it is the GPU diffusion
augmentation engine, run via the `cosmos2.transfer` tool (`--execute` on an
`npa-cosmos2-transfer` image / GPU). The classical structural "hallucination"
check is a CPU checker folded into the grade gate; the model-based attribute
verification is `vlm_eval`.

> **Config → augment MULTIPLY.** The `augment` stage receives the Config-Gen
> manifest via `--configs-uri` and runs **one Cosmos Transfer 2.5 inference per
> sampled combo**: each combo's prompt drives a distinct appearance, published as
> its own per-clip dir (`cosmos_augmented/<clip>/`) with its own `metadata.json`
> `variables` (which drives that clip's Rerun label). So a config with N
> augmentations produces **N scenario variants**, recorded via `variant_count` /
> `multiply_mode` in the augment manifest, curation, and finalize reports.
> Optionally condition each variant on the run's real input clip
> (`NPA_COSMOS_CONDITION_ON_INPUT=1`) so the geometry/motion is preserved while
> only the appearance changes (edge control computed on-the-fly).

## Runtime placement

- **Token Factory (zero-GPU, hosted):** captioning, attribute verification.
- **GPU (Nebius Managed K8s):** Cosmos Transfer 2.5 augmentation only.
- **CPU:** config sampling, structural check, curation, finalize.

## Validate / plan / render

```bash
SPEC=npa/workflows/physical-ai-data-factory.yaml
npa workbench workflow validate-spec "$SPEC" --json
# --var bucket= is what ties the plan to your storage; without it the spec's
# `example-bucket` placeholder is planned (plan-spec warns when that happens).
npa workbench workflow plan-spec "$SPEC" --run-id demo \
  --assume-decision promote_checkpoint --var bucket=<your-bucket> --json
# Render the serial SkyPilot YAML without launching (needs NPA_SRC_S3_URI or --image
# for the CPU tool steps, same as the other Token Factory specs):
NPA_SRC_S3_URI=s3://<your-bucket>/npa-src/npa/ \
  npa workbench workflow submit "$SPEC" --run-id demo --var bucket=<your-bucket> \
    --assume-decision promote_checkpoint --plan-only
```

## Submit (real run)

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-bucket> \
  --stage-src \
  --assume-decision promote_checkpoint \
  --infra k8s/<your-kube-context> \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

`--var bucket=` points `config.bucket` at the artifact bucket your NPA agent
reads; without it the run uses the spec's `example-bucket` placeholder.
`--stage-src` publishes the local `npa` package to
`s3://<your-bucket>/npa-src/npa/` so the CPU/Token-Factory steps (which run on
SkyPilot's default image) can install it — equivalently, run
`npa workbench workflow stage-src --bucket <your-bucket>` once and export
`NPA_SRC_S3_URI`, or pin `--image`. Submit checks all of this up front and
prints anything still missing in a single list; the
[deploy runbook](physical-ai-data-factory-deploy.md) has the full ordered
quickstart (`configure` → `provision-if-absent` → `skypilot bootstrap` →
`stage-src` → submit).

> **Stage input frames first (required).** Before you submit, upload **PNG/JPEG
> frames** (8–16; only the first `config.max_images`, default 8, are captioned) to
> `s3://<bucket>/physical-ai-data-factory/<run-id>/input/`. The first stage,
> `annotate-original`, captions image frames and **fails fast on an empty `input/`**
> (`No images found …/input/`), so the later augment → grade → curate → visualize →
> finalize stages never run. This holds **even for the default stock-Cosmos
> augment**: the augment *video* is the bundled Cosmos example, but captioning
> still needs real image files — a `.mp4` alone is not enough. Add an optional
> `video_0.mp4` (720p–1080p H.264/H.265, 5–15 s) for the Rerun viz and the
> `NPA_COSMOS_CONDITION_ON_INPUT=1` path. Copy-paste `aws s3 cp` staging plus
> `ffmpeg` extract/synthesize one-liners are in the
> [deploy runbook](physical-ai-data-factory-deploy.md) ("Stage input first").

## S3 artifact layout (agent-viewable)

```
s3://<bucket>/physical-ai-data-factory/<run-id>/
  input/               # source clips (.mp4) + frames (.png)   -> video / image
  configs/             # Stage 1 sampled augmentation manifest  -> json
  labeled_original/    # Stage 2a VLM captions                  -> json
  cosmos_augmented/    # Stage 2b augmented clips + metadata    -> video / json
  grade/               # attribute-verify report + decision     -> json
  labeled_augmented/   # Stage 3 VLM captions on augmented      -> json
  curation/            # Stage 4 curation report                -> json
  reports/sim2real.rrd # Rerun recording (input+augmented+captions) -> rerun
  reports/final.json   # finalize summary                       -> json
```

The `visualize` stage builds `reports/sim2real.rrd` from the run's input +
augmented frames and captions (via `npa.workflows.data_factory_viz.build_run_rrd`)
so the run renders in the NPA agent's **embedded Rerun viewer** — the agent
prefers `reports/sim2real.rrd`, so selecting the run and loading it (or clicking
the `.rrd` in the artifact browser) shows it in the Rerun panel.

## View input / intermediate / output in the NPA agent

The agent discovers runs from its artifact bucket. If the agent's base prefix is
`checkpoints`, place the run under `checkpoints/physical-ai-data-factory/<run-id>/`
(or pass the matching discovery prefix). Then:

```bash
# discover runs
GET /api/artifacts/runs?prefix=physical-ai-data-factory
# list a run's artifacts (render hints: video / image / json / text)
GET /api/artifacts/run/<run-id>?prefix=physical-ai-data-factory
# load one artifact into the viewer
POST /api/sim-viz/load-artifact  {"s3_uri":"s3://<bucket>/.../input/video_0.mp4"}
```

Input clips render as `video`, extracted frames as `image`, and every stage's
labels/reports as `json` — so the full input → intermediate → output flow is
browsable in the agent.

> **Voxel51 and the full Rerun recording only appear once the run gets past
> annotate → augment → curate → visualize.** With an empty `input/` the run stops
> at `annotate-original` (only `configs/manifest.json` is written), so there is no
> `cosmos_augmented/`, no `curation/report.json` for the Voxel51 tab, and no
> `reports/sim2real.rrd` for the Rerun panel. Stage input frames first (see the
> callout above) so the pipeline reaches curate/visualize and those panels
> populate.
