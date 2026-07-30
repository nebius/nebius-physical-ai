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
| Stage 2b Augment & Multiply | `augment` | `workbench.cosmos2.transfer_execute` | GPU (Cosmos Transfer 2.5) |
| Evaluate & Validate | `grade` loop (`evaluate` + `quality-gate`) | `workbench.cosmos_evaluator.evaluate` + `data_factory_stages.grade_gate` | Token Factory + CPU |
| Stage 3 Pseudo-Label Augmented | `annotate-augmented` | `npa workbench token-factory caption` (run.shell) | Token Factory |
| Stage 4a Curation | `cosmos-curate` | `workbench.cosmos_curate.curate` | CPU |
| Stage 4b Curation review | `curate` | `data_factory_stages.curate` (FiftyOne Brain) | CPU |
| Visualize | `visualize` | `data_factory_viz.build_run_rrd` | CPU |
| Finalize | `finalize` | `data_factory_stages.finalize` | CPU |

Three NVIDIA components in that table are the real open-source projects:

- **Cosmos Transfer 2.5** augments (GPU diffusion, `npa-cosmos2-transfer` image).
  It is **not** a Token Factory model.
- **[Cosmos Evaluator](https://github.com/nvidia-cosmos/cosmos-evaluator)**
  (Apache-2.0) grades. The `evaluate` stage runs two of upstream's checks per
  variant: *attribute verification* (an LLM writes one multiple-choice question
  per sampled attribute, a VLM answers it from a frame — both on Token Factory,
  since upstream drives them through a configurable OpenAI-compatible endpoint)
  and *hallucination* (per-frame dynamic-mask comparison against the source clip,
  CPU only). The hallucination score only feeds the run score for
  input-conditioned variants; otherwise the clips are different scenes and it stays
  informational.
- **[Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate)**
  (Apache-2.0) curates. The `cosmos-curate` stage drives upstream's own stages —
  `VideoDownloader` → `FixedStrideExtractorStage` → `ClipTranscodingStage` →
  `MotionVectorDecodeStage` + `MotionFilterStage` → `ClipWriterStage` — and writes
  upstream's canonical `clips/` + `metas/v0/` tree with real per-clip motion
  scores. FiftyOne then reviews that set.

See `skills/NOTICE-NVIDIA-COSMOS-OSS` for exactly which upstream code runs and
where NPA substitutes its own endpoint.

**Model roles** (verified available on Nebius Token Factory):

- VLM captioning + the evaluator's attribute answering: `Qwen/Qwen2.5-VL-72B-Instruct`
- Cosmos-family reasoning critic: `nvidia/Cosmos3-Super-Reasoner`
- Prompt / MCQ LLM: `meta-llama/Llama-3.3-70B-Instruct`

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

- **Token Factory (zero-GPU, hosted):** captioning, and the Cosmos Evaluator
  attribute-verification check's LLM + VLM calls.
- **GPU (Nebius Managed K8s):** Cosmos Transfer 2.5 augmentation only.
- **CPU:** config sampling, the evaluator's hallucination check, Cosmos Curator
  curation, FiftyOne review, visualize, finalize.

The `cosmos-curate` stage needs the `npa-cosmos-curate` image, which bakes a
pinned upstream checkout and a conda-forge ffmpeg carrying `libopenh264` —
upstream's transcoding stage accepts only `libopenh264` or `h264_nvenc`, and
Debian/Ubuntu ffmpeg builds have neither. Without it the stage records
`engine: unavailable` plus the reason and the FiftyOne review still runs. Check
what an environment resolves to with:

```bash
npa workbench cosmos-evaluator engine --output text
npa workbench cosmos-curate    engine --output text
```

## Validate / plan / render

```bash
SPEC=npa/workflows/physical-ai-data-factory.yaml
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" --run-id demo --assume-decision promote_checkpoint --json
# Render the serial SkyPilot YAML without launching (needs NPA_SRC_S3_URI or --image
# for the CPU tool steps, same as the other Token Factory specs):
NPA_SRC_S3_URI=s3://<bucket>/npa-src/ \
  npa workbench workflow submit "$SPEC" --run-id demo --assume-decision promote_checkpoint --plan-only
```

## Submit (real run)

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-%Y%m%dt%H%M%sz)" \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

Set `config.bucket` (via `--var bucket=<your-bucket>`) to the artifact bucket
your NPA agent reads. Input source videos/frames go under
`s3://<bucket>/physical-ai-data-factory/<run-id>/input/` (flat `.mp4` H.264/H.265
clips, 720p–1080p, 5–15 s, plus extracted `.png` frames for the VLM stages).

## S3 artifact layout (agent-viewable)

```
s3://<bucket>/physical-ai-data-factory/<run-id>/
  input/               # source clips (.mp4) + frames (.png)   -> video / image
  configs/             # Stage 1 sampled augmentation manifest  -> json
  labeled_original/    # Stage 2a VLM captions                  -> json
  cosmos_augmented/    # Stage 2b augmented clips + metadata    -> video / json
  grade/               # Cosmos Evaluator report + decision     -> json
  labeled_augmented/   # Stage 3 VLM captions on augmented      -> json
  curation/
    cosmos_curator/    # Cosmos Curator output tree             -> video / json
      clips/           #   transcoded clips
      metas/v0/        #   per-clip metadata + motion scores
      processed_videos/
    cosmos_curator.json# Cosmos Curator run summary             -> json
    report.json        # FiftyOne review report (merges the above) -> json
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
