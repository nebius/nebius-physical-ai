# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased

### NVIDIA Cosmos Evaluator + Cosmos Curator in the Physical AI Data Factory

- The blueprint's evaluate/validate gate now grades with the real
  [Cosmos Evaluator](https://github.com/nvidia-cosmos/cosmos-evaluator)
  (Apache-2.0) instead of the generic `vlm_eval` scorer. New
  `npa workbench cosmos-evaluator` runs two of upstream's checks per augmented
  variant: attribute verification (upstream's LLM question generation + VLM
  answering protocol, pointed at Nebius Token Factory) and the hallucination check
  (dynamic-mask motion comparison against the source clip, CPU only). The
  hallucination check delegates to upstream's own `HallucinationProcessor` when a
  checkout is importable and otherwise runs an in-repo port of the same algorithm;
  each result records which engine produced it, and the two agree to ~1e-3.
- Curation now runs the real
  [Cosmos Curator](https://github.com/nvidia-cosmos/cosmos-curate) (Apache-2.0)
  before FiftyOne review. New `npa workbench cosmos-curate` drives upstream's own
  stage classes in-process — no Ray scheduler, no GPU — and produces upstream's
  canonical `clips/` + `metas/v0/` + `processed_videos/` tree with real per-clip
  motion scores. `plan-pipeline` prints upstream's documented `video-pipeline
  split` command for operators running the full curator container.
- Both tools are containerized as mode-based workbench images, and **neither bakes
  model weights**: upstream's source is Apache-2.0 and redistributable, its weights
  are not. Each Dockerfile ends with a check that fails if a weight file is present,
  both upstream checkouts are fetched with `GIT_LFS_SKIP_SMUDGE=1`, and a guardrail
  test fails if a Dockerfile grows a build-time model download.
  - `npa-cosmos-evaluator` (456 MB, CPU) needs no weights at all — its golden eval,
    a real hallucination run against upstream's own processor, passes with
    `--network none`. Upstream's objects check would need EULA-gated Git-LFS
    weights, so it stays unwired.
  - `npa-cosmos-curate` (CPU) bakes a pinned upstream checkout, the dependency
    subset its GPU-free stages import, and a conda-forge ffmpeg carrying
    `libopenh264` (upstream's transcoding stage accepts only `libopenh264` or
    `h264_nvenc`). Its GPU stages' weights are downloaded at run time into a
    `/config/models` volume by the new `fetch-models` mode using the operator's
    `HF_TOKEN`; the model ids and pinned revisions come from upstream's own
    registry, so a pin moves only when the checkout does. Where the curator cannot
    run, the stage records `engine: unavailable` plus the reason rather than
    emitting a report that implies curation happened.
- New `npa workbench cosmos-curate fetch-models` / `models`: download the curator's
  weights with your own Hugging Face token, and report each capability's model set,
  what is already on disk, and which of `HF_TOKEN` / `NGC_API_KEY` is visible.
- **Workbench images now satisfy SkyPilot's Kubernetes provisioner.** Its per-pod
  setup runs `sudo apt install openssh-server rsync` and `service ssh restart` inside
  the image; images lacking those exited and SkyPilot reported the misleading
  `container not found ("ray-node")`, which is why operators disabled image pins
  wholesale with `NPA_E2E_CLEAR_WORKBENCH_IMAGES=1`. All three Cosmos images install
  them with passwordless sudo, and their entrypoints exec the arguments Kubernetes
  passes rather than swallowing them.
- **`npa-cosmos2-transfer`'s inference venv is usable by the non-root user it runs
  as.** `uv` had installed the interpreter under `/root` (0700), so every inference
  call failed with `Permission denied`. The image now keeps the interpreter in
  `UV_PYTHON_INSTALL_DIR`, rewrites `pyvenv.cfg` by directory prefix (uv records it
  through a version symlink, so matching the resolved path left `sys._home` in
  `/root`), and repoints the absolute symlink `cp -a` copies verbatim. The build
  asserts each of those and exercises `distutils.sysconfig`, which is the import path
  that actually broke.
- The workflow submit path refreshes the cluster's Nebius registry pull secret before
  launching, since that secret holds a short-lived IAM token and a stale one fails
  every private image pull with a 401 that SkyPilot reports as
  resources-unavailable.
- `grade_gate` reads `cosmos_evaluator.json` and still accepts the older `vlm_eval`
  report, so runs in flight keep grading. The FiftyOne review report gains a
  `cosmos_curator` block with the curator's run-level summary.
- Attribution: `skills/NOTICE-NVIDIA-COSMOS-OSS` records which upstream modules
  run, which are reimplemented, and where NPA substitutes its own endpoint.

### First-time-user cold-start fixes

- `npa configure --interactive` no longer exits 0 having written nothing. When it
  cannot proceed (no authenticated Nebius CLI profile for provisioning) or is
  cancelled mid-flow (EOF/Ctrl-C), it now exits **non-zero** with actionable
  guidance. **Behavior change:** wrappers/CI that treated a cancelled or aborted
  `npa configure` as success will now see a failure. Setup guidance and the
  interactive prompts also link where to obtain the Hugging Face and NGC keys.
- Added `npa workbench health preflight`: a PASS/WARN/FAIL/SKIP check over
  Hugging Face, NVIDIA NGC, Nebius object storage (S3), and Token Factory
  credentials (`--checks`, `--offline`, `--warn-only`, `--json`). Replaces the
  deprecated hidden `npa workbench health sim2real` in the README preflight
  guidance.
- Added `npa agent preflight` and moved the terraform-binary and SSH-key-pair
  checks (plus the Token Factory 503 warning) ahead of any cloud IAM side effects
  in `npa agent deploy`, so Route C prerequisites fail fast instead of mid-run.

### Repo hardening

- Shipped SkyPilot examples and cookbooks now use the `<your-registry-id>`
  placeholder instead of the first-party registry ID; a guardrail test keeps
  concrete registry IDs out of shipped examples.
- The base `pip install npa` is now lightweight (offline paths only); heavy
  dependencies moved to `npa[data]`, `npa[lancedb]`, `npa[viz]`, with
  `npa[full]` covering the previous monolithic install. Over-narrow version
  pins were relaxed and the previously undeclared `pydantic` dependency is
  declared.
- Added `npa.__version__`, a tag-driven Release workflow that builds and
  attaches sdist/wheel artifacts, and `docs/releasing.md`.

### Cosmos e2e

- Validated Cosmos end-to-end on Nebius via serverless `train --smoke`.
  Run ID: `w13-cosmos-e2e-20260521T233523Z`. Output artifact:
  `s3://${NPA_S3_BUCKET}/w13-cosmos-e2e/w13-cosmos-e2e-20260521T233523Z/checkpoint.json`.
- Closes the 7/8 -> 8/8 Workbench tool verification matrix gap for the
  artifact-bearing Cosmos CLI workflow.
- Known constraints remain documented in `docs/testing/e2e-serverless.md`:
  NIM/Triton are not implemented, `finetune` is a placeholder, and deferred
  visual-generation/rendering paths still depend on the container EGL/DRI gap.

- Validated Isaac Lab bring-your-own-fork path: image override (Run ID:
  `w10-byof-image-only-20260520T232650Z`) and image+command override (Run ID:
  `w10-byof-image-and-cmd-20260520T233113Z`). Worked example at
  `docs/workbench/cookbooks/byof-isaac-lab/`. Checkpoint + sentinel:
  `s3://${NPA_S3_BUCKET}/checkpoints/isaac-lab-byof/w10-byof-image-and-cmd-20260520T233113Z/`.
- Fixed Isaac Lab train command construction to call the RSL-RL training script with `--num_envs` and `--max_iterations`; added SkyPilot single-job and parallel sweep YAMLs plus the Isaac Lab RL runner.
- Added BYOVM post-deploy SSH endpoint strategy persistence and transient SSH tunnel routing for live workbench commands; fixed GR00T S3 env injection/auditing, shortened BYOVM auto public health fallback, printed normal-deploy Hugging Face access status, suppressed successful FiftyOne readiness curl noise, and made template tests cwd-independent.
- Implemented demo pre-staging CLI fixes for shared credential injection, shell-safe and Docker-safe env files, BYOVM project storage inheritance, Hugging Face gated-model validation, BYOVM SSH health fallback, live status/readiness reporting, Cosmos progress output, GR00T gated-model fail-fast handling, FiftyOne video ingestion, deploy dry-runs, credential env audits, and cross-tool smoke-test scaffolding.
- Preserved Genesis BYOVM staging fixes with tests: EGL fallback for multi-GPU demo generation, Docker group/device access for Genesis containers, and BYOVM storage credential reuse.
- Added structured implementation prompts for the 14 NPA CLI demo pre-staging fixes.

## W9-W10 - Workbench maturity sequence

- fix(sonic): default serverless training to H100, not L40S (W12 condensed commit)
- feat(skypilot): `npa skypilot bootstrap/status/verify` with isolated venv
  pattern (W11 condensed commit)
- Isaac Lab SkyPilot orchestration validated end-to-end via BYOF runs
  (W10 condensed commit; see `docs/workbench/cookbooks/byof-isaac-lab/`)
- BYOF mechanism validated: image override and command override surfaces;
  worked example with verified S3 artifacts (run IDs in cookbook)
- Removed SONIC routing entry from `CONTRIBUTING.md` Known Deviations
