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

### Insights + agent: make "which runs used N gpus" answerable from real runs

Found by operating the stack against live infra (8 real runs, 3 of them on 1/2/4
RTX PRO 6000 GPUs) rather than by reading it.

- **Submitted runs are resource-honest.** The insights `gpus` metric had no
  producer: no cluster submit wrote an `npa.workflow.run.v1` manifest, and the
  manifest the local `run-spec --persist-state` path did write carried no
  `resources_profile`, so every run looked CPU-only no matter how many
  accelerators it requested. Step records now carry `resources` +
  `resources_profile` (local, executor-dispatched, and failure paths),
  `persist_submitted_manifest()` writes the manifest after an accepted submit,
  and the `--runtime` tier shares the ledger store so it lands `manifest.json`
  next to `runtime.json`. Ingest refuses to emit `gpus` for a manifest with
  status `planned` — a run that never executed must not report accelerators.
- **The append-only store is safe for concurrent writers.** Appending used to
  read-modify-write one object, so two overlapping ingests both reported success
  while the later write silently dropped the earlier one's rows. Each append now
  writes an immutable shard under `records.d/` / `edges.d/` and readers
  concatenate the base object (legacy stores keep working) plus all shards.
  Note: a reader older than sharding sees only the base object and silently
  reports a truncated store — re-bootstrap deployed agents after upgrading.
- **`failed_check_count` counts as a regression**, not an improvement
  (`LOWER_IS_BETTER_HINTS` matched `failure`/`fail_`, never `failed_check_count`).
- **The agent action loop survives reasoning-model output.** The cheap planner
  tier emits `<think>` blocks containing JSON-looking snippets, and the greedy
  `{.*}` fallback spanned trace + answer, aborting the turn with `no_plan` and
  discarding observations already gathered. Traces are now stripped, candidates
  come from a balanced-brace scan, one bounded corrective re-ask is allowed, and
  a planner failure still answers from what the read-only tools returned — an
  empty result set reports "no runs found" instead of a planner error.
- **Oversized tool observations keep their structure.** A large query result used
  to collapse into a string preview, leaving the planner with no readable run ids
  (it invented a placeholder id, which the tool then rejected). Record-bearing
  observations are downsampled field-wise instead.
- **Final answers must name the observation field they quote.** Measured honestly:
  this did *not* fix scalar selection (4/5 → 0/5 unfaithful on the phrasing
  tested), but replies now cite their source field, which turns a silent wrong
  answer into an auditable one. A verifier pass over the final answer is the
  tracked follow-up.

### Foxglove embedded viewer

- Embedded the official [Foxglove TypeScript SDK](https://docs.foxglove.dev/docs/embed/typescript-sdk)
  (`@foxglove/embed`, MIT) in the NPA agent: a new **Foxglove** viewer tab mounts the
  real SDK from same-origin assets and drives it with the configured embed source,
  organization slug, layout key and data source. The SDK is fetched from npm at
  build/bootstrap time and verified against its pinned sha512 integrity digest;
  nothing is vendored. When the assets or an embed source are missing,
  `GET /api/foxglove/config` reports `available:false` with a reason and the pane
  says so instead of rendering an empty viewer.
- The pane picks a **viewer backend** at runtime, so it renders instead of showing a
  config screen: the official Foxglove app when `NPA_FOXGLOVE_EMBED_SRC` is configured,
  otherwise the self-hosted, Foxglove-compatible OSS viewer the agent already runs
  (Lichtblick) — which plays the recording with no account at all — otherwise an
  explained unavailable state. `NPA_FOXGLOVE_VIEWER_BACKEND` forces either backend.
- `.mcap`, `.bag`, `.db3`, `.ulg` and `.ulog` artifacts classify as `mcap` and are
  published twice from one load: same-origin for the in-page OSS viewer, and on an
  unauthenticated, CORS-enabled, byte-range `/foxglove/data/` path (random file names,
  pruned) for the cross-origin Foxglove app, which cannot send basic-auth credentials.
- New agent endpoints: `GET /api/foxglove/config|status`,
  `POST /api/foxglove/load-artifact|convert-run|live`, a grounded `foxglove_viewer`
  chat intent, and deploy flags `--foxglove-embed-src`, `--foxglove-org-slug`,
  `--foxglove-live-url`. **Describe this** on this pane sends viewer *state* only —
  a cross-origin embed cannot be captured — and says so.
- New workbench tool `npa workbench foxglove` (`convert-run`, `inspect`,
  `install-sdk`, `config`) plus `npa.sdk.workbench.foxglove` and the
  `workbench.foxglove.convert` toolRef: packs a run's real frames, metrics and logs
  into MCAP using Foxglove well-known schemas (a JSON array of records becomes a
  plottable time series). Frame clocks are recorded as `timestamps=synthetic-fps`
  because run artifacts carry no capture time, and frames are encoded by the same
  encoder the Lichtblick writer uses. Needs the new optional extra `npa[foxglove]`.
- New container `npa-foxglove-embed` (caddy, `:8099`, non-root): serves the SDK, the
  shared glue module, a standalone host page, and mounted recordings with CORS +
  byte ranges. Registered in the packaging contract, image registry, supported-tool
  versions, and the golden-eval manifest with a capability smoke that checks the
  SDK, glue, range reads and the CORS preflight. `/data/` has no directory listing
  and is not world-writable.
- Fixed: a relative Lichtblick `ds.url` was never loaded by the viewer (its
  `remote-file` source silently ignores relative URLs), so the recording URL is now
  always pinned onto the browsed origin.

### npa.workflow: real parallel execution and a runtime orchestrator

- **Parallel fan-out.** `npa.workflow/v0.0.1` specs can declare a `parallel:`
  group with an optional `maxConcurrency`. The group renders as a SkyPilot
  **JobGroup** (`execution: parallel`), so its members launch as genuinely
  concurrent jobs, and the group's `next` state is a barrier that starts only
  after every member reaches a terminal state. Groups larger than
  `maxConcurrency` are submitted in batches. Serial remains the default: the
  serial renderer and its guard are untouched, and `--plan-only` still renders
  the flattened serial plan for every spec.
- **`params:` per-state config overlay** so N members of a sweep can share one
  `toolRef` and still differ (learning rate, output prefix, ...).
- **Runtime orchestrator** (`npa workbench workflow submit --runtime`): submits
  each wave, polls it to a terminal state, reads the *real* decision artifact
  from S3 through the existing `decisions.py` contract, and replans — bounded
  loops with true early-exit, data-dependent `goto` branching, wave retry,
  timeout cancellation, and `--resume` on a durable ledger
  (`npa.workflow.runtime.v1` at `<prefix>/npa-workflow/runtime.json`). The
  plan-time `--assume-decision` path is unchanged and remains the offline mode.
- **`trigger:`** on a state makes the runtime driver wait for objects at an S3
  prefix before that state runs (watermark recorded in the ledger).
- **New CLI:** `submit --runtime/--resume/--poll-seconds/--max-wait-seconds/
  --retries/--max-concurrency/--cancel-on-timeout`, `plan-spec --waves`.
- **New specs:** `token-factory-parallel-fanout.yaml` (zero-GPU JobGroup + join
  barrier), `token-factory-gate-loop.yaml` (zero-GPU runtime gate loop with real
  early-exit and branch), `isaac-lab-rl-sweep.yaml` (port of the one
  `execution: parallel` SkyPilot template). All three are registered in
  `SUBMIT_LIVE_MATRIX` and covered by a live runtime e2e tier
  (`scripts/npa-workflow-runtime-live-e2e.sh`).
- Design: repo-root `DESIGN.md`; live evidence: repo-root `EVIDENCE.md`.

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
