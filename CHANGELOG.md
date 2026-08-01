# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased

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
