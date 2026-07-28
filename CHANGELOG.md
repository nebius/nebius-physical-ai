# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased

### Setup re-architecture: project discovery + simple agent deploy

- **`npa configure` discovers projects instead of asking you to type ids.** It
  now enumerates the Nebius projects your CLI profile can reach (via
  `nebius iam tenant list` + `iam project list`) and lets you pick one or more
  from a list, auto-deriving tenant id, project id, and region. New client
  helpers: `npa.clients.nebius.list_tenants()`, `list_projects_in_tenant()`,
  `list_accessible_projects()`.
- **npa is multi-project.** `configure` can select several projects and writes
  each as its own stanza under `projects:` with a chosen `default_project`;
  switch with `-p <alias>`. Falls back to the manual tenant/project/region
  prompts when discovery is unavailable (no CLI / not authenticated / no
  results).
- **Object storage is opt-in in the discovery path.** `configure` sets up the
  Nebius connection and optional model/inference tokens; it only provisions an S3
  bucket + access key when you ask, so first-run is lighter.
- **New `npa agent setup`: a simple interactive deploy.** After `configure`, it
  picks one of your configured projects (prompting when there is more than one)
  and deploys the agent VM — no `--project-id`/`--tenant-id`/`--region` to type.
  `npa agent fresh-setup` remains for scripted deploys.
- **Documented the agent-VM AI Cloud credential model.** The VM authenticates to
  Nebius AI Cloud via an **attached `npa-agent` service account** (granted the
  tenant `editors` role); code on the VM mints short-lived IAM tokens from the
  Nebius metadata endpoint on demand — key-less and auto-rotating. No static AI
  Cloud key is stored on the VM.

### First-time-user cold-start fixes

- **`npa --version` (and `-V`) is now ~6x faster** (~0.72s → ~0.11s). The console
  script now points at a lightweight entry (`npa.cli.entry:main`) that answers a
  bare version request before importing `npa.cli.main`, which eagerly pulls in
  the whole command tree (boto3 / paramiko / rerun / numpy …). In addition,
  `npa/__init__.py` now imports its SDK convenience submodules lazily (PEP 562
  `__getattr__`) instead of eagerly, so any bare `import npa` — which every CLI
  invocation triggers — no longer pays for the full SDK surface. `import npa` and
  `from npa import convert` still work unchanged; the historical
  `NPA_SKIP_EAGER_IMPORTS` flag is now a no-op (imports are always lazy).
- `npa configure --interactive` no longer exits 0 having written nothing. When it
  cannot proceed (no authenticated Nebius CLI profile for provisioning) or is
  cancelled mid-flow (EOF/Ctrl-C), it now exits **non-zero** with actionable
  guidance. **Behavior change:** wrappers/CI that treated a cancelled or aborted
  `npa configure` as success will now see a failure. Setup guidance and the
  interactive prompts also link where to obtain the Hugging Face and NGC keys.
- Removed the dead **Nebius AI Cloud key (`NEBIUS_AI_CLOUD_KEY`)** credential. It
  had no consumer — no code ever used it as an auth header; Nebius AI Cloud
  compute/storage authenticates through the Nebius CLI profile (short-lived IAM
  access token) and the S3 access keys, and hosted inference uses the Token
  Factory key. `npa configure` no longer prompts for it, the `--show` template no
  longer lists it, and it is no longer staged into agent-VM credentials. A stale
  `NEBIUS_AI_CLOUD_KEY` already in `~/.npa/credentials.yaml` is harmless and left
  untouched (nothing reads it), and the `CredentialsConfig.ai_cloud_api_key` /
  `nebius_api_key` aliases and `resolve_ai_cloud_key` helper are removed.
- Added `npa workbench health preflight`: a PASS/WARN/FAIL/SKIP check over
  Hugging Face, NVIDIA NGC, Nebius object storage (S3), and Token Factory
  credentials (`--checks`, `--offline`, `--warn-only`, `--json`). Replaces the
  deprecated hidden `npa workbench health sim2real` in the README preflight
  guidance.
- Added `npa agent preflight` and moved the terraform-binary and SSH-key-pair
  checks (plus the Token Factory 503 warning) ahead of any cloud IAM side effects
  in `npa agent deploy`, so Route C prerequisites fail fast instead of mid-run.

### Repo hardening

- The base `pip install -e npa` is now fully capable: the previous
  `data`/`lancedb`/`viz`/`server` extras (dataframe/reporting, LanceDB, Rerun
  viewer, FastAPI eval/agent server) are folded into the default install, so
  there is no longer an `npa` vs `npa[full]` split. Only GPU/simulation wheels
  (`npa[genesis]`, `npa[groot]`, `npa[sonic]`) and the optional agent adapters
  (`npa[agent-eval]`, `npa[agent-trace]`) remain opt-in. The `full`, `data`,
  `lancedb`, `viz`, and `server` extras are retained as empty no-op aliases so
  existing `npa[full]`/`npa[server]` commands keep working. Supersedes the
  earlier "base install is lightweight" split below.
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
