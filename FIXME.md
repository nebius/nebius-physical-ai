# FIXME

**Purpose**: active operational backlog. Items here are known issues that should
be addressed. Closed items move to `## Resolved (recent)` for short-term
traceability and then to `docs/archive/fixme-closed-work-log.md`.

**Not in scope**: feature requests (use issues), strategic roadmap items (use
`docs/architecture/`), or general TODOs in code (use `TODO:` comments where the
work lives).

**Priorities**: H (operational hazard or partner-incident risk), M
(architectural debt with near-term cost), L (polish or nice-to-have).

## Active

### High

#### [H] Finish validating npa-loop-eval:0.1.3-genuine-sm120 on RTX PRO 6000, then delete stale 0.1.1

- **Surfaced by**: 2026-07-21 live sm_120 GPU smoke on `npa-rtxpro-mk8s`
  (RTX PRO 6000 Blackwell node) during finding-3 validation.
- **Status**: Mostly resolved. Corrected image rebuilt + pushed; default bumped.
  Two follow-ups remain (below).
- **Background**: The old pinned artifact
  `npa-loop-eval:0.1.1-genuine-sm120` bundled `torch 2.6.0+cu124` (arch_list
  sm_50..sm_90 only), so the first torch CUDA op crashed on RTX PRO 6000
  (`sm_120`) with "no kernel image is available for execution on the device".
  Genesis stage-10 held-out eval (`sim_backend=genesis`,
  `npa.genesis.env_pick_place.FrankaPickPlaceEnv`) hard-requires torch CUDA, so
  it died before Genesis physics. `0.1.2-genuine-sm120` was never pushed (404).
- **Done (2026-07-21)**: Rebuilt+pushed
  `cr.eu-north1.nebius.cloud/<repo>/npa-loop-eval:0.1.3-genuine-sm120` from
  `npa-genesis:0.4.6-sm80-sm90-sm120-latest` (torch `2.9.0+cu130`;
  `torch._C._cuda_getArchFlags()` → `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
  compute_120`, i.e. the sm_90-cap root cause is fixed). Bumped
  `DEFAULT_EVAL_TAG` / `SUPPORTED_TOOL_VERSIONS['loop-eval']` / pyproject /
  runbook / README / sm120 catalog / golden-eval table to 0.1.3, and marked both
  0.1.1 and 0.1.2 stale in `audit_workbench_image_tags.py`.
- **Remaining**:
  1. **Full on-GPU rollout validation** of 0.1.3 (torch CUDA matmul +
     `gs.init(backend=gs.gpu)` + a `FrankaPickPlaceEnv` step) on an RTX PRO 6000
     node. This could not be run from the dev VM: the dev-VM service account
     (`npa-mk8s-provisioner`) can push and `docker`/`crane`-pull, but the cluster
     kubelet gets `403 Forbidden` on the registry manifest HEAD (registry in
     eu-north1, cluster in us-central1), so K8s Jobs can't pull the image. Run
     from an operator/codex environment whose cluster pull creds work (the same
     one that pulled 0.1.1 for the original repro). CPU-side archflags already
     prove the torch root cause is fixed; this confirms Genesis JIT kernels too.
  2. **Delete the stale `npa-loop-eval:0.1.1-genuine-sm120`** tag from the
     registry once (1) passes (kept for now as a documented-broken fallback).

#### [H] Parameterize Isaac Lab -> LeRobot formatter

- **Surfaced by**: CC review of commit `2956b72` on 2026-05-10.
- **Status**: Fixed.
- **Current issue**: `npa/src/npa/adapter/isaac_lab_lerobot.py` remains G1-bound
  through hardcoded state names, dimensions, schema shape, and robot type.
- **Next step**: Introduce a `LeRobotFeatureSpec` dataclass and split
  sim-specific extraction from shared LeRobot parquet/schema/metadata writing.

#### [H] Isaac Lab train does not export trajectories or list registered tasks

- **Surfaced by**: 2026-05-09 physical AI demo investigation.
- **Status**: Fixed.
- **Current issue**: `npa workbench isaac-lab train` writes summaries and a
  random-policy checkpoint but not synced observation/action/state episodes;
  the CLI also lacks a public `list-tasks` command.
- **Next step**: Add opt-in trajectory export with the numpy episode contract and
  add a task listing command with tests or documented runtime limitations.

#### [H] GR00T BYOVM env omits inherited S3 credentials

- **Surfaced by**: 2026-05-09 8x H200 validation.
- **Status**: Fixed.
- **Current issue**: GR00T BYOVM env files inherited HF/GPU values but omitted
  project storage credentials present in Cosmos and FiftyOne env files.
- **Next step**: Write merged project storage credentials into
  `/etc/npa-groot-server/env` and include those keys in deploy env audit.

#### [H] Cosmos infer surfaces S3 upload AccessDenied as an uncaught traceback

- **Surfaced by**: 2026-05-09 8x H200 rerun.
- **Status**: Fixed.
- **Current issue**: Cosmos generation can complete successfully, then local S3
  upload failure prints a Python traceback instead of a clean command error.
- **Next step**: Catch upload failures, report the generated VM-local file path,
  and return an actionable storage-credentials error.

### Medium

#### [M] FiftyOne BYOVM auto health fallback waits too long

- **Surfaced by**: 2026-05-09 BYOVM validation.
- **Status**: Still active.
- **Current issue**: Auto mode can exhaust a long public HTTP retry window before
  trying SSH-local readiness even when the app is already healthy on localhost.
- **Next step**: For BYOVM auto mode, try SSH after a short public failure budget
  or run public and SSH checks in parallel.

#### [M] Gated model validation does not report access status during normal deploy

- **Surfaced by**: 2026-05-09 Cosmos and GR00T deploy validation.
- **Status**: Still active.
- **Current issue**: Normal deploy does not print explicit non-secret Hugging
  Face access results, so success is only inferred when deploy continues.
- **Next step**: Print `HF access ok: <repo>` or a clean failure for each checked
  gated repository during normal deploy.

#### [M] GR00T readiness reports not ready after successful HF model serve

- **Surfaced by**: 2026-05-09 8x H200 rerun.
- **Status**: Still active.
- **Current issue**: Readiness stays false when `ngc_credentials_configured` is
  false, even after a Hugging Face/base model is loaded successfully.
- **Next step**: Treat loaded HF/base models as ready without NGC, or downgrade
  missing NGC credentials to an optional warning.

#### [M] Cosmos requires manual serve after deploy/restart with no auto-load

- **Surfaced by**: 2026-05-09 demo runbook dry-run.
- **Status**: Fixed.
- **Current issue**: `cosmos status` reports unloaded after deploy or service
  restart until the operator manually runs `cosmos serve`.
- **Next step**: Add an explicit deploy auto-serve option or make the post-deploy
  `serve` requirement prominent in CLI help and standard runbooks.

#### [M] Add standalone LeRobot library validation test

- **Surfaced by**: CC review of commit `2956b72` on 2026-05-10.
- **Status**: Still active.
- **Current issue**: Adapter tests validate parquet via pyarrow directly, which
  misses failures that real `LeRobotDataset` loading would catch.
- **Next step**: Add an optional `pytest.importorskip("lerobot")` smoke test that
  loads an exported dataset with the LeRobot library and inspects one sample.

#### [M] Lift remote-env upload pattern to shared storage module

- **Surfaced by**: CC review on 2026-05-10.
- **Status**: Still active when a third caller appears.
- **Current issue**: Cosmos and Isaac Lab duplicate the remote-env upload pattern
  based on `set -a; . env_file; python3 - <<PY`.
- **Next step**: Extract `upload_via_remote_env(host, local_path, s3_uri,
  env_file_path)` into `npa.clients.storage` when another tool needs it.

#### [M] SDK_PUBLIC_SURFACE

- **Surfaced by**: Architecture doc follow-up.
- **Status**: Still active.
- **Current issue**: `npa/__init__.py` does not expose a clean public SDK surface
  even though the architecture docs describe one as roadmap.
- **Next step**: Decide public versus internal methods, add re-exports,
  document the API, and cover imports/behavior in tests.

#### [M] CC_REVIEW_M4: document or add `groot infer --allow-host-creds`

- **Surfaced by**: CC review follow-up.
- **Status**: Still active.
- **Current issue**: `groot infer` lacks `--allow-host-creds` while other
  cross-project commands expose it; the difference is not documented.
- **Next step**: Either document the constrained single-credential behavior or
  add the flag with intentionally limited semantics.

#### [M] `<tool> status` without `-p`/`-n` hits an unconfigured default endpoint

- **Surfaced by**: 2026-05-09 validation.
- **Status**: Fixed.
- **Current issue**: Omitted project/name flags can silently hit a stale or
  computed endpoint, producing misleading failure output.
- **Next step**: Error and prompt for an alias, or track and use the most recent
  alias explicitly.

### Low

#### [L] Pytest path assumptions in deploy template tests

- **Surfaced by**: Local test execution from repository root.
- **Status**: Still active.
- **Current issue**: `npa/.venv/bin/pytest npa/tests/test_deploy.py` from repo
  root can fail with `FileNotFoundError` for deploy template paths.
- **Next step**: Resolve Terraform template fixture paths relative to the package
  root or test file instead of the process current working directory.

#### [L] VM `deploy --destroy` runs Terraform destroy with no confirmation

- **Surfaced by**: 2026-06-11 Cosmos VM teardown.
- **Status**: Fixed.
- **Current issue**: For Terraform-managed (`--runtime vm`) aliases,
  `cosmos deploy --destroy` proceeds straight to `terraform destroy` with no
  confirmation prompt; `--yes` only gates `--replace`, so a mistyped `-n` alias
  is destroyed immediately. BYOVM destroy is non-destructive (it only
  unregisters the alias). The same unguarded VM-destroy path likely exists in the
  other workbench deploy commands (groot, fiftyone, isaac-lab).
- **Next step**: Prompt before VM destroy unless `--yes` (or `--dry-run`) is
  passed, and apply the same guard consistently across the workbench tools.

## Resolved (recent)

- 2026-07-19 - Remote install/SSH failures now surface a compact, actionable
  error (step label + exit code + stderr tail) with the full command and output
  behind `NPA_DEBUG=1`. Root-caused in `SSHClient.run_or_raise`
  (`npa.clients.ssh.format_remote_failure`) and the FiftyOne clone; retires the
  full-script dumps across Cosmos install/serve, FiftyOne, GR00T, Isaac Lab,
  LeRobot, and Genesis. Hiding the command by default also stops leaking the
  inlined docker-login `registry_token`. Original FIXME entry: `[M] Cosmos deploy
  install failure dumps the full install script and traceback`.
- 2026-07-19 - Isaac Lab -> LeRobot formatter parameterized via
  `LeRobotFeatureSpec` with a G1 default spec (decoupled state/action dims).
- 2026-07-19 - `npa workbench isaac-lab list-tasks` (remote gym registry) and
  opt-in `train --export-trajectories` (trained-policy rollout, numpy episode
  contract, `npa_isaac_lab_rollout_v2` meta).
- 2026-07-19 - Omitted `-p`/`-n` now errors with available aliases when no
  unambiguous default exists (`npa.clients.config` shared resolvers) instead
  of silently hitting a stale or arbitrary endpoint.

- 2026-07-09 - GR00T BYOVM/project storage credential inheritance + reload-env
  parity with Cosmos (`apply_storage_env_vars`, `_shared_groot_env_or_fail(cfg, ...)`).
- 2026-07-09 - VM `deploy --destroy` confirmation gate via
  `npa.deploy.confirm.confirm_vm_destroy` across workbench tools; e2e scripts
  pass `--yes`.
- 2026-05-22 - BYOVM live commands SSH fallback schema and live routing
  (`9784d25`, W15 Stage B). Original FIXME entry: `BYOVM live commands do not
  SSH-fallback when public endpoints are blocked`.
- 2026-05-22 - Deploy replacement guard with Terraform plan analysis
  (`aa2ad51`, W15 Stage A). Original FIXME entry: ``<tool> deploy` against
  existing alias provisions replacement infrastructure`.
- 2026-05-13 - W7-parallel-tools: generic serverless across Workbench tools
  (see archive).
- 2026-05-13 - LeRobot x Jobs e2e closeout, W2 through W2.7 (see archive).
- 2026-05-13 - NER UX hardening - three surfaces shipped (see archive).

Full historical details are preserved in
`docs/archive/fixme-closed-work-log.md`.

## Archive

Detailed closed work logs, superseded operational notes, low-priority parking
lot items, and the pre-curation FIXME snapshot live in
`docs/archive/fixme-closed-work-log.md`.

Archived headline groups include:

- 2026-05-12 - Serverless Cosmos Endpoint backend with self-discovery + NER
  fallback.
- 2026-05-13 - LeRobot GPU benchmark reproducibility cookbook.
- 2026-05-13 - Cosmos x Jobs e2e closeout, W1 + W1.5.
- 2026-05-13 - NER UX hardening and W4 docs polish.
- 2026-05-13 - LeRobot x Jobs e2e closeout and W7 generic serverless work.
- W7 LanceDB, SONIC, platform-pattern, image-manifest, and subnet follow-ups.
- Low-priority CLI polish and cleanup items deferred out of the active
  operational backlog.
