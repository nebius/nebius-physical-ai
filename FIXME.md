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

### Medium

#### [M] SkyPilot's runtime bootstrap stalled once in npa-cosmos-curate

- **Surfaced by**: a full-pipeline `physical-ai-data-factory` submit on
  `npa-rtxpro-mk8s`, 2026-08-01 (managed job 255, image `npa-cosmos-curate:0.1.1`).
  Stages 1-5 ran through SkyPilot; the `cosmos-curate` stage never left `STARTING`.
- **What happened**: the pod came up healthy and SkyPilot's setup script completed,
  but its own runtime bootstrap — `bash --login -c -i '... ~/.sky/.runtime_files
  ...'` — then sat for over an hour with no ray, pip, or uv child process. That is a
  different failure from the image-setup one already fixed: the container stays up,
  so SkyPilot reports `STARTING` rather than `container not found`.
- **Status: has not recurred.** Two later full submits on `npa-cosmos-curate:0.1.2`
  (jobs 279 and 282) provisioned and ran the stage through SkyPilot, job 282
  completing all ten stages. Downgraded from high because the observed impact is now
  a one-off stall rather than a reproducible block, and left open because the cause
  was never identified.
- **Mechanism confirmed statically, 2026-09-18.** The PATH hypothesis was right,
  but not in the shape it was written, and the corrected version explains the one
  observation that did not fit. Read from SkyPilot 0.12.2's
  `sky/skylet/constants.py` and the published image, with no cluster:
  - `RAY_INSTALLATION_COMMANDS` version-checks with
    `SKY_UV_PIP_CMD list | grep "ray " | grep $SKY_REMOTE_RAY_VERSION`, which runs
    against SkyPilot's **own** venv (`VIRTUAL_ENV=~/skypilot-runtime`). The image
    venv's ray is invisible to that check, so it is not what skips the install.
  - When that check fails — which it does on a fresh pod, before SkyPilot has
    installed anything — the next clause is
    `|| RAY_STATUS ||`, and `RAY_STATUS` is
    `RAY_ADDRESS=127.0.0.1:6380 $SKY_RAY_CMD status`.
  - `SKY_RAY_CMD` is
    `$SKY_PYTHON_CMD $([ -s ~/.sky/ray_path ] && cat ~/.sky/ray_path || which ray)`,
    and `~/.sky/ray_path` is only written by the *last* clause of the same
    command. So on a fresh pod the fallback fires and **bare `which ray` is
    resolved against the container's PATH**.
  - The published `npa-cosmos-curate:0.1.2-skypilot-v1-20260813T164700Z` config
    sets `PATH=/opt/cosmos-curate/venv/bin:...` first
    (`npa/docker/workbench/cosmos-curate/Dockerfile:63`), and that venv contains
    `ray` **2.57.0**, verified from the image layers. SkyPilot 0.12.2 pins
    `SKY_REMOTE_RAY_VERSION = '2.9.3'`, so the mismatch is 2.57.0 against 2.9.3,
    not the patch-level difference originally guessed.
  - Net: SkyPilot's python executes the *cosmos-curate* ray script and asks a
    non-existent GCS on 127.0.0.1:6380 for status, before any install has run.
    That also resolves the observation that looked like a contradiction — there
    was "no ray, pip, or uv child process" because the process is
    `$SKY_PYTHON_CMD` running ray's entry script, so it is named `python`.
  - The evaluator image never stalling is consistent: it has no `ray` on PATH, so
    `which ray` finds nothing and the clause fails fast into the install.
- **Not yet reproduced.** This is a static read of the two artifacts, so it
  establishes the mechanism, not the hang. Confirm by running
  `RAY_ADDRESS=127.0.0.1:6380 ray status` inside the image with the venv on PATH
  and timing it, which needs a pod and is cheap once there.
- **Next step, once reproduced**: three candidate fixes, none applied, because an
  image change needs a rebuild plus live validation and catalog reconciliation:
  1. Drop the unused `ray` console script from the image venv. `cosmos-xenna`
     imports ray as a library and never shells out to it, so `which ray` would
     find nothing and the clause would fail fast, exactly like the evaluator
     image. Smallest change; verify no stage calls the `ray` CLI first.
  2. Move the venv off the baked `ENV PATH` and activate it in
     `entrypoint.sh` (`COSMOS_CURATE_VENV` is already exported). This leaves
     SkyPilot's bootstrap shell clean, but SkyPilot's `run` block does not go
     through the entrypoint, so every stage command would have to activate the
     venv itself.
  3. Pre-seed a non-empty `~/.sky/ray_path` so the `[ -s ... ]` branch wins.
     Rejected unless something else changes: the only ray to point it at is the
     2.57.0 one, which is the problem.


#### [M] Lift remote-env upload pattern to shared storage module

- **Surfaced by**: CC review on 2026-05-10.
- **Status**: Still active; the trigger condition is **not** met, and the proposed
  signature is now known to be insufficient.
- **Current issue**: Cosmos and Isaac Lab duplicate the remote-env upload pattern
  based on `set -a; . env_file; python3 - <<PY`.
- **Re-checked 2026-09-18**: still two tools, so the "another tool needs it"
  trigger has not fired. But they hold *three* helpers with three different
  payload shapes, not two copies of one:
  `_upload_local_directory_via_remote_env` and
  `_upload_existing_remote_directory_via_remote_env`
  (`npa/src/npa/cli/isaac_lab/__init__.py`) and `_upload_remote_file_via_env`
  (`npa/src/npa/cli/cosmos/__init__.py`), each with its own tool-specific
  `env_file` default. What actually repeats is the env-sourcing preamble and the
  boto3-client-from-env construction; the payload walk differs every time.
  (`set -a` in `npa/src/npa/clients/ssh.py` is a separate concern — sourcing
  tokens for an arbitrary command, not uploading.)
- **Next step**: the originally proposed
  `upload_via_remote_env(host, local_path, s3_uri, env_file_path)` serves only
  the local-directory case, so extracting it would add a fourth shape rather
  than remove duplication. Extract the shared preamble plus client construction
  and keep the three payload walks, or cover all three shapes in one signature.
  Still wait for a third tool.

## Resolved (recent)

#### [M] SDK_PUBLIC_SURFACE

- **Resolved**: 2026-09-18. `npa.__all__` is the public surface and `dir()`
  reports exactly that, for `npa`, `npa.sdk`, `npa.sdk.workbench`, and
  `npa.workbench`; the surface and its v0 stability are documented in
  `docs/sdk/README.md`. Three defects the audit found are fixed: `npa.sdk` was
  absent from the top-level lazy list, so the documented `npa.sdk.workbench.<tool>`
  entrypoint raised `AttributeError` after a plain `import npa`; both SDK
  namespaces imported eagerly, so `import npa.sdk` pulled 349 npa modules plus
  pyarrow and boto3 where it now pulls two and no third-party dependency; and
  `npa.sdk.workbench.lancedb` resolved to either the implementation module or the
  SDK client depending on import order. `npa/tests/test_sdk_surface.py` covers
  each, and fails on any public SDK module left out of `__all__`.

#### [M] Add standalone LeRobot library validation test

- **Resolved**: 2026-09-18. `npa/tests/test_adapter.py::TestLeRobotLibraryLoad`
  loads a converted dataset with the real `LeRobotDataset` and inspects a
  sample — loaded metadata, state/action shapes, and both video-backed cameras
  decoding to normalized channels-first frames with non-zero variance, which is
  the mp4 path the 25 parquet tests never touch. A third case downgrades the
  exported `codebase_version` to `v2.1` and asserts the loader refuses it, which
  pins that field as load-bearing. Skips unless `lerobot` is installed (it pulls
  in torch and is not a test dependency), with the `importorskip` in an autouse
  fixture so a skip costs 0.4s rather than three discarded conversions. Verified
  against `lerobot==0.5.1`: 3 passed in 51s. `skills/atomic/testing-conventions/SKILL.md`
  records how to run it.

#### [H] Agent VM S3 credentials were readable from cloud-init user data

- **Resolved**: 2026-08-27. Terraform/cloud-init no longer receives or renders
  S3 access or secret keys for any workbench type. NPA verifies the created VM
  and SSH channel, then stages scoped runtime credentials into owner-only files;
  the attached service-account metadata identity remains the rotating
  control-plane authentication path. Rendered-template tests cover every
  workbench variant, and ingress is independently fail-closed.
- **Migration**: redeploy affected VMs and rotate keys retained in historical
  provider user-data. See `docs/security/workbench-service-boundaries.md`.

#### [H] npa-cosmos2-transfer's inference venv was unusable as its own non-root user

- **Surfaced by**: a live 4-GPU `physical-ai-data-factory` run on `npa-rtxpro-mk8s`,
  2026-07-31. **Resolved** 2026-08-01 in
  `npa/docker/workbench/cosmos2-transfer/Dockerfile`.
- `uv` had installed the venv's CPython as root, so the venv's `bin/python`
  symlinked into `/root/.local/share/uv/...`; `/root` is 0700, so the image's own
  `USER ubuntu` got `Permission denied` (exit 126) for every inference call.
- The fix took three passes, each surfacing the next layer: set
  `UV_PYTHON_INSTALL_DIR` so a fresh build keeps the interpreter out of `/root`;
  rewrite `pyvenv.cfg` by directory *prefix*, because uv records the interpreter
  through a version symlink whose name differs from the resolved path (leaving
  `sys._home` in `/root`, which failed far away inside `distutils.sysconfig`); and
  repoint the absolute version symlink that `cp -a` copies verbatim.
- The build now asserts all three, its usability check exercises
  `distutils.sysconfig` rather than just importing torch, and
  `npa/tests/docker/test_cosmos_oss_images.py` runs the relocation block itself
  against a fixture shaped like the published image.

- 2026-07-26 - First-install `SyntaxWarning` + silent embedded-regex corruption.
  `npa --version` (the README verify step) printed `SyntaxWarning: invalid
  escape sequence '\s'` from `npa/src/npa/cli/agent.py`. Root cause: the
  `_bootstrap_agent_stack` `setup_script` f-string embeds the agent backend
  source verbatim, so single-backslash regex escapes are processed by the outer
  f-string — `\s`/`\d` emitted a warning while `\b` silently collapsed to a
  backspace (0x08), corrupting the deployed backend's word-boundary regexes
  (e.g. `_maybe_stage_count_numeric_reply`, the `find_artifacts` run-id matcher,
  and the small-sim2real chat shortcut never matched). Doubled the backslashes
  in the six affected embedded regexes and hardened
  `tests/cli/test_agent_backend_render.py` to compile the rendered backend with
  `SyntaxWarning` as an error and assert no backspace byte / intact regex
  classes. Whole-package compile scan now emits zero escape warnings.
- 2026-07-22 - FIXME Active bookkeeping: removed seven stale `Status: Fixed`
  entries that were already covered under Resolved (Isaac Lab→LeRobot
  formatter, Isaac Lab train trajectories/`list-tasks`, GR00T BYOVM S3
  inheritance, Cosmos infer S3 AccessDenied, Cosmos auto-serve after deploy,
  `<tool> status` without `-p`/`-n`, VM `deploy --destroy` confirmation).
- 2026-07-22 - LeRobot serverless train/profile-train fail fast on empty S3
  credentials via `require_s3_credentials` before `create_job` (parity with
  Cosmos/FiftyOne/Genesis/GR00T/Isaac Lab/SONIC).
- 2026-07-22 - `npa configure` prompts for a project alias (default = region)
  and echoes `-p <alias>` usage after writing `~/.npa/config.yaml`.
- 2026-07-21 - FiftyOne BYOVM auto health fallback now spends a short public
  budget (`FIFTYONE_AUTO_PUBLIC_HEALTH_RETRIES = 3`, ~21s) before falling back
  to SSH-local readiness (`npa/src/npa/cli/fiftyone/__init__.py`).
- 2026-07-21 - Gated-model access reporting: normal Cosmos and GR00T deploys
  print `HF access ok: <repo>` (or a clean failure) per checked gated repo
  (`cosmos/__init__.py`, `groot/__init__.py`).
- 2026-07-21 - GR00T readiness now reports `ready` from the loaded/served model;
  missing NGC/HF credentials are downgraded to non-blocking notes instead of
  forcing `ready: false` (`groot/__init__.py`).
- 2026-07-21 - `groot infer` single-credential constraint is documented in CLI
  help (`--source-project`/`--target-project`) and a clear runtime error, in
  lieu of an `--allow-host-creds` flag (`groot/__init__.py`).
- 2026-07-21 - Deploy template tests resolve fixture paths from the package root
  (`PACKAGE_ROOT = Path(__file__).resolve().parents[1]`), not the process CWD
  (`npa/tests/test_deploy.py`).
- 2026-07-21 - Sim2Real eval image rebuilt for Blackwell. The pinned
  `npa-loop-eval:0.1.1-genuine-sm120` shipped `torch 2.6.0+cu124` (sm_50..sm_90),
  so torch CUDA crashed on RTX PRO 6000 (`sm_120`) before Genesis physics.
  Rebuilt + pushed `npa-loop-eval:0.1.3-genuine-sm120` from
  `npa-genesis:0.4.6-sm80-sm90-sm120-latest` (torch `2.9.0+cu130`), bumped every
  pin/doc + build default and marked 0.1.1/0.1.2 stale in the tag audit.
  **Validated end-to-end on an RTX PRO 6000 node in `npa-rtxpro-mk8s`**: torch
  sm_120 matmul + `gs.init(backend=gs.gpu)` + a `FrankaPickPlaceEnv` step all pass
  with no "no kernel image" error (digest
  `sha256:9ae0ca513a7cf03af3562c91a6e811cd2b68abe168e36899d37f7cb4cb4ebaaa`). The
  superseded broken `0.1.1-genuine-sm120` tag was deleted from the registry.
- 2026-07-19 - Remote install/SSH failures now surface a compact, actionable
  error (step label + exit code + stderr tail) with the full remote stderr
  behind `NPA_DEBUG=1` (the failing command is never echoed, so inlined
  registry tokens stay out of operator logs). Root-caused in
  `SSHClient.run_or_raise` (`npa.clients.ssh.format_remote_failure`) and the
  FiftyOne clone; retires the full-script dumps across Cosmos install/serve,
  FiftyOne, GR00T, Isaac Lab, LeRobot, and Genesis. Original FIXME entry:
  `[M] Cosmos deploy install failure dumps the full install script and traceback`.
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
