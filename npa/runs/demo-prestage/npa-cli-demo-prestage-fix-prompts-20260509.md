# NPA CLI demo pre-staging fix prompts

Date: 2026-05-09

Source report: `npa/runs/demo-prestage/physical-ai-workbench-demo-prestage-20260508.md`

Preflight source-drift handling:
- Checked `npa/src/npa/cli/genesis/__init__.py` and `npa/src/npa/deploy/configurator.py` against the previous clean commit.
- The changes were genuine staging-time bug fixes, not formatting drift.
- Preserved them with focused tests in commit `e124da1` (`genesis: preserve byovm staging fixes`).

Recommended implementation order:
1. Shared credential/env utilities: Fixes 1, 2, 3, 4, 11, and 13.
2. BYOVM health/status behavior: Fixes 5 and 6.
3. Tool-specific operator UX: Fixes 7, 8, 9, and 10.
4. Cross-tool release safety: Fixes 12 and 14.

General requirements for every implementation prompt:
- Add focused unit tests.
- Update e2e or integration tests when a workflow boundary changes.
- Preserve existing `--input-path` and `--output-path` conventions.
- Do not print credential values to stdout, stderr, logs, or test failure messages.
- Add or update `CHANGELOG.md`.
- Add newly discovered workaround-worthy issues to `FIXME.md` in the requested format.

## Fix 1 - Unified credential injection on deploy

### Problem
`npa workbench cosmos deploy` injected `HF_TOKEN` into the Cosmos service env, but `npa workbench groot deploy` did not receive the same shared token. The operator manually copied the token into the GR00T env.

### Goal
All `npa workbench <tool> deploy` commands should inject the full shared credential set from `~/.npa/credentials.yaml` unless the operator opts out.

### Scope
- Shared credential loader.
- Deploy commands for `lerobot`, `genesis`, `isaac-lab`, `cosmos`, `groot`, and `fiftyone`.
- Any container, VM, BYOVM, and serverless env writers.

### Acceptance Criteria
- Read shared credentials from `~/.npa/credentials.yaml`.
- Supported shared keys include `HF_TOKEN`, S3 access key, S3 secret key, and S3 endpoint.
- Every tool deploy receives the full shared set by default.
- Add `--no-shared-creds` to every deploy command.
- If `HF_TOKEN` is missing, print exactly:
  `Warning: HF_TOKEN not found in ~/.npa/credentials.yaml. Gated model downloads will fail.`
- Never print credential values.

### Tests
- Unit test shared credential loading from credentials.yaml.
- Unit test each deploy command calls the shared injection utility by default.
- Unit test `--no-shared-creds` omits shared keys.
- Unit test missing `HF_TOKEN` warning.
- Regression test that stdout/stderr do not contain raw credential values.

## Fix 2 - Shell-safe credential escaping in env injection

### Problem
S3 secret keys containing `$`, `!`, backticks, quotes, or backslashes were written into env files or shell exports in a way that allowed shell interpolation to mangle values.

### Goal
Credential env injection must round-trip byte-for-byte through every runtime mode.

### Scope
- Env file writer utilities.
- Remote shell commands that source env files.
- Docker `--env-file` paths.
- VM, container, BYOVM, and serverless env handling.

### Acceptance Criteria
- Prefer env files with single-quoted shell-safe values.
- Escape literal single quotes as `'\''`.
- Example output:
  `S3_SECRET_KEY='abc$def'\''ghi'`
- If a runtime uses Docker `--env-file` semantics without shell sourcing, document and test that path separately.
- No credential value may be constructed with unquoted `export KEY=VALUE`.

### Tests
- Unit test `shell_quote_env_value()` or equivalent for `$`, `!`, backticks, `'`, `"`, and `\`.
- Unit test env file round-trip through the same parser/source path the service uses.
- Unit test all deploy env writers use the shared safe writer.

## Fix 3 - BYOVM aliases inherit project S3 storage settings

### Problem
BYOVM workbench aliases created by deploy did not inherit project object-storage settings. The operator manually copied storage metadata into each alias in `~/.npa/config.yaml`.

### Goal
When a BYOVM alias is created or updated, it should inherit the parent project's object-storage settings.

### Scope
- BYOVM deploy path for every workbench tool.
- Config writer/updater for project and workbench storage metadata.

### Acceptance Criteria
- On `--runtime byovm`, copy `object-storage` or equivalent storage settings from the project selected by `-p` / `--project`.
- Copy endpoint, access key, secret key, and bucket into the alias config or into the resolved workbench storage model, following existing schema conventions.
- If the project has no storage block, print exactly:
  `Warning: Project {project} has no object-storage settings. S3 operations on this workbench will fail unless configured manually.`

### Tests
- Unit test BYOVM deploy under a project with S3 settings writes matching alias storage.
- Unit test BYOVM deploy under a project without S3 settings emits the warning.
- Regression test for at least Cosmos, GR00T, and FiftyOne BYOVM deploy.

## Fix 4 - Gated model access validation at deploy time

### Problem
GR00T deploy succeeded, but model loading failed later because `HF_TOKEN` lacked access to `nvidia/Cosmos-Reason2-2B`. The failure surfaced only after lengthy runtime setup.

### Goal
Fail fast before provisioning when a known gated HF model cannot be accessed.

### Scope
- Shared HF access validation utility.
- Cosmos deploy.
- GR00T deploy.
- Future gated-model tools.

### Acceptance Criteria
- Maintain a tool-to-gated-model mapping:
  - `cosmos`: `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`
  - `groot`: `nvidia/GR00T-N1.7-3B`, `nvidia/Cosmos-Reason2-2B`
- If `--model <custom-model>` is provided, validate that model repo instead of the default model repo when appropriate.
- Use a HEAD request to `https://huggingface.co/api/models/{repo}` with the HF token.
- On 401 or 403, fail before provisioning with:
  `Error: HF_TOKEN does not have access to {repo}. Request access at https://huggingface.co/{repo} and retry.`
- Missing `HF_TOKEN` warns per Fix 1 but does not block deploy.
- Add `--skip-model-check`.

### Tests
- Unit test mocked HF API responses for 200, 401, and 403.
- Unit test `--skip-model-check`.
- Unit test custom model validation.
- Unit test deploy fails before provisioning on denied access.

## Fix 5 - BYOVM health-check falls back to SSH

### Problem
BYOVM deploy health checks used public ports. The VM firewall blocked those ports, so deploy reported `install_failed` even though services were healthy on VM-local localhost.

### Goal
Default BYOVM health checks should verify local service health through SSH when public ports fail.

### Scope
- BYOVM deploy health-check logic.
- Shared health-check utility.
- Status commands for all service-backed workbench tools.

### Acceptance Criteria
- Add `--health-check-mode` with `public`, `ssh`, and `auto`.
- `auto` is the default.
- In `auto`, check public endpoint first, then SSH to `127.0.0.1:<port>` on timeout or connection refused.
- Reuse the deploy command's SSH host, key, and user.
- If SSH succeeds, print:
  `Public port {port} unreachable; service healthy via SSH on {host}.`
- If SSH succeeds, mark the app healthy/provisioned, not `install_failed`.
- If public and SSH checks fail, mark `install_failed`.

### Tests
- Unit test public timeout plus SSH success returns healthy.
- Unit test public failure plus SSH failure returns `install_failed`.
- Unit test explicit `public` does not use SSH.
- Unit test explicit `ssh` skips public check.

## Fix 6 - `app_status` reflects live service health

### Problem
Deploy-time public health-check failure persisted `app_status: install_failed`. Later `status` commands replayed that stale value despite healthy live services.

### Goal
`status` should compute live health every time.

### Scope
- Status commands for Cosmos, GR00T, FiftyOne, Genesis, Isaac Lab, and LeRobot where applicable.
- Config writer/update logic.

### Acceptance Criteria
- Do not persist deploy-time `app_status` as the primary source of truth for status output.
- If the service is reachable and healthy, report `app_status: healthy`.
- If service is reachable but the model is not loaded, report `app_status: degraded` and include `reason: model not loaded`.
- If service is unreachable, report `app_status: unreachable`.
- Reserve `install_failed` for actual deploy/install/container-start failure.
- Health lookup should honor Fix 5 health-check mode and SSH fallback.

### Tests
- Unit test healthy state.
- Unit test degraded model-not-loaded state.
- Unit test unreachable state.
- Unit test install failure when container/install failed.
- Regression tests for Cosmos and GR00T examples from the staging report.

## Fix 7 - FiftyOne video ingestion from S3

### Problem
`fiftyone load-dataset --format auto` failed for an S3 `.mp4` Cosmos output because the loader assumed image-directory input.

### Goal
FiftyOne should load S3 video files or video prefixes directly.

### Scope
- FiftyOne `load-dataset` command.
- Format detection utility.
- S3 download/temp staging path if FiftyOne cannot load S3 video natively.

### Acceptance Criteria
- Detect video file extensions: `.mp4`, `.avi`, `.mov`, `.mkv`.
- Detect S3 prefixes containing video files.
- Add `--format video`.
- Use `fo.Dataset.from_videos()` or `fo.Dataset.from_dir(..., dataset_type=fo.types.VideoDirectory)`.
- If S3 video paths are unsupported by FiftyOne, download to a local temp directory first.
- In `auto`, infer video mode for a video file path.
- If a prefix mixes images and videos, warn and default to image mode with a suggestion to use `--format video`.

### Tests
- Unit test format detection for S3 URIs and Hugging Face Hub dataset refs.
- Unit test video loader command construction.
- Unit test mixed-prefix warning behavior.
- E2E test loading a short `.mp4` from the test S3 bucket and asserting sample count > 0.

## Fix 8 - Cosmos `infer` progress during generation

### Problem
`cosmos infer` printed nothing during a 7.4 minute generation, making it unclear whether the job was running or stalled.

### Goal
Print useful progress at each poll interval and fail immediately on server-side errors.

### Scope
- Cosmos infer poll loop.
- Shared polling utility if one exists.

### Acceptance Criteria
- Add `--quiet` to suppress progress.
- At each poll interval, print:
  `[42s] Generating... (status: processing)`
- If the server exposes percent or step counts, include them:
  `[42s] Generating... 63% (step 126/200)`
- If the server returns an error, stop polling and exit non-zero immediately.
- On success, print:
  `Generation complete in 447.9s`

### Tests
- Unit test three intermediate statuses print progress lines.
- Unit test mid-poll error fails immediately.
- Unit test `--quiet` suppresses progress but still prints final result.
- Unit test total generation time is printed on completion.

## Fix 9 - GR00T `serve` fails fast on gated-model errors

### Problem
`groot serve` kept polling after the server encountered a gated-model 401 error. The operator had to inspect container logs to find the cause.

### Goal
Surface model-load errors directly in CLI output and stop polling.

### Scope
- GR00T serve command.
- Model-load status polling.

### Acceptance Criteria
- Add `--timeout` to `groot serve`, default `600s`.
- If status includes an error, stop polling immediately.
- Print:
  `Model load failed: {error_message}`
- If the error mentions gated access or authentication, append:
  `Request access at https://huggingface.co/{model_repo} and ensure HF_TOKEN has the required permissions.`
- Exit non-zero on load failure.
- Keep normal success behavior unchanged.

### Tests
- Unit test mocked 401 gated-access error exits non-zero with the actionable message.
- Unit test successful load exits zero.
- Unit test timeout exits non-zero with a concise timeout message.

## Fix 10 - GR00T and Cosmos readiness checks in `status`

### Problem
`groot status` returned raw fields such as `loaded: False` and `ngc_credentials_configured: False` without clearly indicating that the workbench was not demo-ready.

### Goal
Expose readiness as a first-class structured section.

### Scope
- GR00T status.
- Cosmos status.
- Shared readiness utility for model-backed tools.

### Acceptance Criteria
- GR00T readiness includes:
  - `hf_token_present`
  - `ngc_credentials_configured`
  - `model_loaded`
  - `ready`
  - `blockers`
- Cosmos readiness includes at least:
  - `hf_token_present`
  - `model_loaded`
  - `ready`
  - `blockers`
- `ready` is true only when all required checks pass.
- Blockers are human-readable and actionable.

### Tests
- Unit test every readiness combination for GR00T.
- Unit test every readiness combination for Cosmos.
- Unit test text and JSON output preserve the same readiness facts.

## Fix 11 - Deploy dry-run mode

### Problem
Debugging deploy credentials requires full provisioning, which burns GPU time and hides env-generation errors from CI.

### Goal
All deploy commands should support a dry-run that renders redacted env and validation results without provisioning.

### Scope
- Deploy commands for all workbench tools.
- Shared redaction utility.
- Shared dry-run output formatter.

### Acceptance Criteria
- Add `--dry-run` to every workbench deploy command if missing.
- Dry-run skips VM/container/serverless provisioning.
- Print the env file that would be written, with credential values redacted as first 4 chars plus `****`.
- Print HF gated-model validation result from Fix 4.
- Print GPU type, region, runtime, and target host if applicable.
- Exit 0 when validation succeeds.
- Exit 1 when validation fails, including missing required creds for the selected tool.

### Tests
- Unit test complete credentials produce exit 0 with redacted keys.
- Unit test missing `HF_TOKEN` produces exit 1 and the Fix 1 warning when the selected tool requires gated models.
- Unit test raw credential values are absent from output.

## Fix 12 - Cross-tool integration smoke tests

### Problem
Existing e2e tests validate isolated tools. The pre-staging failures were workflow-boundary failures between tools and services.

### Goal
Add integration smoke tests for multi-tool workflows that must work without manual intervention.

### Scope
- Integration test suite or tagged e2e tests.
- CI release/pre-staging workflow.

### Acceptance Criteria
- Add a Cosmos -> FiftyOne pipeline test:
  - Deploy Cosmos with shared credentials.
  - Run `cosmos infer` with a short prompt to S3.
  - Run `fiftyone load-dataset` against the Cosmos output path.
  - Assert sample count > 0.
  - Assert no credential or format errors.
- Add or extend Genesis -> LeRobot coverage if distill e2e does not exercise credential propagation.
- Tests may be tagged `integration` or placed in a separate directory because they require GPU VMs.
- Failure messages identify the seam: credential propagation, format mismatch, missing env var, or service health.

### Tests
- Integration test for Cosmos -> FiftyOne.
- Integration test or explicit coverage note for Genesis -> LeRobot.
- CI workflow path that runs these before release/pre-staging.

## Fix 13 - Post-deploy credential audit assertion

### Problem
Utility-level tests can pass even when a deploy command forgets to wire shared credentials into a specific service.

### Goal
Deploy commands can audit the actual deployed service env and fail if shared credentials are missing or mismatched.

### Scope
- Shared credential audit utility.
- Runtime-specific env readback:
  - SSH/systemd env for VM.
  - Docker exec/env file for container/BYOVM.
  - Serverless env metadata when available.
- Integration tests from Fix 12.

### Acceptance Criteria
- Add `--verify-env`.
- Default `--verify-env` on in CI and off in production.
- Verify `HF_TOKEN`, S3 access key, S3 secret key, and S3 endpoint.
- Compare source credentials to deployed service env byte-for-byte.
- Print only PASS/FAIL per key.
- On failure, print:
  `Credential audit failed: {KEY_NAME} missing or mismatched in {tool} service env. Deploy may have skipped shared credential injection.`
- Never print credential values.

### Tests
- Unit test one correct and one missing credential.
- Unit test mismatch detection.
- Unit test output redaction.
- Add audit step to the Fix 12 integration tests.

## Fix 14 - Source drift guardrail after agent sessions

### Problem
The staging run left source changes in `npa/src/` that were unrelated to the demo staging task and not visible until manually reported.

### Goal
Add a warning guardrail that makes source drift visible after agent-assisted sessions.

### Scope
- `.pre-commit-config.yaml` or equivalent local hook.
- CI workflow if pre-commit is not used.

### Acceptance Criteria
- Run `git diff --name-only npa/src/`.
- Warn if any files under `npa/src/` are modified.
- Do not block commits by default; this is visibility, not enforcement.
- CI may surface this as a non-blocking warning or a separate informational step.

### Tests
- Unit or script test that a modified file under `npa/src/` is reported.
- Unit or script test that no source modifications produce a clean result.
- Documentation note explaining when the guardrail should be run.
