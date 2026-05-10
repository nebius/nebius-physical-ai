# FIXME

## Pytest path assumptions in deploy template tests
**Symptom:**
Running `npa/.venv/bin/pytest npa/tests/test_deploy.py` from the repository root fails with `FileNotFoundError` for paths such as `src/npa/deploy/terraform/cloud_init.yaml.tpl`.

**Workaround:**
Run those tests from the `npa/` package directory, for example: `cd npa && .venv/bin/pytest tests/test_deploy.py`.

**Proper fix:**
Resolve Terraform template fixture paths relative to the package root or the test file instead of the process current working directory.


## BYOVM live commands do not SSH-fallback when public endpoints are blocked
**Symptom:**
On the 2026-05-09 8x H200 validation target `185.82.71.252`, deploy-time health checks for Cosmos, GR00T, and FiftyOne succeeded by falling back from blocked public ports to SSH-local checks. Subsequent live commands still used the stored public endpoints directly and failed:

```bash
npa/.venv/bin/npa workbench cosmos -p eu-north1 -n demo-cosmos-8gpu-h200-20260509 serve --port 8081
# Error: Cosmos serve request failed: Failed to reach http://185.82.71.252:8081/serve after 1 attempts: [Errno 60] Operation timed out

npa/.venv/bin/npa workbench groot -p eu-north1 -n demo-groot-8gpu-h200-20260509 serve --model nvidia/GR00T-N1.7-3B --robot-embodiment REAL_G1 --port 8082
# Error: Model load failed: Failed to reach http://185.82.71.252:8082/serve after 1 attempts: [Errno 60] Operation timed out

npa/.venv/bin/npa workbench cosmos -p eu-north1 -n demo-cosmos-8gpu-h200-20260509 status
# app_status: unreachable
# Error: Cannot reach Cosmos endpoint at http://185.82.71.252:8081/health: timed out

npa/.venv/bin/npa workbench groot -p eu-north1 -n demo-groot-8gpu-h200-20260509 status
# app_status: unreachable
# Error: Cannot reach GR00T endpoint at http://185.82.71.252:8082/health: timed out

npa/.venv/bin/npa workbench fiftyone -p eu-north1 -n demo-fiftyone-8gpu-h200-20260509 status --port 5151
# app_status: unreachable
# Error: Cannot reach FiftyOne app at http://185.82.71.252:5151: timed out
```

Cosmos inference also failed at submit for the same reason, so progress reporting and 8x H200 generation timing were not exercised.

**Workaround:**
Manual endpoint changes or SSH tunnels would make the services reachable, but those were intentionally not applied during this zero-intervention validation.

**Proper fix:**
For BYOVM workbenches, live commands such as `status`, `serve`, and `infer` should reuse the SSH fallback/proxy path when the public endpoint is unreachable, or the deploy should persist an endpoint strategy that live commands can use without manual tunnel setup.


## GR00T BYOVM env omits inherited S3 credentials
**Symptom:**
The GR00T BYOVM alias inherited S3 bucket and endpoint settings in `~/.npa/config.yaml`, but `/etc/npa-groot-server/env` on `185.82.71.252` contained only HF/GPU-related values and did not contain `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `NEBIUS_S3_ENDPOINT`, or `NEBIUS_S3_BUCKET`. Cosmos and FiftyOne env files on the same run contained those values.

**Workaround:**
No manual env edits were applied during validation.

**Proper fix:**
Write the merged project storage credentials into the GR00T service env the same way Cosmos and FiftyOne do, and include those keys in the deploy env audit.


## FiftyOne BYOVM auto health fallback waits too long
**Symptom:**
`npa workbench fiftyone deploy --runtime byovm --health-check-mode auto` found the app healthy through SSH, but only after exhausting a long public HTTP retry window. The app was reachable on VM localhost while the deploy remained in `HTTP check on http://185.82.71.252:5151...`; the deploy took `real 908.65`.

**Workaround:**
Passing `--health-check-mode ssh` would avoid the delay, but the validation intentionally used default auto mode.

**Proper fix:**
For BYOVM auto mode, attempt the SSH-local check after a short public failure budget or run public and SSH checks in parallel.


## Gated model validation does not report access status during normal deploy
**Symptom:**
The 2026-05-09 Cosmos and GR00T deploy commands used default gated model validation, but the normal deploy logs did not print an explicit Hugging Face access status. Because deploy continued, access success is only inferred. This does not satisfy the validation requirement to report access status at deploy time.

**Workaround:**
No workaround was applied during validation. `--dry-run` appears to print `HF access ok`, but the live deploy path should be self-reporting.

**Proper fix:**
Print a non-secret access result for each checked gated repository during normal deploy, for example `HF access ok: <repo>`, without exposing tokens.


## FiftyOne load-dataset prints curl failures on successful load
**Symptom:**
The 2026-05-09 validation run successfully loaded `s3://YOUR_S3_BUCKET/demo-prestage/cosmos/fiftyone-ranked/` with `--format auto` and reported 5 samples, but the command also printed repeated stderr lines like:

```text
curl: (7) Failed to connect to 127.0.0.1 port 5151 after 0 ms: Couldn't connect to server
```

The command exited successfully after `NPA_FIFTYONE_APP_READY`.

**Workaround:**
Treat the JSON status and exit code as authoritative.

**Proper fix:**
Suppress or scope transient readiness curl failures once the ready marker is observed, or print them only when the command fails.


## Cosmos infer surfaces S3 upload AccessDenied as an uncaught traceback
**Symptom:**
The 2026-05-09 8x H200 rerun completed Cosmos generation successfully in `448.1s`, then failed while saving to the requested S3 URI:

```bash
npa/.venv/bin/npa workbench cosmos -p eu-north1 -n demo-cosmos-8gpu-h200-20260509 infer \
  --prompt "A robot arm gently stacks colored cubes on a lab table." \
  --output-path s3://YOUR_S3_BUCKET/demo-8gpu-h200/cosmos/test-8gpu-timing.mp4
```

The generated VM-local file existed at `/opt/cosmos-data/outputs/cosmos-0d62b401962849ac87d7372ae6bcfea9.mp4`, but the local storage client raised `AccessDenied` on `PutObject`. The CLI printed a Python traceback instead of a concise command error.

**Workaround:**
Use a local output path or credentials with write access. No manual upload was applied during validation.

**Proper fix:**
Catch storage upload failures in Cosmos `infer`, report the generated VM-local output path, and return a clean actionable error. Also verify that the storage credentials used for local S3 uploads match the project bucket write policy.


## GR00T readiness reports not ready after successful HF model serve
**Symptom:**
The 2026-05-09 8x H200 rerun successfully served `nvidia/GR00T-N1.7-3B` with `REAL_G1`, and status showed `loaded: True`, `loaded_model: nvidia/GR00T-N1.7-3B`, and `embodiment_tag: REAL_G1`. The readiness section still reported `ready: False` because `ngc_credentials_configured: False`.

**Workaround:**
Treat `loaded: True` for the HF model as the effective serve readiness signal. Configuring NGC credentials clears the symptom but does not fix the underlying logic.

**Proper fix:**
Do not require NGC credentials for readiness when the active model is a Hugging Face/base model that has already loaded successfully, or distinguish optional credential warnings from readiness blockers.


## `<tool> deploy` against existing alias provisions replacement infrastructure
**Symptom:**
Running `npa workbench groot -p eu-north1 -n <existing-alias> deploy` to update environment variables (specifically to add `NGC_API_KEY` to `/etc/npa-groot-server/env`) caused Terraform to plan and begin creating REPLACEMENT infrastructure for the alias rather than updating the existing VM in place. Plan output showed `9 to add, 0 to change, 0 to destroy`. Partial Nebius resources (network, subnet, security group, boot disk) were created before the operator interrupted with Ctrl-C. The running demo VM was not destroyed only because the abort happened mid-flight.

**Workaround:**
Do not run `npa workbench groot deploy` against an existing alias. Use direct SSH + `systemctl restart npa-<tool>-server` for env updates, accepting the model unload+reload cost.

**Proper fix:**
`deploy` against an existing alias should default to idempotent in-place update of environment and configuration, NOT create replacement infrastructure. Replacement provisioning should require an explicit `--replace` flag with confirmation. Same applies to cosmos and isaac-lab deploy paths.


## No `reload-env` command for env updates without model unload
**Symptom:**
Updating credentials or environment variables on a running tool server (`NGC_API_KEY` for GR00T, `AWS_*` keys for Cosmos, etc.) requires:
1. Manual SSH to the host
2. Manual edit/append to `/etc/npa-<tool>-server/env`
3. `systemctl restart npa-<tool>-server` — unloads the model from GPU memory
4. Manual `npa workbench <tool> serve` to re-load the model with the correct embodiment/config

The model reload tax is substantial on large models. Step 4 requires the operator to remember which model + embodiment was previously loaded (e.g. `REAL_G1` vs `NEW_EMBODIMENT` for GR00T — the wrong embodiment fails for the base checkpoint).

**Workaround:**
Manual SSH + systemctl restart + remember-and-re-serve. Operator must track model and embodiment state out-of-band.

**Proper fix:**
Add `npa workbench groot reload-env` and `npa workbench cosmos reload-env` commands that update env from `~/.npa/credentials.yaml` + project config, restart the service, and automatically re-serve the previously-loaded model with the previously-loaded embodiment by reading last-known-good state. If/when servers support signal-based reload, prefer that path to avoid model unload entirely.


## No cleanup path for partial deploys
**Symptom:**
When `npa workbench <tool> deploy` is interrupted mid-flight, partial Nebius resources are orphaned: network, subnet, security group/rules, and boot disk remain in the project. The CLI provides no command to detect or clean up these orphaned resources. Operator must use `terraform destroy` directly or the Nebius console to remove them, or they continue to incur cost and may cause name conflicts on future deploys.

**Workaround:**
Manual cleanup via Nebius console or direct `terraform destroy` in the workbench's terraform working directory.

**Proper fix:**
Add `npa workbench <tool> cleanup-partial` (or a `--cleanup-on-failure` flag on deploy) that detects orphaned terraform-managed resources for a given alias and tears them down with explicit confirmation.


## Cosmos requires manual `serve` after deploy/restart with no auto-load
**Symptom:**
After `npa workbench cosmos deploy` or a service restart, `cosmos status` reports `loaded: False` and `ready: False` until the operator manually runs `cosmos serve` to load the model. The 1-GPU H200 demo runbook initially did not include a `cosmos serve` step, leading to a status check during dry-run showing `loaded: False, blockers: ['Model nvidia/Cosmos-1.0-Diffusion-7B-Text2World not loaded']`.

**Workaround:**
Always run `cosmos serve` after deploy or restart. Add the step explicitly to runbooks.

**Proper fix:**
Either `cosmos deploy` auto-serves the default model (parameterized via flag), or `cosmos serve` is documented as a required step post-deploy in CLI help and standard runbook templates. The same auto-serve pattern probably applies to GR00T after env updates (see `reload-env` entry).


## `mongosh` not available in FiftyOne container
**Symptom:**
Diagnostic commands referenced in Codex investigation prompts use `mongosh` to query MongoDB state inside the FiftyOne container (`docker exec npa-fiftyone mongosh ...`). The container image (`npa-fiftyone:1.15.0`) does not include `mongosh`:

```text
OCI runtime exec failed: exec failed: unable to start container process:
exec: "mongosh": executable file not found in $PATH
```

Codex worked around this by using PyMongo from inside the container's Python environment.

**Workaround:**
Use PyMongo via `docker exec npa-fiftyone python -c "..."` for diagnostic queries.

**Proper fix:**
Either include `mongosh` in the FiftyOne container image, or document the PyMongo equivalent as the canonical diagnostic path in operator runbooks and investigation prompts.


## FiftyOne load-dataset prints duplicate progress/JSON block
**Symptom:**
On successful `npa workbench fiftyone load-dataset` runs, the CLI streams the same progress bar and final JSON status block twice. Cosmetic noise; complicates output parsing for downstream automation.

**Workaround:**
None needed; treat any one of the JSON status blocks as authoritative.

**Proper fix:**
Deduplicate the streamed output so each load operation produces a single progress sequence and a single JSON status block.

## `<tool> status` without `-p`/`-n` hits an unconfigured default endpoint
**Symptom:**
Running `npa workbench groot status` with no `-p` or `-n` flags returned an endpoint at port 8080 on the project's first BYOVM host (matching no configured alias — the 1-GPU demo alias uses port 18082 via SSH tunnel; the 8-GPU alias uses port 8082):

```text
app_status: degraded
readiness: {... model_loaded: False, ngc_credentials_configured: False, ready: False,
            blockers: ['NGC credentials not configured', 'Model nvidia/GR00T-N1.7-3B not loaded']}
```

No `-n` was provided. The CLI silently fell back to a hardcoded or computed default endpoint rather than erroring or using a sensible default like the most-recently-used alias.

**Workaround:**
Always pass `-p <project>` and `-n <alias>` to `status`, `serve`, `infer`, and similar commands.

**Proper fix:**
When `-p` and `-n` are omitted, status should either error and prompt for the alias, or default to the most-recently-used alias for the project (tracked in `~/.npa/state` or similar). Silently hitting a stale or hardcoded default produces misleading output that looks like a real failure.


## Isaac Lab train does not export trajectories or list registered tasks
**Symptom:**
The 2026-05-09 investigation for the Monday physical AI demo found that `npa workbench isaac-lab train` only writes `npa_isaac_lab_train_summary.json` and `npa_isaac_lab_random_policy_checkpoint.json`. It does not persist synced observations, actions, or states. `npa adapter convert` expects a numpy episode folder contract:

```text
episode_NNNN/obs_workspace.npy
episode_NNNN/obs_wrist.npy
episode_NNNN/state.npy
episode_NNNN/actions.npy
```

The Isaac Lab CLI also has no `list-tasks` command, so operators cannot enumerate humanoid/G1-compatible tasks through the public CLI.

**Workaround:**
No supported CLI-only workaround. Do not fake the Isaac Lab -> LeRobot -> GR00T demo path with Franka, quadruped, or other non-REAL_G1 data.

**Proper fix:**
Add `--export-trajectory / --no-export-trajectory` to `npa workbench isaac-lab train` defaulting off. When enabled, write one episode folder per exported episode under `--output-path` using the numpy contract above, with synced observations, actions, and states. Add `npa workbench isaac-lab list-tasks` to print registered Isaac Lab gym task names, one per line. Add unit coverage for exported folder structure/array shapes and for a non-empty task listing in the test environment. If the Isaac Lab gym registry cannot be accessed without the Isaac Lab container/runtime, document that limitation explicitly in CLI help and tests.
