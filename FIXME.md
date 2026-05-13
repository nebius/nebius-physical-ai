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

---

## [L] SDK_V1_STABILIZATION

The public SDK surface now exists as v0:
`from npa import convert, demo, rerun, workbench, network, workflow, errors`.
Before declaring v1 stability, add type stubs or stronger typing coverage,
settle semver policy, and bake the wrapper signatures with customer usage.

## [L] ADAPTER_NAMESPACE_CONSOLIDATION

`npa adapter convert` is the only command in the adapter namespace. Either grow
more adapter commands or fold this into another namespace, such as
`npa convert <adapter-name>`.

## [L] NETWORK_NAMESPACE_CONSOLIDATION

`npa network ensure-ingress` is the only command in the network namespace.
Either grow more network commands or fold this into a different command shape.

---

## [M] CC_REVIEW_M4

`groot infer` is missing `--allow-host-creds` while other cross-project commands
have it. The difference is semantically defensible because GR00T inference runs
remote S3 I/O under a single credential set, but the inconsistency is not
documented in cross-project docs or command help. Add that note, or add the flag
with intentionally constrained behavior.

## [L] CC_REVIEW_M7

`_ProjectBoundaryS3._side_for_bucket` silently routes to `target` when a bucket
appears in both source and target sets. Edge case, but it repeats the silent
collapse pattern from I1. Either error on ambiguous membership or split dispatch
by operation intent.

## [L] CC_REVIEW_N1

`npa.cli.viz.lerobot` has no-op `as <same-name>` re-exports of private adapter
functions to silence linting. `npa/tests/cli/test_viz_cli.py` imports the
privates through this path. Cleaner migration: update test imports to point at
`npa.adapter.lerobot.render` and delete the re-exports.

## [L] CC_REVIEW_N2

`npa init` is a literal alias for `npa configure`. Either drop the alias or add
a one-line help distinction.

---

## CLOSED 2026-05-12 (deploy lifecycle safety run)

- `<tool> deploy against existing alias provisions replacement infrastructure` — CLOSED
  Added existing-alias detection for groot, cosmos, isaac-lab, and fiftyone.
  Re-running deploy against a saved alias now avoids Terraform by default;
  replacement provisioning requires explicit `--replace` and confirmation
  unless `--yes` is set.

- `No reload-env command for env updates without model unload` — CLOSED FOR COSMOS
  Added `npa workbench cosmos reload-env`; groot already had reload-env.
  isaac-lab and fiftyone remain lower-priority future candidates.

- `Add --dry-run to groot reload-env` — CLOSED
  Added `--dry-run` to both groot and cosmos reload-env. Dry-run reads the
  remote env file, prints a redacted unified diff, and shows the commands that
  would run without applying changes.

- `No cleanup path for partial deploys` — CLOSED
  Added `npa workbench <tool> cleanup-partial` for cosmos, groot, isaac-lab,
  and fiftyone. The command classifies alias state and only runs Terraform
  destroy for `partial`; fully deployed aliases are refused and BYOVM aliases
  are skipped. Confirmation is required unless `--yes` is passed.

---

## CLOSED 2026-05-12 (serverless Cosmos Endpoint backend with self-discovery + NER fallback)

- Added `--runtime serverless` for Cosmos workbench. `deploy` creates a Nebius
  Serverless AI Endpoint, `status` reads endpoint state and HTTP health,
  `serve` is an optional pre-warm, `infer` uses the saved endpoint URL, and
  `teardown` deletes the endpoint plus local alias.
- Added `ServerlessClient` for `nebius ai endpoint` with explicit
  `NotEnoughResourcesError` classification. Auth errors are not treated as NER.
- E2E setup self-discovers a sandbox project from `~/.npa/config.yaml`, prefers
  the project ID whose region is `eu-north1`, sets `NPA_INTEGRATION_E2E=1` and
  `NPA_E2E_SERVERLESS_PROJECT=<project-id>` in the run environment, and builds a
  run-wide NER fallback chain across non-production project IDs.
- Added mocked unit tests, fake-`nebius` smoke tests, and real-Nebius e2e tests
  for Cosmos serverless endpoint flows. Follow-up validation on 2026-05-12
  closed the real-Nebius loop with `8 passed, 1 skipped`; the skip is the
  documented forced-NER scenario gated by `NPA_E2E_FORCE_NER`. Report:
  `/tmp/npa-serverless-cosmos-infer-timeout-20260512T210106Z.md`.
- Resolved `NOVEL_ISSUE_PHASE5_CREATE_OUTPUT_PARSE`: the create parser fallback
  was validated against real Nebius, endpoint URLs are read from
  `status.public_endpoints`/`status.publicEndpoints`, and bare addresses are
  normalized before CLI inference.
- Cosmos serverless e2e now propagates operator Hugging Face auth to deployed
  Endpoints and polls long-running prompt inference asynchronously so the real
  7B text-to-world workload can finish without the previous 600s CLI timeout.
  No `npa-e2e-*` endpoint or alias leak remained after validation.
- Operator-ratified Decision 3 is implemented: changing the served Cosmos model,
  image, platform, preset, env, or volumes requires redeploying the endpoint.
- Deferred: Jobs substrate, GR00T/LeRobot/FiftyOne serverless, K8s backends,
  DevPods, and pipeline orchestration.

---

## CLOSED 2026-05-13 (LeRobot GPU benchmark reproducibility cookbook published)

- New cookbook at `docs/cookbooks/lerobot-gpu-benchmarks.md` documents
  reproduction of the May 2026 LeRobot GPU benchmark research via Nebius
  Workbench.
- Covers all four policy architectures (VQ-BeT 38M, ACT 52M, Diffusion 263M,
  SmolVLA 450M) across four NVIDIA GPUs (L40S, H200, B300, RTX PRO 6000).
- GPU recommendation guide with PTX JIT caveat for B300 + Diffusion.
- Profiling methodology section warns about torch.profiler per-stage sync
  misrepresentation (up to 40%).
- Forward-compatible: current path uses `--runtime container`; serverless path
  uses `--runtime serverless` when LeRobot Jobs support is present in the
  installed `npa` version.
- Partner-facing artifact: suitable for customer evaluation, partner demos, and
  internal SA reference.

---

## CLOSED 2026-05-13 (Cosmos × Jobs e2e closeout, W1 + W1.5)

- Fixed `s3://` scheme bug in `_serverless_train_output_path` (W1: `b7149e0`).
- Added unit test for output path scheme (W1: `1e2f3ae`).
- Parameterized NER test platform via `NPA_E2E_NER_PLATFORM` env var (W1.5: `2781039`).
- Validated 5/6 Cosmos Jobs e2e hardening dimensions against real Nebius:
  happy-path completion, cancel, status lifecycle, HF token propagation, and
  idempotent submission.
- NER fixture is now resilient to Nebius platform catalog changes, but the
  sandbox accepted the largest valid discovered H200 request, so NER fallback
  was not reproduced in W1.5.

Outstanding observations (deferred):
- Need a deterministic NER trigger for Jobs e2e; current `gpu-h200-sxm`
  `8gpu-128vcpu-1600gb` request created Jobs successfully in eu-north1.
- Cancel-internal-error finding from May 12 remains deferred; W1.5 saw
  transient internal cancel errors during NER fixture teardown, but follow-up
  cleanup removed all `npa-e2e-jobs-*` Jobs.
- Delete-operation auth polling from May 13 remains a transient to monitor.

---

## CLOSED 2026-05-13 (NER UX hardening - three surfaces shipped)

- Move 1 (CLI): top-level error handler formats typed exceptions with
  actionable messages, suggested alternatives, and consistent exit codes.
  Stack traces are hidden by default; `NPA_DEBUG=1` enables them.
- Move 2 (Status): `npa workbench <tool> status` distinguishes `scheduled`
  from `waiting_for_capacity`, with hints.
- Move 3 (SDK): typed exceptions carry structured fields (`project_id`,
  `platform`, `preset`, `gpu_count`, `suggested_alternatives`, `error_class`)
  for agent and orchestrator branching logic.
- JSON output mode is supported for typed CLI errors and status payloads.
- Backward compatible: existing exception class names and `str(exc)` behavior
  are unchanged; new fields are additive.
- Customer-facing docs: `docs/cli-errors.md`, `docs/sdk/errors.md`.
- E2E NER test fixture remains deferred as a separate issue; UX coverage
  shipped independently.

---

## CLOSED 2026-05-13 (W4-NER-UX docs polish)

- Added installation + import section to `docs/sdk/errors.md` so SDK consumers
  know how to install the package and import typed exceptions.
- Added cross-references between `docs/cli-errors.md` and
  `docs/sdk/errors.md` so readers landing on either find the other.
- Added `docs/README.md` with an index of the docs directory structure.
- W4-NER-UX deliverable is now fully shippable for external partner
  distribution.
