# FIXME Closed And Archived Work Log

This file preserves the pre-curation FIXME.md content moved out of the active
backlog during W15 operational-safety cleanup on 2026-05-22. The snapshot below
is intentionally verbatim except for this header, so closed work logs and
deferred low-priority entries remain traceable without crowding the active
operational backlog.

---

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
On the 2026-05-09 8x H200 validation target `203.0.113.10`, deploy-time health checks for Cosmos, GR00T, and FiftyOne succeeded by falling back from blocked public ports to SSH-local checks. Subsequent live commands still used the stored public endpoints directly and failed:

```bash
npa/.venv/bin/npa workbench cosmos -p eu-north1 -n demo-cosmos-8gpu-h200-20260509 serve --port 8081
# Error: Cosmos serve request failed: Failed to reach http://203.0.113.10:8081/serve after 1 attempts: [Errno 60] Operation timed out

npa/.venv/bin/npa workbench groot -p eu-north1 -n demo-groot-8gpu-h200-20260509 serve --model nvidia/GR00T-N1.7-3B --robot-embodiment REAL_G1 --port 8082
# Error: Model load failed: Failed to reach http://203.0.113.10:8082/serve after 1 attempts: [Errno 60] Operation timed out

npa/.venv/bin/npa workbench cosmos -p eu-north1 -n demo-cosmos-8gpu-h200-20260509 status
# app_status: unreachable
# Error: Cannot reach Cosmos endpoint at http://203.0.113.10:8081/health: timed out

npa/.venv/bin/npa workbench groot -p eu-north1 -n demo-groot-8gpu-h200-20260509 status
# app_status: unreachable
# Error: Cannot reach GR00T endpoint at http://203.0.113.10:8082/health: timed out

npa/.venv/bin/npa workbench fiftyone -p eu-north1 -n demo-fiftyone-8gpu-h200-20260509 status --port 5151
# app_status: unreachable
# Error: Cannot reach FiftyOne app at http://203.0.113.10:5151: timed out
```

Cosmos inference also failed at submit for the same reason, so progress reporting and 8x H200 generation timing were not exercised.

**Workaround:**
Manual endpoint changes or SSH tunnels would make the services reachable, but those were intentionally not applied during this zero-intervention validation.

**Proper fix:**
For BYOVM workbenches, live commands such as `status`, `serve`, and `infer` should reuse the SSH fallback/proxy path when the public endpoint is unreachable, or the deploy should persist an endpoint strategy that live commands can use without manual tunnel setup.


## GR00T BYOVM env omits inherited S3 credentials
**Symptom:**
The GR00T BYOVM alias inherited S3 bucket and endpoint settings in `~/.npa/config.yaml`, but `/etc/npa-groot-server/env` on `203.0.113.10` contained only HF/GPU-related values and did not contain `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `NEBIUS_S3_ENDPOINT`, or `NEBIUS_S3_BUCKET`. Cosmos and FiftyOne env files on the same run contained those values.

**Workaround:**
No manual env edits were applied during validation.

**Proper fix:**
Write the merged project storage credentials into the GR00T service env the same way Cosmos and FiftyOne do, and include those keys in the deploy env audit.


## FiftyOne BYOVM auto health fallback waits too long
**Symptom:**
`npa workbench fiftyone deploy --runtime byovm --health-check-mode auto` found the app healthy through SSH, but only after exhausting a long public HTTP retry window. The app was reachable on VM localhost while the deploy remained in `HTTP check on http://203.0.113.10:5151...`; the deploy took `real 908.65`.

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
The 2026-05-09 validation run successfully loaded `s3://your-bucket-name/demo-prestage/cosmos/fiftyone-ranked/` with `--format auto` and reported 5 samples, but the command also printed repeated stderr lines like:

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
  --output-path s3://your-bucket-name/demo-8gpu-h200/cosmos/test-8gpu-timing.mp4
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

# FIXME entries from CC code review of 4 Codex commits — 2026-05-10

These entries come from the Sunday code review of commits 13264a6,
2956b72, c7cb3cb, 1acfddc. Items here are non-blocking for Monday demo
but should be addressed before any of this code lands on a stable
branch / customer-facing version.

Priority: H = High (security or architectural debt with near-term cost),
M = Medium (cleanup that compounds), L = Low (nice-to-have)

---

## [H] Parameterize Isaac Lab → LeRobot formatter (2956b72)

**Source:** CC review, "Architecture quality / Extract / format separation"

**Symptom:** `isaac_lab_lerobot.py` is fully G1-bound: hardcodes
`G1_STATE_NAMES_43`, `G1_STATE_DIM`, schema list size, `unitree_g1` robot
type. Future Genesis/MuJoCo/CARLA adapters cannot reuse the formatter
half without forking the whole module.

**Fix:** parameterize `convert()` over a `FeatureSpec` dataclass:

```python
@dataclass
class LeRobotFeatureSpec:
    state_names: list[str]
    action_names: list[str]
    robot_type: str
    state_dim: int
    action_dim: int
```

Sim-specific extractors (Isaac Lab, Genesis, MuJoCo) stay separate.
Format half (parquet writing, schema generation, episode metadata)
becomes shared and parameterized.

**Site:** `npa/src/npa/adapter/isaac_lab_lerobot.py` (or wherever the
G1 constants live)

**Priority:** H — every new sim adapter blocks on this. Today the
abstraction is fine because there's only one sim adapter; ships gracefully
to two; collapses at three.

**Recommended pairing:** do this work alongside the LeRobot library
validation test below (single coherent refactor session).

---

## [M] Add standalone LeRobot library validation test (2956b72 follow-up)

**Source:** CC review, "LeRobot output validates standalone: no"

**Symptom:** Current adapter tests in `test_groot_adapter.py` validate
parquet output via `pq.read_table` directly — this is circular validation
because both writer and reader are pyarrow. The c7cb3cb sidecar parquet
bug demonstrated that real-world `groot convert` runs catch failures
that the test suite doesn't.

**Fix:** add a smoke test that loads the produced LeRobotDataset via the
LeRobot library directly:

```python
def test_isaac_lab_export_loads_via_lerobot_library(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    # ... run isaac-lab export to tmp_path ...

    dataset = LeRobotDataset.load(tmp_path)
    assert len(dataset) > 0
    # validate at least one sample structure
```

**Site:** `npa/tests/test_groot_adapter.py` or new `test_lerobot_library_compat.py`

**Priority:** M — paired with the formatter refactor above. Catches a
class of failures the current test suite cannot.

---

## [M] Cosmos and Isaac Lab share remote-env upload pattern — lift to shared module

**Source:** CC review, "Duplication"

**Symptom:** `_upload_local_file_via_remote_env` (cosmos) and
`_upload_local_directory_via_remote_env` (isaac_lab) share the same
shape: `set -a; . env_file; python3 - <<PY`. As more tools need
remote-env upload (groot? fiftyone? future tools?), this gets more
copies.

**Fix:** lift to `npa.clients.storage` as `upload_via_remote_env(host,
local_path, s3_uri, env_file_path)`. Both cosmos and isaac_lab call
into it.

**Priority:** M — refactor when adding the third caller. Don't
preemptively refactor before there's pressure.

---

## [L] Add `--dry-run` to `groot reload-env`

**Source:** CC review, "Specific concerns" on 13264a6

**Symptom:** `groot reload-env` immediately applies env changes,
restarts the systemd service, and waits for `/health`. There's no way
to preview what would change without committing. Useful for incident
response or operator confidence-building.

**Fix:** add `--dry-run` flag that:
- Reads credentials.yaml
- Diffs against current `/etc/npa-groot-server/env`
- Prints the diff and the would-execute commands
- Exits without applying

**Site:** `npa/src/npa/cli/groot/__init__.py` near `_build_reload_env_command`

**Priority:** L — operational nice-to-have, not blocking.

---

## [L] Untested edge cases in `groot reload-env` (13264a6)

**Source:** CC review, "Specific concerns" on 13264a6

**Symptom:** Tests cover happy path, missing-credentials, and
command-construction. Untested edge cases: SSH transport failure,
sudo failure, restart-then-health-timeout (service restarts but
`/health` stays down).

**Fix:** add tests for the three SSH/systemd failure paths above.
Mock SSH client to raise transport errors; mock systemctl to fail;
mock /health to time out. Each should produce a clean error message
not a stack trace.

**Site:** `npa/tests/cli/test_groot_cli.py`

**Priority:** L — current behavior is "raise raw exception which user
sees as a stack trace." Not pretty but not dangerous.

---

## [L] Isaac Lab placeholder ego-view defaults to True (2956b72)

**Source:** CC review, "Architectural Notes / placeholder ego-view default-on"

**Symptom:** `--placeholder-video=True` is the default on `isaac-lab
export-lerobot`. This writes synthetic frames as
`observation.images.ego_view`. A downstream visual policy reading
`observation.images.ego_view` and assuming real video would train on
garbage.

**Fix options:**
1. Default `--placeholder-video=False`. Operator opts in explicitly.
2. Rename modality to `observation.images.placeholder_ego_view` so
   downstream consumers can't conflate it with real ego-view data.
3. Add a `metadata.json` flag indicating placeholder data is present.

Pick one (probably option 2 — most explicit, lowest behavior change risk).

**Site:** `npa/src/npa/adapter/isaac_lab_lerobot.py`

**Priority:** L — no current user is reading this field as real video.
Bites if/when one appears.

---

## [L] Cosmos fallback test missing both-paths-fail case (1acfddc)

**Source:** CC review, "Tests meaningful? partial" on 1acfddc

**Symptom:** Existing test covers single fallback (local fails, remote
succeeds). Missing: a test where BOTH local and remote upload fail.
Currently this propagates raw SSHError to the operator, which is
neither the cleanest UX nor the most informative.

**Fix:** add a test that mocks both local and remote upload to fail.
Wrap the resulting error in a clean exception type
(`UploadFailedAllPaths` or similar) with a message naming both
attempts and their failure modes.

**Site:** `npa/tests/cli/test_cosmos_cli.py`

**Priority:** L — error case, but not silently wrong.

---

## [L] Unused `temp_dirs` parameter in `_upload_local_file_via_remote_env` (1acfddc)

**Source:** CC review, "Minor"

**Symptom:** `_upload_local_file_via_remote_env` accepts a `temp_dirs`
parameter that isn't used inside the function.

**Fix:** remove the parameter and update callers.

**Site:** `npa/src/npa/cli/cosmos/__init__.py`

**Priority:** L — dead code. Five-minute cleanup whenever the file
is next touched.

---
## Claude code review
CC review pending on network primitive commits (1bc5bb3, 01e449b,
  12ac80f, 2724160, plus Phase 1 commit). Inner loop discipline applied,
  Phase 5 verification passed, but no independent code review. Run CC
  review before merging to stable branch / customer-facing version.
  Pattern: same as cosmos review on 2026-05-10 morning that caught
  silent IAM scope expansion in upload fallback.

---
[L] Repo-wide ruff check/format fails on pre-existing unrelated lint debt. New code passes targeted checks. Bulk lint cleanup deferred — should be done as a single focused commit ("apply ruff format/check to entire codebase") to avoid muddying functional commits.

## [M] SDK_PUBLIC_SURFACE

Wire up `npa/__init__.py` to expose a clean public SDK surface, such as
`from npa import convert, demo, rerun` with stable public methods. The
architecture doc now states this SDK surface is roadmap, not current behavior.
Likely scope: decide what is public versus internal, build re-exports, document
the API, and add import/behavior tests.

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

---

## CLOSED 2026-05-13 (LeRobot × Jobs e2e closeout, W2 through W2.7)

- Substrate landed by W2 (commits `52ba1f6` through `b3c7825`):
  ServerlessJobConfig persistence, `--runtime serverless` for LeRobot train,
  helpers, 17 unit tests, and 7 smoke tests.
- W2.5 (`1dbbb3c`) fixed e2e harness HOME isolation to preserve Nebius CLI
  auth while protecting local npa config.
- W2.6 diagnostic identified root cause of workload failures: spurious
  `mkdir -p /tmp/lerobot_output` in `_lerobot_train_container_command`
  triggered LeRobot 0.5.1's `FileExistsError` on `output_dir`.
- W2.7 fix (`48ca938`) removed the spurious mkdir; LeRobot's own training
  script creates `output_dir`. Regression test added.
- W2.7 validated 7/10 LeRobot × Jobs e2e hardening dimensions against real
  Nebius. Happy path, status lifecycle, HF propagation, idempotent submit,
  HF dataset training, Diffusion on H200, and `--submit-only` passed.
- First tool-axis expansion of Jobs substrate is shippable with caveats:
  Cosmos and LeRobot both run under `--runtime serverless`.
- GPU recommendations encoded from May 2026 benchmark research; PTX JIT
  warning fires for `--gpu-type b300 --policy-type diffusion`.

Outstanding (deferred):
- W5-refactor: extract generic `ServerlessJobSubmitter` and
  `ServerlessEndpointDeployer` to eliminate per-tool serverless duplication.
- Deterministic NER e2e fixture needs a quota-bound sandbox project or a valid
  platform/preset combination that reliably produces `NotEnoughResources`.
- LeRobot cancel e2e still hits intermittent Nebius internal cancel errors;
  cleanup succeeds, but the hardening dimension is not validated.
- Dataset-from-S3 remains skipped until `NPA_E2E_LEROBOT_S3_DATASET` is
  provided.

---

## CLOSED 2026-05-13 (W7-parallel-tools: generic serverless across Workbench tools)

- Extracted shared serverless infrastructure to `npa/src/npa/serverless_common/`
  for env construction, secret splitting, GPU platform mapping, output upload,
  and S3 output validation.
- Subnet resolution and Job polling intentionally remain per-tool; polling uses
  the existing `ServerlessClient.poll_job` API directly.
- Added `--runtime serverless` to Cosmos (shared-infra refactor), Isaac Lab
  `train`, FiftyOne `load-dataset`, Genesis `train-teacher`, and GR00T `infer`.
- Nebius Serverless smoke results: 4/5 PASS. Cosmos, Isaac Lab, FiftyOne, and
  Genesis uploaded real artifacts to
  `s3://your-bucket-name/w7p-fresh/20260513T225839Z/`.
- GR00T code and unit tests landed, but smoke attempts failed before container
  logs with Nebius internal Job errors on H200 and L40S.
- Cross-tool docs landed at `docs/cookbooks/serverless-tools-coverage.md`.
- LeRobot serverless integration remained owned by W7-full-reproduction; this
  run did not edit LeRobot CLI/tests/cookbook files.
- Did not touch `npa/src/npa/clients/serverless.py` to avoid W7 collision.

Outstanding (deferred):
- GR00T smoke remains open. W7p-groot-debug fixed the missing default image tag
  (`npa-groot:n1.7` -> pushed `npa-groot:0.1.0`), but the single post-fix H200
  retry stalled in `STARTING` with no logs and required `delete --async`
  cleanup. Nebius handoff:
  `/tmp/w7pgd-20260514T001207Z/NEBIUS-SUPPORT-HANDOFF.md`.
- LeRobot can optionally migrate to `npa.serverless_common` after W7 lands.
- Subnet resolution extraction can be revisited if the per-tool patterns
  stabilize.
- Multi-region storage credentials remain single-block operator state.

---

## W7-lancedb deferred follow-ups

- `lancedb`: LanceDB Cloud and Enterprise provisioning remains partnership-gated;
  v1 is connection-only for already-provisioned Cloud endpoints.
- `lancedb`: Parent `npa workbench lancedb` registration requires a follow-up
  edit to `npa/src/npa/cli/workbench/__init__.py`, which was outside the
  W7-lancedb write allowlist.
- `lancedb`: VM and BYOVM app deployment should be completed once parent
  Workbench registration is allowed; local container smoke is validated.
- `lancedb`: Cross-tool integration with FiftyOne as an embeddings backend is
  deferred to a separate run.
- `lancedb`: Backup and restore commands are deferred to v2; use storage-level
  snapshot or prefix replication for now.
- groot: smoke validation pending Nebius support investigation; W7pgdvr-20260514T011023Z confirmed pre-delete retry logs were captured and empty; see /tmp/w7pgd-20260514T001207Z/NEBIUS-SUPPORT-HANDOFF.md

---

## W7-sonic deferred follow-ups

- `sonic`: GR00T+SONIC composition using PolicyServer plus SONIC decoder is
  deferred to separate orchestration discovery and build.
- `sonic`: additional embodiments beyond Unitree G1 are deferred until customer
  signal or hardware availability justifies qualification.
- `sonic`: NIM distribution path was not confirmed in discovery; the Workbench
  integration uses the Hugging Face distribution path and should revisit NIM if
  it becomes the preferred customer path.
- `sonic`: GR00T cookbook should reference SONIC for the full VLA-to-actuator
  pipeline pattern in a separate cross-tool documentation update.
- `sonic`: Dockerfile build is blocked locally after three attempts. The final
  state pins linux/amd64 for Nebius L40S, but the image still needs an amd64
  rebuild, C++ deploy dependency validation, and registry push before serverless
  smoke can run.

---

## W7-lancedb-e2e deferred follow-ups

- `lancedb`: Full Nebius VM-mode e2e remains blocked. `deploy --runtime vm
  --dry-run` accepts the plan, but the non-dry-run path currently exits with
  "VM/BYOVM app deploy needs Workbench parent registration outside this run
  allowlist" before provisioning. The W7-lancedb-e2e fallback validated the
  public CLI through container deploy, table creation, basic and filtered
  query, S3-backed Lance files, and container teardown against
  `eu-north1`/`your-bucket-name`. Implement managed CPU VM provisioning and
  teardown in `npa.cli.workbench.lancedb.deploy` before claiming full VM
  infrastructure validation.

---

## W7-sonic-build-fix deferred follow-ups

- `sonic`: Dockerfile build now succeeds locally as a self-contained
  `linux/amd64` image with `isaaclab==2.3.2.post1`, and the pushed fix image
  passes local import/entrypoint smoke. The older W7-sonic "Dockerfile build is
  blocked" note is superseded by this follow-up status.
- `sonic`: Nebius L40S serverless smoke using the pushed fix image reached Job
  `ERROR` before `started_at` and produced no container logs; classify this as
  `FAIL_PLATFORM` and hand off the job ID/report artifacts to Nebius support if
  it recurs.
- `sonic`: `BUILD_SONIC_DEPLOY=1` still needs a separate deploy-image
  validation pass for TensorRT and ONNX Runtime discovery before claiming the
  C++ deploy stack is production-ready.

---

## Added by W7-all-fixes (20260514T160548Z)

### SONIC orphan Job submission race
- **Surfaced by:** W7-platform-pattern-investigation (20260514T160548Z)
- **Symptom:** Job `aijob-test-00000000000` was run-owned but unexpected in one run, then appeared again as a fresh retry submission in a later W7-platform-pattern run.
- **Hypothesis:** CLI submission idempotency bug, polling collision, stale handle reuse, or Nebius-side ID collision.
- **Priority:** Investigate near-term; could affect future SONIC submissions.
- **Investigation:** Trace `npa workbench sonic train` submission and polling paths for hidden retries, stale handles, and idempotency gaps.

### Workbench tool image manifest hygiene rule
- **Surfaced by:** W7-platform-pattern-investigation + W7-all-fixes (20260514T160548Z)
- **Rule:** Workbench tool images should be pushed as single `linux/amd64` manifests where practical, not OCI image indexes with `unknown/unknown` attestation children.
- **Status:** Defensive best practice, not a proven fix. W7-platform-pattern `20260514T143718Z` refuted the universal manifest-failure hypothesis: GR00T completed with the OCI-index shape, while SONIC failed on L40S with a single `linux/amd64` image.
- **Current state:** SONIC `-amd64` tag fixed; GR00T repush skipped as `SKIPPED_LOW_PRIORITY`; LeRobot, FiftyOne, Genesis, Isaac Lab, and Cosmos were already verified compliant in W7-platform-pattern.
- **TODO:** Add a CI/build check that asserts new serverless image pushes are single-platform unless an explicit exception is documented.

### GR00T subnet resolution defect
- **Surfaced by:** W7-platform-pattern-investigation (20260514T160548Z)
- **Status:** Fixed in commit `aee8484` (`groot: honor configured GPU workbench subnet`).
- **Symptom:** GR00T serverless retries without `--subnet-id` resolved a discovered subnet instead of the configured subnet at `projects.eu-north1.workbenches.h200.serverless_job.subnet_id`.
- **Validated workaround:** Explicit `--subnet-id vpcsubnet-test-00000000000` completed in W7-platform-pattern job `aijob-test-00000000000`.
- **Validation:** W7-all-fixes job `aijob-test-00000000000` passed without `--subnet-id`; live spec contained `vpcsubnet-test-00000000000` and GR00T artifacts were uploaded.

### SONIC L40S Nebius platform issue
- **Surfaced by:** W7-platform-pattern-investigation (20260514T160548Z)
- **Status:** Workaround in place (route SONIC to H100); awaiting Nebius support response.
- **Ticket draft:** `/tmp/w7all-20260514T160548Z/nebius-support-ticket-sonic-l40s-draft.md`
