# Changelog

Releases are git tags `vX.Y.Z` matching `npa/pyproject.toml`; artifacts are
built and attached by `.github/workflows/release.yml`. See `docs/releasing.md`
for the release process. Entries accumulate under "Unreleased" and move under
a versioned heading when a release is cut.

## Unreleased

### Retiring the raw SkyPilot task catalog (36 → 0 templates (of the 36; two arrived mid-sweep from #234/#235))

`npa.workflow/v0.0.1` specs are becoming the only workflow authoring surface.
SkyPilot remains the execution engine, and `npa workbench workflow submit` still
accepts a customer's own SkyPilot YAML — what is going away is the shipped catalog
under `npa/src/npa/workflows/skypilot/`.

- **Retired 23 templates**, each only after its spec reached a terminal `SUCCEEDED` on
  real infrastructure (run ids in `EVIDENCE.md` §R2–R6, §R10, §R22): `cosmos3-reason.yaml`,
  `isaac-lab-rl-sweep.yaml`, `sonic-export.yaml`, `sonic-eval.yaml`,
  `sonic-export-eval.yaml`, `token-factory-caption.yaml`,
  `token-factory-generate.yaml`, `token-factory-cosmos-reason.yaml`,
  `vlm-eval-token-factory.yaml`, `mjlab-eval.yaml`, `retargeting.yaml`,
  `vlm-eval.yaml`, `vlm-eval-benchmark.yaml`, `sim-to-real-loop.yaml`,
  `scenario-gen-adversarial.yaml`, `sim2real-envgen-split.yaml`, `cosmos3-ea-fetch.yaml`,
  `tokenfactory-train-triage.yaml`, `tokenfactory-rollout-judge.yaml`,
  `tokenfactory-scene-to-rollout-judge.yaml`, `sim2real-actions.yaml`,
  `isaac-franka-capture-reason.yaml`, `cosmos2-transfer.yaml`,
  `sim-to-real-pipeline.yaml`, `sim-to-real-trigger.yaml`, `dataset-ingest-curate.yaml`,
  `cosmos3-text-to-image-inference.yaml`, `bdd100k-pipeline.yaml`.
  `test_skypilot_catalog_retirement.py` pins the remaining set, so the tally is
  machine-checked and a new raw template needs a deliberate edit.
- **Multi-node stages.** A resource profile can declare `num_nodes`, so a spec can ask
  for a real gang-scheduled block; previously that was only reachable through
  `npa burst submit --nodes`, outside the workflow surface. Additive: a 1-node profile
  renders exactly as before. Reference spec `npa-workflows/multi-node-probe.yaml`
  verifies one report per rank from distinct hosts.
- **`isaac-lab-cosmos-sdg-burst-smoke.yaml` relocated** to `npa/src/npa/burst/examples/`:
  it is a single-task input to `npa burst submit-yaml`, not a workflow (no plan, no stage
  graph, nothing for a `toolRef` to describe), and the template said so itself. A guardrail
  pins one-task-per-file and the survival of its `${VAR}` placeholders, and proves burst
  accepts it offline.
- **BYOF resource profiles relocated** from `npa/src/npa/workflows/skypilot/` to
  `npa/src/npa/workflows/byof/profiles/` (they are pod shapes reached through
  `byof.yaml`, not workflow templates), and the three BYOF runner scripts gained
  `--secret-env`, defaulting to the S3 credentials their profiles need for uploads —
  without which a run provisioned, trained, and then died on `NoCredentialsError`.
- **Live-matrix coverage:** the two insights specs (`insights-smoke`,
  `insights-aggregate`) gained entries and now run live — the harness seeds the two
  artifact shapes `workbench.insights.ingest_run` recognises, which is what was
  missing; and the two dataset-of-record specs, for which
  `npa.workflows.dataset_fixture` generates raw sensor records satisfying both specs'
  quality gates. Uncovered specs: 17 -> 12; matrix cases: 24 -> 31.
  `scenario-gen-smoke` needed no fixture at all — its adversary backend is
  deterministic and GPU-free. `dataset-ingest-curate` is `plan_only` for a stated
  infrastructure reason (its `register` stage needs the LanceDB workbench service,
  which is not deployed); its other four stages did pass live.
- **New test fixture:** `npa.workflows.motion_fixture` +
  `scripts/stage-sonic-motion-fixture.sh` synthesize a valid SOMA-CSV G1 motion clip
  using only the standard library, so the retargeting-backed specs are live-testable
  without NVIDIA's dual-licensed motion dataset. `retargeting.yaml`'s live case was
  previously **failing** for lack of input data.
- **New spec:** `npa-workflows/vlm-eval-token-factory.yaml` — zero-GPU VLM scoring
  through the hosted `api` backend. This is the VLM eval path that needs no vLLM
  server, and it is registered in the live matrix as a `cpu` case.
- **`outputs:` declarations corrected in eight specs (eleven stages).** A stage can
  succeed while writing its result somewhere other than the URI the spec declares —
  `vlm-eval` writes `vlm_eval_stub.json`, `mjlab eval` writes `mjlab_eval.json`, the
  Cosmos reasoner writes `scene_reasoning.json`, and several specs declared
  `report.json` / `plan.json`. `test_spec_declared_outputs.py` now compares every
  stage's declared artifact against the tool's own `*_result_uri_for()` helper.
- `npa workbench {mjlab,retargeting,token-factory,vlm-eval} workflow|status` print
  npa.workflow spec paths instead of raw SkyPilot template paths, and
  `vlm-eval workflow|status` gain a `token_factory_workflow` key. A guardrail asserts
  every advertised path is a real file.
- **User-facing behaviour changes:**
  - `npa workbench sonic export` and `npa workbench sonic eval` now accept `s3://`
    URIs for `--checkpoint`, `--onnx`, `--obs-spec`, `--action-spec`, `--config` and
    `--output`, downloading and uploading as needed (including an ONNX's
    `<name>.onnx.data` external weights). Local paths behave exactly as before.
    `sonic eval` adds an `onnx_uri` field to its result when the input was an object
    URI.
  - `npa workbench isaac-lab train` is now invoked correctly by
    `workbench.rl.policy_train`: the toolRef passed `--learning-rate`, `--batch-size`
    and `--input-path`, none of which exist on that command. Trainer hyper-parameters
    go through Isaac Lab's repeatable Hydra `--override KEY=VALUE`, `batch_size`
    becomes the real `--num-envs`, and `--input-path` becomes `--data-path`. The three
    specs that use it rename their `batch_size` config key to `num_envs`.
  - `workbench.rl.evaluate_policy` passed `--episodes`; the CLI option is
    `--num-episodes`.
  - `workbench.sonic.eval` passed `--output json`, conflating the **result path** with
    the output format, so the eval result was written to a relative `json/` directory
    inside the pod and the artifact the spec declared never appeared. It now passes
    `--output <eval_uri> --output-format json`; `sonic-eval.yaml` and
    `sonic-export-eval.yaml` gain an `eval_uri` config key.
  - `solutions.toml`'s `sonic-locomotion-finetuning` solution now submits the
    npa.workflow spec instead of the raw template.
- **`npa workbench vlm-eval loop`** — score every rollout under a prefix and write the
  aggregate `task_success_report.json` the sim-to-real loop gates on. `vlm-eval run`
  scores *one* rollout (it discovers frames recursively), so this capability existed only
  as ~80 lines of bash inside `sim-to-real-loop.yaml` and, separately, as Python inside a
  gated GPU test. The report is field-compatible with the template's, including the
  distinction that `task_success` gates on the **mean** score rather than the pass rate.
  New spec `npa-workflows/vlm-eval-loop.yaml`.
- **A self-hosted VLM stage now serves the model it calls.** `vlm_backend: self-hosted`
  makes the tool POST to localhost, and nothing in a spec started a server — the stage
  failed with `Connection refused`. The renderer gained a per-`toolRef` **run preamble**
  (the sibling of its setup hook; a background service cannot start in `setup:`, which
  SkyPilot runs in a different shell) that starts vLLM, health-checks `/health`, fails
  fast with the server log if it dies, and traps `EXIT` so no GPU-resident server leaks.
  It requires nothing of the task image: `ninja` and a CUDA compiler both come from pip,
  and the JIT-dependent sampler falls back to its pure-PyTorch equivalent.
  `config.vlm_serve_ready_seconds` (default 900 s) tunes the readiness window.
- **`detection-training train` gained `--wait` and `--label-map`** — the poll-until-done
  loop and the category map that `bdd100k-pipeline.yaml`'s template did in bash and that
  no spec could reach. `--wait` is opt-in; the BDD100K and AV night-scene specs use it, so
  their eval stages no longer race a checkpoint that does not exist yet.
- **`vlm-eval-benchmark.yaml`'s twin matched its template in name only:** it passed a
  **repo path** as `--dataset` (unresolvable in a pod) and ran the `stub` backend, so it
  never touched a VLM. Both fixed, and the repo-path class of bug is now machine-checked
  by `test_spec_paths_are_not_repo_relative.py`, which immediately found five `byof-*`
  specs doing the same thing. `resolve_byof_profile_path()` accepts a packaged profile
  **name**, so an installed wheel resolves what a checkout does.
- **`detection-training eval` gained `--discover-checkpoint` and
  `--write-canonical-metrics`**, and now fails on a non-numeric `mAP`. All three were bash
  and `jq` inside `bdd100k-pipeline.yaml`, so no spec could reach them: without discovery the
  eval stage scored the training *directory* instead of the checkpoint training wrote, and
  without the canonical write the BDD100K spec declared a `metrics.json` nothing produced.
- **`run_bdd100k_pipeline.py` renders the spec** (`--spec`, with `--yaml` kept as an alias)
  instead of injecting env vars into raw SkyPilot documents, and its `--mock-endpoints`
  validation now executes **each plan step's resolved argv** against stand-in services and
  checks the call *order* — every `POST /train` followed by `GET /status`, every `POST /eval`
  preceded by `GET /runs`. That drive immediately found two real defects: the
  `create_failure_views` toolRef passed `--table` to a command whose option is
  `--source-table` (so `curate-views` could never have run), and the eval prefixes lacked a
  trailing slash, so the declared artifact URI was
  `…/eval/bdd100k_rider_train` + `metrics.json` concatenated.
- **`workbench.sim2real_envgen.raw_shard` could never have run.** It omitted `--run-id`,
  which the module's parser requires, so every stage using it died on a usage error; three
  shipped specs referenced it. It was also handed the raw-env prefix where the module expects
  the **run root** (from which it derives `envs/raw`, `envs/train`, `envs/heldout`,
  `envs/manifest`), and the four specs using it declared a `manifest.json` that subcommand
  never writes. All fixed, plus a new `workbench.sim2real_envgen.split` toolRef.
- **New spec `sim2real-envgen-shards.yaml`** declares the shard fan-out the retired template
  drove from a Kubernetes Job completion index: a `parallel:` group whose members differ only
  through `params.shard_index`, with the split as a barrier. Live proof records
  `max_concurrent_observed: 2` and a split manifest that saw all 64 envs, 32 from each shard.
- **A `toolRef` can declare third-party CLIs it shells out to**
  (`TOOL_REF_PIP_REQUIREMENTS`), installed only when `command -v` cannot find them.
  `cosmos fetch` runs `huggingface-cli`, which the retired template pip-installed in its
  setup — the one load-bearing line of a ~35-line preamble. New spec `cosmos-fetch.yaml` plus
  `workbench.cosmos.{check,fetch}` toolRefs; the template's hand-rolled `test -n` token checks
  are dropped because `cosmos check` reports which access is missing and continues.
- **Every `toolRef` now invokes `python3`, not `python`.** Five did the latter, which some
  images do not provide: a stage died with `bash: python: command not found` inside the LeRobot
  vendor image, having passed on SkyPilot's default image (miniconda supplies `python` there).
  A guardrail pins it.
- **`lerobot policy_container train --artifacts-s3-uri`** publishes a run's whole output tree,
  not just the checkpoint `--checkpoint-s3-uri` uploads, so a downstream stage can read the run.
- **New `npa.workflows.token_factory_triage`** makes the triage stage executable: it digests a
  run's textual artifacts and has a hosted text model write the report, replacing ~45 lines of
  inline bash that ended in `token-factory generate --system-prompt "$(cat …)"`. It fails loudly
  rather than triaging nothing when a run has no readable text.
- **A `toolRef` can declare its vendor image's interpreter** (`TOOL_REF_VENDOR_INTERPRETERS`).
  Setup installs npa into it — with `--no-deps`, so a vendor's pinned stack is never perturbed —
  and records it as the stage interpreter, so a tool and the vendor library it imports share one
  environment. Without this, a stage on a vendor image runs the system python and fails with
  `No module named 'lerobot'`; the probe checks `import npa.workbench` rather than `import npa`,
  because these images bake a partial npa on `PYTHONPATH` that would otherwise mask the problem.
- **`npa-lerobot` is SkyPilot-hostable** (plus a `Dockerfile.k8s-prereqs` for repairing a
  published tag). Note that `0.5.1` fails LeRobot training at step 0 with a torch/torchcodec ABI
  mismatch; use `0.6.0` or later.
- **`lerobot eval` gained `--rollouts-s3-uri`** (publish the rendered episodes) and resolves a
  remote `--checkpoint-path`: a local path, an `s3://` prefix or a Hugging Face model id, because
  a stage's pod starts empty. Note LeRobot >= 0.6 requires the processor format, so a
  pre-0.6 public policy such as `lerobot/diffusion_pusht` needs migrating first.
- **`npa workbench cosmos3 text-to-image`** turns the retired-in-spirit
  `cosmos3-text-to-image-inference.yaml` bash block into a real command: fetch, uv sync, run the
  framework's inference as an argv, verify the image, publish it with a manifest. Its template is
  NOT yet retired — see EVIDENCE.md §R39 for the one remaining blocker.
- **Hugging Face and uv are resolved as modules, not PATH lookups**, so "installed it, still
  cannot find it" stops being a failure mode.
- **PATH ordering (`/usr/bin` first) is an Isaac requirement, not a universal one.** Forcing it
  on an image whose own python carries npa breaks setup.
- **`npa` is shimmed to the recorded interpreter**, like `python3` already was, so a vendor
  image's baked console script cannot run a stale CLI against a fresh library.
- **The staged npa source goes ahead of a baked one on `PYTHONPATH`.** A source tree on
  PYTHONPATH shadows every install; three vendor images ship one.
- **`npa workbench cosmos2 transfer` publishes its manifest to S3** beside the augmented clip
  instead of only echoing it, so the provenance of a synthetic clip survives the pod.
- **`npa workbench detection-training deploy` defaults to the ambient kubeconfig.** It used to
  default `--cluster-name` to a specific profile, so deploys silently landed on another cluster.
  It also learned the RTX PRO 6000 node label and mints its pull secret instead of copying a
  docker login that expires.
- **`detection-training eval` takes `--label-map`,** like `train` already did. Without it a
  dataset with string categories fails on `int('train')`.
- **`npa workbench lancedb deploy --runtime kubernetes`** puts the LanceDB service in the
  cluster, where a workflow stage can reach it. Previously the only runtimes were a local docker
  daemon, a blocked VM path, and LanceDB Cloud.
- **The LanceDB wrapper serves `/index` and `/query`**, the paths the dataset-of-record has
  always posted, and `dataset ingest` can populate the index it later queries.
- **`sonic train --accept-nvidia-eula`** carries the operator's licence acceptance from the
  spec (`sonic_accept_nvidia_eula`, empty by default) to the vendor entrypoint, which refuses
  until it is given. Acceptance is the operator's to make, so nothing asserts it for them.
- **`npa workbench sonic train --runtime in-job`** trains in the pod the stage is already
  running in, instead of provisioning a Nebius Job from inside it.
- **The legacy `sim_to_real` stack is retired.** The watcher survives and submits
  `npa-workflows/sim2real-vlm-rl.yaml`; `scripts/run_sim_to_real_pipeline.py` and
  `run_sim_to_real_quickstart.py` are gone with it.
- **Isaac Lab frame capture is a package module** (`npa.workflows.isaac_capture`) instead of a
  repo script, so it runs in a pod with no checkout; `npa/scripts/capture_isaac_lab_scene_frames.py`
  remains as a shim. It also owns its camera framing (`--camera-eye`/`--camera-target`) and
  renders at 512x512 for VLM consumption.
- **Isaac images give Kit writable data/cache/log directories.** Without them Isaac boots and
  then stalls indefinitely without rendering; the k8s-prereqs guardrail now separates "can be
  scheduled" from "can render".
- **The vendor npa install falls back to installing dependencies** when `--no-deps` leaves npa
  unimportable, which is the case in Isaac's kit python.
- **`vlm-eval run --task-from <artifact>`** scores a rollout against the plan an earlier
  reasoning stage wrote, reading its `analysis` field. The retired scene-to-rollout-judge
  template did this with `--task "$(python3 … )"`; without it a three-stage combo's judge would
  score against a literal string.
- **`npa.workflow.submit` gained `image=`**, mirroring the CLI's `--image` (including `"none"` to
  clear workbench pins). A spec that pins images could not previously be submitted from Python
  against anything else.
- **The `npa` console script is found when PEP 668 pushes it to `--user`**, which previously made
  a stage fail with `bash: npa: command not found`.
- **A LeRobot training failure carries its log** (last 60 lines) instead of naming a path inside
  a pod that no longer exists.
- **New guardrails** (none weakened): a catalog-wide check that every `toolRef` argv
  names real CLI options and passes values its options can mean — including the `npa …`
  commands **inside** a `bash -c` toolRef, a blind spot where a real defect had shipped;
  a check that a `python -m` toolRef argv parses against its module's own argparse parser,
  where a second one hid (a missing required `--run-id`); a
  check that no spec hands a stage a path inside the repo checkout; a check that the
  reference-workflows skill's template list matches the directory; the three-tier
  contract's third tier moved from SkyPilot `envs` onto the spec + toolRef argv, with
  each contract pinning and *classifying* the parameters a spec cannot set yet; a
  live-matrix check that each case declares the secrets its plan hints at; and a
  `solutions.toml` check that every advertised `workflow submit <path>` exists.
- **Engine:** a `toolRef` can declare an npa extra (`TOOL_REF_PIP_EXTRAS`), installed
  from the same source tree npa came from, so a SONIC stage runs on SkyPilot's default
  image without a vendor image.
- **Images:** the SONIC Dockerfile gains the four SkyPilot-on-Kubernetes prerequisites
  the Isaac Lab image needed, plus a `Dockerfile.k8s-prereqs` for repairing a
  published tag in-cluster. The image guardrail now covers `sonic`.
- **Test fixtures:** `npa.workflows.sonic_fixture` + `scripts/stage-sonic-export-fixture.sh`
  build a real, tiny SONIC policy checkpoint **in-cluster**, so the SONIC twins are
  live-testable without NVIDIA's gated `GEAR-SONIC` weights.
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
