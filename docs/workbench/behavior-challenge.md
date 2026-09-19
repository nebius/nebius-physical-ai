# BEHAVIOR 2026 challenge evaluation

[Workbench](README.md) · [Workflow](../../workflows/testing/behavior-challenge-eval.yaml) · [Readiness](../../workflows/testing/behavior-challenge-eval.readiness.json)

Run the official BEHAVIOR evaluator on a Nebius L40S through Workbench and
SkyPilot. A separately served policy receives RGB, depth, and robot
proprioception and returns robot actions over WebSocket. This workflow evaluates
a fixed policy; policy serving uses pinned upstream implementations.

**Latest experiment, September 19:** both
[matched training arms](behavior-matched-results-2026-09-19.md) completed 3,600
updates, full holdout selection, and selected-model export. GPU validation found
one native correlation-statistics precision difference, then proved the selected
update-3599 model's typed state, fixed-batch metrics, fixed-RNG actions, and
existing serving wrapper agree after restoring that receipt-bound intermediate.
The replay-selected model then completed the ten radio development cases with
mean Q=0.10 and one full success, compared with stock's Q=0.40 and four
successes on the same cases. Trash scored Q=0.30 versus stock's Q=0.366667,
with two full successes each. The other four matched cells remain pending, so
there is no six-task aggregate or policy-quality win claim.

**Earlier comparison, September 18:** the
[76-rollout follow-up](behavior-followup-results-2026-09-18.md) scored the
published RLC controller at Q=0.404444 and the balanced three-task fine-tune at
Q=0.227778 on the same development panel. The fine-tune improved four cases and
worsened fifteen. Meta100 scored Q=0.20 versus stock's Q=0.30 on radio; its
six-case trash prefix matched stock's mean Q, with two more full successes.
Meta's shoe task remains unscored and its final publication was incomplete. The
[earlier 80-rollout experiment](behavior-experiment-results-2026-09-17.md)
remains separately recorded; the radio press-phase checkpoint is still
unevaluated. These measurements cover three tasks and do not establish
competitive full-challenge performance or 24 GB serving compliance.

**Earlier validation, September 16:** the published RLC policy completed the
ten-instance radio development selection with **two successes and mean Q = 0.20**,
compared with the official baseline's **one success and Q = 0.10** on the same
cases. All failures are retained.
The official baseline's separate reporting selection scored **Q = 0.00**.
This is a measured improvement on one task's development selection; performance on
the other 99 tasks and 24 GB serving compliance require separate validation.
Organizer submission is a separate step. The runtime image, licensed
asset volume, and policy endpoint are required operator inputs. This integration
does not add a published BEHAVIOR container to the Workbench image catalog.

A private development run has authorized inputs, verified storage access, and a
verified radio policy server on an RTX PRO 6000 worker. The official radio checkpoint archive has SHA-256
`169c5d8c6dfc6aa463bfc162da983ea26e4cf82eca1d63de3719d3157a61f3be`.
The simulator runtime is Isaac Sim 5.1.0.0 from the pinned upstream installer.
CUDA passed in both policy and simulator environments. The first development
attempt stopped during simulator startup because its compute-only cluster lacked
NVIDIA graphics libraries. Its original logs and incomplete summary are retained;
no rollout score or video was produced. The runner now checks dynamic GLX/EGL
loading and NVIDIA Vulkan enumeration before claiming an evaluation attempt.
The replacement rendering worker passed Workbench's stability, CUDA,
GLX/EGL loading, and NVIDIA Vulkan device checks. It loaded the official scene
and connected to the policy, then failed on its first observation: the evaluator
uses `robot_r1` while the pinned baseline expects `robot`. That failed attempt's
original evidence is retained. The managed policy now adapts those lookup names.
The resulting rollouts produced original JSON and MP4 artifacts that passed full
video decoding and SHA-256 readback checks.

## Recorded development result

The September 16, 2026 validation completed all prescribed development instances
311–320 with one rollout each, the frozen official checkpoint, and the observation
name adapter described below. Workbench reported the workflow as succeeded.

| Instances | Count | Official task result | Q per trial | Steps and decoded frames per trial |
| --- | ---: | --- | ---: | ---: |
| 317 | 1 | Success | 1.0 | 1,921 |
| 311–316, 318–320 | 9 | Failure | 0.0 | 3,225 |

An independent check of the downloaded artifacts verified the exact case and
attempt lists, original JSON/MP4 hashes, and every video frame. Representative
frames from the successful rollout show the robot grasping and manipulating the
radio. The success label comes from the official evaluator.

The **0.10 development mean is not a reporting challenge score**. Development
artifacts have `challenge_score: null` and contain no submission ZIP. Reporting
requires its separate prescribed instances and the full 1,000-case denominator.

## Developing a stronger policy

The [recent-research candidate](behavior-policy-research.md) keeps the existing
π0.5-derived backbone and adds training-only camera consistency helpers inspired
by September 2026 research. It records the failed adaptation run and the
timestamp and annotation-alignment audits. The separate
[matched stage-conditioning implementation](behavior-matched-training.md)
freezes the parent task and stage modules while comparing teacher-bin and parent
stage-replay action conditioning. Native GPU training and holdout selection
completed for both arms. Selected-model serving validation passed; the first
matched radio and trash cells regressed from stock's Q=0.40 and Q=0.366667 to
Q=0.10 and Q=0.30, respectively. Four matched cells remain pending. The published
RLC comparison below is a separate measured result.

`--policy-kind rlc` selects a development-only transfer of the published
[RLC 2025 winning solution](https://github.com/IliaLarchenko/behavior-1k-solution).
Its learned task/stage memory, rolling action inpainting, compression and
proprioception-based recovery replace the radio-only baseline controller.
The measured 2026 radio comparison is below. The authors' 2025 results do not
establish 2026 performance on the full challenge.

### Recorded RLC comparison

Both policies ran all ten prescribed development instances, 311–320, once each
through the same unchanged 2026 evaluator and official task timeout.

| Policy | Successes | Mean Q | Successful instances |
| --- | ---: | ---: | --- |
| Official radio baseline | 1/10 | 0.10 | 317 |
| Published RLC checkpoint 2 with the 2026 adapter | 2/10 | 0.20 | 319, 320 |

RLC succeeded in 2,556 steps on instance 319 and 1,057 steps on instance 320.
Its other eight cases failed with Q=0 after 3,225 steps each. It lost the baseline's
success on instance 317, so the paired result is two gains and one loss, with a
mean-Q increase of 0.10. Ten cases from one task do not establish a reliable gain
across the 100-task challenge.

An independent download check matched every original JSON and MP4 hash and
decoded all **29,413 video frames**. The complete summary has SHA-256
`dff5e8aecd31d1531db22eed5dc909d7cfb84e1b46e0b58b4c524152de68d9f6`.
The evaluated checkpoint ZIP has SHA-256
`9e7e078a721e5a0db60ca180e8ed6ace57d66da03d9884923b5d88304b5f98ea`.
These are development results, not an organizer-verified score or submission.

### Reproduce the transfer

The candidate requires the clean source checkout at
`ca556f74a455cef7987a2be4537b5ac85cc56dd7`, including its pinned OpenPI and
BEHAVIOR submodules. The historical BEHAVIOR submodule supplies task names only;
**simulation still uses the unchanged official 2026 v3.9.2 checkout**.
All original 50 task names match the first 50 entries in the 2026 registry.
The remaining 50 tasks are unsupported by these checkpoints.

Use the existing four managed-policy arguments with `--policy-kind rlc`:

- `--policy-root`: the pinned RLC source checkout.
- `--policy-python`: a Python 3.11 environment with the pinned OpenPI dependencies.
- `--policy-checkpoint`: the published checkpoint directory selected by the
  upstream task mapping, such as `checkpoint_2` for radio.
- `--policy-archive`: a ZIP with that directory as its top-level prefix. Put
  its SHA-256 in the recipe's `policy_checkpoint_sha256` field.

The default `--policy-kind official` retains the original radio policy.
RLC requires a one-task `development` recipe and rejects reporting selections.
The runner checks both extracted weights against the ZIP and every file against
Hugging Face revision `89545bc1b7aa7f2e687bc0032d091f132d715d4e` of
[`IliaLarchenko/behavior_submission`](https://huggingface.co/IliaLarchenko/behavior_submission).
Source and weights remain runtime inputs; Workbench does not redistribute them.
Retain upstream attribution and satisfy the existing OpenPI/Gemma use terms.

For a holdout-selected matched-training export, use
`--policy-kind rlc-selected` with the same four paths and three additional
regular files: `--policy-selected-export-receipt`,
`--policy-correlation-manifest`, and `--policy-validation-receipt`. The archive
must use `selected-model/` as its top-level prefix. The validation receipt must
bind the selected step, exact export receipt, exact BF16 correlation artifact,
and the unchanged RLC server and observation-adapter sources. The runner loads
that intermediate before policy/JIT construction and rejects partial or mixed
receipt sets. A successful serving validation permits development evaluation;
it does not establish a policy-quality improvement.

Selected mode defaults to the exact native execution wrapper. The optional
`--policy-execution-variant transition-refresh` experiment keeps the native
26-action prediction prefix, executes 20 steps, retains four actions for
inpainting, and keeps the native two-of-three stage vote. Before an action is
computed, two consecutive fully open or fully closed onboard gripper readings
can clear the queued plan. An accepted stage change clears the remaining queue
and inpainting state after returning the action that detected the change, so the
next observation replans. The variant receives only the same RGB images and
61-element proprioception as native selected serving; episode ordinals and
transition counts are logging metadata and never policy inputs. Its public
integration has unit and differential coverage but has not itself passed a GPU
run or produced measured rollout scores. The radio result above used the native
selected execution path; it does not validate this optional transition variant.

This selected export remains limited to the 16 tasks assigned to published
checkpoint 2: task IDs 0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43,
and 45. Fine-tuning used IDs 0, 1, and 22; the matched development panel uses
IDs 0, 1, 8, 22, 30, and 45. Selected mode rejects every other task, and tasks
inside the inherited set remain unevaluated unless the panel says otherwise.

The launcher creates a temporary policy-source copy with exactly three import
changes to use the 2026 proprioception layout. It accepts only the three onboard
RGB images and the 61-element proprioception vector. The task is configured from
the public registry; incoming task/instance metadata cannot override it.
Episode resets clear policy memory and queued actions without adding an extra
WebSocket response. Original checkpoints and evaluator code remain unchanged.

One transfer limitation needs measurement: the 2025 training vector's base
velocity convention differs from the robot-local velocity supplied in 2026.
The adapter passes the permitted local velocity directly, without consulting
simulator pose. Fine-tuning on the 2026 training demonstrations is the next
experiment; this comparison uses the published weights without fine-tuning.
Full-task coverage, a new reporting score, and the policy memory requirement
remain unverified.

## Rules and pinned source

Reviewed against the official [2026 overview](https://behavior.stanford.edu/challenge/index.html),
[evaluation rules](https://behavior.stanford.edu/challenge/evaluation.html),
[submission instructions](https://behavior.stanford.edu/challenge/submission.html),
and [baseline guide](https://behavior.stanford.edu/challenge/baselines.html)
on September 15, 2026. The advertised deadline is October 16, 2026; recheck
organizer updates before submitting.

| Requirement | Integration behavior |
| --- | --- |
| BEHAVIOR-1K v3.9.2 | Requires commit `b1979916ec1549b10a4e65e630bc6504a9af1b00` and unchanged tracked source |
| RGB + depth + proprioception | Uses `omnigibson.eval.wrappers.RGBDFullResWrapper` and the bundled R1Pro config unchanged |
| Reporting instances | Public indices 0–9, which this evaluator writes as actual instance IDs 301–310 |
| Development instances | Public indices 10–19; development outputs never produce a submission ZIP or challenge score |
| One rollout per reported instance | Rollout 0 only; no best-of selection or retries |
| Task-specific timeout | Omits `--max-steps`, preserving the official 1.5× mean human demonstration length |
| Original outputs | Stores JSON and MP4 bytes unchanged, decodes videos, hashes files, and verifies S3 readback |
| Partial submissions | Freeze the selected tasks before evaluation; missing cases contribute zero to the full 1,000-case score |
| Reproducibility | Snapshots the evaluator, wrappers, default robot config, exact commands, policy instructions, and declared checkpoint hash |

The official evaluator computes Q, including its own treatment of predicates
that were true initially. NPA reads `q_score.final`; it does not recompute goals
or replace the benchmark with a VLM judgment. Failed tasks with Q=0 are retained.
Upstream `Infinity` values for normalized distance at zero movement are retained.

The workflow rejects custom wrappers, robot configuration, hidden/train splits,
rollout counts, and timeout overrides. These are deliberate boundaries of this
first integration, even where the challenge permits broader customization.
The organizer still reviews the policy and wrapper for privileged simulator
access or environment manipulation. A local artifact check cannot certify that
a remote policy uses its declared checkpoint or complies with every rule.

## Prepare the licensed runtime and policy

1. Complete [Workbench setup](getting-started.md). Use an RT-core simulator GPU
   such as L40S; H100, H200, B200, and B300 are unsuitable for OmniGibson rendering.
   The checked-in simulator profile requests 16 CPU and 128 GiB host RAM. The
   policy service has its own compute allocation.
   The Kubernetes target must also provide the `nvidia` runtime class and mount
   NVIDIA graphics libraries. For RTX PRO 6000, use Workbench's
   [`rtx-rendering` profile](mk8s-gpu-driver-strategy.md#rtx-rendering-workload-profile)
   and its CUDA plus GLX/EGL/Vulkan readiness gates. A passing CUDA test alone
   does not establish rendering readiness. The reference pod requests that runtime
   class and `NVIDIA_DRIVER_CAPABILITIES=all`; the evaluator also requires
   `vulkaninfo` in its prepared runtime.
2. Check the upstream [asset license and installation prompts](https://github.com/StanfordVL/BEHAVIOR-1K/blob/v3.9.2/setup.sh).
   BEHAVIOR's Data Bundle explicitly limits use to non-commercial academic
   research and prohibits redistribution of its data and key. Resolve eligibility
   or obtain separate permission before installing it. Its acceptance is separate
   from NVIDIA, Conda, and policy-model terms. The workflow never accepts terms,
   downloads a key, or fetches assets on the operator's behalf.
3. Prepare an operator-controlled, digest-pinned runtime with NPA's dependencies,
   Git, the unmodified checkout at `/opt/BEHAVIOR-1K`, and the installed upstream
   evaluator environment at `/opt/conda/envs/behavior/bin/python`. Follow the
   [official installation](https://behavior.stanford.edu/getting_started/installation.html)
   under the appropriate upstream acceptance mechanisms. Upstream currently says
   its prebuilt Docker installation is unavailable. Do not substitute a current
   NPA Isaac image and assume its runtime matches this evaluator. Keep licensed
   simulator assets and decryption keys outside image layers; follow
   [container packaging](container-packaging.md) before distributing any runtime.
4. Mount the previously authorized data at `/data/behavior` using a read-only
   PVC in the selected Kubernetes namespace. The volume must contain the data
   bundle and `2026-challenge-task-instances/metadata/B100_task_misc.csv`; prepare
   any required key and writable simulator caches separately under the upstream
   installation's paths. The workflow does not create this volume, install
   NVIDIA drivers, or provision the policy service. Configure image pull access
   for the operator image before submitting.
5. Choose an official baseline. The [baseline guide](https://behavior.stanford.edu/challenge/baselines.html)
   provides task-specific `turning_on_radio` checkpoints for π0.5 and GR00T N1.7.
   Use the guide's BEHAVIOR forks and `scripts/b1k/serve_b1k.py`; NPA's existing
   DROID OpenPI checkpoint and generic GR00T endpoint have different embodiment
   and serving contracts. Verify `/healthz`, the msgpack WebSocket `action`
   response, and reset behavior on development instances. Freeze the checkpoint
   and record its SHA-256 plus exact serving command and source revisions.
6. To extend beyond that task, fine-tune the official `pi05_b1k` or GR00T baseline
   on the released 2026 LeRobot v3 demonstrations for the selected tasks. Keep
   all reporting and hidden instances out of data collection, training, and
   checkpoint selection. Use development instances for iteration. A radio
   checkpoint does not establish performance on the other 99 tasks.

NPA's OpenPI path requires a run-scoped Gemma terms opt-in before building or
fetching Gemma-derived checkpoints; see the
[third-party terms skill](../../skills/atomic/third-party-eula-preflight/SKILL.md).
GR00T's gated dependencies require exact artifact access. Neither choice is
authorized by the BEHAVIOR Data Bundle acceptance alone.

### Supervise the official radio policy in the worker

An operator workflow can append these four arguments to the internal `evaluate`
command and set `--host 127.0.0.1`:

| Argument | Prepared worker-local input |
| --- | --- |
| `--policy-root` | Unchanged `wensi-ai/openpi` checkout at `0cc8e355f7bac0976db1cc3139b1ff0379feea60` |
| `--policy-python` | That checkout's installed policy interpreter |
| `--policy-checkpoint` | Extracted `pi05_turn_on_the_radio` directory |
| `--policy-archive` | Original official `pi05TurningOnRadio.zip` download |

Forward the OpenPI acceptance through `--secret-env NPA_OPENPI_ACCEPT_GEMMA_TERMS`
after completing the run-scoped opt-in. The worker checks the recipe's archive
SHA-256, compares every loaded checkpoint file with the archive, rejects extra
files and an occupied policy endpoint, then starts the official server. It uses
`--repo-id turning_on_radio`: the provided archive stores normalization under
`assets/turning_on_radio`, rather than the demonstration repository name in the
generic training example. No checkpoint or normalization data is rewritten.

The pinned baseline expects `robot::proprio` and camera keys starting with
`robot::robot:`. The official v3.9.2 R1Pro evaluator emits `robot_r1` in both
positions. A small NPA launcher changes only the policy registry's robot name
and three camera lookup keys before running the unchanged upstream serving
script. Camera ordering, images, proprioception, action indices, checkpoint,
and the official evaluator remain unchanged. The launcher source is saved as
`policy-server.py`, with its hash and mapping in `policy-provenance.json`.
Reporting archives include its Apache 2.0 license as `policy-server.LICENSE`,
alongside the official evaluator's separate license.

This option supports only `turning_on_radio`. It records `policy-provenance.json`
and `policy.log`, waits for the real health endpoint, and stops its own server
on success or failure. Startup failures preserve diagnostics in S3 with zero
completed cases. The default command still connects to an independently served
policy. Provision sufficient simulator and policy resources; this convenience
does not establish compliance with the challenge's 24 GB model requirement.

Managed policy startup has a ten-minute deadline, checked after each bounded
health probe. A process that remains alive without a timely successful
`/healthz` response raises a startup error pointing to `policy.log` and is
terminated before any evaluation case begins. This applies to both managed
official and RLC policies.

## Freeze the evaluation selection

Prepare `recipe.json` privately. Set `policy_checkpoint_sha256` to the SHA-256
of the exact checkpoint archive used by the policy service. Use `development`
while iterating. Switch to `report` only after freezing the policy. Use
`"tasks": "all"` for the full 100-task suite, or list selected official task IDs
for an explicitly partial submission. Every selected task receives its prescribed
ten instances; there is no user-defined rollout or time budget.

For multiple tasks, also provide `policy_ports`, a mapping from every selected
task ID to a distinct port on `policy_host`. The official baseline server fixes
its task at startup; the evaluator does not send a task prompt. Start each
task-configured endpoint before evaluation. The integration refuses multi-task
recipes without this mapping so it cannot silently run the radio policy prompt
on all 100 tasks. A single task can use the default `policy_port` of 8000.

```json
{
  "schema": "npa.behavior.recipe.v1",
  "tasks": ["turning_on_radio"],
  "split": "development",
  "policy_checkpoint_sha256": "REPLACE_WITH_CHECKPOINT_SHA256"
}
```

The placeholder hash intentionally fails validation. Local planning needs only
the source checkout and the completed recipe, with no simulator asset access:

```bash
npa/.venv/bin/python -m npa.workflows.behavior_challenge plan \
  --recipe-path /path/to/recipe.json \
  --upstream-root /path/to/BEHAVIOR-1K
npa workbench workflow validate-spec workflows/testing/behavior-challenge-eval.yaml
npa workbench workflow plan-spec workflows/testing/behavior-challenge-eval.yaml --run-id preview
```

Store the completed recipe and a `policy.md` serving runbook in the selected
private S3 bucket. The runbook must give the exact policy source revision,
checkpoint identity, server command, dependencies, and Docker launch instructions
or service details. Provision a reachable policy service before the evaluator.
The default R1Pro policy must return the action shape and controller semantics
defined by the pinned upstream robot configuration.

## Submit through Workbench

Use your configured project and verified artifact bucket. Run credential checks
and the image preflight in the [workflow operations guide](npa-workflow-guide.md)
with the same image and storage overrides. Before a GPU submit, verify the
licensed runtime, asset mount, fixed policy, and sufficient writable output disk.
The workflow requests `source_overlay: true`, so submit stages the current NPA
checkout and installs it over the prepared image before running the evaluator.
Full-suite videos can be large; this runner retains local originals until upload
verification and uses S3 as the durable evidence store.

Set `PROJECT`, `BUCKET`, `RUN_ID`, `RUNTIME_IMAGE`, `ASSETS_CLAIM`, and
`POLICY_HOST` in your private shell from the prepared resources:

```bash
npa workbench health preflight --checks nebius,hf
npa workbench health preflight --project "$PROJECT" --checks s3
npa workbench workflow submit workflows/testing/behavior-challenge-eval.yaml \
  --project "$PROJECT" --run-id "$RUN_ID" \
  --var "bucket=$BUCKET" \
  --var "runtime_image=$RUNTIME_IMAGE" \
  --var "assets_claim=$ASSETS_CLAIM" \
  --var "policy_host=$POLICY_HOST" \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

Other configuration keys are worker paths `upstream_root`, `evaluator_python`,
and `data_root`; `policy_port` defaults to 8000. `input_uri` and
`policy_readme_uri` select the staged files. `output_uri` defaults to the run's
`behavior/` prefix. `assets_claim` names an existing PVC; changing `data_root`
also requires changing the mount path in the spec. These worker paths do not
refer to files on the submitting laptop. The all-zero runtime image digest and
`.invalid` policy hostname are planning placeholders that must be replaced.

## Evidence, failure handling, and handoff

The runner creates `claim.json` using an atomic S3 create before simulation.
It writes `attempts.json` before each rollout and publishes completed originals
after every case. A duplicate invocation against that output prefix fails;
the runner never automatically retries a reporting rollout. Keep failed attempts
and discuss any infrastructure rerun with the organizers. Creating another run
ID does not make cherry-picking permissible; audit attempts across run IDs.
If evaluation or publication fails, the runner prints and retains its worker-local
evidence directory. Recover it before deleting the pod or worker. Successful
execution removes its local temporary copy only after verified publication.

Outputs include `json/`, `videos/`, evaluator logs, `evaluator/`, `plan.json`,
`policy.md`, `README.md`, and `summary.json`. For a completed reporting selection,
`submission.zip` contains the original metrics, evaluator sources, robot config,
and reproduction instructions. Videos remain separate, as required by the
submission portal. Development runs never create a submission ZIP.

NPA's `challenge_score` divides the sum of reported Q scores by **1,000**, even
for a partial task selection. `evaluated_mean_q` describes only evaluated
rollouts and must not be presented as the full challenge score.

Before submitting to the organizers, independently verify the policy's inputs,
data provenance, frozen checkpoint, and reproduction instructions. Docker policy
submissions must run on one 24 GB GPU; the workflow's L40S simulator allocation
does not prove this. IP-based submissions instead require at least 50 available
ports. Provide the required MP4 link through the official portal. This workflow
does not upload to the organizer portal or register an entry.

Use Workbench `status`, `logs`, and `artifacts` to inspect the run. Preserve all
evidence before cleanup. Cancel owned active jobs before removing owned GPU
resources, and stop the separately deployed policy service when finished; follow
[teardown](../teardown.md). Shared asset PVCs and other users' resources remain
operator-managed.

## Validation

The focused tests generate synthetic two-frame MP4s and synthetic metrics to
exercise decoding, exact-byte archives, invalid artifacts, failure preservation,
split isolation, and atomic duplicate refusal. They are not robot policy results.

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_behavior_challenge.py \
  npa/tests/workflows/test_behavior_policy.py -q
```

The opt-in live test is `npa/tests/e2e/test_behavior_challenge_live.py`. In the
prepared GPU runtime, set `NPA_INTEGRATION_E2E=1` and
`NPA_BEHAVIOR_LIVE_CONFIG` to a private JSON file with `input_path`,
`output_path`, `policy_readme_uri`, `upstream_root`, `evaluator_python`,
`data_root`, `host`, and `port`, then run that test. To supervise the official
radio server, also supply `policy_root`, `policy_python`, `policy_checkpoint`,
and `policy_archive`, with the same run-scoped terms opt-in. Use a fresh development
selection; invoking it consumes the prescribed cases and claims its prefix.
The submit matrix plans this recipe because the template needs an
operator-prepared runtime, asset volume, and policy. The recorded live validation
used the standard workflow CLI. The opt-in test exercises the same evaluator
entrypoint and requires its own fresh development selection.
