# BEHAVIOR 2026 challenge evaluation

[Workbench](README.md) · [Workflow](../../workflows/testing/behavior-challenge-eval.yaml) · [Readiness](../../workflows/testing/behavior-challenge-eval.readiness.json)

Run the official BEHAVIOR evaluator on a Nebius L40S through Workbench and
SkyPilot. A separately served policy receives RGB, depth, and robot
proprioception and returns robot actions over WebSocket. This workflow evaluates
a fixed policy; training and serving use the challenge's upstream baseline
implementations.

**Status:** protocol planning and local artifact tests are implemented. GPU
evaluation, a trained challenge solution, policy memory measurements, and an
official submission have not been demonstrated. The runtime image, licensed
asset volume, and policy endpoint are required operator inputs. This integration
does not add a published BEHAVIOR container to the Workbench image catalog.

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
npa/.venv/bin/python -m pytest npa/tests/workflows/test_behavior_challenge.py -q
```

The opt-in live test is `npa/tests/e2e/test_behavior_challenge_live.py`. In the
prepared GPU runtime, set `NPA_INTEGRATION_E2E=1` and
`NPA_BEHAVIOR_LIVE_CONFIG` to a private JSON file with `input_path`,
`output_path`, `policy_readme_uri`, `upstream_root`, `evaluator_python`,
`data_root`, `host`, and `port`, then run that test. Use a fresh development
selection; invoking it consumes the prescribed cases and claims its prefix.
The submit matrix currently plans this recipe because runtime and asset
readiness remain unresolved. Live coverage must pass before this draft is
presented as a validated challenge integration.
