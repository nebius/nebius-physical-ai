# Slurm policy training with two evaluation gates

[Workflow catalog](../../../workflows/README.md) ·
[Workflow YAML](../../../workflows/testing/policy-training-slurm.yaml)

This workflow runs CPU orchestration on Nebius Managed Kubernetes and submits
training, evaluation, and policy-test jobs to an existing Slurm cluster. It
requires operator-provided model scripts; it does not bundle a foundation model,
simulation benchmark, robot controller, or synthetic-data generator.

```mermaid
flowchart LR
  A[Real and optional synthetic episodes] --> B[FiftyOne curation]
  B --> C[Split source groups 90/5/5]
  C --> D[Slurm pretraining]
  D --> E[Evaluate holdout 1]
  E --> F{All systems pass?}
  F -->|retry| D
  F -->|pass| G[Slurm fine-tuning]
  G --> H[Evaluate holdout 2]
  H --> I{All systems pass?}
  I -->|retry| G
  I -->|pass| J[RTX-class policy test]
```

Both holdouts remain evaluation-only. Fine-tuning uses the training partition;
it never trains on either evaluation holdout. Repeated decisions on a holdout
make it a development evaluation set, so reserve a separate untouched dataset
for any final generalization claim. The split operates on source-lineage groups,
not frames. Related synthetic variants must carry the same group as their real
source. Approximately 90/5/5 of groups are assigned to training and the two
holdouts: each holdout gets `floor(groups / 20)`, with the remainder in training.
Unequal group sizes can produce different episode percentages.

Each failed gate repeats only its own training phase. Later attempts resume the
previous candidate; the first fine-tuning attempt starts from the checkpoint
approved by the first gate. Each attempt has distinct candidate, evaluation,
request, and decision artifacts. All required evaluation systems must meet their
configured success-rate thresholds. There is no automatic retry-count or time
limit, and no invented success score.

## Prepare private inputs

Use non-customer data for validation. Keep all runtime manifests, script paths,
checkpoints, endpoints, and raw logs in access-controlled storage outside Git.
The source presentation is not a pipeline input and must not be staged.

Populate the four input URIs under the selected run prefix before submission.
Override them with `--var` when inputs already exist elsewhere. `bucket` and
`prefix` choose the run's private artifact destination; `split_seed` fixes group
assignment. `slurm_service_account` selects the service account for Slurm client
pods, defaulting to the generic name `policy-batch-client`.

1. `episodes_uri`: a `npa.policy.episodes.v1` JSON manifest. Each episode needs
   `dataset_uri`, integer `episode_index`, stable `group_id`, `source` (`real` or
   `synthetic`), `preview_uri`, `detections` (provided detector label strings),
   and boolean `inhouse_keep` (the existing review result).
2. `curation_policy_uri`: numeric `min_brightness`, `max_brightness`,
   `min_sharpness`, and integer `min_detections`. Brightness is mean grayscale in
   `[0,1]`; sharpness is the variance of the four-neighbor Laplacian on that
   normalized preview. Supply thresholds appropriate to the camera and task.
3. `gate_policy_uri`: phase-to-system threshold maps, for example
   `{"pretrain":{"system_1":0.8,"system_2":0.9},"finetune":{"simulation":0.9}}`.
   These values illustrate the format; the operator chooses the real systems
   and thresholds. Empty policies, invalid probabilities and missing systems fail.
4. `batch_settings_uri`: private Slurm transport and script selection, described
   below. No credentials belong in this JSON.

The curation stage invokes real FiftyOne dataset queries. It compares its
selection with `inhouse_keep` and reports the disagreement count. It checks
LeRobot `meta/info.json` for `codebase_version: v3.0` and episode-count bounds.
It retains references to the source dataset, rather than rewriting LeRobot
Parquet/video shards. Training scripts must load precisely the listed episode
indices. Missing source files and source-version incompatibility remain errors
for the consuming trainer.

Quality checks cover the supplied preview frame. Object detections are supplied
labels, not a detector run by this stage. A full-video quality or detection
pipeline should produce those reviewed inputs upstream. Curation fails if
FiftyOne is missing or selects no episodes; splitting fails with fewer than
twenty independent groups. There is no fallback selection engine.

## Slurm runtime and script contract

For Soperator, use `transport: soperator` and set `namespace`, `login_pod`, and
`login_container` in the private settings. The worker uses the Kubernetes Python
client and its service account to execute `chroot /mnt/jail sbatch` in that login
container. Grant only the needed pod access and `pods/exec` permissions. An
optional `context` selects a mounted kubeconfig for a separate cluster. Without
it, the client uses in-cluster authentication. The workflow does not create or
change clusters, service accounts, RBAC, or credentials.

For execution on an existing Slurm login host, `transport: local` invokes the
installed Slurm clients directly. In either mode, `scripts` must map all five
names to absolute script paths on the login host:

```json
{
  "transport": "local",
  "scripts": {
    "pretrain": "/shared/jobs/pretrain.sh",
    "evaluate-pretrain": "/shared/jobs/evaluate-pretrain.sh",
    "finetune": "/shared/jobs/finetune.sh",
    "evaluate-finetune": "/shared/jobs/evaluate-finetune.sh",
    "deploy": "/shared/jobs/test-policy.sh"
  }
}
```

Scripts own their `#SBATCH` resource requests, model environment, evaluation
implementations and storage credentials. Pretraining can request multiple Slurm
nodes; the policy-test script should request the intended RTX-class partition
and operator-configured robot or simulator. The Workbench pods request only CPU
resources. Slurm burst capacity is managed by the existing cluster configuration;
this workflow does not substitute a SkyPilot GPU job for a Slurm batch job.

Each script receives `--request-uri URI`. The JSON request supplies `run_id`,
`stage`, `iteration`, `partition`, a `dataset` manifest URI and digest, an input
`checkpoint` where applicable, and a unique `result_uri`. Retrieve input data
explicitly: Slurm does not transfer dataset files for the script. The adapter
uses [`sbatch --wait`](https://slurm.schedmd.com/sbatch.html) and propagates batch
failure. An interrupted or failed client requests cancellation of its uniquely
named job. If the control connection is lost, inspect that job in private Slurm
evidence and confirm cancellation before retrying or tearing down the cluster.

Write a result only after the work and artifact uploads complete:

```json
{
  "schema": "npa.policy.batch-result.v1",
  "status": "completed",
  "request_sha256": "<canonical-request-sha256>",
  "checkpoint": {"uri": "<private-checkpoint-uri>", "sha256": "<checkpoint-sha256>"},
  "systems": {
    "system_1": {"successes": 8, "trials": 10},
    "system_2": {"successes": 9, "trials": 10}
  }
}
```

Compute `request_sha256` with `npa.workflows.policy_training.contracts.digest`.
The hash binds the full JSON request, including its unique result destination.
Training results require the checkpoint identity. Evaluation results additionally
require each configured system's integer task-success counts and must identify
the exact requested checkpoint. The deployment test additionally requires
`policy_test_report_uri`. Batch scripts must compute the actual checkpoint hash
and report measurements honestly; the adapter verifies the result contract and
provenance, not model correctness or the semantics of arbitrary external code.

Source scripts, weights, datasets, benchmark access, image validation, and robot
execution permission are operator prerequisites. Staging NPA source does not
stage these dependencies. The existing FiftyOne workbench image serves curation;
use a validated immutable image override when required by the deployment.

## Validate and run

Validate and preview without submitting jobs:

```bash
npa/.venv/bin/npa workbench workflow validate-spec workflows/testing/policy-training-slurm.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec workflows/testing/policy-training-slurm.yaml \
  --run-id policy-preview --assume-decision promote_checkpoint --check-render --json
npa/.venv/bin/npa workbench workflow run-spec --run-id policy-preview \
  --plan-only --scheduler-plan --assume-decision promote_checkpoint --json \
  workflows/testing/policy-training-slurm.yaml
```

The plan previews one iteration per condition-only loop. Runtime submission is
mandatory; a static submission cannot flatten the measured retries correctly.
After storage, access, worker images, service-account access, and scripts have
been verified, use the generic submission command:

```bash
npa/.venv/bin/npa workbench workflow submit workflows/testing/policy-training-slurm.yaml \
  --runtime --stage-src --run-id "$POLICY_RUN_ID" \
  --var "bucket=$NPA_S3_BUCKET" --var "slurm_service_account=$POLICY_SERVICE_ACCOUNT"
```

Review `curation/episodes.json`, `splits/index.json`, each numbered training and
evaluation attempt, and `deployment/result.json` under the run prefix. A planning
success is not a training result. The
[readiness record](../../../workflows/testing/policy-training-slurm.readiness.json)
records which execution prerequisites have actually been verified.

## Tests and cleanup

Focused unit tests exercise split isolation, failure handling, request binding,
checkpoint identity, thresholds, and independent retry loops:

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_policy_training.py \
  npa/tests/orchestration/npa_workflow/test_policy_training_spec.py -q
```

The real curation test uses locally generated images and a generated metadata
fixture. It does not prove model training or complete dataset ingestion. Install
the same FiftyOne runtime used by the worker; the local validation environment
used FiftyOne 1.22.0 with `graphql-core<3.3` for Strawberry compatibility.

```bash
NPA_INTEGRATION_E2E=1 FIFTYONE_DO_NOT_TRACK=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_policy_training_live.py -q
```

The second test submits the real Slurm pipeline only when
`NPA_E2E_POLICY_TRAINING_SPEC`, `NPA_E2E_POLICY_TRAINING_RUN_ID`, and
`NPA_E2E_POLICY_TRAINING_RESULT_URI` identify a privately prepared spec and run.
It checks the terminal policy-test result. The general submit matrix retains
an explicit plan-only entry because these operator-owned inputs are not shipped.

Cancel the workflow before deleting its artifacts or changing the cluster.
Confirm any submitted Slurm job has stopped, especially after a disconnected
client. Preserve checkpoints and decision evidence needed for review; remove
only the selected run's private output prefix when it is no longer needed.
The workflow neither destroys the shared cluster nor publishes model artifacts.
