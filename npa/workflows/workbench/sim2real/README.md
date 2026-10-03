# Sim2Real workflow

> **Naming:** `sim2real` is the staged 14-stage VLM-to-RL loop described
> here. `sim-to-real` (hyphenated) is the older, separate H100 pipeline.

Use the single canonical spec and complete the operator runbook before submit:

- [onboarding, preflight, submit, and remediation](../../../../docs/workbench/guides/sim2real-workflow.md)
- [data and customer-asset contracts](../../../../docs/workbench/guides/sim2real-customer-assets.md)
- [canonical RobotSpec and URDF example](../../../../docs/workbench/guides/sim2real-robot-spec.md)
- [architecture and durable resume](../../../../docs/architecture/sim2real-compositional-workflow.md)

## Start with your inputs

| Prepare | Contract |
| --- | --- |
| Robot embodiment and capture assets | [Customer assets](../../../../docs/workbench/guides/sim2real-customer-assets.md) and [RobotSpec](../../../../docs/workbench/guides/sim2real-robot-spec.md) |
| Project, cluster, bucket, immutable images, and Isaac cache | [Operator runbook](../../../../docs/workbench/guides/sim2real-workflow.md) |
| Expected 14-stage graph and recovery behavior | [Architecture](../../../../docs/architecture/sim2real-compositional-workflow.md) |

From the repository root, validate the canonical spec locally:

```bash
npa workbench workflow validate-spec workflows/main/sim2real.yaml
```

This checks the declaration; it does not prove image access, model access, GPU
placement, or customer-data compatibility. Continue with the operator runbook's
preflight and complete `submit --runtime` command using your verified inputs.
Keep the returned run ID for status, logs, artifacts, and durable resume.

## One seam, one value

Every BYO seam is one value addressed four ways. This table is generated from
the canonical `SIM2REAL_SEAMS` tuple in
`npa/src/npa/workflows/sim2real_health.py`; if a flag, kwarg, or env name
drifts, the coherence guardrail test fails.

| Seam | CLI flag (`npa workbench sim2real run`) | SDK kwarg (`sim2real.run()`) | YAML env |
| --- | --- | --- | --- |
| `s3_endpoint` | `--s3-endpoint` | `s3_endpoint` | `AWS_ENDPOINT_URL` |
| `s3_bucket` | `--s3-bucket` | `s3_bucket` | `NPA_SIM2REAL_BUCKET` |
| `s3_prefix` | `--s3-prefix` | `s3_prefix` | `NPA_SIM2REAL_PREFIX` |
| `trigger_dataset_uri` | `--trigger-dataset-uri` | `trigger_dataset_uri` | `NPA_SIM2REAL_TRIGGER_DATASET_URI` |
| `trigger_dataset_id` | `--trigger-dataset-id` | `trigger_dataset_id` | `NPA_SIM2REAL_TRIGGER_DATASET_ID` |
| `assets_uri` | `--assets-uri` | `assets_uri` | `ASSETS_URI` |
| `scene_spec_uri` | `--scene-spec-uri` | `scene_spec_uri` | `SCENE_SPEC_URI` |
| `augment_image` | `--augment-image` | `augment_image` | `AUGMENT_IMAGE` |
| `policy_image` | `--policy-image` | `policy_image` | `POLICY_IMAGE` |
| `trainer_image` | `--trainer-image` | `trainer_image` | `TRAINER_IMAGE` |
| `vlm_image` | `--vlm-image` | `vlm_image` | `VLM_IMAGE` |
| `eval_image` | `--eval-image` | `eval_image` | `EVAL_IMAGE` |
| `k8s_isaac_cache_pvc` | `--k8s-isaac-cache-pvc` | `k8s_isaac_cache_pvc` | `NPA_SIM2REAL_ISAAC_CACHE_PVC` |
| `vlm_model` | `--vlm-model` | `vlm_model` | `VLM_MODEL` |
| `threshold` | `--threshold` | `threshold` | `SUCCESS_THRESHOLD` |
| `inner_iterations` | `--inner-iterations` | `inner_iterations` | `INNER_ITERATIONS` |
| `outer_iterations` | `--outer-iterations` | `outer_iterations` | `OUTER_ITERATIONS` |
| `loop_of_loops_iterations` | `--loop-of-loops-iterations` | `loop_of_loops_iterations` | `LOOP_OF_LOOPS_ITERATIONS` |
| `rollout_count` | `--rollout-count` | `rollout_count` | `ROLLOUT_COUNT` |
| `steps_per_rollout` | `--steps-per-rollout` | `steps_per_rollout` | `STEPS_PER_ROLLOUT` |
| `heldout_env_count` | `--heldout-env-count` | `heldout_env_count` | `HELDOUT_ENV_COUNT` |

## Runtime and outputs

The YAML exposes all 14 stages and runs through the standard workflow runtime.
Each real solution has its own image/resource state, S3 inputs and outputs, and
ComponentRecord. Parallel Stage 4 leaves publish attributable lane records;
Stage 8 is one CPU-only hosted Cosmos3 evaluator with a direct Stage 9 barrier. Stage 11 early
exit is explicit (`allow_early_exit`), Stage 13/14 use the completed loop
iteration, shard cardinality is validated before submission, and visualization
downloads only its declared artifact set into cleaned ephemeral storage.
Runtime values are operator inputs; this directory contains no
tenant, project, registry, bucket, cluster, credential, or run identifier.

`controller_image` must be the small CPU-only image built from
[npa/docker/workbench/sim2real-control/Dockerfile](../../../docker/workbench/sim2real-control/Dockerfile). It contains the exact source and
pinned S3 dependencies but no Genesis, Isaac, CUDA, trainer, or injected source
bootstrap. GPU solution images remain attached only to their corresponding
workflow states.

The legacy `npa.workflows.sim2real` controller modules have a finite compatibility
window for archived callers and artifacts. They are lazy, are not called by the
canonical workflow, and cannot materialize or submit its retired controller.

The submit path fails before launch when storage, secret propagation, gated
model access, the dedicated CPU capacity, Isaac cache PVC,
immutable images, or real image pulls are not ready. The linked runbook gives
copy-paste setup, expected results, and remediation without duplicating it here.
