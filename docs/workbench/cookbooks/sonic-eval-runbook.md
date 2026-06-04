# SONIC Export and Eval Runbook

This runbook covers the end-to-end SONIC locomotion path:

```text
policy checkpoint -> npa workbench sonic export -> npa workbench sonic eval
```

The SkyPilot blueprint is
`npa/workflows/workbench/skypilot/sonic-export-eval.yaml`.

## Prerequisites

- SkyPilot is bootstrapped and Nebius is enabled:

  ```bash
  npa skypilot bootstrap
  export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
  "$NPA_SKYPILOT_BIN" check
  ```

- The policy checkpoint is readable from the SkyPilot task. Use an `s3://`
  checkpoint URI for normal runs, or a pre-mounted local path for development.
- Object storage credentials are available to the task, and the endpoint is
  `https://storage.eu-north1.nebius.cloud`.
- The first-party SONIC image is built and pushed as
  `${NPA_REGISTRY}/npa-sonic:0.1.0`:

  ```bash
  export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
  npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push
  docker manifest inspect "${NPA_REGISTRY}/npa-sonic:0.1.0"
  ```

## One Command

Edit the `envs` block in the YAML for your checkpoint, output prefix, registry,
and image tag, then submit it:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/sonic-export-eval.yaml \
  --run-id sonic-export-eval-$(date -u +%Y%m%dT%H%M%SZ)
```

The default run requests `H100:1` and uses the reference backend:

```yaml
POLICY_CKPT: s3://<your-bucket-name>/sonic-locomotion/<run-id>/training/last.pt
OUTPUT_DIR: s3://<your-bucket-name>/sonic-locomotion/<run-id>/export-eval/
EVAL_BACKEND: reference
EVAL_ENV: sonic-locomotion-smoke
EPISODES: "8"
```

## Inputs

- `POLICY_CKPT`: trained SONIC checkpoint. `s3://` URIs are downloaded before
  export; local paths are used as-is.
- `SONIC_CONFIG`: optional YAML/JSON config when the checkpoint does not carry
  policy class, observation, action, normalization, or control-rate metadata.
- `SONIC_OBS_SPEC` and `SONIC_ACTION_SPEC`: optional explicit layout specs.
- `EVAL_BACKEND`: `reference` by default, or `container` for an external
  evaluator.
- `EPISODES`: rollout count for eval.

## Outputs

`OUTPUT_DIR` receives:

- `sonic_policy.onnx`
- `sonic_policy.metadata.json`
- `sonic_export_result.json`
- `sonic_eval_results.json`
- `sonic_eval_stdout.json`

The metrics JSON uses format `npa_sonic_eval_result_v1`. For a reference run,
check:

- `backend`: `reference`
- `mode`: `sim`
- `smoke_level`: `false`
- `metrics.distance_mean`: positive rollout distance
- `metrics.fall_rate`: expected to stay near `0.0` for a stable policy
- `metrics.valid_action_rate`: expected `1.0`
- `episodes`: per-episode rollout records

## BYO External Eval Container

Switch to a config-driven evaluator without changing the workflow code:

```yaml
EVAL_BACKEND: container
CONTAINER_IMAGE: cr.eu-north1.nebius.cloud/<your-registry-id>/<eval-image>:<tag>
CONTAINER_POLICY_PATH: /npa/eval/input/policy.onnx
CONTAINER_METADATA_PATH: /npa/eval/input/metadata.json
CONTAINER_OUTPUT_PATH: /npa/eval/output/sonic_eval_results.json
```

The container receives:

- `NPA_SONIC_ONNX`
- `NPA_SONIC_METADATA`
- `NPA_SONIC_OUTPUT`
- `NPA_SONIC_EPISODES`
- `NPA_SONIC_ENV`
- `NPA_SONIC_RESULT_FORMAT`

It must read the mounted ONNX and sidecar files, then write JSON to
`NPA_SONIC_OUTPUT`. If the JSON already uses `npa_sonic_eval_result_v1`, the CLI
preserves the supplied metrics. Otherwise, the raw payload is embedded under
`external_result`.

This image is BYO/customer-provided and is not the Workbench first-party
`npa-sonic` image. Leave `EVAL_BACKEND=reference` and `CONTAINER_IMAGE=""` when
you want the supported built-in evaluator.

## Troubleshooting

- `metadata sidecar missing`: keep `SONIC_METADATA=sidecar`, or pass the sidecar
  path explicitly to eval if using a custom export.
- `checkpoint not found`: confirm `POLICY_CKPT` is visible from the SkyPilot task
  and that S3 credentials are present.
- `ONNX parity check failed`: rerun with `SONIC_VERIFY=0` only to isolate the
  eval path; keep verification enabled for production exports.
- `container eval failed`: validate `CONTAINER_IMAGE` and the three container
  path variables. The container must write the result JSON before exiting.
- S3 upload/download errors: use
  `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud` and confirm bucket
  permissions with `aws s3 ls --endpoint-url "$AWS_ENDPOINT_URL"`.
