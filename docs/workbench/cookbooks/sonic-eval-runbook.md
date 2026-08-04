# SONIC Export and Eval Runbook

This runbook covers the end-to-end SONIC locomotion path:

```text
policy checkpoint -> npa workbench sonic export -> npa workbench sonic eval
```

The blueprint is the `npa.workflow` spec
`npa/workflows/workbench/npa-workflows/sonic-export-eval.yaml` (its raw SkyPilot
template is retired).

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
- The first-party SONIC image is built and pushed with the variant that matches
  your GPU target. Use `0.1.2` for L40S VM targets and `0.1.2-k8s-runtime` for
  RTX PRO 6000 Blackwell Kubernetes targets:

  ```bash
  export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
  npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push --variant baked
  npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push --variant k8s --tag 0.1.2-k8s-runtime
  docker manifest inspect "${NPA_REGISTRY}/npa-sonic:0.1.2"
  docker manifest inspect "${NPA_REGISTRY}/npa-sonic:0.1.2-k8s-runtime"
  ```

  See `docs/workbench/sonic-image-catalog.md` for the compatibility matrix.

## One Command

Submit through the generic workflow command. The SONIC materializer fills the
first-party image and S3 endpoint as literal YAML values before SkyPilot sees
the workflow:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/sonic-export-eval.yaml \
  --run-id sonic-export-eval-$(date -u +%Y%m%dT%H%M%SZ) \
  --registry "${NPA_REGISTRY}" \
  --var bucket=<bucket> \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

The default run requests `H100:1` for both stages and uses the reference eval
backend. The spec's `config` block is the whole surface:

```yaml
config:
  bucket: example-bucket
  prefix: "runs/{{run.id}}/sonic-export-eval"
  episodes: "8"
  env: smoke
  checkpoint_uri: "s3://{{config.bucket}}/{{config.prefix}}/checkpoint.pt"
  onnx_uri: "s3://{{config.bucket}}/{{config.prefix}}/sonic_policy.onnx"
  eval_uri: "s3://{{config.bucket}}/{{config.prefix}}/eval.json"
```

Override any of them with `--var key=value` (e.g.
`--var checkpoint_uri=s3://...`) instead of editing a copy of the spec.

## Inputs

- `checkpoint_uri`: trained SONIC checkpoint. `s3://` URIs are downloaded by the
  tool before export; local paths are used as-is.
- `onnx_uri`: where the export stage writes the ONNX. Its sidecar metadata and any
  `<name>.onnx.data` external weights are published next to it, and the eval stage
  reads exactly this URI.
- `eval_uri`: where the eval stage writes `eval.json`.
- `episodes`: rollout count for eval. `env`: reference env (`smoke` by default).
- `--config` / `--obs-spec` / `--action-spec` and the container backend are CLI/SDK
  options that the toolRef argv does not carry yet (a pinned `spec_gap`); use
  `npa workbench sonic export|eval` directly for those.

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

- Docker GPU injection defaults to `CONTAINER_GPUS=all`, which uses Docker's
  `--gpus` flag. For NVIDIA CDI sidecars, set
  `CONTAINER_GPUS=nvidia.com/gpu=all`; the CLI then uses the NVIDIA runtime and
  passes that CDI device through `NVIDIA_VISIBLE_DEVICES` instead of `--gpus`.

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

- `metadata sidecar missing`: the export stage writes `<stem>.metadata.json` next
  to the ONNX and the eval stage stages both; if you point `--onnx` at an ONNX you
  produced elsewhere, publish its sidecar alongside it.
- `checkpoint not found`: confirm `config.checkpoint_uri` exists and that
  `--secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY` were passed,
  so the pod can read it.
- `observation dimension is required`: the checkpoint's policy exposes no
  `obs_dim`/`observation_dim`; pass `--obs-spec` on the CLI, or export from a
  policy that carries the attribute.
- the stage SUCCEEDS but `eval.json` is missing: check that the toolRef passes
  `--output <uri> --output-format json` and not `--output json` — that exact
  conflation silently wrote the result inside the pod (EVIDENCE §R5), and
  `test_tool_catalog_argv.py` now guards it.
- S3 upload/download errors: use
  `AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud` and confirm bucket
  permissions with `aws s3 ls --endpoint-url "$AWS_ENDPOINT_URL"`.
