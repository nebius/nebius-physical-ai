# `npa workbench sonic`

## Command Tree

```text
Usage: npa workbench sonic [OPTIONS] COMMAND [ARGS]...

NVIDIA GEAR-SONIC whole-body-control workbench.

Options
--project  -p  TEXT  Project alias from ~/.npa/config.yaml.
--name  -n  TEXT  Workbench instance name within the project.
--help  Show this message and exit.
Commands
deploy  Prepare or plan a SONIC runtime.
train  Run SONIC Isaac Lab training or smoke validation.
export  Export a SONIC locomotion policy to deterministic-action ONNX.
eval  Evaluate an exported SONIC ONNX locomotion policy.
serve  Launch or describe a SONIC serving path.
status  Inspect SONIC runtime state.
list  List configured SONIC workbenches and default model artifacts.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  TEXT  Project alias from ~/.npa/config.yaml. |
| `--name` | -n  TEXT  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Prepare or plan a SONIC runtime. |
| `train` | Run SONIC Isaac Lab training or smoke validation. |
| `export` | Export a SONIC locomotion policy to deterministic-action ONNX. |
| `eval` | Evaluate an exported SONIC ONNX locomotion policy. |
| `serve` | Launch or describe a SONIC serving path. |
| `status` | Inspect SONIC runtime state. |
| `list` | List configured SONIC workbenches and default model artifacts. |

## Examples

```bash
npa workbench sonic --help
npa workbench sonic deploy --help
```

## Eval Result JSON

`npa workbench sonic eval` writes `npa_sonic_eval_result_v1` JSON. Required
top-level fields are `format`, `status`, `backend`, `mode`, `smoke_level`,
`policy`, `eval`, `metrics`, `episodes`, and `warnings`. Required metrics are
`episode_return_mean`, `distance_mean`, `fall_rate`, `termination_rate`,
`episode_length_mean`, and `valid_action_rate`.

The reference backend runs real rollout loops for `locomotion-smoke` and
`sonic-locomotion-smoke` with the built-in locomotion simulator
(`mode=sim`, `smoke_level=false`). It can also run a configured local simulator
when `--env` names one that is importable through gymnasium. Without a wireable
simulator it emits a clearly marked smoke-level result (`mode=smoke`,
`smoke_level=true`) after feeding representative observations through the ONNX
policy.

## Container Eval Contract

The container backend stages the exported policy and sidecar, then runs:

```bash
npa workbench sonic eval \
  --onnx exported/sonic_policy.onnx \
  --metadata exported/sonic_policy.metadata.json \
  --backend container \
  --container-image <eval-image> \
  --output sonic_eval_results.json
```

By default the container reads `/npa/eval/input/policy.onnx` and
`/npa/eval/input/metadata.json`, writes
`/npa/eval/output/sonic_eval_results.json`, and receives the same paths through
`NPA_SONIC_ONNX`, `NPA_SONIC_METADATA`, and `NPA_SONIC_OUTPUT`. It also receives
`NPA_SONIC_EPISODES`, `NPA_SONIC_ENV`, and `NPA_SONIC_RESULT_FORMAT`. The paths
can be changed with `--container-policy-path`, `--container-metadata-path`, and
`--container-output-path` without code changes.

Regenerate this page with `bash scripts/build_docs.sh` after changing `sonic`.
