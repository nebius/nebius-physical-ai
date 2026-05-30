# `npa workbench vlm-eval`

## Command Tree

```text
Usage: npa workbench vlm-eval [OPTIONS] COMMAND [ARGS]...

VLM evaluation for sim-to-real pipeline gating.

Options
--help  Show this message and exit.
Commands
benchmark  Sweep VLM-eval configs over a labeled rollout benchmark set.
run  Score a rollout artifact with a VLM backend.
workflow  Show the SkyPilot YAML template for VLM evaluation.
status  Show VLM eval backend status.
list  List available VLM eval backends.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `benchmark` | Sweep VLM-eval configs over a labeled rollout benchmark set. |
| `run` | Score a rollout artifact with a VLM backend. |
| `workflow` | Show the SkyPilot YAML template for VLM evaluation. |
| `status` | Show VLM eval backend status. |
| `list` | List available VLM eval backends. |

## Examples

```bash
npa workbench vlm-eval --help
npa workbench vlm-eval run --help
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `vlm-eval`.
