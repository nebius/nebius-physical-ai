# `npa workbench vlm-eval`

## Command Tree

```text
Usage: npa workbench vlm-eval [OPTIONS] COMMAND [ARGS]...

VLM evaluation for sim-to-real pipeline gating.

Options
--help  Show this message and exit.
Commands
run  Score a rollout artifact with a VLM backend.
benchmark  Sweep VLM-eval configs over a labeled rollout benchmark set.
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
| `run` | Score a rollout artifact with a VLM backend. |
| `benchmark` | Sweep VLM-eval configs over a labeled rollout benchmark set. |
| `workflow` | Show the SkyPilot YAML template for VLM evaluation. |
| `status` | Show VLM eval backend status. |
| `list` | List available VLM eval backends. |

## Examples

```bash
npa workbench vlm-eval --help
npa workbench vlm-eval run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `vlm-eval`.
