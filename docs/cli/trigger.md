# `npa workbench workflow trigger`

## Command Tree

```text
Usage: npa workbench workflow trigger [OPTIONS] COMMAND [ARGS]...

Watch S3-compatible data prefixes and retrigger Workbench workflows.

Options
--help  Show this message and exit.
Commands
list-presets  List explicit workflow trigger/seed presets.
stage-preset  Runtime-fetch and stage a verified preset into a run-scoped trigger prefix.
run  Poll once and launch one pipeline run if new LeRobot data is present.
watch  Poll continuously and launch one pipeline run per new LeRobot data batch.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list-presets` | List explicit workflow trigger/seed presets. |
| `stage-preset` | Runtime-fetch and stage a verified preset into a run-scoped trigger prefix. |
| `run` | Poll once and launch one pipeline run if new LeRobot data is present. |
| `watch` | Poll continuously and launch one pipeline run per new LeRobot data batch. |

## Examples

```bash
npa workbench workflow trigger --help
npa workbench workflow trigger list-presets --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `trigger`.
