# `npa burst`

## Command Tree

```text
Usage: npa burst [OPTIONS] COMMAND [ARGS]...

Submit and inspect cold-start multi-node SkyPilot GPU jobs.

Options
--help  Show this message and exit.
Commands
submit  Submit one coupled multi-node burst job.
submit-yaml  Submit one rendered workbench YAML task through the burst path.
status  Query a burst job status.
logs  Stream or fetch burst job logs.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `submit` | Submit one coupled multi-node burst job. |
| `submit-yaml` | Submit one rendered workbench YAML task through the burst path. |
| `status` | Query a burst job status. |
| `logs` | Stream or fetch burst job logs. |

## Examples

```bash
npa burst --help
npa burst submit --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `burst`.
