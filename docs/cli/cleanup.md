# `npa cleanup`

## Command Tree

```text
Usage: npa cleanup [OPTIONS]

Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

Local only. Cloud resources (agent VM, cluster, bucket, IAM) are removed by
the commands in the printed runbook -- `--yes` never deletes anything in the
cloud, which the report says explicitly so it is not mistaken for a teardown.

Options
--yes  -y  Remove the local caches (otherwise just report). Local only: this never
deletes cloud resources -- see the printed runbook for those.
--include-sky  --keep-sky  Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs
run).
[default: include-sky]
--project  <str>  Scope the empty per-alias state-dir report to this alias.
--skip-jobs  Do not query the SkyPilot managed-job queue.
--sky-bin  <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--yes` | -y  Remove the local caches (otherwise just report). Local only: this never |
| `--include-sky` | --keep-sky  Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs |
| `--project` | <str>  Scope the empty per-alias state-dir report to this alias. |
| `--skip-jobs` | Do not query the SkyPilot managed-job queue. |
| `--sky-bin` | <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa cleanup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cleanup`.
