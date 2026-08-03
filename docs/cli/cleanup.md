# `npa cleanup`

## Command Tree

```text
Usage: npa cleanup [OPTIONS]

Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

Local only. Cloud resources (agent VM, cluster, bucket, IAM) are removed by
the commands in the printed runbook. Existing `--yes` keeps credentials and
config; add `--full` to remove known shared-service credentials and empty
NPA-owned local state. Neither scope deletes anything in the cloud.

Options
--yes  -y  Remove the local caches (otherwise just report). Local only: this never
deletes cloud resources -- see the printed runbook for those.
--include-sky  --keep-sky  Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs
run).
[default: include-sky]
--full  Broaden --yes to also remove locally saved HF, Token Factory, and NGC
credentials, then prune empty config/known ~/.npa state. Non-empty and
unrelated data is preserved.
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
| `--full` | Broaden --yes to also remove locally saved HF, Token Factory, and NGC |
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
