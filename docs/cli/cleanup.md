# `npa cleanup`

## Command Tree

```text
Usage: npa cleanup [OPTIONS]

Report (or with --yes remove) local NPA/SkyPilot residue left after teardown.

Options
--yes  -y  Remove the local caches (otherwise just report).
--include-sky  --keep-sky  Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs
run).
[default: include-sky]
--project  <str>  Scope the empty per-alias state-dir report to this alias.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--yes` | -y  Remove the local caches (otherwise just report). |
| `--include-sky` | --keep-sky  Also remove SkyPilot's own ~/.sky state cache (safe once no clusters/jobs |
| `--project` | <str>  Scope the empty per-alias state-dir report to this alias. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa cleanup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cleanup`.
