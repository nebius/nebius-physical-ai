# `npa cleanup`

## Command Tree

```text
Usage: npa cleanup [OPTIONS] COMMAND [ARGS]...

Report and remove NPA teardown leftovers.

Options
--yes  Remove the local caches listed in the report. Never touches cloud resources.
--keep-sky  With --yes, keep ~/.sky (SkyPilot's own state) in place.
--skip-jobs  Do not query the SkyPilot managed-job queue.
--sky-bin  <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution.
--json  Emit a JSON report.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--yes` | Remove the local caches listed in the report. Never touches cloud resources. |
| `--keep-sky` | With --yes, keep ~/.sky (SkyPilot's own state) in place. |
| `--skip-jobs` | Do not query the SkyPilot managed-job queue. |
| `--sky-bin` | <str>  SkyPilot executable path. Defaults to NPA_SKYPILOT_BIN or PATH resolution. |
| `--json` | Emit a JSON report. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa cleanup --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cleanup`.
