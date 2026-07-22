# `npa workbench byof`

## Command Tree

```text
Usage: npa workbench byof [OPTIONS] COMMAND [ARGS]...

Onboard an OSS repo as a BYOF container (Tier 0 of the OSS ladder).

Options
--help  Show this message and exit.
Commands
run  Build/push a BYOF image and optionally run a live workload.
ladder  Show the OSS onboarding ladder (Tier 0 -> Tier 2).
status  Report BYOF packaging surfaces (CLI / SDK / YAML).
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Build/push a BYOF image and optionally run a live workload. |
| `ladder` | Show the OSS onboarding ladder (Tier 0 -> Tier 2). |
| `status` | Report BYOF packaging surfaces (CLI / SDK / YAML). |

## Examples

```bash
npa workbench byof --help
npa workbench byof run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `byof`.
