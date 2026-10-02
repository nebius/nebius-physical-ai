# `npa workbench workflow challenge`

## Command Tree

```text
Usage: npa workbench workflow challenge [OPTIONS] COMMAND [ARGS]...

Prepare a BEHAVIOR DEV evaluation from one setup file.

Options
--help  Show this message and exit.
Commands
init  Create an editable BEHAVIOR setup file without provisioning resources.
check  Check local prerequisites; GPU readiness requires runtime preflight.
prepare  Freeze ten prescribed DEV cases and generate standard Workbench commands.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `init` | Create an editable BEHAVIOR setup file without provisioning resources. |
| `check` | Check local prerequisites; GPU readiness requires runtime preflight. |
| `prepare` | Freeze ten prescribed DEV cases and generate standard Workbench commands. |

## Examples

```bash
npa workbench workflow challenge --help
npa workbench workflow challenge init --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `challenge`.
