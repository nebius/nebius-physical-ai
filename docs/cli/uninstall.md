# `npa uninstall`

## Command Tree

```text
Usage: npa uninstall [OPTIONS]

Dry-run or safely remove only this repository's ``npa/.venv``.

Ordinary ``npa cleanup`` never removes the invoking environment. Actual
uninstall requires both --remove-environment and --yes and is performed by a
one-time helper after this process exits.

Options
--remove-environment  Opt in to deferred removal of the exact invoking repository-local venv.
--yes  -y  Confirm the exact environment-removal plan.
--status  <str>  Show one uninstall receipt by id and exit.
--retry  <str>  Retry a failed deferred uninstall receipt by id.
--json  Emit machine-readable output.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--remove-environment` | Opt in to deferred removal of the exact invoking repository-local venv. |
| `--yes` | -y  Confirm the exact environment-removal plan. |
| `--status` | <str>  Show one uninstall receipt by id and exit. |
| `--retry` | <str>  Retry a failed deferred uninstall receipt by id. |
| `--json` | Emit machine-readable output. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa uninstall --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `uninstall`.
