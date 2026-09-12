# `npa workbench isaac-arena`

## Command Tree

```text
Usage: npa workbench isaac-arena [OPTIONS] COMMAND [ARGS]...

Isaac Lab-Arena policy evaluation with durable results.

Options
--install-completion  Install completion for the current shell.
--show-completion  Show completion for the current shell, to copy it or customize the installation.
--help  Show this message and exit.
Commands
capabilities  Print the pinned upstream surface and NPA support status as JSON.
evaluate  Evaluate a zero, replay, or RSL-RL policy with upstream's runner.
terms  Describe source and runtime redistribution boundaries.
```

## Options

| Option | Description |
| --- | --- |
| `--install-completion` | Install completion for the current shell. |
| `--show-completion` | Show completion for the current shell, to copy it or customize the installation. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `capabilities` | Print the pinned upstream surface and NPA support status as JSON. |
| `evaluate` | Evaluate a zero, replay, or RSL-RL policy with upstream's runner. |
| `terms` | Describe source and runtime redistribution boundaries. |

## Examples

```bash
npa workbench isaac-arena --help
npa workbench isaac-arena capabilities --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `isaac-arena`.
