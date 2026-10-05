# `npa workbench flux-action`

## Command Tree

```text
Usage: npa workbench flux-action [OPTIONS] COMMAND [ARGS]...

Fine-tune FLUX 3 Action on a declared robot embodiment.

Options
--help  Show this message and exit.
Commands
finetune  Index demonstrations, train the native policy, and export a verified checkpoint.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `finetune` | Index demonstrations, train the native policy, and export a verified checkpoint. |

## Examples

```bash
npa workbench flux-action --help
npa workbench flux-action finetune --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `flux-action`.
