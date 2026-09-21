# `npa workbench flex-pi`

## Command Tree

```text
Usage: npa workbench flex-pi [OPTIONS] COMMAND [ARGS]...

Flex-Pi multi-stream policy inference and public training.

Options
--help  Show this message and exit.
Commands
train  Train the pinned real public YAM workload on exactly four GPUs.
infer  Run genuine action-only inference and publish verified artifacts.
terms  Print separately applicable source, model, and public-input terms.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `train` | Train the pinned real public YAM workload on exactly four GPUs. |
| `infer` | Run genuine action-only inference and publish verified artifacts. |
| `terms` | Print separately applicable source, model, and public-input terms. |

## Examples

```bash
npa workbench flex-pi --help
npa workbench flex-pi train --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `flex-pi`.
