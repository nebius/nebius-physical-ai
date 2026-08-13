# `npa registry`

## Command Tree

```text
Usage: npa registry [OPTIONS] COMMAND [ARGS]...

Inspect and tear down exact registries in NPA-created projects.

Options
--help  Show this message and exit.
Commands
delete  Delete one exact registry only with durable NPA project-creation proof.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `delete` | Delete one exact registry only with durable NPA project-creation proof. |

## Examples

```bash
npa registry --help
npa registry delete --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `registry`.
