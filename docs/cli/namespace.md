# `npa workbench namespace`

## Command Tree

```text
Usage: npa workbench namespace [OPTIONS] COMMAND [ARGS]...

Manage team namespaces and researcher access.

Options
--help  Show this message and exit.
Commands
apply  Apply team access; omitted members lose this command's previous grants.
context  Prepare private client configuration without changing the source context.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `apply` | Apply team access; omitted members lose this command's previous grants. |
| `context` | Prepare private client configuration without changing the source context. |

## Examples

```bash
npa workbench namespace --help
npa workbench namespace apply --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `namespace`.
