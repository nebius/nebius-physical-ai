# `npa storage service-account`

## Command Tree

```text
Usage: npa storage service-account [OPTIONS] COMMAND [ARGS]...

NPA-owned object-storage service accounts.

Options
--help  Show this message and exit.
Commands
delete  Delete the storage service account only when NPA recorded creating it.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `delete` | Delete the storage service account only when NPA recorded creating it. |

## Examples

```bash
npa storage service-account --help
npa storage service-account delete --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `service-account`.
