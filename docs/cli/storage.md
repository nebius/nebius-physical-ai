# `npa storage`

## Command Tree

```text
Usage: npa storage [OPTIONS] COMMAND [ARGS]...

Inspect and tear down npa-managed object storage.

Options
--help  Show this message and exit.
Commands
bucket  Object-storage buckets.
service-account  NPA-owned object-storage service accounts.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `bucket` | Object-storage buckets. |
| `service-account` | NPA-owned object-storage service accounts. |

## Examples

```bash
npa storage --help
npa storage bucket --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `storage`.
