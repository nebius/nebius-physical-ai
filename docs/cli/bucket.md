# `npa storage bucket`

## Command Tree

```text
Usage: npa storage bucket [OPTIONS] COMMAND [ARGS]...

Object-storage buckets.

Options
--help  Show this message and exit.
Commands
list  List the object-storage buckets in a project, marking the configured one.
cors  Check or configure the app.rerun.io browser CORS contract.
delete  Delete an object-storage bucket npa provisioned, contents and versions included.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list` | List the object-storage buckets in a project, marking the configured one. |
| `cors` | Check or configure the app.rerun.io browser CORS contract. |
| `delete` | Delete an object-storage bucket npa provisioned, contents and versions included. |

## Examples

```bash
npa storage bucket --help
npa storage bucket list --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `bucket`.
