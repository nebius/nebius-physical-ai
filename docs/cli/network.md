# `npa network`

## Command Tree

```text
Usage: npa network [OPTIONS] COMMAND [ARGS]...

Network operations for Nebius resources.

Options
--help  Show this message and exit.
Commands
delete-project-default  Delete only the unique default topology of an NPA-created project.
ensure-ingress  Ensure TCP ingress to a VM security group.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `delete-project-default` | Delete only the unique default topology of an NPA-created project. |
| `ensure-ingress` | Ensure TCP ingress to a VM security group. |

## Examples

```bash
npa network --help
npa network delete-project-default --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `network`.
