# `npa network`

## Command Tree

```text
Usage: npa network [OPTIONS] COMMAND [ARGS]...

Network operations for Nebius resources.

Options
--help  Show this message and exit.
Commands
ensure-ingress  Ensure TCP ingress to a VM security group.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `ensure-ingress` | Ensure TCP ingress to a VM security group. |

## Examples

```bash
npa network --help
npa network ensure-ingress --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `network`.
