# `npa soperator`

## Command Tree

```text
Usage: npa soperator [OPTIONS] COMMAND [ARGS]...

Deploy and manage Nebius soperator (Slurm-on-Kubernetes) clusters.

Options
--help  Show this message and exit.
Commands
plan  Show a public-safe, provider-free Soperator capacity plan.
deploy  Deploy or reconcile a pinned-contract Soperator cluster spec.
destroy  Destroy an npa-managed soperator cluster by name.
status  Show a soperator cluster's Slurm partitions/nodes via kubectl.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `plan` | Show a public-safe, provider-free Soperator capacity plan. |
| `deploy` | Deploy or reconcile a pinned-contract Soperator cluster spec. |
| `destroy` | Destroy an npa-managed soperator cluster by name. |
| `status` | Show a soperator cluster's Slurm partitions/nodes via kubectl. |

## Examples

```bash
npa soperator --help
npa soperator plan --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `soperator`.
