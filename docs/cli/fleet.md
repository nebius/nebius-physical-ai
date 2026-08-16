# `npa fleet`

## Command Tree

```text
Usage: npa fleet [OPTIONS] COMMAND [ARGS]...

Deploy and manage fleets of Nebius Managed Kubernetes clusters across projects.

Options
--help  Show this message and exit.
Commands
plan  Show the resolved deployment plan without touching infrastructure.
deploy  Deploy the fleet: resolve/create projects and apply each cluster.
destroy  Destroy the fleet's spec-declared clusters (best-effort, per-target).
status  Show the last-known deployment state for the fleet.
verify-mig  Verify exact RTX PRO 6000 MIG labels, operands, and kubelet resources.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `plan` | Show the resolved deployment plan without touching infrastructure. |
| `deploy` | Deploy the fleet: resolve/create projects and apply each cluster. |
| `destroy` | Destroy the fleet's spec-declared clusters (best-effort, per-target). |
| `status` | Show the last-known deployment state for the fleet. |
| `verify-mig` | Verify exact RTX PRO 6000 MIG labels, operands, and kubelet resources. |

## Examples

```bash
npa fleet --help
npa fleet plan --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `fleet`.
