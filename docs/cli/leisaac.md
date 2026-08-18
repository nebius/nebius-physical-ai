# `npa workbench leisaac`

## Command Tree

```text
Usage: npa workbench leisaac [OPTIONS] COMMAND [ARGS]...

LeIsaac SO101 browser teleoperation on the RTX PRO 6000 Kubernetes pool.

Options
--help  Show this message and exit.
Commands
list-tasks  List the pinned SO101 tasks that support browser keyboard control.
export-paidf  Export one finalized episode directly from S3 into a PAIDF run input.
materialize-paidf  Create an immutable derived dataset after strict PAIDF video alignment.
launch  Launch a supported SO101 task and publish its secure collector capability.
reconnect-agent  Reconnect one existing private LeIsaac run to a replacement agent.
status  Report the live Kubernetes objects for a LeIsaac run.
destroy  Delete this run's transient GPU deployment and LBs, preserving S3 evidence.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list-tasks` | List the pinned SO101 tasks that support browser keyboard control. |
| `export-paidf` | Export one finalized episode directly from S3 into a PAIDF run input. |
| `materialize-paidf` | Create an immutable derived dataset after strict PAIDF video alignment. |
| `launch` | Launch a supported SO101 task and publish its secure collector capability. |
| `reconnect-agent` | Reconnect one existing private LeIsaac run to a replacement agent. |
| `status` | Report the live Kubernetes objects for a LeIsaac run. |
| `destroy` | Delete this run's transient GPU deployment and LBs, preserving S3 evidence. |

## Examples

```bash
npa workbench leisaac --help
npa workbench leisaac list-tasks --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `leisaac`.
