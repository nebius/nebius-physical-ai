# `npa workbench intrinsic`

## Command Tree

```text
Usage: npa workbench intrinsic [OPTIONS] COMMAND [ARGS]...

Intrinsic Core host-only, read-only validation: preflight, ICON status, and digital-twin reachability (mutating operations not exposed).

Options
--help  Show this message and exit.
Commands
preflight  Check local Intrinsic Core prerequisites.
icon-status  Report read-only ICON real-time control status.
world-probe  Probe digital-twin (world) reachability without mutating state.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `preflight` | Check local Intrinsic Core prerequisites. |
| `icon-status` | Report read-only ICON real-time control status. |
| `world-probe` | Probe digital-twin (world) reachability without mutating state. |

## Examples

```bash
npa workbench intrinsic --help
npa workbench intrinsic preflight --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `intrinsic`.
