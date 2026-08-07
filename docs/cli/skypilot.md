# `npa skypilot`

## Command Tree

```text
Usage: npa skypilot [OPTIONS] COMMAND [ARGS]...

Manage the isolated SkyPilot runtime used by NPA workflows.

Options
--help  Show this message and exit.
Commands
bootstrap  Install SkyPilot into an isolated, idempotent virtualenv.
uninstall  Remove the isolated SkyPilot venv and clear the saved sky binary.
status  Report the isolated SkyPilot runtime status.
cleanup-controller  Tear down NPA's shared jobs controller after its managed jobs drain.
bind-controller  Bind the shared jobs controller to one immutable project/cluster identity.
verify  Run `sky check` against the isolated SkyPilot runtime.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `bootstrap` | Install SkyPilot into an isolated, idempotent virtualenv. |
| `uninstall` | Remove the isolated SkyPilot venv and clear the saved sky binary. |
| `status` | Report the isolated SkyPilot runtime status. |
| `cleanup-controller` | Tear down NPA's shared jobs controller after its managed jobs drain. |
| `bind-controller` | Bind the shared jobs controller to one immutable project/cluster identity. |
| `verify` | Run `sky check` against the isolated SkyPilot runtime. |

## Examples

```bash
npa skypilot --help
npa skypilot bootstrap --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `skypilot`.
