# `npa cluster node-group`

## Command Tree

```text
Usage: npa cluster node-group [OPTIONS] COMMAND [ARGS]...

Manage GPU node groups attached to NPA Workbench cluster targets.

Options
--help  Show this message and exit.
Commands
add  Attach a GPU node-group profile to an NPA Workbench cluster target.
add-cpu  Attach a CPU node-group profile to an NPA Workbench cluster target.
remove  Remove an NPA GPU node-group profile and clean up its local cache.
status  Show NPA GPU node-group target state from Nebius and the local cache.
list  List GPU node-group profiles attached to NPA Workbench cluster targets.

`npa cluster` manages NPA Workbench cluster targets and profiles. For raw MK8s administration (edit, update, upgrade,
operation inspection, version listing, compatibility matrix), use `nebius mk8s` directly.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `add` | Attach a GPU node-group profile to an NPA Workbench cluster target. |
| `add-cpu` | Attach a CPU node-group profile to an NPA Workbench cluster target. |
| `remove` | Remove an NPA GPU node-group profile and clean up its local cache. |
| `status` | Show NPA GPU node-group target state from Nebius and the local cache. |
| `list` | List GPU node-group profiles attached to NPA Workbench cluster targets. |

## Examples

```bash
npa cluster node-group --help
npa cluster node-group add --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `node-group`.
