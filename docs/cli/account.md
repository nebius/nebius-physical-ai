# `npa workbench team account`

## Command Tree

```text
Usage: npa workbench team account [OPTIONS] COMMAND [ARGS]...

Operator-managed users, personal keys, and optional SSO links.

Options
--help  Show this message and exit.
Commands
create  Create a permanent user with optional locally managed groups.
list  List local users and credential IDs without disclosing keys.
issue-key  Deliver a new personal access key to a new mode-0600 file.
revoke-key  Revoke a credential and the browser sessions created with it.
update  Disable an account or replace its local group membership.
link  Link a verified SSO subject to an existing account without changing ownership.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `create` | Create a permanent user with optional locally managed groups. |
| `list` | List local users and credential IDs without disclosing keys. |
| `issue-key` | Deliver a new personal access key to a new mode-0600 file. |
| `revoke-key` | Revoke a credential and the browser sessions created with it. |
| `update` | Disable an account or replace its local group membership. |
| `link` | Link a verified SSO subject to an existing account without changing ownership. |

## Examples

```bash
npa workbench team account --help
npa workbench team account create --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `account`.
