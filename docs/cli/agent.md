# `npa agent`

## Command Tree

```text
Usage: npa agent [OPTIONS] COMMAND [ARGS]...

Deploy and operate a public NPA chat agent VM.

Options
--help  Show this message and exit.
Commands
auth-profile  Complete a human Nebius CLI profile on this operator/dev VM.
list  List agent deployments recorded in ~/.npa/config.yaml.
preflight  Check Route C prerequisites before `npa agent deploy` / `fresh-setup`.
deploy  Provision VM + bootstrap the public NPA agent stack.
fresh-setup  Initialize fresh project config and deploy a new agent from scratch.
setup  Interactively deploy an agent VM into a project you already configured.
bootstrap  Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform).
status  Show agent status, URLs, and health checks.
destroy  Destroy agent VM/resources by exact identity; receipts need no project stanza.
verify-live  Exit 0 only when live infra checks and tests pass.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `auth-profile` | Complete a human Nebius CLI profile on this operator/dev VM. |
| `list` | List agent deployments recorded in ~/.npa/config.yaml. |
| `preflight` | Check Route C prerequisites before `npa agent deploy` / `fresh-setup`. |
| `deploy` | Provision VM + bootstrap the public NPA agent stack. |
| `fresh-setup` | Initialize fresh project config and deploy a new agent from scratch. |
| `setup` | Interactively deploy an agent VM into a project you already configured. |
| `bootstrap` | Re-bootstrap agent UI/backend/nginx on an existing VM (refresh without Terraform). |
| `status` | Show agent status, URLs, and health checks. |
| `destroy` | Destroy agent VM/resources by exact identity; receipts need no project stanza. |
| `verify-live` | Exit 0 only when live infra checks and tests pass. |

## Examples

```bash
npa agent --help
npa agent auth-profile --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `agent`.
