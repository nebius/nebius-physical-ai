# `npa workbench health`

## Command Tree

```text
Usage: npa workbench health [OPTIONS] COMMAND [ARGS]...

Preflight health checks for workbench workflows.

Options
--help  Show this message and exit.
Commands
preflight  Validate service credentials and optional Nebius CLI authentication.
access  Check HF + NGC access to every gated model the workbench capabilities need.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `preflight` | Validate service credentials and optional Nebius CLI authentication. |
| `access` | Check HF + NGC access to every gated model the workbench capabilities need. |

## Examples

```bash
npa workbench health --help
npa workbench health preflight --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `health`.
