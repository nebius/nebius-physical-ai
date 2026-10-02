# `npa workbench namespace`

## Command Tree

```text
Usage: npa workbench namespace [OPTIONS] COMMAND [ARGS]...

Create or select Kubernetes namespaces.

Options
--help  Show this message and exit.
Commands
apply  Create a missing namespace; reuse existing namespaces without changing access.
context  Select a namespace privately, preserving existing credentials and access.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `apply` | Create a missing namespace; reuse existing namespaces without changing access. |
| `context` | Select a namespace privately, preserving existing credentials and access. |

## Examples

```bash
npa workbench namespace --help
npa workbench namespace apply --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `namespace`.
