# `npa workbench ltx2`

## Command Tree

```text
Usage: npa workbench ltx2 [OPTIONS] COMMAND [ARGS]...

LTX-2.5 licensing surface: declare the LTX-2.x Community License terms, stamp them onto generated video, and gate downstream training on them.

Options
--help  Show this message and exit.
Commands
terms  Print the LTX-2.x licence terms and the declaration this workbench requires.
declare  Validate the operator's licensing declaration from the environment.
stamp  Stamp the accepted licence terms onto the artifacts a run generated.
gate  Refuse or allow a downstream trainer to consume LTX-2.5 output.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `terms` | Print the LTX-2.x licence terms and the declaration this workbench requires. |
| `declare` | Validate the operator's licensing declaration from the environment. |
| `stamp` | Stamp the accepted licence terms onto the artifacts a run generated. |
| `gate` | Refuse or allow a downstream trainer to consume LTX-2.5 output. |

## Examples

```bash
npa workbench ltx2 --help
npa workbench ltx2 terms --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `ltx2`.
