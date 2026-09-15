# `npa workbench alpamayo2-super`

## Command Tree

```text
Usage: npa workbench alpamayo2-super [OPTIONS] COMMAND [ARGS]...

NVIDIA Alpamayo 2 Super trajectory-inference workbench.

Options
--help  Show this message and exit.
Commands
sweep  Sweep scenarios with Ray; optionally refine a baseline report's hard cases.
infer  Run the real upstream expert trajectory inference and publish artifacts.
terms  Print separately applicable source, model, and dataset terms.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `sweep` | Sweep scenarios with Ray; optionally refine a baseline report's hard cases. |
| `infer` | Run the real upstream expert trajectory inference and publish artifacts. |
| `terms` | Print separately applicable source, model, and dataset terms. |

## Examples

```bash
npa workbench alpamayo2-super --help
npa workbench alpamayo2-super sweep --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `alpamayo2-super`.
