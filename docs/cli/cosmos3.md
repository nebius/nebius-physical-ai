# `npa workbench cosmos3`

## Command Tree

```text
Usage: npa workbench cosmos3 [OPTIONS] COMMAND [ARGS]...

Cosmos3 omni-model generation and reasoning workflow contracts.

Options
--help  Show this message and exit.
Commands
checkpoint-eval  Evaluate Cosmos3 text-to-image checkpoints in guarded, load-once batches.
generate  Run a real Cosmos 3 generation with the omni model.
reason  Build the Cosmos3 reason stage manifest.
text-to-image  Generate an image from a prompt with the Cosmos3 framework, and publish it.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `checkpoint-eval` | Evaluate Cosmos3 text-to-image checkpoints in guarded, load-once batches. |
| `generate` | Run a real Cosmos 3 generation with the omni model. |
| `reason` | Build the Cosmos3 reason stage manifest. |
| `text-to-image` | Generate an image from a prompt with the Cosmos3 framework, and publish it. |

## Examples

```bash
npa workbench cosmos3 --help
npa workbench cosmos3 checkpoint-eval --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cosmos3`.
