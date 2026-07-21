# `npa workbench token-factory`

## Command Tree

```text
Usage: npa workbench token-factory [OPTIONS] COMMAND [ARGS]...

Nebius Token Factory hosted inference (zero-GPU, OpenAI-compatible).

Options
--help  Show this message and exit.
Commands
caption  Caption a folder of images with a hosted Token Factory vision model.
generate  Generate completions for each prompt in a JSONL/text file.
reason  Reason over scene images for physical understanding and a plan of action.
models  List models available to the configured Token Factory API key.
verify  Verify Token Factory authentication with a live models call.
status  Show Token Factory connection status (no network call).
list  List Token Factory tool capabilities.
workflow  Show the checked-in Token Factory SkyPilot workflow templates.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `caption` | Caption a folder of images with a hosted Token Factory vision model. |
| `generate` | Generate completions for each prompt in a JSONL/text file. |
| `reason` | Reason over scene images for physical understanding and a plan of action. |
| `models` | List models available to the configured Token Factory API key. |
| `verify` | Verify Token Factory authentication with a live models call. |
| `status` | Show Token Factory connection status (no network call). |
| `list` | List Token Factory tool capabilities. |
| `workflow` | Show the checked-in Token Factory SkyPilot workflow templates. |

## Examples

```bash
npa workbench token-factory --help
npa workbench token-factory caption --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `token-factory`.
