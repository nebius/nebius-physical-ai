# `npa workbench molmoact`

## Command Tree

```text
Usage: npa workbench molmoact [OPTIONS] COMMAND [ARGS]...

MolmoAct VLA: validate fine-tune/serve/eval configs (planning only; execution not implemented).

Options
--help  Show this message and exit.
Commands
finetune  Fine-tune a MolmoAct policy on a demonstration dataset.
serve  Serve a MolmoAct policy behind an HTTP endpoint.
eval  Evaluate a MolmoAct policy on an evaluation dataset.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `finetune` | Fine-tune a MolmoAct policy on a demonstration dataset. |
| `serve` | Serve a MolmoAct policy behind an HTTP endpoint. |
| `eval` | Evaluate a MolmoAct policy on an evaluation dataset. |

## Examples

```bash
npa workbench molmoact --help
npa workbench molmoact finetune --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `molmoact`.
