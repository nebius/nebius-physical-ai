# `npa workbench cosmos-evaluator`

## Command Tree

```text
Usage: npa workbench cosmos-evaluator [OPTIONS] COMMAND [ARGS]...

Cosmos Evaluator checks plus NPA source-relative temporal and protected-appearance diagnostics.

Options
--help  Show this message and exit.
Commands
evaluate  Grade every augmented variant of a run and write one evaluator report.
hallucination  Score hallucinated motion in one augmented clip.
attribute-verify  Verify one clip's augmented attributes with an LLM + VLM question pass.
engine  Report which evaluator engine this environment resolves to.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `evaluate` | Grade every augmented variant of a run and write one evaluator report. |
| `hallucination` | Score hallucinated motion in one augmented clip. |
| `attribute-verify` | Verify one clip's augmented attributes with an LLM + VLM question pass. |
| `engine` | Report which evaluator engine this environment resolves to. |

## Examples

```bash
npa workbench cosmos-evaluator --help
npa workbench cosmos-evaluator evaluate --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `cosmos-evaluator`.
