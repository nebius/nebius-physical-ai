# `npa workbench vlm-eval`

## Command Tree

```text
Usage: npa workbench vlm-eval [OPTIONS] COMMAND [ARGS]...

VLM evaluation for sim-to-real pipeline gating.

Options
--help  Show this message and exit.
Commands
run  Score a rollout artifact with a VLM backend.
compare-judges  Compare two hosted judges without averaging their outcomes.
compare-preference  Compare two images under blinded labels in both orders.
review-visual  Write a separate audit-only rich visual review.
loop  Score every rollout under a prefix and write an aggregate task-success report.
benchmark  Sweep VLM-eval configs over a labeled rollout benchmark set.
workflow  Show the npa.workflow specs for VLM evaluation.
status  Show VLM eval backend status.
list  List available VLM eval backends.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Score a rollout artifact with a VLM backend. |
| `compare-judges` | Compare two hosted judges without averaging their outcomes. |
| `compare-preference` | Compare two images under blinded labels in both orders. |
| `review-visual` | Write a separate audit-only rich visual review. |
| `loop` | Score every rollout under a prefix and write an aggregate task-success report. |
| `benchmark` | Sweep VLM-eval configs over a labeled rollout benchmark set. |
| `workflow` | Show the npa.workflow specs for VLM evaluation. |
| `status` | Show VLM eval backend status. |
| `list` | List available VLM eval backends. |

## Examples

```bash
npa workbench vlm-eval --help
npa workbench vlm-eval run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `vlm-eval`.
