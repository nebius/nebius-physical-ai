# `npa workbench newton`

## Command Tree

```text
Usage: npa workbench newton [OPTIONS] COMMAND [ARGS]...

Newton physics engine: teacher training, demo generation, evaluation.

Options
--help  Show this message and exit.
Commands
train-teacher  Train a teacher policy in Newton simulation.
generate-demos  Generate demonstration rollouts from a Newton teacher policy.
eval  Evaluate a policy in Newton simulation.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `train-teacher` | Train a teacher policy in Newton simulation. |
| `generate-demos` | Generate demonstration rollouts from a Newton teacher policy. |
| `eval` | Evaluate a policy in Newton simulation. |

## Examples

```bash
npa workbench newton --help
npa workbench newton train-teacher --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `newton`.
