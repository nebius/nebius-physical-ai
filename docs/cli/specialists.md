# `npa workbench specialists`

## Command Tree

```text
Usage: npa workbench specialists [OPTIONS] COMMAND [ARGS]...

Self-hosted specialist agents and durable task monitoring.

Options
*  --config  <file>  [required]
    --help  Show this message and exit.
Commands
serve  Serve the browser monitor and optionally supervise independent workers.
worker  Run one specialist independently of the monitor.
submit  Enqueue a goal and emit its durable JSON receipt.
status  Emit observed team or task state as JSON.
pause  Pause a task/profile, or resume it with --resume.
reconcile  Record an inspected operation result or explicitly authorize retry.
cancel  Stop a task at its next durable boundary; external workloads remain independent.
```

## Options

No command-specific options are listed by `--help`.

## Subcommands

| Command | Description |
| --- | --- |
| `serve` | Serve the browser monitor and optionally supervise independent workers. |
| `worker` | Run one specialist independently of the monitor. |
| `submit` | Enqueue a goal and emit its durable JSON receipt. |
| `status` | Emit observed team or task state as JSON. |
| `pause` | Pause a task/profile, or resume it with --resume. |
| `reconcile` | Record an inspected operation result or explicitly authorize retry. |
| `cancel` | Stop a task at its next durable boundary; external workloads remain independent. |

## Examples

```bash
npa workbench specialists --help
npa workbench specialists serve --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `specialists`.
