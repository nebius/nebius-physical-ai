# `npa destroy`

## Command Tree

```text
Usage: npa destroy [OPTIONS]

Plan or execute project-scoped teardown through guarded NPA commands.

Options
--project  <str>  Exact configured project alias.
--receipt  <str>  Opaque durable receipt for post-forget exact project deletion.
--all  Required acknowledgement to plan the full project lifecycle.
--yes  -y  Execute the rendered plan.
--delete-project  Also delete an exact, empty project with durable NPA creation proof.
--json  Emit JSON.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | <str>  Exact configured project alias. |
| `--receipt` | <str>  Opaque durable receipt for post-forget exact project deletion. |
| `--all` | Required acknowledgement to plan the full project lifecycle. |
| `--yes` | -y  Execute the rendered plan. |
| `--delete-project` | Also delete an exact, empty project with durable NPA creation proof. |
| `--json` | Emit JSON. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa destroy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `destroy`.
