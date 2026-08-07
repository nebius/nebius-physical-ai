# `npa destroy`

## Command Tree

```text
Usage: npa destroy [OPTIONS]

Plan or execute project-scoped teardown through guarded NPA commands.

Options
*  --project  <str>  Exact configured project alias. [required]
    --all  Required acknowledgement to plan the full project lifecycle.
    --yes  -y  Execute the rendered plan.
    --delete-project  Report provider project-deletion support (currently plan-only/unsupported).
    --json  Emit JSON.
    --help  Show this message and exit.
```

## Options

No command-specific options are listed by `--help`.

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa destroy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `destroy`.
