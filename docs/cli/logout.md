# `npa logout`

## Command Tree

```text
Usage: npa logout [OPTIONS]

Forget a local Workbench connection without revoking accounts or cloud login.

Args:
profile: Optional saved connection name.
output_format: Human text or JSON.
Returns:
None; prints whether a saved connection was removed.
Raises:
TeamError: Session storage cannot be safely updated.

Options
--profile  <str>  Connection to forget; defaults to the active connection.
--output-format  <str>  text or json. [default: text]
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--profile` | <str>  Connection to forget; defaults to the active connection. |
| `--output-format` | <str>  text or json. [default: text] |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa logout --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `logout`.
