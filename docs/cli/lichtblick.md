# `npa workbench lichtblick`

## Command Tree

```text
Usage: npa workbench lichtblick [OPTIONS] COMMAND [ARGS]...

Lichtblick (MPL-2.0) - an open-source, Foxglove-compatible MCAP / ROS-bag log viewer.

Options
--help  Show this message and exit.
Commands
serve  Serve a robotics log in Lichtblick, staged from S3.
launch  Alias for ``serve``: stage and view a robotics log in Lichtblick.
status  Show Lichtblick tool status.
list  List artifact formats the Lichtblick viewer can open.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `serve` | Serve a robotics log in Lichtblick, staged from S3. |
| `launch` | Alias for ``serve``: stage and view a robotics log in Lichtblick. |
| `status` | Show Lichtblick tool status. |
| `list` | List artifact formats the Lichtblick viewer can open. |

## Examples

```bash
npa workbench lichtblick --help
npa workbench lichtblick serve --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `lichtblick`.
