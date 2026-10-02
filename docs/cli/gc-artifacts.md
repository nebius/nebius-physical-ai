# `npa workbench gc-artifacts`

## Command Tree

```text
Usage: npa workbench gc-artifacts [OPTIONS] COMMAND [ARGS]...

Garbage-collect expired workbench run artifacts from S3 (dry-run by default).

Options
--bucket  <str>  S3 bucket holding run artifacts. Defaults to NPA_S3_BUCKET.
--prefix  <str>  Scan only under this key prefix. Defaults to NPA_S3_PREFIX.
--retention-days  <int>  Delete terminal runs older than this many days. [default: 90]
--pin-marker  <str>  Marker file under a run prefix that exempts it from deletion. [default: .npa-retain]
--max-depth  <int>  How deep to search for run prefixes under --prefix. [default: 3]
--apply,--yes  Execute the plan. Without this flag nothing is deleted.
--json  Emit the plan as JSON instead of a table.
--journal  <path>  Write the JSON plan journal to this path.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--bucket` | <str>  S3 bucket holding run artifacts. Defaults to NPA_S3_BUCKET. |
| `--prefix` | <str>  Scan only under this key prefix. Defaults to NPA_S3_PREFIX. |
| `--retention-days` | <int>  Delete terminal runs older than this many days. [default: 90] |
| `--pin-marker` | <str>  Marker file under a run prefix that exempts it from deletion. [default: .npa-retain] |
| `--max-depth` | <int>  How deep to search for run prefixes under --prefix. [default: 3] |
| `--apply,--yes` | Execute the plan. Without this flag nothing is deleted. |
| `--json` | Emit the plan as JSON instead of a table. |
| `--journal` | <path>  Write the JSON plan journal to this path. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa workbench gc-artifacts --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `gc-artifacts`.
