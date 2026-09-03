# `npa workbench encord`

## Command Tree

```text
Usage: npa workbench encord [OPTIONS] COMMAND [ARGS]...

Encord curation SaaS: register-in-place push, headless curation, and curated pull.

Options
--help  Show this message and exit.
Commands
push  Register S3 media in Encord and optionally link a dataset.
curate  Headlessly curate a folder into a Collection via Encord quality filters.
pull  Pull curated media + labels + lineage manifest back to S3.
cleanup  Tear down run-scoped Encord state created by push/curate.
system-info  Show the Encord tool's SDK pin, API domain, and configured credentials.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `push` | Register S3 media in Encord and optionally link a dataset. |
| `curate` | Headlessly curate a folder into a Collection via Encord quality filters. |
| `pull` | Pull curated media + labels + lineage manifest back to S3. |
| `cleanup` | Tear down run-scoped Encord state created by push/curate. |
| `system-info` | Show the Encord tool's SDK pin, API domain, and configured credentials. |

## Examples

```bash
npa workbench encord --help
npa workbench encord push --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `encord`.
