# `npa rerun`

## Command Tree

```text
Usage: npa rerun [OPTIONS] COMMAND [ARGS]...

Host and share Rerun .rrd recordings through app.rerun.io.

Options
--help  Show this message and exit.
Commands
host  Upload or reference a Rerun .rrd and print an app.rerun.io URL.
share  Create a durable S3-backed Rerun share URL, capped at 7 days.
list-shares  List shared Rerun recordings stored in the operator bucket.
revoke  Delete matching shared Rerun recordings from S3.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `host` | Upload or reference a Rerun .rrd and print an app.rerun.io URL. |
| `share` | Create a durable S3-backed Rerun share URL, capped at 7 days. |
| `list-shares` | List shared Rerun recordings stored in the operator bucket. |
| `revoke` | Delete matching shared Rerun recordings from S3. |

## Examples

```bash
npa rerun --help
npa rerun host --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `rerun`.
