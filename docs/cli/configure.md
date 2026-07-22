# `npa configure`

## Command Tree

```text
Usage: npa configure [OPTIONS]

Interactive credential and config setup guidance.

Options
--show  Print the credential/config file layout instead of prompting.
--interactive  --no-interactive  Force or disable interactive prompting (defaults to auto-detect
TTY).
--provision  --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key
(default). Reuse an existing bucket by name, or press Enter to
create a default npa-bucket with standard storage and a size cap.
Use --no-provision to enter existing S3 credentials.
[default: provision]
--token-factory-key  <str>  Store a Nebius Token Factory API key in ~/.npa/credentials.yaml
under tokens.NEBIUS_TOKEN_FACTORY_KEY (skips interactive setup).
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--show` | Print the credential/config file layout instead of prompting. |
| `--interactive` | --no-interactive  Force or disable interactive prompting (defaults to auto-detect |
| `--provision` | --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key |
| `--token-factory-key` | <str>  Store a Nebius Token Factory API key in ~/.npa/credentials.yaml |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa configure --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `configure`.
