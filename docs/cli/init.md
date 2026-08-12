# `npa init`

## Command Tree

```text
Usage: npa init [OPTIONS]

Interactive credential and config setup guidance.

Options
--show  Print the credential/config file layout instead of prompting.
--interactive  --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY).
--provision  --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key (default). Reuse an existing bucket by name, or press Enter to create a default
    npa-bucket with standard storage and a size cap. Use --no-provision to enter existing S3 credentials.
    [default: provision]
--save-env-credentials  Persist supported environment credentials atomically with mode 0600.
--env  Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show
    --env)".
--src-s3-uri  <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without
    re-exporting it in every shell (skips interactive setup).
--tenant-id  <str>  Known Nebius tenant ID for prompt-free configure.
--project-id  <str>  Known Nebius project ID for prompt-free configure.
--region  <str>  Known Nebius project region for prompt-free configure.
--project-alias  <str>  Local NPA alias for prompt-free configure.
--container-registry  <str>  Optional non-secret registry override.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--show` | Print the credential/config file layout instead of prompting. |
| `--interactive` | --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY). |
| `--provision` | --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key (default). Reuse an existing bucket by name, or press Enter to create a default npa-bucket with standard storage and a size cap. Use --no-provision to enter existing S3 credentials. [default: provision] |
| `--save-env-credentials` | Persist supported environment credentials atomically with mode 0600. |
| `--env` | Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show --env)". |
| `--src-s3-uri` | <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without re-exporting it in every shell (skips interactive setup). |
| `--tenant-id` | <str>  Known Nebius tenant ID for prompt-free configure. |
| `--project-id` | <str>  Known Nebius project ID for prompt-free configure. |
| `--region` | <str>  Known Nebius project region for prompt-free configure. |
| `--project-alias` | <str>  Local NPA alias for prompt-free configure. |
| `--container-registry` | <str>  Optional non-secret registry override. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa init --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `init`.
