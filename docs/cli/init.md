# `npa init`

## Command Tree

```text
Usage: npa init [OPTIONS]

Interactive credential and config setup guidance.

Options
--show  Print the credential/config file layout instead of prompting.
--interactive  --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY).
--provision  --no-provision  Explicitly create or reuse Nebius object storage. Interactive setup offers provisioning by default; prompt-free known-project setup is
    project-only unless --provision is passed. --no-provision performs no provider calls or storage adoption.
--save-env-credentials  Persist supported environment credentials atomically with mode 0600.
--env  Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show
    --env)".
--src-s3-uri  <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without
    re-exporting it in every shell (skips interactive setup).
--tenant-id  <str>  Known Nebius tenant ID for prompt-free configure.
--project-id  <str>  Known Nebius project ID for prompt-free configure.
--region  <str>  Known Nebius project region for prompt-free configure.
--project-alias  <str>  Local NPA alias for prompt-free configure.
--bucket-storage-class  <str>  Storage class for a newly created known-project bucket: standard, enhanced, or intelligent.
--bucket-size-gb  <str>  GiB cap for a newly created known-project bucket; 0 means unlimited.
--prepare-catalog-access  Audit full HF/NGC catalog access.
--open-approval-pages  Affirmatively open missing official HF/NGC approval pages.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--show` | Print the credential/config file layout instead of prompting. |
| `--interactive` | --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY). |
| `--provision` | --no-provision  Explicitly create or reuse Nebius object storage. Interactive setup offers provisioning by default; prompt-free known-project setup is project-only unless --provision is passed. --no-provision performs no provider calls or storage adoption. |
| `--save-env-credentials` | Persist supported environment credentials atomically with mode 0600. |
| `--env` | Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show --env)". |
| `--src-s3-uri` | <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without re-exporting it in every shell (skips interactive setup). |
| `--tenant-id` | <str>  Known Nebius tenant ID for prompt-free configure. |
| `--project-id` | <str>  Known Nebius project ID for prompt-free configure. |
| `--region` | <str>  Known Nebius project region for prompt-free configure. |
| `--project-alias` | <str>  Local NPA alias for prompt-free configure. |
| `--bucket-storage-class` | <str>  Storage class for a newly created known-project bucket: standard, enhanced, or intelligent. |
| `--bucket-size-gb` | <str>  GiB cap for a newly created known-project bucket; 0 means unlimited. |
| `--prepare-catalog-access` | Audit full HF/NGC catalog access. |
| `--open-approval-pages` | Affirmatively open missing official HF/NGC approval pages. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa init --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `init`.
