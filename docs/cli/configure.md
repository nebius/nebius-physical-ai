# `npa configure`

## Command Tree

```text
Usage: npa configure [OPTIONS]

Interactive credential and config setup guidance.

Options
--show  Print the credential/config file layout instead of prompting.
--interactive  --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY).
--provision  --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key (default). Reuse an existing bucket by name, or press Enter to create a default
    npa-bucket with standard storage and a size cap. Use --no-provision to enter existing S3 credentials.
    [default: provision]
--token-factory-key  <str>  Store a Nebius Token Factory API key in ~/.npa/credentials.yaml under tokens.NEBIUS_TOKEN_FACTORY_KEY, then continue the rest of setup.
--hf-token  <str>  Store a Hugging Face token in ~/.npa/credentials.yaml under tokens.HF_TOKEN without prompting (for scripted setup).
--ngc-api-key  <str>  Store an NVIDIA NGC API key in ~/.npa/credentials.yaml under ngc.api_key without prompting (for scripted setup).
--env  Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show
    --env)".
--forget-project  <str>  Remove a project stanza (and its terraform_state) from ~/.npa/config.yaml, then exit - the inverse of writing it. Use `npa storage bucket delete`
    and `npa agent destroy` to clean up the cloud resources and their credentials first.
--src-s3-uri  <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without
    re-exporting it in every shell (skips interactive setup).
--tenant-id  <str>  Known Nebius tenant ID for prompt-free configure (requires the other known-project flags).
--project-id  <str>  Known Nebius project ID for prompt-free configure (requires the other known-project flags).
--region  <str>  Known Nebius project region for prompt-free configure.
--project-alias  <str>  Local NPA alias to write for the known project (prompt-free configure).
--container-registry  <str>  Optional non-secret registry override for prompt-free configure.
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--show` | Print the credential/config file layout instead of prompting. |
| `--interactive` | --no-interactive  Force or disable interactive prompting (defaults to auto-detect TTY). |
| `--provision` | --no-provision  Auto-create a Nebius S3 bucket (when missing) and an access key (default). Reuse an existing bucket by name, or press Enter to create a default npa-bucket with standard storage and a size cap. Use --no-provision to enter existing S3 credentials. [default: provision] |
| `--token-factory-key` | <str>  Store a Nebius Token Factory API key in ~/.npa/credentials.yaml under tokens.NEBIUS_TOKEN_FACTORY_KEY, then continue the rest of setup. |
| `--hf-token` | <str>  Store a Hugging Face token in ~/.npa/credentials.yaml under tokens.HF_TOKEN without prompting (for scripted setup). |
| `--ngc-api-key` | <str>  Store an NVIDIA NGC API key in ~/.npa/credentials.yaml under ngc.api_key without prompting (for scripted setup). |
| `--env` | Print the saved project/bucket/kube-context values as NPA_* shell assignments (no secrets) instead of prompting: eval "$(npa configure --show --env)". |
| `--forget-project` | <str>  Remove a project stanza (and its terraform_state) from ~/.npa/config.yaml, then exit - the inverse of writing it. Use `npa storage bucket delete` and `npa agent destroy` to clean up the cloud resources and their credentials first. |
| `--src-s3-uri` | <str>  Persist the staged npa source prefix (s3://bucket/prefix/npa) in ~/.npa/config.yaml so workflow submits resolve NPA_SRC_S3_URI without re-exporting it in every shell (skips interactive setup). |
| `--tenant-id` | <str>  Known Nebius tenant ID for prompt-free configure (requires the other known-project flags). |
| `--project-id` | <str>  Known Nebius project ID for prompt-free configure (requires the other known-project flags). |
| `--region` | <str>  Known Nebius project region for prompt-free configure. |
| `--project-alias` | <str>  Local NPA alias to write for the known project (prompt-free configure). |
| `--container-registry` | <str>  Optional non-secret registry override for prompt-free configure. |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa configure --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `configure`.
