# Set up the Nebius AI Cloud key (optional)

The **Nebius AI Cloud key** (`NEBIUS_AI_CLOUD_KEY`) is an **optional** credential.
Nothing in the workbench requires it: `npa configure` writes it only when you
paste a value, and `npa` injects it into a job's environment only when it is
present. You can leave it blank and everything — GPU clusters, managed
Kubernetes, object storage, and hosted inference — still works.

> **TL;DR:** you almost certainly don't need this. Access to Nebius AI Cloud is
> authenticated by the **Nebius CLI profile** that `npa configure` sets up (a
> short-lived IAM access token), not by a static API key. Press **Enter** to skip
> the `NEBIUS_AI_CLOUD_KEY` prompt unless you were explicitly handed a Nebius AI
> Cloud API key to call a specific API.

## How Nebius AI Cloud auth actually works

Nebius AI Cloud does **not** use a long-lived static API key for its primary
API/CLI/Terraform authentication. Instead it uses short-lived **IAM access
tokens** (valid ~12 hours), which the `nebius` CLI mints on demand:

```bash
nebius iam get-access-token
```

`npa configure` creates or reuses a Nebius CLI profile for you, so `npa` already
authenticates to AI Cloud (compute, Kubernetes, storage provisioning) through
that profile. There is nothing extra to paste for normal use. See the Nebius docs
on [access tokens](https://docs.nebius.com/iam/authorization/access-tokens) and
[CLI setup](https://docs.nebius.com/cli/configure).

## Don't confuse it with these

Three Nebius credentials are easy to mix up — most workbench users only need the
first two:

| Credential | What it is | Where it lives in `npa` |
| --- | --- | --- |
| **IAM access token** | Short-lived (~12 h) token for AI Cloud API / CLI / Terraform. Minted by `nebius iam get-access-token`; used automatically via your CLI profile. | Nebius CLI profile (set up by `npa configure`) |
| **Object Storage access key** | AWS-compatible access key **id/secret** pair for S3 (object storage). Auto-created by `npa configure` when you provision a bucket. | `storage.access_key_id` / `storage.secret_access_key` |
| **Token Factory key** | Starts with `v1.`; OpenAI-compatible **hosted inference** key. Separate console. | `tokens.NEBIUS_TOKEN_FACTORY_KEY` — see [token-factory-key.md](token-factory-key.md) |

The optional `NEBIUS_AI_CLOUD_KEY` is a passthrough for any additional Nebius AI
Cloud API key you may have been issued for a specific API. If you don't have one,
skip it.

## If you do have a Nebius AI Cloud key

Set it interactively:

```bash
npa configure
# ... paste it at the "Nebius AI Cloud API key (NEBIUS_AI_CLOUD_KEY, optional)"
#     prompt, or press Enter to skip.
```

**Or by hand** in `~/.npa/credentials.yaml`:

```yaml
tokens:
  NEBIUS_AI_CLOUD_KEY: <paste-your-nebius-ai-cloud-api-key>
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

**Or via environment variable** (good for CI / one-off shells):

```bash
export NEBIUS_AI_CLOUD_KEY=<paste-your-nebius-ai-cloud-api-key>
```

When present, `npa` exports `NEBIUS_AI_CLOUD_KEY` into the environment of jobs it
launches; when absent it is simply omitted.

## See also

- [Hugging Face token](huggingface-token.md) — model + dataset downloads (incl. gated repos).
- [Nebius Token Factory key](token-factory-key.md) — zero-GPU hosted inference.
- [NVIDIA NGC API key](ngc-api-key.md) — GR00T / Cosmos container + model pulls.
- [Quickstart § credentials](../quickstart.md#4-configure-credentials)
