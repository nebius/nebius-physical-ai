# Set up a Hugging Face token

Many workbench models and datasets are hosted on [Hugging Face](https://huggingface.co).
A Hugging Face access token lets `npa` download them (including **gated** models
such as Llama and several NVIDIA Cosmos / GR00T assets). This guide creates one,
accepts the gated licenses, and wires it into `npa`.

> **TL;DR:** create a **Read** token at
> <https://huggingface.co/settings/tokens>, run `npa configure` and paste it at
> the `HF_TOKEN` prompt, then click **Agree and access repository** on each
> gated model page while signed in.

## 1. Create the token

1. Sign in (or create a free account) at <https://huggingface.co>.
2. Go to **Settings → Access Tokens**: <https://huggingface.co/settings/tokens>.
3. Click **Create new token**. A **Read** token is enough for downloads. (If you
   plan to *push* datasets/models, use a **Write** token.)
4. Name it (e.g. `npa-workbench`) and click **Create token**.
5. **Copy it now** — Hugging Face shows the value once. It starts with `hf_`.

## 2. Accept gated-model licenses (required for gated repos)

Gated repositories require **interactive** license acceptance — there is no API
to accept on your behalf. For each gated model you plan to use, open its page
while signed in and click **Agree and access repository**. The workbench's gated
models include, among others:

- <https://huggingface.co/nvidia/GR00T-N1.7-3B>
- <https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B>
- <https://huggingface.co/nvidia/Cosmos-Reason2-2B>
- <https://huggingface.co/nvidia/Cosmos-Reason2-8B>
- <https://huggingface.co/nvidia/Cosmos-Reason1-7B>
- <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> (only needed if you
  self-host it; Nebius Token Factory serves it hosted with no HF gating)

`npa` prints the exact acceptance URL for anything still gated when you run the
access check below, so you can also just react to that.

## 3. Give the token to `npa`

**Recommended — interactive setup:**

```bash
npa configure
# ... paste the token at the "Hugging Face token (HF_TOKEN)" prompt
```

**Or by hand** in `~/.npa/credentials.yaml`:

```yaml
tokens:
  HF_TOKEN: hf_XXXXXXXXXXXXXXXXXXXX   # your real token, pasted verbatim
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

**Or via environment variable** (good for CI / one-off shells):

```bash
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXX
```

`npa` resolves the token in this order: explicit CLI flag → environment
variable → `~/.npa/credentials.yaml`.

## 4. Verify access

```bash
npa workbench health access          # checks HF (and NGC) access to gated models
npa workbench health access --capability paidf    # Cosmos Transfer only
# or, credentials-presence only (no network):
npa workbench health preflight --offline
```

`health access` reports `HF access ok: <repo>` for each model your token can
reach and, for anything still gated, the exact **Agree and access repository**
URL to open.

## Troubleshooting

- **`401`/`403` on a gated repo** — you have not accepted that model's license.
  Open the model page while signed in and click **Agree and access repository**.
- **Token rejected everywhere** — the token is wrong or was revoked. Regenerate
  it at <https://huggingface.co/settings/tokens> and re-run `npa configure`.
- **Downloads are slow / rate-limited without a token** — public repos work
  without a token but authenticated downloads are faster and higher-limit; set
  `HF_TOKEN` anyway.

## See also

- [NVIDIA NGC API key](ngc-api-key.md) — for GR00T / Cosmos container + model pulls.
- [Nebius Token Factory key](token-factory-key.md) — zero-GPU hosted inference.
- [Quickstart § credentials](../quickstart.md#4-configure-credentials)
