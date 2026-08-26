# Set up a Hugging Face token

Many workbench models and datasets are hosted on [Hugging Face](https://huggingface.co).
A Hugging Face access token lets `npa` download private or **gated** assets and
raises rate limits. Public assets work anonymously. A token authenticating an
account that already has repository access is the complete automated preflight;
NPA does not add another acceptance switch.

> **TL;DR:** for gated or private assets, create a **Read** token at
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

Gated repositories require account access upstream — there is no API for NPA to
grant it. For each gated model you plan to use, open its page while signed in
and complete its access request. As reverified against the authoritative
Hugging Face API on 2026-08-14, the workbench's gated models include:

- <https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B>
- <https://huggingface.co/nvidia/Cosmos-Reason2-2B>
- <https://huggingface.co/nvidia/Cosmos-Reason2-8B>
- <https://huggingface.co/nvidia/Cosmos-Guardrail1>
- <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
- <https://huggingface.co/nvidia/Cosmos-1.0-Diffusion-7B-Text2World>
- <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> (only needed if you
  self-host it; Nebius Token Factory serves it hosted with no HF gating)

The public defaults `nvidia/GR00T-N1.7-3B`, `nvidia/GEAR-SONIC`,
`nvidia/Cosmos-Reason1-7B`, `nvidia/Cosmos3-Nano`, and the
`nvidia/PhysicalAI-NuRec-PPISP` dataset work anonymously. `npa` prints the exact
access URL for anything gated when you run the
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
npa workbench health access          # checks selected HF assets and NGC entitlement
npa workbench health access --capability paidf    # Cosmos Transfer only
# authenticate the configured token itself (no model download):
npa workbench health preflight --checks hf
# or, credentials-presence only (no network):
npa workbench health preflight --offline
```

`health access` reports `HF access ok: <repo>` for each model your token can
reach and, for anything still gated, the exact **Agree and access repository**
URL to open. Interactive `npa configure` runs the same repository-aware probe
(including the dataset API for gated datasets) and prints a bounded advisory
summary; use `health access` when you need an enforcing PASS/FAIL gate.
Generic online preflight calls Hugging Face's authenticated `whoami-v2`
endpoint; public repository metadata is not accepted as token proof.

## Troubleshooting

- **`401`/`403` on a gated repo** — you have not accepted that model's license.
  Open the model page while signed in and click **Agree and access repository**.
- **Token rejected everywhere** — the token is wrong or was revoked. Regenerate
  it at <https://huggingface.co/settings/tokens> and re-run `npa configure`.
- **Downloads are slow / rate-limited without a token** — public repos work
  without a token but authenticated downloads are faster and higher-limit; set
  `HF_TOKEN` anyway.

## See also

- [NVIDIA NGC API key](ngc-api-key.md) — for paths that actually pull entitlement-controlled NGC artifacts.
- [Nebius Token Factory key](token-factory-key.md) — zero-GPU hosted inference.
- [Quickstart § credentials](../quickstart.md#4-configure-credentials)
