# Set up a Hugging Face token

Many workbench models and datasets are hosted on [Hugging Face](https://huggingface.co).
A Hugging Face access token lets `npa` download private or **gated** assets and
raises rate limits. Public assets work anonymously. Tokens inherit the owning
account's access; tokens do not own or accept licences. NPA verifies entitlement
with an exact-revision payload-byte authorization probe and does not add another
acceptance switch.

> **TL;DR:** for gated or private assets, create a **Read** token at
> <https://huggingface.co/settings/tokens>, run `npa configure` and paste it at
> the `HF_TOKEN` prompt, then run `npa configure --prepare-catalog-access`.
> NPA can open exact official pages after you consent, but only you can complete
> an access request while signed in.

## 1. Create the token

1. Sign in (or create a free account) at <https://huggingface.co>.
2. Go to **Settings → Access Tokens**: <https://huggingface.co/settings/tokens>.
3. Click **Create new token**. A **Read** token is enough for downloads. (If you
   plan to *push* datasets/models, use a **Write** token.)
4. Name it (e.g. `npa-workbench`) and click **Create token**.
5. **Copy it now** — Hugging Face shows the value once. It starts with `hf_`.

## 2. Accept gated-model licenses (required for gated repos)

Gated repositories require account access upstream — there is no API for NPA to
grant it. The catalog evolves, so use its source-of-truth audit instead of a
copied list:

```bash
npa configure --prepare-catalog-access
```

Token Factory is a separate optional hosted product and is not included in this
HF/NGC approval plan.

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
npa workbench health access --prepare
npa workbench health access --capability paidf --prepare
npa workbench health access --capability paidf --prepare --json  # never prompts/opens
# authenticate the configured token itself (no model download):
npa workbench health preflight --checks hf
# or, credentials-presence only (no network):
npa workbench health preflight --offline
```

`health access --prepare` probes a catalogued payload path at the exact model or
dataset revision with HEAD or a one-byte Range request; it never downloads the
payload. Repository/revision metadata, README, model-card, licence, tokenizer,
and config files cannot produce Ready. Ready means exact technical fetch
entitlement only, never legal acceptance. The command records only
Ready/Pending/Denied/Unavailable evidence and prints official pages plus a safe
resume command. It asks before opening pages in a terminal; JSON mode never
prompts or opens a browser. NPA never performs acceptance or claims that it did.
Generic online preflight calls Hugging Face's authenticated `whoami-v2`
endpoint; public repository metadata is not accepted as token proof.

## Troubleshooting

- **`401`/`403` on a gated payload** — the token cannot fetch that exact payload;
  its account may still need approval, or the token may be invalid. Verify identity,
  then open the printed official page while signed in and complete any user-bound step.
- **Token rejected everywhere** — the token is wrong or was revoked. Regenerate
  it at <https://huggingface.co/settings/tokens> and re-run `npa configure`.
- **Downloads are slow / rate-limited without a token** — public repos work
  without a token but authenticated downloads are faster and higher-limit; set
  `HF_TOKEN` anyway.

## See also

- [NVIDIA NGC API key](ngc-api-key.md) — for paths that actually pull entitlement-controlled NGC artifacts.
- [Nebius Token Factory key](token-factory-key.md) — zero-GPU hosted inference.
- [Quickstart § credentials](../quickstart.md#4-configure-credentials)
