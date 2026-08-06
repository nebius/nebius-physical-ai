# Set up a Nebius Token Factory key

[Nebius Token Factory](https://tokenfactory.nebius.com/) is an OpenAI-compatible
hosted-inference API for open text and vision models. It is the cheapest way to
get a real result on Nebius — **zero GPU, no cluster** — so it's a great way to
confirm your credentials and connectivity work end to end.

This is the quick "create the key" guide. For the full integration reference
(captioning, batch generation, VLM scoring, SkyPilot templates) see
[token-factory.md](token-factory.md).

> **TL;DR:** create a key at <https://tokenfactory.nebius.com/> → **API keys**,
> run `npa configure`, and paste it at the `NEBIUS_TOKEN_FACTORY_KEY` prompt. The
> key is a long opaque token that **starts with `v1.`**.

## ⚠️ It is not your Nebius IAM / CLI token

The Token Factory key is a **separate credential** from your Nebius Cloud IAM
token:

- Token Factory key → starts with `v1.`, minted in the Token Factory console,
  lives in `NEBIUS_TOKEN_FACTORY_KEY`.
- Nebius IAM token → a `nebius …` access token from `nebius iam get-access-token`
  used by the `nebius` CLI / Terraform.

Pasting an IAM token where a Token Factory key belongs returns `403` on Token
Factory requests. `npa configure` warns you if the value you paste does not look
like a `v1.` key.

## 1. Create the API key

Token Factory has its own console, separate from the main Nebius Cloud console:

1. Go to <https://tokenfactory.nebius.com/> and sign up or sign in (Google /
   GitHub / email all work).
2. Make sure the project has credit. Token Factory is pay-per-token; new accounts
   usually get trial credit. A project with no balance returns `402`/`403`.
3. Open **API keys → Create API key**, name it (e.g. `npa-workbench`), and click
   **Create**.
4. **Copy it now** — it is shown once. It is a long opaque Bearer token starting
   with `v1.`.

Optional 10-second self-test from your terminal:

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer <PASTE_KEY>" | head
```

## 2. Give the key to `npa`

**Recommended — interactive setup:**

```bash
npa configure
# ... paste the key at the "Nebius Token Factory API key" prompt
```

You can also store just this key non-interactively:

```bash
export NEBIUS_TOKEN_FACTORY_KEY='v1.XXXXXXXXXXXXXXXXXXXX'
npa configure --no-interactive --save-env-credentials
```

**Or by hand** in `~/.npa/credentials.yaml`:

```yaml
tokens:
  NEBIUS_TOKEN_FACTORY_KEY: v1.XXXXXXXXXXXXXXXXXXXX   # your real key, verbatim
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

**Or via environment variable:**

```bash
export NEBIUS_TOKEN_FACTORY_KEY=v1.XXXXXXXXXXXXXXXXXXXX
```

## 3. Verify

```bash
npa workbench token-factory verify   # confirms the key authenticates + lists models
```

Expected output reports `authenticated: True` with a non-zero model count. Then
run your first hosted generation:

```bash
printf 'Explain sim-to-real transfer in one sentence.\n' > /tmp/prompts.txt
npa workbench token-factory generate \
  --input-path /tmp/prompts.txt --output-path /tmp/tf-generations.jsonl --output json
```

## Troubleshooting

- **`403` on Token Factory requests** — you likely pasted a Nebius IAM token
  instead of the `v1.` Token Factory key (see the note above), or the project has
  no balance. Mint a real key at <https://tokenfactory.nebius.com/> → API keys.
- **`401 Unauthorized`** — the key is wrong or revoked; create a new one.
- **A specific model 404s** — confirm it is enabled for your key with
  `npa workbench token-factory models`.

## See also

- [token-factory.md](token-factory.md) — full integration reference.
- [Hugging Face token](huggingface-token.md) · [NVIDIA NGC API key](ngc-api-key.md)
- [Quickstart § verify the path works](../quickstart.md#5a-verify-the-path-works-zero-gpu-inference-nebius-token-factory)
