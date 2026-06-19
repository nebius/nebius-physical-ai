# Nebius Token Factory Integration Guide

> **Just want the fastest serverless copy-paste path (e.g. for a hackathon)?**
> See [../hackathon-cosmos3-reasoner.md](../hackathon-cosmos3-reasoner.md). This
> page is the full reference.

Nebius Token Factory is an OpenAI-compatible hosted-inference API for open text
and vision models. NPA uses it natively for zero-GPU workflows: captioning,
batch text generation, and VLM-based rollout scoring all call the hosted API
instead of starting a GPU server.

Authentication is a single API key sent as an `Authorization: Bearer <key>`
header. NPA reads it from the `NEBIUS_API_KEY` environment variable (or
`~/.npa/credentials.yaml`). The default endpoint is
`https://api.tokenfactory.nebius.com/v1/`.

> **The Token Factory key is its own credential.** It is a long opaque token
> that starts with `v1.` (not a `nebius_…` string). It is **not** your Nebius
> IAM / `nebius` CLI access token — an IAM token returns `403` against Token
> Factory. You must mint a Token Factory key in the Token Factory console
> (step 1 below); having `nebius` CLI access is not enough.

## 1. Register and get an API key

Token Factory has its own console, separate from the main Nebius cloud console.
Registering and minting a key takes about two minutes:

1. **Create an account / sign in.** Go to <https://tokenfactory.nebius.com/> and
   sign up (Google/GitHub/email all work) or sign in. New accounts land in a
   default organization and project.
2. **Make sure the project has credit.** Token Factory is pay-per-token. New
   accounts usually get trial credit; otherwise open **Billing** and add a
   payment method or redeem credits. A project with no balance returns
   `402`/`403` on inference calls. If your team already has a Token Factory
   project, just confirm you're switched into it (top-left project switcher).
3. **Create the API key.** In the left nav open **API keys** → **Create API
   key**, give it a name (e.g. `npa-workbench`), and click **Create**.
4. **Copy it now.** The key is shown **once** and cannot be reopened later. It's
   a long opaque Bearer token. Store it somewhere safe (a password manager) —
   you'll paste it into NPA in step 2.

> Tip: each Token Factory project also has an **AI project ID** (looks like
> `aiproject-...`, visible in the project switcher / settings). You don't need
> it for normal use — the API key already scopes requests to its project — but
> it's handy when filtering models by project in the console.

Optional 10-second self-test from your own terminal (proves the key works and
shows the served catalog, including whether `nvidia/Cosmos3-Super-Reasoner` is
enabled for you):

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer <PASTE_KEY>" | head
```

## 2. Give the key to NPA

Pick one (NPA checks them in this order: explicit arg → env var → credentials file).

**A. Interactive setup (recommended)**

```bash
npa configure
# ... answer the "Nebius Token Factory API key (NEBIUS_API_KEY)" prompt
```

**B. Credentials file by hand** — `~/.npa/credentials.yaml`:

```yaml
tokens:
  NEBIUS_API_KEY: v1.XXXXXXXXXXXXXXXXXXXXXXXX   # your real key, paste it verbatim
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

**C. Environment variable** (good for CI / one-off shells):

```bash
export NEBIUS_API_KEY=v1.XXXXXXXXXXXXXXXXXXXXXXXX
```

## 3. Verify authentication

```bash
npa workbench token-factory verify
```

Expected output confirms the key authenticated and lists a few available models:

```
  authenticated: True
  base_url: https://api.tokenfactory.nebius.com/v1/
  model_count: 42
  sample_models: ['meta-llama/Llama-3.3-70B-Instruct', ...]
```

`npa workbench token-factory status` shows the resolved base URL and whether a
key is configured **without** making a network call.
`npa workbench token-factory models` lists the full model catalog.

## 4. Run something

Caption a folder of images (local or S3):

```bash
npa workbench token-factory caption \
  --input-path ./frames \
  --output-path /tmp/captions \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --output json
```

Batch text generation from a JSONL prompt file (`{"id": ..., "prompt": ...}`
per line) or a `.txt` file (one prompt per line):

```bash
npa workbench token-factory generate \
  --input-path ./prompts.jsonl \
  --output-path /tmp/generations \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --output json
```

Reason over a scene with NVIDIA Cosmos3-Super-Reasoner — point it at scene
images and ask what a robot should do (scene understanding + plan of action):

```bash
npa workbench token-factory reason \
  --input-path ./scene \
  --output-path /tmp/scene-reasoning \
  --task "What is in this scene and how should a robot pick up the red box?" \
  --model nvidia/Cosmos3-Super-Reasoner \
  --output json
```

Score a rollout with a hosted VLM (no GPU, no vLLM):

```bash
npa workbench vlm-eval run \
  --input-path ./rollouts/episode-000 \
  --output-path /tmp/vlm-eval \
  --backend api \
  --api-key-env NEBIUS_API_KEY \
  --output json
```

## 5. Run on Nebius (SkyPilot)

The checked-in CPU-only workflows pass the key as a SkyPilot secret:

```bash
sky jobs launch \
  --secret NEBIUS_API_KEY \
  --secret AWS_ACCESS_KEY_ID \
  --secret AWS_SECRET_ACCESS_KEY \
  npa/workflows/workbench/skypilot/token-factory-caption.yaml
```

Other workflows: `token-factory-generate.yaml`,
`token-factory-cosmos-reason.yaml`, `vlm-eval-token-factory.yaml`.

## Live testing (first-class)

The mocked unit tests never touch the network. The *live* tests that actually
authenticate against Token Factory are in `npa/tests/e2e/test_token_factory_e2e.py`
and are marked `token_factory_e2e`. They self-skip when no key is configured, so
the only thing you need to run them is a real key:

```bash
NEBIUS_API_KEY=nebius_xxx npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_token_factory_e2e.py -v
```

They cover: `list_models` authenticates, a text chat completion returns text,
and `nvidia/Cosmos3-Super-Reasoner` produces a scene plan (that last one skips if
the model is not available for your key). For a quick manual check use
`npa workbench token-factory verify`.

## Use it in Python

NPA's client is a thin OpenAI-compatible wrapper, so you can call it directly:

```python
from npa.clients.token_factory import TokenFactoryClient

client = TokenFactoryClient()  # reads NEBIUS_API_KEY
text = client.chat_completion_text(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Give me one robot task instruction."}],
)
print(text)
```

Override the endpoint with `NEBIUS_TOKEN_FACTORY_BASE_URL` (or `NEBIUS_BASE_URL`)
if you are pointed at a non-default deployment.

## Troubleshooting

- **`NEBIUS_API_KEY is not set`** — provide the key via step 2; confirm with
  `npa workbench token-factory status`.
- **`Token Factory request failed (401)`** — the key is invalid or revoked;
  create a new one.
- **`Token Factory request failed (404)` on a model** — the model id is wrong or
  retired; check `npa workbench token-factory models`.
- **Workflow exits with "NEBIUS_API_KEY is required"** — you did not pass
  `--secret NEBIUS_API_KEY` to `sky jobs launch`.
