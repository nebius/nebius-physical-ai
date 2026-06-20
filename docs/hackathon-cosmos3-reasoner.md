# Hackathon Quickstart — Cosmos3 Reasoner on Nebius Token Factory (serverless)

This is the **foolproof, copy‑paste** path to call NVIDIA
**`nvidia/Cosmos3-Super-Reasoner`** as a *serverless* backend for your hackathon
project. Inference is hosted by **Nebius Token Factory** (an OpenAI‑compatible
API), so there is **no GPU to rent, no VM to start, no container to build, and
no Kubernetes**. You need exactly two things: an API key and an internet
connection.

Every command block below was run end‑to‑end against the live API. Work top to
bottom and stop at the first ❌ checkpoint that fails.

> **Never paste your key into chat, screenshots, git, or Slack.** Treat it like
> a password. Every block uses the `NEBIUS_TOKEN_FACTORY_KEY` environment variable so the
> key itself never appears in a command you might copy somewhere public.

---

## 0. What you're building

You send a question (and optionally **scene images**) to a hosted physical‑AI
reasoning model and get back scene understanding + a step‑by‑step plan a robot
could follow. That's the whole loop. Three ways to call it, pick one:

| Path | Use when | Needs |
|------|----------|-------|
| **A. `curl`** | quickest possible check, any language | just `curl` |
| **B. OpenAI Python SDK** | you're writing Python | `pip install openai` |
| **C. `npa` CLI** | you want the batteries‑included workbench (S3 in/out, image folders, JSON artifacts) | `pip install -e npa` |

---

## 1. Get a Token Factory API key (~2 minutes)

The Token Factory key is **its own credential**. It is a long opaque token that
**starts with `v1.`**. It is **NOT**:

- ❌ your Nebius IAM / `nebius` CLI login token (that returns `403` here), or
- ❌ a `nebius_…` string (older docs show that shape — ignore it).

Steps:

1. Go to <https://tokenfactory.nebius.com/> and sign in (Google/GitHub/email).
2. Confirm your project has credit (new accounts usually get trial credit; a
   project with $0 balance returns `402`/`403` on calls).
3. Left nav → **API keys** → **Create API key** → name it `hackathon` → **Create**.
4. **Copy it now** — it is shown only once. It looks like `v1.CmQK…` (very long).

---

## 2. Put the key in your shell

```bash
# Paste your real key between the quotes. It starts with v1.
export NEBIUS_TOKEN_FACTORY_KEY='PASTE_YOUR_KEY_HERE'
```

✅ **Checkpoint** — this must print `key length: <a few hundred>` and **not** `0`:

```bash
echo "key length: ${#NEBIUS_TOKEN_FACTORY_KEY}"
```

---

## 3. Prove the key works (10‑second auth check)

```bash
curl -s https://api.tokenfactory.nebius.com/v1/models \
  -H "Authorization: Bearer ${NEBIUS_TOKEN_FACTORY_KEY}" \
  | grep -o '"id":"[^"]*"' | grep -i cosmos
```

✅ **Checkpoint** — you should see:

```
"id":"nvidia/Cosmos3-Super-Reasoner"
```

❌ If you get `{"detail":"You don't have access ..."}` or HTTP `401/403`: the key
is wrong, revoked, or you pasted an IAM token instead of a Token Factory key.
Re‑mint it (step 1). If you see `nvidia/...` models but *not* `Cosmos3`, ask the
Token Factory console to enable it for your project, or use another reasoner
from the list (e.g. `nvidia/Nemotron-3-Ultra-550b-a55b`).

---

## Path A — `curl` (text reasoning, any language)

```bash
curl -s https://api.tokenfactory.nebius.com/v1/chat/completions \
  -H "Authorization: Bearer ${NEBIUS_TOKEN_FACTORY_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos3-Super-Reasoner",
    "max_tokens": 300,
    "temperature": 0.2,
    "messages": [
      {"role": "user", "content": "A robot arm sees a red cube on a table next to an open drawer. Give a 3-step plan to put the cube in the drawer."}
    ]
  }' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

✅ You get a numbered plan back. That's a working serverless backend — done.

> The model may wrap its private reasoning in `<think>…</think>` before the
> final answer. If you only want the answer, strip everything up to the last
> `</think>` in your app.

---

## Path B — OpenAI Python SDK

Token Factory is OpenAI‑compatible, so the standard `openai` client works by
just changing `base_url`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip openai
```

```python
# reason.py
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_TOKEN_FACTORY_KEY"],   # never hardcode the key
)

resp = client.chat.completions.create(
    model="nvidia/Cosmos3-Super-Reasoner",
    temperature=0.2,
    max_tokens=300,
    messages=[
        {"role": "system", "content": "You are a physical-AI reasoning assistant for a robot."},
        {"role": "user", "content": "How should a robot grasp a ceramic mug without dropping it?"},
    ],
)
print(resp.choices[0].message.content)
```

```bash
python3 reason.py
```

✅ Prints a grasp plan. Build your hackathon logic around this `client`.

### Reason over an image (vision)

`Cosmos3-Super-Reasoner` is multimodal — pass images as base64 data URLs:

```python
# reason_image.py
import base64, os
from openai import OpenAI

client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/",
                api_key=os.environ["NEBIUS_TOKEN_FACTORY_KEY"])

with open("scene.png", "rb") as f:                       # your scene photo
    data_url = "data:image/png;base64," + base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="nvidia/Cosmos3-Super-Reasoner",
    max_tokens=400, temperature=0.2,
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "What objects are here and what should the robot do?"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]}],
)
print(resp.choices[0].message.content)
```

```bash
python3 reason_image.py
```

---

## Path C — `npa` CLI (batteries included)

The `npa` workbench wraps the same API with a folder/S3 in‑and‑out contract and
writes a clean JSON artifact. Best if you have a folder of scene frames.

### One‑time setup

```bash
# In the nebius-physical-ai repo:
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e npa
```

Store the key once (writes `~/.npa/credentials.yaml`, mode 0600 — no env var
needed afterward):

```bash
npa configure --token-factory-key "$NEBIUS_TOKEN_FACTORY_KEY"
```

✅ **Checkpoint** — authenticates and lists models (no key value printed):

```bash
npa workbench token-factory verify
```

Expected:

```
  authenticated: True
  base_url: https://api.tokenfactory.nebius.com/v1/
  model_count: 35
  sample_models: ['nvidia/Llama-3_1-Nemotron-Ultra-253B-v1', ...]
```

> `npa workbench token-factory status` shows the same info **without** a network
> call. `npa workbench token-factory models` lists the full catalog.

### Run the reasoner over a folder of scene images

Don't have a scene image handy? Generate a throwaway one:

```bash
mkdir -p scene out
python3 - <<'PY'
from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 480), (225, 220, 210))
d = ImageDraw.Draw(img)
d.rectangle([0, 360, 640, 480], fill=(150, 110, 70))        # table
d.rectangle([250, 300, 360, 380], fill=(200, 40, 40))       # red cube
d.rectangle([420, 330, 560, 470], outline=(60, 60, 60), width=6)  # open drawer
img.save("scene/frame_00.png")
print("wrote scene/frame_00.png")
PY
```

```bash
npa workbench token-factory reason \
  --input-path ./scene \
  --output-path ./out \
  --task "What is in this scene and how should the robot put the red cube in the drawer?" \
  --model nvidia/Cosmos3-Super-Reasoner \
  --max-images 4 \
  --max-tokens 400 \
  --output json
```

✅ The command prints a JSON object with an `analysis` field and writes
`./out/scene_reasoning.json`. `--input-path` / `--output-path` also accept
`s3://…` URIs if your data lives in object storage.

### Batch text generation (e.g. synthesize task prompts)

```bash
printf '%s\n' \
  '{"id":"t1","prompt":"Write one pick-and-place instruction for a robot arm."}' \
  '{"id":"t2","prompt":"Write one instruction for a mobile robot to tidy a desk."}' \
  > prompts.jsonl

npa workbench token-factory generate \
  --input-path ./prompts.jsonl \
  --output-path ./gen \
  --model nvidia/Cosmos3-Super-Reasoner \
  --max-tokens 200 --output json
# -> writes ./gen/generations.jsonl  ({"id","prompt","completion"} per line)
```

---

## Knobs you'll actually tune

| Flag / field | What it does | Sane default |
|--------------|--------------|--------------|
| `model` | which hosted model | `nvidia/Cosmos3-Super-Reasoner` |
| `temperature` | randomness (0 = deterministic) | `0.2` for reasoning |
| `max_tokens` | answer length cap | `300`–`1024` |
| `--max-images` (CLI) | scene frames sent per request | `4`–`8` |
| `--task` (CLI) | the question asked of the scene | your task |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `echo ${#NEBIUS_TOKEN_FACTORY_KEY}` prints `0` | key not exported in this shell | re‑run step 2 (new terminals don't keep it unless you persisted it) |
| `403` / `"You don't have access"` on `/models` | not a Token Factory key (e.g. an IAM token), or wrong project | mint a real key (step 1); the key starts with `v1.` |
| `401` | key invalid/revoked | create a new key |
| `402` / quota / billing error | project has no credit | add credit/payment in the Token Factory console |
| `404` on a model id | model name typo or not enabled for your key | `curl …/v1/models` (step 3) and copy the exact id |
| `NEBIUS_TOKEN_FACTORY_KEY is not set` (CLI) | CLI can't find the key | `npa configure --token-factory-key "$NEBIUS_TOKEN_FACTORY_KEY"` or `export NEBIUS_TOKEN_FACTORY_KEY=…` |
| CLI: `No scene images found` | `--input-path` has no `.png/.jpg/.jpeg/.webp/.bmp/.ppm` | point it at a folder that contains images |
| answer contains `<think>…</think>` | model's visible reasoning | keep it, or strip up to the last `</think>` |

---

## Why no VM / GPU / Kubernetes?

Token Factory **hosts** the model. Your code only makes an HTTPS call, so the
"serverless backend" is just the API at `https://api.tokenfactory.nebius.com/v1/`.
You do **not** need to provision compute for the reasoner.

> The repo also ships heavier SkyPilot templates
> (`npa/workflows/workbench/skypilot/token-factory-cosmos-reason.yaml`) that run
> the *orchestration* on Nebius Kubernetes. Those require a container image in
> your registry, an S3 bucket, and a cluster — **skip them for a hackathon.**
> The three paths above are all you need.

---

## Going deeper

- [hackathon-isaac-token-factory.md](hackathon-isaac-token-factory.md) — **Isaac Lab Franka sim
  + Token Factory** (GPU capture → serverless reasoner; workflow + SDK example with visuals).
- `docs/workbench/token-factory.md` — full integration reference (all tools,
  Python client, live tests).
- `npa workbench token-factory --help` — every subcommand.
- Model catalog for your key: `curl -s https://api.tokenfactory.nebius.com/v1/models -H "Authorization: Bearer $NEBIUS_TOKEN_FACTORY_KEY"`.
