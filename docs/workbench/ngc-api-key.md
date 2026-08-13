# Set up an NVIDIA NGC API key

NVIDIA distributes several workbench assets — GR00T and Cosmos container images
and models — through [NGC](https://ngc.nvidia.com) (and the NGC-backed container
registry `nvcr.io`). An NGC API key lets `npa` pull them. This guide creates the
key and wires it into `npa`.

> **TL;DR:** create a key at <https://org.ngc.nvidia.com/setup/api-key>, run
> `npa configure`, and paste it at the `NGC_API_KEY` prompt. The key starts with
> `nvapi-`.

You only need this for the **GR00T** and **Cosmos** capabilities. Token Factory,
vlm-eval, LeRobot, Genesis, and Isaac Lab do not require it.

## 1. Create the API key

1. Sign in (or create a free account) at <https://ngc.nvidia.com>.
2. Go to **Setup → Generate API Key**:
   <https://org.ngc.nvidia.com/setup/api-key>.
3. Click **Generate Personal Key** (or **Generate API Key**). Give it a name and,
   for a personal key, select the services you need (include **NGC Catalog**).
4. **Copy it now** — NGC shows the value once. It starts with `nvapi-`.

> Personal keys are scoped to your account and are the simplest choice. If your
> organization uses **org/team**-scoped keys, note the org and team names — you
> can set them alongside the key (see below).

## 2. Give the key to `npa`

**Recommended — interactive setup:**

```bash
npa configure
# ... paste the key at the "NVIDIA NGC API key (NGC_API_KEY)" prompt
```

**Or by hand** in `~/.npa/credentials.yaml`:

```yaml
ngc:
  api_key: nvapi-XXXXXXXXXXXXXXXXXXXX   # your real key, pasted verbatim
  # org: your-ngc-org    # optional, only for org/team-scoped keys
  # team: your-ngc-team  # optional
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

**Or via environment variable:**

```bash
export NGC_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXX
```

## 3. Verify

```bash
npa workbench health access          # checks NGC key presence/format + HF access
# or presence-only, no network:
npa workbench health preflight --offline
```

A well-formed key (starting with `nvapi-`) reports as configured; GR00T and
Cosmos deploys additionally confirm gated Hugging Face access for their NVIDIA
models (see [Hugging Face token](huggingface-token.md)).

## Troubleshooting

- **`NGC_API_KEY is set but does not look like an NGC key`** — NGC keys start
  with `nvapi-`. You likely pasted a different value; regenerate at
  <https://org.ngc.nvidia.com/setup/api-key>.
- **`401 Unauthorized` pulling `nvcr.io/...`** — the key is missing, wrong, or
  lacks catalog access. Re-generate it and include the NGC Catalog service.
- **Org/team model not found** — set `org` (and `team`) in the `ngc:` block so
  the key resolves the right namespace.

## See also

- [Hugging Face token](huggingface-token.md) — GR00T / Cosmos also need HF access.
- [Nebius Token Factory key](token-factory-key.md) — zero-GPU hosted inference.
- [Quickstart § credentials](../quickstart.md#4-configure-credentials)
