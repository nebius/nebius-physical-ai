# Cosmos 3 access preflight: accounts, tokens, and gated-repo diagnostics

Clearing gated Hugging Face access is the most common reason a Cosmos 3
generation run fails before it produces anything. This page is the full
checklist: create the account, scope the token, accept the right license for
the serving path you are using, and read the two failure signatures that look
like credential problems but are not the same one.

See [`cosmos3-generate.md`](cosmos3-generate.md) for the credential table this
extends and for `--dry-run`, which resolves everything below except the
network calls themselves, so you can confirm mode and checkpoint on a laptop
before spending GPU time.

## Hugging Face account and token

1. Create a Hugging Face account at <https://huggingface.co/join> if you do
   not already have one.
2. Create a token at <https://huggingface.co/settings/tokens>. A read-scoped
   token is enough; generation only downloads weights, it never pushes.
3. Export it as `HF_TOKEN` (or set `NPA_COSMOS3_HF_TOKEN_ENV` to the name of
   an env var you already use for a different token).
4. Accept the license for every gated repo your run touches before the run,
   not during it. `require_model_access` only checks that a token is present;
   it cannot check that the token's account has accepted a given repo's
   license, so an unaccepted license surfaces later, mid-download, as the 403
   case in the diagnostic below.

## Two gated guardrail repos, split by serving path

Cosmos 3's guardrail models are gated separately from the base checkpoint,
and **which guardrail repo you need depends on how you are serving the
model**, not on which Cosmos 3 checkpoint you picked. Accepting one license
does not accept the other.

| Serving path | Guardrail repo | Accept the license at |
| --- | --- | --- |
| `npa workbench cosmos3 generate` (this repo, cosmos-framework) | `nvidia/Cosmos-Guardrail1` | <https://huggingface.co/nvidia/Cosmos-Guardrail1> |
| vLLM-Omni serving (e.g. the `vllm/vllm-omni:cosmos3` container, outside npa) | `nvidia/Cosmos-1.0-Guardrail` | <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail> |

An operator who has cleared the vLLM-Omni guardrail repo for a serving
deployment and then runs `npa workbench cosmos3 generate` for the first time
still fails mid-download until they separately accept
`nvidia/Cosmos-Guardrail1`, and the reverse is also true. Both paths can skip
the guardrail download entirely: pass `--no-guardrails` to
`npa workbench cosmos3 generate` per run, or to `vllm serve` at server launch
on the vLLM-Omni path.

## The 401-vs-403 diagnostic

A gated-repo failure during download can come from two different causes that
produce two different HTTP status codes. Distinguishing them tells you
whether the problem is the token or the license:

| Request | Status | Meaning |
| --- | --- | --- |
| Anonymous request to the gated repo URL | `401` | No valid token reached Hugging Face: it is missing, empty, malformed, or revoked. |
| Authenticated request with the token you configured | `403` | The token is valid and reached Hugging Face; the account behind it has not accepted this repo's license yet. |

Measured signature, from the vLLM-Omni serving path (`nvidia/Cosmos-1.0-Guardrail`,
before the license was accepted; shown because the same status-code
diagnostic applies on either path). The status below is `401` rather than the
`403` this section describes because the container held no token, so its fetch
was anonymous: the first row of the table above. The `403` came from a manual
authenticated request against the same URL:

```
GatedRepoError: 401 Client Error
Cannot access gated repo for url
https://huggingface.co/nvidia/Cosmos-1.0-Guardrail/resolve/.../blocklist/custom/branding
Access to model nvidia/Cosmos-1.0-Guardrail is restricted.
RuntimeError: Orchestrator initialization failed: 401 Client Error.
```

Before the license was accepted, an authenticated fetch of the same URL
returned `403` while an anonymous fetch returned `401`; that split is what
identified the problem as a missing license acceptance rather than a bad
token, and license acceptance resolved it immediately. On the npa path,
`require_model_access` catches the no-token case before any download starts;
if you already have `HF_TOKEN` set and generation still fails partway through
fetching the guardrail weights, read the HTTP status in the traceback against
this table before assuming the token itself is bad.

## Xet download failure and workaround

With a valid token and an accepted license, download can still fail inside
Hugging Face's Xet transfer client:

```
huggingface_hub/file_download.py:1920 _download_to_tmp_and_move
  -> xet_get(...)
    -> huggingface_hub/file_download.py:563  session.new_file_download_group(...)
RuntimeError: Task error: Unable to parse string as hex hash value
```

This is [huggingface/xet-core#895](https://github.com/huggingface/xet-core/issues/895),
closed 2026-07-28. It reproduces on the exact pin pair `hf-xet 1.5.1` plus
`huggingface_hub 1.23.0`. It was measured here on a gated guardrail-repo
download; on the same environment and pins, a large ungated checkpoint had
downloaded cleanly beforehand, so not every download on the affected pair
triggers it. Treat the failure signature, not the download type, as the
indicator.

**Workaround:** set `HF_HUB_DISABLE_XET=1` in the environment before the run
starts. This falls back to the standard HTTP downloader instead of the Xet
client. Measured on the affected pair: the 146-file, 17 GB guardrail repo
that failed under Xet downloaded in 1m52s with `HF_HUB_DISABLE_XET=1` set,
and the run proceeded normally afterward.

This is scoped to the affected pin pair, not a default for every
environment: newer `huggingface_hub`/`hf-xet` releases fix the issue, and
setting the variable unconditionally would silently disable a faster
download path for environments that do not need it. `npa workbench cosmos3
generate` checks the installed `hf-xet` and `huggingface_hub` versions in the
runtime environment and prints a warning naming this workaround only when it
detects the affected pair; it does not set the variable for you.

## Checklist

1. Create a Hugging Face account and a read-scoped token.
2. Export the token as `HF_TOKEN` (or point `NPA_COSMOS3_HF_TOKEN_ENV` at it).
3. Accept `nvidia/Cosmos-Guardrail1` for the npa generation path, and
   separately accept `nvidia/Cosmos-1.0-Guardrail` if you also serve through
   vLLM-Omni.
4. If a download fails, check whether an anonymous request to the same URL
   returns `401` (bad/missing token) or an authenticated one returns `403`
   (unaccepted license) before assuming the token is wrong.
5. If the failure is `Unable to parse string as hex hash value` from
   `huggingface_hub`'s Xet client, set `HF_HUB_DISABLE_XET=1` and retry.
