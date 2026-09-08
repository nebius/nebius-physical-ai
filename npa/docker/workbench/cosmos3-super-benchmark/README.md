# Cosmos3-Super public development runtime

This image contains a public bootstrap and SkyPilot worker tools. CUDA, vLLM,
vLLM-Omni, and model weights are fetched into operator storage at runtime. The
image overlays the reviewed serving recipe and pins vLLM-Omni revision `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c` and
checks dependency and source archive hashes. See [REDISTRIBUTION.md](REDISTRIBUTION.md).

The default entrypoint without arguments waits for SkyPilot's worker command.
Ordinary commands forward unchanged for SkyPilot's CPU worker bootstrap.
With `--runtime`, `/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint` checks
runtime acceptance, serializes cache installation, then executes the command in
the fetched virtual environment. A worker that overrides the image entrypoint
must explicitly invoke this wrapper with `--runtime` around its workload. Mount a
writable cache at `/opt/npa-cosmos3-serving/runtime`, owned by uid 1000. Selecting
this runtime defaults to noninteractive delivery under its vendor terms.
`NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE` defaults to `YES` only inside the
running wrapper. Explicit `NO` opts out before any fetch; empty or other values
fail closed. The image config contains no acceptance value, and the runtime does
not persist one beside cached files.

The smallest single GPU development service uses:

```bash
/usr/local/bin/npa-cosmos3-super-benchmark-entrypoint --runtime \
  vllm serve nvidia/Cosmos3-Super \
  --revision e0262be9d8f7586bc24c069a2aed2b665bdff266 \
  --omni --tensor-parallel-size 1 --no-guardrails --init-timeout 1800 \
  --host 127.0.0.1 --port 8000
```

The public model is fetched anonymously unless the operator supplies an HF
credential. `NPA_COSMOS3_SERVE_GUARDRAILS` defaults to `off`, matching the
benchmark's explicit `--no-guardrails`; only `on` and `off` are valid. Enabling
guardrails fetches their separately gated pinned blocklist and requires actual
upstream access. It must also be enabled in the server command.

Generate a real video through `/v1/videos/sync`, retain the exact request, image
digest and runtime versions, and fully decode the resulting MP4. A single
development video is functional evidence. It is not a measurement of the fixed
eight-GPU benchmark or its 24-attempt single-H200 suite, whose contracts remain
unchanged. The benchmark workflow and measured cells are documented in
[the benchmark skill](../../../../skills/tools/cosmos3-super-benchmark/SKILL.md).

For a fixed benchmark on replacement bytes, set
`NPA_COSMOS3_BENCHMARK_RUNTIME_IMAGE` to the actual immutable `repository@sha256:...`
reference. The benchmark records it in its plan and resume contract; mutable
tags are rejected. Without the override it retains the historical vendor image
identity, so do not omit it when benchmarking this replacement.

`build.sh` produces only a local candidate named `dev-<current-full-git-sha>`.
It refuses publication flags. Use the trusted repository workflow for security
scans, SBOM/provenance, anonymous development publication, and exact-digest GPU
validation.

The accepted public parent contains an older runtime recipe. Both replacement
images explicitly copy the current reviewed serving lock and bootstrap scripts,
override their source/checksum environment values, and verify all 11 effective
bootstrap files with `bootstrap-source-sha256s.txt` during the non-root build.
The enabled guardrail materializer keeps the reviewed source bytes; the separate
policy wrapper decides whether disabled guardrails request any assets. Debian
packages are upgraded from the pinned `20260907T000000Z` snapshot and Python
packaging tools use the reviewed hash lock. The inherited interpreter remains
Python 3.12.12; the resulting image still requires its own complete security scans.

The fetched serving closure includes `cosmos-guardrail==0.3.1` and
`nltk==3.10.3` even with guardrails disabled.
[CVE-2026-81726](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp)
affects NLTK's model-artifact path-security APIs, with no upstream patched release
listed at review. The benchmark disables guardrails and does not offer those
model import/export APIs. Validation of video generation does not establish that
every API in the downloaded runtime is safe. Image-layer scans cover the
published bootstrap; they do not qualify packages fetched after launch.
