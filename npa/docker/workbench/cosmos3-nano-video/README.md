# Cosmos3-Nano public development runtime

This image ships NPA video adapters and a pinned runtime bootstrap. CUDA,
vLLM, Ray, models, and media are fetched or supplied only after launch. See
[REDISTRIBUTION.md](REDISTRIBUTION.md) and the
[deployment and workload guide](../../../deploy/cosmos3-nano-video/README.md).

The default entrypoint materializes the runtime and starts authenticated local
Ray Serve. Supply a private inference token through `NPA_COSMOS3_VIDEO_TOKEN`.
The launcher creates its own distinct owner-only Ray management credential.
Set `NPA_COSMOS3_VIDEO_REPLICAS=1` for a single GPU development run; the default
remains 16. The Ray application builder also accepts an integer `num_replicas`
argument, which takes precedence over the environment variable.

Mount writable operator storage at `/opt/npa-cosmos3-serving/runtime` for the
serving/Ray cache, `/models` for staging, and `/outputs` for generated artifacts;
the image runs as uid 10001. Selecting the runtime defaults to noninteractive
delivery under its vendor terms. `NPA_COSMOS3_ACCEPT_NVIDIA_SOFTWARE_LICENSE`
defaults to `YES` only in the running wrapper. Explicit `NO` opts out before
fetch; an empty or invalid value fails closed. No acceptance is baked or persisted. No HF token is required for the public
pinned Nano weights. `NPA_COSMOS3_SERVE_GUARDRAILS=off` skips unused guardrail
assets; Nano's serving adapter always disables guardrails.

Stage weights before serving with the image entrypoint and command
`python -m npa.workbench.cosmos.nano_video_server --stage-weights`. The default
command then starts `--serve`. `/usr/local/bin/npa-cosmos3-nano-bootstrap` also
wraps arbitrary commands. KubeRay's `ray` executable uses that bootstrap because
the operator replaces the image entrypoint. The CPU staging Job prepares the
shared runtime once; the head and workers reuse it under a filesystem lock.

Validate real continuation and source-conditioned augmentation through the
documented SDK/CLI, fully decode the MP4s, and inspect joins and source alignment.
A one-replica result does not qualify the 16-replica routing/concurrency matrix.

The accepted public parent contains an older runtime recipe. Both replacement
images explicitly copy the current reviewed serving lock and bootstrap scripts,
override their source/checksum environment values, and verify all 11 effective
bootstrap files with `bootstrap-source-sha256s.txt` during the non-root build.
The enabled guardrail materializer keeps the reviewed source bytes; the separate
policy wrapper decides whether disabled guardrails request any assets. Debian
packages are upgraded from the pinned `20260907T000000Z` snapshot and Python
packaging tools use the reviewed hash lock. The inherited interpreter remains
Python 3.12.12; the resulting image still requires its own complete security scans.

The Nano additions lock resolves only `ray[serve]==2.56.0` and `httpx==0.28.1`,
using the serving lock as constraints. It excludes the historical vendor-image
NLTK override because Ray Serve does not require NLTK. This is a dependency-scope
correction, not a runtime vulnerability fix: the unchanged serving closure still
fetches `cosmos-guardrail==0.3.1`, which requires `nltk==3.10.3` after resolution.
[CVE-2026-81726](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp)
affects that version; upstream lists no patched release. The advisory concerns
NLTK model-artifact APIs bypassing its path-security restrictions on
caller-selected paths. Nano disables guardrails and exposes video generation,
not those NLTK APIs; real validation of that path does not certify other uses of
the fetched runtime. Do not treat a public-image scan as a scan of packages
downloaded after launch. No scanner exception or package metadata rewrite is
used to remove this finding.
