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

## Verified public development image

On 2026-09-08, `dev-1bd1b00330e37b3a8916b2a740c2578aca25651e`
(`sha256:7e237a1b8fbf27bf08422263e64a70f870d8ebd28bf4f99b0cfbed946a598d78`)
passed its [trusted publication gates](https://github.com/nebius/nebius-physical-ai/actions/runs/34193230368)
and real workloads on **one B200**: a 720-frame, 30-second continuation and a
144-frame, six-second structural augmentation, both at 832×480 and 24 fps.
All 27 generated artifacts passed storage readback, with complete media decoding.
The actual GPU pod's image and installed module hashes matched the published
bytes; the runtime used no source overlay.

Visual review found recognizable orange robot and blue racks with no obvious
reset at the sampled joins. Wheel details and floor markings changed, and the
requested dampness was unclear: prompt fidelity is partial. This result does not
qualify exact asset preservation, physical realism, the 16-replica matrix,
30-second source augmentation or B300. Supported-release defaults remain unchanged.
See the [measured development scope](../../../deploy/cosmos3-nano-video/README.md#measured-public-development-validation)
for timing and fresh-execution evidence; the historical 16-replica results apply
to their recorded operator image.

## Runtime compatibility

The video adapter launches the pinned Omni serving command through its Python
parser so `sound_gen=false` and `guardrails=false` remain explicit model settings.
Omni 0.28 removed `--stage-configs-path` and the `stage_args` schema. Its Cosmos
diffusion fallback does not apply deployment-file overrides, so changing only
the flag to `--deploy-config` would lose the video-only checkpoint contract.
The launcher retains upstream parsing, validation, BF16 precision, TP1 placement,
profiling, and the real serving implementation; it does not patch vendor code.

The augmentation evidence validator includes Omni 0.28's recorded resolver
defaults: `emphasize_control_in_prompt=true` and edge `control_weight=1.0`.
Both must have those exact values; missing defaults, altered values, and unknown
resolver fields still fail validation. A completed diffusion request alone does
not qualify an augmentation whose final evidence validation failed.

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
