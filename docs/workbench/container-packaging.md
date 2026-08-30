# Workbench Container Packaging

Canonical contract for packaging workbench containers correctly, securely, and
with the right runtime features exposed. Machine-readable rules live in
`npa/docker/workbench/packaging-contract.yaml` (enforced by unit tests).

## SkyPilot worker bootstrap contract

Every workflow image must satisfy version `skypilot-0.12.2-v1`: a usable
effective user; root or verified passwordless sudo; `openssh-server`, `rsync`,
and compatible service/init behavior; writable `/tmp` and home; and an
entrypoint that forwards orchestrator arguments. Compliant first-party images
record `org.nebius.npa.skypilot-bootstrap-contract=skypilot-0.12.2-v1` in OCI
config, with Dockerfile behavior covered by build tests.

The canonical `npa-groot:0.1.0` is not in that attested publication set. It has
the non-root `ubuntu` user, system Python, `rsync`, an SSH client, and
passwordless sudo, but lacks `openssh-server`, runtime host-key generation, and
an argument-forwarding entrypoint. `groot/Dockerfile.k8s-prereqs` is the exact
derived source for those missing capabilities. Its source contract is checked
from `packaging-contract.yaml`, but its OCI label is only a declaration: because
the canonical and repaired artifacts share the `npa-groot` repository, submit
ignores label-backed (including cached) evidence and requires a capability probe
against the selected immutable digest. Public-image verification continues to
check only canonical Dockerfiles that independently satisfy the declared
publication contract; GR00T is deliberately absent from that set.

Submit resolves the selected tag to an immutable digest and validates metadata
on that digest. Missing/mismatched first-party evidence fails before launch.
Arbitrary unattested images get one bounded exact-context capability pod; probe
cleanup failure also fails closed. Cache keys use digest plus contract version,
never a tag. Image-byte licensing scans remain mandatory before registry push.
For a multi-tool spec, repeat `--image-override TOOL_REF=IMAGE` to select each
tool's artifact independently; the preflight and renderer share that same map.

## Inventory

All first-class images live under `npa/docker/workbench/`:

| Image / role | Dockerfile | Default exposure |
| --- | --- | --- |
| `npa-lerobot` | `lerobot/Dockerfile` | FastAPI server `:8080` |
| `npa-lerobot-policy` | `lerobot-policy/Dockerfile` | entrypoint modes (serve/train/eval) |
| `npa-genesis` | `genesis/Dockerfile` | job shell (CLI supplies command) |
| `npa-isaac-lab` | `isaac-lab/Dockerfile` | job shell |
| `npa-leisaac` | `leisaac/Dockerfile` | teleoperation/status service `:8080`, WebRTC TCP `:49100`, UDP `:47998` |
| `npa-cosmos` | `cosmos/Dockerfile` | job shell; server built but not default CMD |
| `npa-groot` | `groot/Dockerfile` | job shell; `EXPOSE 8080` |
| `npa-fiftyone` | `fiftyone/Dockerfile` | command-passthrough job entrypoint; `EXPOSE 5151` |
| `npa-lancedb` | `lancedb/Dockerfile` | uvicorn `:8686` |
| `npa-sonic` | `sonic/Dockerfile` | `/entrypoint.sh` modes |
| `npa-detection-training` | `detection-training/Dockerfile` | uvicorn `:8790` |
| `npa-retargeting` | `retargeting/Dockerfile` | job shell |
| `npa-foxglove-embed` | `foxglove-embed/Dockerfile` | static host `:8099` (Foxglove embed SDK + MCAP data) |
| Sim2Real stack | `sim2real-*/`, `cosmos3-reason/`, `lerobot-vlm-rl/` | workflow modules |
| Base CUDA 13 | `base/cuda13-b300/Dockerfile` | build base only |

BYOF images (`npa-byof:<run-id>`) are **ad-hoc** and are not registered in
`CONTAINER_IMAGE_NAMES` until promoted to Tier 2 (see
`docs/architecture/oss-onboarding-ladder.md`). The canonical BYOF builder still
installs and byte-checks the shared SkyPilot prerequisites, removes build-time
SSH host keys, forwards orchestrator arguments, and records the same
`skypilot-0.12.2-v1` OCI attestation. Ad-hoc means the solution is not a catalog
image; it does not exempt its runtime bytes from the worker bootstrap contract.

## Packaging tiers

Every Dockerfile must declare one of:

| Tier | `kind` | ENTRYPOINT expectation | Examples |
| --- | --- | --- | --- |
| **Service** | `service` | Starts the HTTP service (or entrypoint that does) | lerobot, lancedb, detection-training, lerobot-policy, leisaac |
| **Job** | `job` | Runs a workflow/CLI module with explicit CMD or an exec-only command-passthrough entrypoint | sonic, fiftyone, sim2real-eval, cosmos3-reason, lerobot-vlm-rl |
| **Interactive** | `interactive` | `/bin/bash` allowed only when CLI always overrides CMD | genesis, isaac-lab, cosmos, groot, retargeting |

Do not ship a service-capable image as `interactive` without documenting why
(deploy path must override CMD). Prefer promoting Cosmos/GR00T to `service`
when the FastAPI server is the primary product surface.

## Security baseline

Required for all workbench images:

1. **Non-root runtime** — final `USER` is non-root (`ubuntu` / uid 1000 or
   documented equivalent). Build stages may use root.
2. **No secrets in layers** — credentials via env / K8s `secretRef` only.
3. **Digest-pinned bases** where the registry allows anonymous or CI digest
   resolution (see `docs/security/image-reproducibility.md`). Document tag-only
   exceptions (e.g. NGC Isaac Lab) with a TODO.
4. **CVE scanning** — Trivy via `.github/workflows/image-security-scan.yml`.
5. **Capability drops at deploy** — K8s `securityContext` should drop `ALL`,
   set `allowPrivilegeEscalation: false`, and use `RuntimeDefault` seccomp
   (detection-training is the reference template).

Strongly recommended for `service` images:

- `HEALTHCHECK` against `/health` (or documented probe path)
- Bind to an explicit address; do not assume public `0.0.0.0` without auth
- Token auth when the service is network-reachable (LanceDB pattern)

## Redistribution (who may pull, and how widely)

The workbench is open source, and images should be pullable widely — but "widely"
has a license boundary that the contract encodes in a `redistribution` field per
image (`public` | `restricted`), enforced by
`npa/tests/docker/test_packaging_contract.py`.

- **`public`** — OSS-redistributable. Code is under OSI-approved licenses
  (Apache-2.0 / BSD-3 / MIT / MPL-2.0), the CUDA/PyTorch base images
  (`nvidia/cuda`, `pytorch/pytorch` on Docker Hub) are freely redistributable,
  and model weights are pulled at runtime. Public GR00T N1.7, GEAR-SONIC,
  Cosmos Reason1, and Cosmos3 Nano assets work anonymously; gated Cosmos assets require a token at
  **runtime** by the operator, never baked into the image. These may be published
  to a public/anonymous registry.
- **`restricted`** — bakes a runtime we are not licensed to redistribute. Such an
  image may be built and run by the operator who owns the registry (internal R&D,
  build-your-own), but hosting it **prebuilt on a public/anonymous registry** would
  make us the third-party redistributor.

  `cosmos3-serving` and the replacement `sonic-mujoco` are public only because
  their old restricted parents were removed. Cosmos serving now ships a
  zero-payload bootstrap and performs its hash-locked CUDA Python/source/model
  delivery at runtime under operator credentials and explicit terms. SONIC
  MuJoCo now builds independently from a digest-pinned public Python base and
  contains only SONIC source, MuJoCo, and the CUDA Toolkit runtime libraries
  expressly listed as redistributable by their included SDK terms. Both current
  releases are bound to exact public development digests with accepted real-GPU
  evidence.

## Manual gate audit (2026-08-16)

The image, model, and configuration surfaces below were audited for local
`ACCEPT_*` booleans, terms flags, confirmation prompts, empty acceptance
placeholders, and duplicated model-entitlement switches. The resulting contract
has two independent mechanisms: Isaac routes use the one public `ACCEPT_EULA`
variable with unset meaning `Y` and a reliable explicit opt-out; runtime-fetched
gated assets use a real upstream access probe with the operator's credential.
Neither mechanism grants redistribution rights or enables privacy/telemetry.

| Audited surface | Assets/images covered | Outcome |
| --- | --- | --- |
| Isaac runtime images and routes | `isaac-lab`, `sonic`, `groot`, Isaac-backed `sim2real` builders and raw-shell sweep states | No Isaac/Kit bytes are baked. Resolved-image routing injects canonical `ACCEPT_EULA=Y` by default; empty/negative values opt out, affirmative legacy values normalize, and invalid values fail before pull/provision/scheduling. No second public EULA variable exists. |
| GR00T deployment | `nvidia/GR00T-N1.7-3B`, `nvidia/Cosmos-Reason2-2B` | Both runtime dependencies are probed before every deploy/update path. Gated access is determined only by the operator's HF token and actual upstream permission; there is no skip or NPA terms flag. |
| Cosmos and Physical AI Data Factory | `nvidia/Cosmos-Transfer2.5-2B`, `nvidia/Cosmos-Reason2-2B`, `nvidia/Cosmos-Reason2-8B`, `nvidia/Cosmos-Reason1-7B`, `nvidia/Cosmos3-Nano`, `nvidia/Cosmos-Guardrail1`, `nvidia/Cosmos-1.0-Guardrail`, `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` | Weights stay out of image layers. Public repositories may be fetched anonymously; gated repositories require a successful upstream HF probe with the operator's token. Deploy has no bypass or duplicate consent flag. |
| Other runtime-fetched NVIDIA assets | `nvidia/GEAR-SONIC`, `nvidia/PhysicalAI-NuRec-PPISP`; NuRec NRE runtime | Public HF assets remain anonymous. NuRec's NGC-hosted NRE runtime requires a real `NGC_API_KEY` repository probe; no local EULA boolean substitutes for vendor access. |
| OpenPI / Gemma | `pi05_droid_jointpos_polaris` | The exact operator-confirmed `NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` value is forwarded only to accepted runtime jobs; refusal is attempt-scoped, and acceptance, weights, and credentials are never baked or persisted. |
| Other non-NVIDIA comparison surfaces | `Wan-AI/Wan2.2-TI2V-5B`, LeRobot, Qwen, self-hosted Llama | No local terms boolean or interactive confirmation duplicates upstream entitlement; external vendor terms still apply at the source. |
| Separate controls retained | privacy/telemetry, image redistribution classification, third-party dataset delivery | Privacy and telemetry remain independently off by default. Packaging contracts and built-image scans still control redistribution. The public PAIDF starter asset remains `acceptance_required: false`; its generic third-party dataset-license mechanism is separate from NVIDIA image/model access. |

Empty `ACCEPT_EULA` remains meaningful only as the explicit Isaac opt-out. Raw
Isaac examples now state `Y`; non-Isaac tasks do not receive the variable.

## Runtime-fetched Isaac Sim (why the Isaac images are publishable)

The four Isaac images used to bake **NVIDIA Omniverse Kit (Isaac Sim)**. The Isaac Sim
*source* is Apache-2.0, but the shipped binary bundles the Kit SDK and NVIDIA assets,
and — this is the part that is easy to get wrong — **both** the `isaacsim` *and*
`isaaclab` PyPI packages declare `License: NVIDIA Proprietary Software`, with
`isaaclab/__init__.py` carrying an explicit no-redistribution header. The
`isaac-sim/IsaacLab` **GitHub repo** is BSD-3-Clause; the wheel is a
differently-licensed repackaging of it. So "Isaac Lab is BSD-3, we can bake that half"
is wrong, and read-the-metadata beats read-the-badge.

A runtime token could not have rescued a baked image: a token gates a *download*, and
Kit was already in the layers. So the images were changed to make the statement true.

**How it works.** The images contain **no NVIDIA Isaac bytes**. On first use of
`/isaac-sim/python.sh`, `npa/docker/workbench/common/isaac_bootstrap.sh`:

1. Applies NPA's non-interactive Isaac default when `ACCEPT_EULA` is unset, then
   validates the value before download. `Y`, `YES`, `1`, and `TRUE` normalize to
   acceptance case-insensitively. Empty, `N`, `NO`, `0`, and `FALSE` are explicit
   opt-outs and exit 78; any other value is reported separately as invalid.
   The run-scoped default is not baked into image layers. The bootstrap parser
   enforces the default and explicit opt-out before downloading, while
   `npa/tests/docker/test_packaging_contract.py` fails the build if an image
   reintroduces a baked `*_ACCEPT_EULA` marker.
2. Installs the pinned `isaacsim`/`isaaclab` wheels from `https://pypi.nvidia.com` into
   a **cache volume**, not the image layers. Every wheel is pinned to a committed
   `sha256` and installed with `--no-deps --require-hashes` against `--index-url` (not
   `--extra-index-url`, so the set cannot be shadowed from PyPI).
3. Also fetches the **BSD-3 Isaac Lab source tree** at a pinned commit, because the
   wheel ships the library but no `scripts/`, and the SkyPilot Isaac tasks invoke
   `scripts/reinforcement_learning/rsl_rl/train.py`.
4. Is idempotent, concurrency-safe (`flock` + a version-stamped tree + atomic rename +
   `.complete` written last), and verifies the install before publishing it.

The shared OSS dependency layer keeps the BSD-2-Clause `imageio-ffmpeg` Python
wrapper but deletes its wheel-bundled static executable and resolves video work
through Ubuntu's dynamically packaged `/usr/bin/ffmpeg`. The built-image payload
scanner fails if that bundled executable returns.

`pypi.nvidia.com` serves these wheels **anonymously**, so the credential was never the
gate — acceptance is. NVIDIA delivers Isaac to each operator under that operator's own
acceptance, and we redistribute nothing. This is the same pattern already used for gated
model weights (Cosmos, GR00T N1, Cosmos-Reason).

**What it costs.** Runtime size and cold-bootstrap cost are version-specific.
Do not apply the old Isaac Lab 2 / Isaac Sim 5 cache measurements to the Isaac
Lab 3 / Isaac Sim 6 runtime. The current paired measurements and exact method
are in [Isaac Lab 3 workbench](isaac-lab-3.md). A per-pod `emptyDir` makes every
pod pay its generation's cold fetch. Warm a **shared** volume once instead:

```bash
kubectl apply -f npa/docker/workbench/common/warm-isaac-cache.yaml
kubectl wait --for=condition=complete job/npa-warm-isaac-cache --timeout=30m
```

Then run workload pods against the same volume with `NPA_ISAAC_CACHE_READONLY=1`, so the
runtime user never needs write access to the cache at all. Other knobs:
`NPA_ISAAC_CACHE_DIR`, `NPA_ISAAC_INDEX_URL` (point it at an internal mirror; the wheels
are sha256-pinned, so a mirror is verifiable), `NPA_ISAAC_BOOTSTRAP_OFFLINE=1`.

**Why the published image is not "Isaac Sim with Omniverse Kit".** Because it does not
contain Isaac Sim. That is verified mechanically against the *built* image, not by
reading the Dockerfile:

```bash
npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py \
    <your-registry>/<namespace>/npa-isaac-lab:3.0.0b2.post1
```

The scanner streams the image's flattened filesystem and its layer history, matching Kit
payload signatures (`libcarb`, `kit/kernel/`, `omni.*` extension dirs, `extscache`,
`*.kit`, `site-packages/isaacsim`) against a short, explicit allowlist. It cannot simply
grep for "isaac": the images deliberately keep a `/isaac-sim/python.sh` **shim**, because
~30 call sites already invoke Isaac through that path and Kubernetes pods override
`ENTRYPOINT`, making the shim the only reliable bootstrap trigger. On
The trusted generation 3 publication job scanned 126,709 local-layer entries
(35 reviewed allowlist hits) through a streaming
`docker save` pipe before push, then scans the exact pushed digest as a flattened
filesystem through the anonymous registry path (122,593 entries and 29
reviewed allowlist hits). Both scans reported `VERDICT: clean`. This avoids an image-sized
temporary archive without weakening the pre-publication layer-byte check. Both
reports must finish with `VERDICT: clean`.

**Build-your-own still works, and no longer needs NGC credentials at all**, because there
is nothing credentialed left to pull:

```bash
npa/docker/workbench/isaac-lab/build.sh --registry <your-registry>/<namespace> --push
npa/docker/workbench/sonic/build.sh    --registry <your-registry>/<namespace> --push --variant k8s
npa/docker/workbench/groot/build.sh    --registry <your-registry>/<namespace> --push
```

### Public development and release — mind the order

Official images use one public namespace:
`ghcr.io/nebius/nebius-physical-ai`. A reviewed build first receives the
immutable tag `dev-<full-git-sha>` on its normal `npa-<tool>` package. Public
development images are anonymously downloadable immediately; deleting a failed
tag cannot revoke downloads.

Before that first public push, run every packaging, licensing, base-pin,
non-root, secret, customer-data, infrastructure-data, proprietary-payload,
gated-asset, SBOM, vulnerability, provenance, source-revision, and bootstrap-
contract gate. Restricted images never enter official GHCR. After the push,
resolve the digest, repeat exact-digest scans, prove anonymous pullability, and
run a real functional workflow on compatible physical hardware.

Promote only the validated digest to its supported release tag:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai \
  --development-sha <full-git-sha> --dry-run
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai \
  --development-sha <full-git-sha>
```

The publisher resolves each development tag once and copies only by immutable
digest. It checks config, licensing, payload scans, SBOM/provenance attestations,
and any declared bootstrap contract before exposing a release tag, then verifies
anonymous pullability and exact digest parity.

**The honest trade.** The payload-clean image still carries the selected CUDA,
PyTorch, and OSS training stack; only the proprietary runtime moves to the
operator cache. Image size and bootstrap-cache size are therefore separate
measurements, and neither should be inferred from the other.

Model weights are a separate axis and are never baked into any image: Cosmos,
GR00T N1, and Cosmos-Reason weights (and VLMs) are downloaded at **runtime**
using the customer's own HF/NGC token when the upstream repository is gated.
Every required gated repository is probed before provisioning; there is no NPA
manual terms flag or access-check bypass. Public repositories work anonymously.
The upstream license still applies, and we never redistribute weights.

The manual `Publish public images` workflow can build selected public development
images and can separately preflight/promote selected validated tools. It uses the
repository-scoped `GITHUB_TOKEN` with `packages: write`; no second registry token
or namespace is part of the official flow. Run `--preflight` before promotion.

| Code | Meaning | Fix |
| --- | --- | --- |
| `UNAUTHORIZED` on **every** image | the GHCR credential resolved to no identity | replace the scoped credential |
| `UNAUTHORIZED` on **some** images | the identity lacks package read access | fix package access, not the tag |
| `NAME_UNKNOWN` | no such repository — the image was never built and pushed | build and push it, or `--skip-missing` |
| `MANIFEST_UNKNOWN` | the repository exists but not this tag — the pin points at an unpushed build | correct the pin, or `--skip-missing` |

Locally, authenticate only when a registry operation needs a write-capable identity:

```bash
printf '%s' "$GHCR_TOKEN" | crane auth login ghcr.io -u "$GHCR_USER" --password-stdin
```

Prove the development image anonymously with an empty Docker config:

```bash
export DOCKER_CONFIG="$(mktemp -d)"
crane manifest ghcr.io/nebius/nebius-physical-ai/npa-lerobot:dev-<full-git-sha> >/dev/null
```

### The plan is what we build, not what is pushed

The publish plan is derived from the packaging contract, which records what this repo
**builds**. The registry holds what someone actually **pushed**. Those two diverge every
time a new tool lands — Dockerfile, contract entry and version pin merge together, while
building and pushing the image is a separate manual step (there is no build-and-push
automation). A brand-new tool is therefore *expected* to be absent from the registry for a
while, and the preflight reports it as `NAME_UNKNOWN`.

By default that blocks the publish, which is the right default: silently mirroring a subset
would make a pin regression that dropped an image look exactly like success. When the gap is
known and intended, publish the ready images anyway:

```bash
python -m npa.deploy.publish_public --skip-missing            # or the workflow's skip_missing input
```

It drops only the images the registry has no copy of, prints each one with the reason, and
copies the rest. Two properties matter here:

- **A denial is never skipped.** `UNAUTHORIZED` / `DENIED` stops the run even with
  `--skip-missing`, because a credential or role fault would otherwise quietly shrink the
  published set. Denial also wins when a registry answers `NAME_UNKNOWN` for a repository
  the identity cannot see.
- **Skipped release tags are absent**, so a consumer using the supported pin
  gets a pull failure for those tags until the image is built and the workflow re-run.

The copy itself is incremental. After the complete digest-pinned source preflight,
bootstrap attestation, and licensing gates,
the publisher compares each source and target manifest digest. An exact match prints
``Already current; skipping copy`` and performs no registry write; only a missing or changed
target runs ``crane copy``. This makes it safe to re-run the full guarded plan when one new
image lands without republishing every existing image. Consumers then pull by
pointing the resolver at the public release channel:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai   # OSS images, any tenant
```

Both development and release tags must pass the unauthenticated check:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public --verify-accepted-releases
```

This read-only health check resolves every accepted release tag anonymously and
compares it with `public_release_manifest.json`'s `published_digest`. Use
`--verify-parity` separately when a retained development source tag exists and
source-to-release parity is the question; a missing historical dev tag is not a
release-byte verdict.

Never add a `restricted` image to official GHCR.
`publish_public` and `development_image_for_tool` refuse every member of the
general restricted-image inventory. Separately, license-eligible candidates
remain in `UNVALIDATED_PUBLICATION_TOOLS` until exact-digest GPU evidence is
recorded, so a classification change alone cannot create a supported release.

> **Publishing is a business decision.** The engineering makes publication defensible —
> the images contain no NVIDIA-proprietary bytes, and NVIDIA delivers Isaac to each
> operator under that operator's own acceptance — but dispatching the workflow with
> `dry_run=false` should wait on sign-off from someone with the authority to accept it.

## Feature exposure

| Access mode | Contract |
| --- | --- |
| Container | ENTRYPOINT/CMD matches packaging tier; ports via `EXPOSE` |
| API | FastAPI endpoints: `/health`, `/status`, `/system-info`, `/list` (+ tool verbs) |
| CLI | `npa workbench <tool> ...` |
| SDK | `npa.sdk.workbench.<tool>` |
| YAML | `toolRef` in `catalog.py` + SkyPilot `image_id` from manifests |

Cross-tool data moves through S3 (`--input-path` / `--output-path`), never
direct service-to-service file coupling.

## Build and tag

1. Resolve registry with `npa.clients.config.resolve_container_registry`.
2. Build from the checked-in Dockerfile (`skills/atomic/build-and-push-image`).
3. Tag from `npa/pyproject.toml` `[tool.npa.supported-tools]` and
   `npa/docker/workbench/tags.yaml` (`cuda12` vs `cuda13-b300`).
4. SONIC variants: `npa/src/npa/deploy/sonic_image_manifest.json`.
5. Blackwell fleet digests: `npa/docker/workbench/sm120-images.json`.
6. Update golden evals when the image’s “does its job” command changes.

## Operator checklist (new or changed image)

- [ ] Dockerfile under `npa/docker/workbench/<tool>/`
- [ ] Packaging tier chosen and matches ENTRYPOINT
- [ ] Non-root final USER
- [ ] Base digest pinned or exception documented
- [ ] Registered in `CONTAINER_IMAGE_NAMES` + supported-tools version
- [ ] Golden eval entry present and passing offline validate
- [ ] CLI/SDK/YAML surfaces updated together
- [ ] Skill + `skills/index.yaml` smoke updated

## Related docs

- Cosmos Transfer 2.5 has an artifact-by-artifact redistribution record at
  `npa/docker/workbench/cosmos2-transfer/REDISTRIBUTION.md`; its `public`
  classification is valid only after the registry-image audits named there pass.
- PAIDF's CC-BY-4.0 RoboPro starter is not an image payload. The operator-side
  CLI runtime-fetches the immutable object, verifies its SHA-256, and stages it
  to the run. `skills/NOTICE-PAIDF-STARTER-MEDIA` records attribution and the
  source/model/media license boundary; Cosmos repository example media remains
  excluded because its asset-level rights and provenance are not uniform.

- `docs/security/container-golden-evals.md` — usefulness + safety contract
- `docs/security/image-reproducibility.md` — digests and tag families
- `docs/architecture/oss-onboarding-ladder.md` — OSS → marketplace promotion
- `docs/workbench/cli-sdk-yaml-walkthrough.md` — three-access pattern
