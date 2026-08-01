# Workbench Container Packaging

Canonical contract for packaging workbench containers correctly, securely, and
with the right runtime features exposed. Machine-readable rules live in
`npa/docker/workbench/packaging-contract.yaml` (enforced by unit tests).

## Inventory

All first-class images live under `npa/docker/workbench/`:

| Image / role | Dockerfile | Default exposure |
| --- | --- | --- |
| `npa-lerobot` | `lerobot/Dockerfile` | FastAPI server `:8080` |
| `npa-lerobot-policy` | `lerobot-policy/Dockerfile` | entrypoint modes (serve/train/eval) |
| `npa-genesis` | `genesis/Dockerfile` | job shell (CLI supplies command) |
| `npa-isaac-lab` | `isaac-lab/Dockerfile` | job shell |
| `npa-cosmos` | `cosmos/Dockerfile` | job shell; server built but not default CMD |
| `npa-groot` | `groot/Dockerfile` | job shell; `EXPOSE 8080` |
| `npa-fiftyone` | `fiftyone/Dockerfile` | job shell; `EXPOSE 5151` |
| `npa-lancedb` | `lancedb/Dockerfile` | uvicorn `:8686` |
| `npa-sonic` | `sonic/Dockerfile` | `/entrypoint.sh` modes |
| `npa-detection-training` | `detection-training/Dockerfile` | uvicorn `:8790` |
| `npa-retargeting` | `retargeting/Dockerfile` | job shell |
| `npa-foxglove-embed` | `foxglove-embed/Dockerfile` | static host `:8099` (Foxglove embed SDK + MCAP data) |
| Sim2Real stack | `sim2real-*/`, `cosmos3-reason/`, `lerobot-vlm-rl/` | workflow modules |
| Base CUDA 13 | `base/cuda13-b300/Dockerfile` | build base only |

BYOF images (`npa-byof:<run-id>`) are **ad-hoc** and are not registered in
`CONTAINER_IMAGE_NAMES` until promoted to Tier 2 (see
`docs/architecture/oss-onboarding-ladder.md`).

## Packaging tiers

Every Dockerfile must declare one of:

| Tier | `kind` | ENTRYPOINT expectation | Examples |
| --- | --- | --- | --- |
| **Service** | `service` | Starts the HTTP service (or entrypoint that does) | lerobot, lancedb, detection-training, lerobot-policy |
| **Job** | `job` | Runs a workflow/CLI module with explicit CMD | sonic, sim2real-eval, cosmos3-reason, lerobot-vlm-rl |
| **Interactive** | `interactive` | `/bin/bash` allowed only when CLI always overrides CMD | genesis, isaac-lab, fiftyone, cosmos, groot, retargeting |

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
  and any gated model weights (Cosmos, GR00T N1, Cosmos-Reason) are pulled at
  **runtime** by the operator, never baked into the image. These may be published
  to a public/anonymous registry.
- **`restricted`** — bakes **NVIDIA Omniverse Kit (Isaac Sim)** binaries
  (`isaac-lab`, `sonic`, `sonic-mujoco`, `groot`). The Isaac Sim *source* is
  Apache-2.0, but the shipped binary bundles the Omniverse Kit SDK + NVIDIA
  assets, which are NVIDIA-proprietary (the `isaacsim` PyPI package's own license
  field reads *"NVIDIA Proprietary Software"*).

  The compliant way customers get these is **build-your-own**: each deployment
  builds the image into its **own** registry (`build.sh --registry
  cr.<region>.nebius.cloud/<your-registry-id> --push`, `NPA_REGISTRY_ID` is
  per-operator), pulling the `nvcr.io/nvidia/isaac-lab` base / `isaacsim` wheels
  with the operator's **own NGC credentials + EULA acceptance**
  (`OMNI_KIT_ACCEPT_EULA=YES`). NVIDIA therefore delivers Omniverse Kit to each
  operator under that operator's own acceptance, and we ship only the Dockerfile
  + orchestration — not the proprietary binaries. This is why using their own
  NGC/HF tokens keeps customers compliant.

  The **one** thing that is *not* allowed is hosting these images **prebuilt on a
  public/anonymous registry**: a pull from such a registry needs no NGC token and
  bypasses the EULA gate, which would make us the third-party redistributor of
  Omniverse Kit (that needs an NVIDIA AI Enterprise license). So keep the
  `restricted` set off any public registry.

Model weights are a separate axis and are never baked into any image: Cosmos,
GR00T N1, and Cosmos-Reason weights (and VLMs) are downloaded at **runtime**
using the customer's own HF/NGC token, so the customer accepts each model
license (e.g. the NVIDIA Open Model License) directly. We never redistribute
weights.

Access model today (both regions): each workbench registry
(`cr.eu-north1.nebius.cloud/…` primary, `cr.us-central1.nebius.cloud/…` mirror)
is **already readable org/tenant-wide** — the tenant `viewers`/`editors` groups
hold `viewer`/`editor`, which cascades to image pull — so developers inside the
owning org can pull every image, including the `restricted` ones (internal R&D
use).

**Pulling from any Nebius tenant / publicly.** Nebius Container Registry cannot
express "any Nebius tenant can pull": it has **no anonymous/public mode** and
**no `allAuthenticatedUsers` / cross-tenant grant**. Every pull needs an
authenticated identity, tenants are strictly isolated, and the only way to admit
an out-of-tenant identity is to invite that specific account into the owning
tenant and add it to a group — which does not scale to "anyone from any tenant."
The only way to make the images pullable by every Nebius tenant (which is also
pullable by anyone) is therefore to **mirror to a public-capable registry** —
GHCR (`ghcr.io`, the default), Docker Hub, or Quay.

Only the `public`-classified subset may be mirrored. Use the license-guarded
publisher, which copies exactly `publicly_publishable_tools()` (16 images) and
hard-refuses the Omniverse-Kit images:

```bash
# defaults to $NPA_PUBLIC_REGISTRY, else ghcr.io/nebius/nebius-physical-ai
python -m npa.deploy.publish_public --dry-run
python -m npa.deploy.publish_public --target ghcr.io/<org>/<repo>
```

or the `Publish public images` GitHub Actions workflow (manual dispatch,
dry-run by default). **Consumers in any tenant** then pull the OSS images by
pointing the resolver at the public mirror:

```bash
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai   # OSS images, any tenant
```

Never add the `restricted` images to a public target — that redistributes NVIDIA
Omniverse Kit to third parties (needs an NVIDIA AI Enterprise license). Those
stay build-your-own (each operator builds with their own NGC credentials + EULA).

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

- `docs/security/container-golden-evals.md` — usefulness + safety contract
- `docs/security/image-reproducibility.md` — digests and tag families
- `docs/architecture/oss-onboarding-ladder.md` — OSS → marketplace promotion
- `docs/workbench/cli-sdk-yaml-walkthrough.md` — three-access pattern
