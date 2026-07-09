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
