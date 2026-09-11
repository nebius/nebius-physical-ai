# Habitat-Sim BYOF registry candidate

This candidate packages pinned Habitat-Sim source into an operator-private BYOF
image and exercises the simulator through NPA, SkyPilot, and Kubernetes. Its
admission gate is a non-interactive embodied-agent traversal of the official
Skokloster Castle test scene with RGB and depth sensors, headless NVIDIA EGL,
and Bullet physics. An import, image build, scene download, or GPU visibility
check by itself does not pass.

Status: **implementation complete; live acceptance pending**. Schema validation
and planning are recorded separately from execution readiness in
[`byof-habitat-sim.readiness.json`](../../workflows/testing/byof-habitat-sim.readiness.json).
Do not describe this candidate as accepted until the exact private image digest
passes the live gate on the run-owned STRICT target.

## Pinned upstream and maintenance warning

- Source: [`facebookresearch/habitat-sim`](https://github.com/facebookresearch/habitat-sim)
  at `57ee4941dc4765240f0f91f70b2c97a919bf9038`.
- Source license: MIT, as recorded in the pinned upstream `LICENSE` and project
  metadata.
- Packaging: the linux/amd64 Ubuntu 22.04 base is pinned to immutable manifest
  digest `sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986`.
  Both the generic BYOF bootstrap and the solution packages resolve through the
  Ubuntu snapshot `20260801T053000Z`. The Python 3.10 closure contains 35 exact
  wheels with reviewed SHA-256 hashes and installs with `--require-hashes`,
  `--only-binary=:all:`, and `--no-deps`; the source build also disables build
  isolation and dependency resolution, including the S3 upload client. The
  image retains the lock plus sorted Python and Debian package inventories for
  digest-bound review.
  The build fetches only the gitlink-pinned OSS submodules needed for RGB-D,
  pathfinding, Bullet, and EGL. It deliberately excludes the unused GUI,
  documentation, and non-commercial audio-propagation submodules. It does not
  bake scenes, semantic data, weights, credentials, or caches.

Upstream's pinned README warns: **Beyond v0.3.4, Meta internal teams do not
officially maintain releases or provide active development.** This integration
therefore pins a full commit, exercises the actual supported interfaces, and
does not imply a newer officially maintained Meta release.

## Scene and license boundary

The hard gate fetches the official Meta-hosted
[`habitat-test-scenes.zip`](http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip)
archive referenced by the pinned upstream
[`examples/settings.py`](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/examples/settings.py).
That URL is mutable, so it is not treated as the asset identity. The workflow
requires the complete 94,590,970-byte archive to match SHA-256
`1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06`,
passes a complete ZIP integrity check, and then requires these exact members:

| Archive member | Bytes | CRC32 | SHA-256 |
| --- | ---: | --- | --- |
| `data/scene_datasets/habitat-test-scenes/skokloster-castle.glb` | 38,295,764 | `7a0ced74` | `b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56` |
| `data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh` | 28,192 | `a694cab0` | `1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d` |

The pinned Habitat-Sim
[`README`](https://github.com/facebookresearch/habitat-sim/blob/57ee4941dc4765240f0f91f70b2c97a919bf9038/README.md)
identifies this demo as [*The King's Hall* by Skokloster
Castle](https://sketchfab.com/3d-models/the-kings-hall-d18155613363445b9b68c0c67196d98d),
links it under the [Creative Commons Attribution 4.0
license](https://creativecommons.org/licenses/by/4.0/legalcode.en) (**CC BY 4.0**),
and credits the
scan to Erik Lernestål. Outputs preserve the creator and scan attribution,
license and original-asset links, and a modification notice stating that the
official Habitat-ready GLB/navmesh were derived from the original scan and are
copied byte-for-byte by NPA. There is no separate EULA, click-through, credential,
or local acceptance proxy for these public CC BY bytes.

The archive is a runtime-fetched data artifact, never an image layer. It is
downloaded to an ephemeral mode-0600 file, verified as a whole, opened without
path-based extraction, and deleted after the two selected members pass exact
name, size, CRC32, and SHA-256 checks. No unrelated archive member is extracted
or retained. The selected GLB and navmesh form an ephemeral worker cache and are
not permission to persist or redistribute any other scene bytes.

No Matterport3D, HM3D, Replica, Gibson, or other separately licensed scene data
is used. There is no EULA click-through or gated repository needed for the
selected source and two public scene files.

## Hard-gate contract

The exact required artifact is
`$NPA_SMOKE_OUTPUT_DIR/habitat-sim-smoke.json`. A passing record contains:

- the requested and pod-observed source commit;
- scene ID, mutable archive URL role, immutable archive/member hashes, exact
  member names/sizes/CRC32 values, license, attribution, original identity, and
  modification provenance;
- every saved RGB PNG, depth NumPy array, and depth preview PNG with shapes and
  paths, byte sizes, and SHA-256 hashes, plus aggregate hashes and finite depth
  statistics; live acceptance downloads and re-hashes every declared object;
- the pathfinder-selected start, goal, end, action trace, geodesic distance,
  collision count, and nonzero displacement;
- Bullet build/enable evidence, physics world-time advancement, and the count
  of `Simulator.step(dt=1/60)` physics steps;
- measured rendered frames per second;
- a live NVIDIA OpenGL vendor/renderer/version query and the EGL/NVIDIA EGL
  libraries loaded into the renderer process;
- exactly one pod-visible RTX PRO 6000 Blackwell, its `12.0` compute capability,
  and architecture classification;
- the `repository@sha256:…` reference injected into the pod by the BYOF runner,
  its digest, retained dependency inventories, and `exit_status: 0`. Before
  teardown, the live test independently reads Kubernetes
  `containerStatuses[].imageID` and requires the pushed digest there too. It
  preserves that observation with the proof and STRICT-provider-receipt hashes
  in the private `habitat-sim-live-validation.json` acceptance record.

The smoke fails before success output if archive access fails, the mutable
archive differs from its pinned size or hash, ZIP integrity or selected-member
identity fails, the GPU count or model is wrong, a B200 or non-Blackwell device
is selected, NVIDIA EGL is not actually loaded, the image is not digest-pinned,
RGB/depth output is absent or static, depth has no finite samples, Bullet time
does not advance, or the agent moves no meaningful distance.

The companion profile requests exactly
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`. Habitat-Sim is a renderer and must
run on the manager-provided, reservation-backed RTX PRO target—**never B200**.
The profile selects the pod shape; the owner-only cluster provenance must
separately prove the backing Capacity Block policy is `STRICT` before launch.

## Validate and plan

From the repository root, after installing the development environment:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  workflows/testing/byof-habitat-sim.yaml --json
npa/.venv/bin/npa workbench workflow plan-spec \
  workflows/testing/byof-habitat-sim.yaml \
  --run-id habitat-sim-plan --json
npa/.venv/bin/npa workbench workflow submit \
  workflows/testing/byof-habitat-sim.yaml \
  --run-id habitat-sim-plan --plan-only
```

Planning renders the `workbench.byof.repo` invocation; it does not build an
image, download the scene, prove the target, or run Habitat-Sim.

## Live qualification

Do not use an ambient/default project or shared cluster. First obtain the
owner-only manager runtime context proving the task-owned project, private
registry, storage prefix, Kubernetes context, and the exact RTX PRO Capacity
Block bound with policy `STRICT`. Keep those identifiers out of commits, logs,
PR text, and public artifacts.

Before building or submitting:

```bash
npa/.venv/bin/npa configure --show
npa/.venv/bin/npa workbench health preflight --checks nebius --json
npa/.venv/bin/npa workbench health preflight --checks s3 --json
```

The live E2E does not accept an ambient context, default registry, or boolean
STRICT attestation. The manager must create an owner-only runtime receipt
outside the repository and bind it to the exact run. It references a second
owner-only provider-readback JSON whose SHA-256 is recorded in the runtime
receipt. The provider receipt must prove the active Capacity Block and the
cluster node-group policy independently:

```json
{
  "schema_version": "npa.byof.habitat-sim.runtime-context.v1",
  "solution": "habitat-sim",
  "run_id": "<unique-habitat-run-id>",
  "task_owned": true,
  "manager_published": true,
  "project": "<npa-project-alias>",
  "registry": "<authorized-private-registry-path>",
  "registry_visibility": "private",
  "registry_push_authorized": true,
  "docker_config_path": "<absolute-owner-only-config-json>",
  "bucket": "<run-owned-bucket>",
  "output_prefix": "oss-solutions/habitat-sim/<unique-habitat-run-id>",
  "s3_endpoint": "<task-owned-s3-endpoint>",
  "kubeconfig_path": "<absolute-owner-only-kubeconfig>",
  "kubernetes_context": "<exact-task-owned-context>",
  "kubernetes_namespace": "<exact-namespace>",
  "skypilot_config_path": "<absolute-owner-only-skypilot-config>",
  "reservation": {
    "policy": "STRICT",
    "state": "ACTIVE",
    "accelerator": "RTX PRO 6000 Blackwell",
    "gpu_count": 1,
    "kubernetes_context": "<exact-task-owned-context>",
    "capacity_block_group_id": "<private-capacity-block-id>",
    "provider_receipt_path": "<absolute-owner-only-provider-readback-json>",
    "provider_receipt_sha256": "<sha256-of-provider-readback-json>",
    "verified_at": "<provider-readback-timestamp>"
  }
}
```

The referenced provider JSON uses schema
`npa.nebius.strict-capacity-binding.v1` and repeats the project, Kubernetes
context, capacity-block ID, active state, accelerator, and GPU count. Its
`node_group_reservation_policy` must be exactly
`{"policy":"STRICT","reservation_ids":["<same-capacity-block-id>"]}`.
All referenced files must be absolute, use no symlinked path component, remain
outside this checkout, and be readable only by their owner. Their immediate
parent directories must also be owner-only. The private registry must not
resolve to the public NPA GHCR namespace. The run ID must be a bounded,
DNS-safe name beginning with `habitat-`; option-like values are rejected before
the runner or teardown receives them.

Use a fresh run ID and replace only `config.bucket` in an owner-private copy of
the workflow. The normal `workbench.byof.repo` state then builds the source,
pushes only to the authorized private registry, resolves the pushed tag to an
immutable digest, and launches the dedicated resource profile. It uploads the
summary, exact proof, rendered observations, and diagnostic logs beneath that
unique run prefix. Run the dedicated E2E with
`NPA_INTEGRATION_E2E=1`, `NPA_BYOF_HABITAT_SIM_RUN_ID`,
`NPA_BYOF_HABITAT_SIM_RUNTIME_RECEIPT`, and
`NPA_BYOF_HABITAT_SIM_LIVE=1`:

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_byof_onboarding_live_e2e.py \
  -k test_live_habitat_sim_private_digest_rgb_depth_bullet_traversal -v
```

It captures the Kubernetes-observed image ID, downloads and re-hashes all
RGB/depth and inventory objects, then tears down only the exact SkyPilot-owned
pod/cluster. Do not publish or promote the BYOF image to GHCR.

## Explicitly deferred

- proprietary or gated datasets, including Matterport3D and HM3D;
- Replica, Gibson, or any other scene pack beyond the two pinned Skokloster
  files;
- semantic annotations and semantic-sensor claims;
- Habitat-Lab installation, policy learning, and distributed Habitat-Lab
  training;
- interactive viewers, GUI workflows, audio sensors, and multi-GPU rendering;
- a stable public NPA image or first-class CLI/SDK tool.

These are outside the smallest correct BYOF candidate and need separate license,
data, packaging, resource, and live-capability reviews.
