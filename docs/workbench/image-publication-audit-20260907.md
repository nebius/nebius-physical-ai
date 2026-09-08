# Public image publication audit — 2026-09-07

The findings below record the September 7 baseline. The subsequent publication
work replaces the Cosmos inherited vendor containers with runtime-fetch
bootstraps, replaces OpenPI/RoboCasa cuDNN development ancestry, pins the
RoboCasa closure, and repairs its package import defect. These replacement
sources are eligible for guarded public development builds; they remain outside
the supported-release selection pending qualification. Historical tags, scan
results and private runtime evidence do not qualify the replacement bytes.

## September 7 baseline

The claim was correct about the audited public release inventory: **32 of 37
canonical images are selected for publication, and all 32 accepted release tags
resolve anonymously to their recorded digests**. It needs a narrower statement
about the other five: four anonymous probes received HTTP 403, which cannot
distinguish a private package from a nonexistent package; the OpenPI candidate
tag received HTTP 404. None of the five was verified anonymously pullable.

This audit used `origin/main` at
`b08b6b789b3f433584e002a6ea68ab3015e63412`, including the cuRobo Python 3.10
compatibility and Cosmos Ray request-integrity fixes. The older checkout SHA in the original claim is not the
audited source. No image was built, promoted, republished, or removed from
quarantine by this audit.

## What the counts mean

| Evidence boundary | Observation |
| --- | --- |
| Canonical resolver inventory | 37 tool/image mappings in `npa.deploy.images.CONTAINER_IMAGE_NAMES` |
| Public release selection | 32 members of `publicly_publishable_tools()` |
| Explicit redistribution restriction | 2 members of `RESTRICTED_PUBLICATION_TOOLS` |
| Publication quarantine | 2 unvalidated tools plus 1 validation candidate |
| Packaging contract | 36 entries: 34 marked `public`, 2 `restricted` |
| Anonymous accepted-release check | 32/32 digest matches; 0 mismatches |
| Omitted candidate references | 0/5 anonymously verified; 4 HTTP 403, 1 HTTP 404 |

The packaging inventory and canonical tool map describe different objects:
`loop-eval` uses the `sim2real-eval` build and `reference-policy` is derived.
Foundation Dockerfiles and historical variants are not additional canonical
public releases. The catalog's old 35-packaging/36-tool/one-restricted totals
were stale. The `public` packaging field records intended redistribution
eligibility; quarantine still requires review of the actual built layers.

The registry command was:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai --verify-accepted-releases
```

It compares anonymous manifest content digests directly with
[`public_release_manifest.json`](../../npa/src/npa/deploy/public_release_manifest.json).
This proves current accepted tag/digest parity. It does not repeat complete
image-layer security scans or establish that newer source is present in older
release bytes. Historical aliases and OCI config claims retain the catalog's
separately dated inspection.

In particular, the latest native Cosmos Ray request/batch-result integrity and
runtime tokenizer changes are source changes. Its retained accepted digest
identifies earlier release bytes; this audit does not claim those newer changes
are present in that public image.

## Why each image is excluded

| Exact candidate reference | Current control | Concrete reason and remaining publication evidence |
| --- | --- | --- |
| `npa-cosmos3-nano-video:0.1.0` | Restricted; anonymous HTTP 403 | Inherits the pinned vLLM-Omni vendor runtime, including CUDA/cuDNN components and vendor fixtures. The extension's source license and runtime-fetched model do not establish redistribution rights for those inherited layers. Retain operator-private builds until the complete component rights are established or the runtime is replaced with an audited closure; then require exact-digest packaging/security and real capability evidence. |
| `npa-cosmos3-super-benchmark:0.1.0` | Restricted; anonymous HTTP 403 | An operator-private SkyPilot bootstrap wrapper around the pinned vLLM-Omni runtime. Its redistribution record requires rights for every inherited layer to be established separately. Both current benchmark YAMLs require a private `runtime_image` digest and pull Secret. The old skill's direct-public-image instructions were incorrect. |
| `npa-curobo:0.8.0-cuda13-b300-unbuilt` | Unvalidated publication quarantine; anonymous HTTP 403 | Pinned cuRobo V2 source/assets and benchmark data have permissive licenses; no model weights or gated downloads are required. Source now uses a non-cuDNN CUDA base and filters the installed cuDNN wheel before layer creation. Actual layer inventories, retained notices, security/SBOM/provenance evidence and separate B200/RTX PRO 6000 physical qualification remain required. The `b300` tag text is not a B300 result. |
| `npa-openpi:pi05-full-droid-rlds-cu128-unbuilt` | Unvalidated publication quarantine; anonymous HTTP 404 | The full-DROID recipe keeps weights and datasets at runtime, but inherits a cuDNN development base and installs cuDNN wheels. Exact-layer rights remain to be resolved alongside byte/security gates. The specified eight-node RTX PRO 6000 qualification is missing; older small Polaris/BYOF inference or optimizer tests do not qualify this full-DROID candidate. |
| `npa-robocasa:0.1.0` | Validation candidate quarantine; anonymous HTTP 403 | No accepted first-class image digest and GPU evidence. RoboCasa and robosuite source use MIT, correcting the earlier Apache-2.0 claim. The mutable cuDNN development base, source tag and partially constrained dependencies need exact-byte review. A reproduced package-copy defect also prevents service import. Historical BYOF results do not validate this service image. |

The two Cosmos restrictions are conservative repository distribution decisions,
not proof that every inherited component is legally prohibited. Their
[Nano](../../npa/docker/workbench/cosmos3-nano-video/REDISTRIBUTION.md) and
[Super](../../npa/docker/workbench/cosmos3-super-benchmark/REDISTRIBUTION.md)
records do not provide a complete component-by-component grant analysis. That
missing evidence must be resolved before relaxing the controls. The existing
public `cosmos3-ray-serve` and `cosmos3-serving` images use different packaging;
their acceptance does not transfer to these wrappers.

For OpenPI and RoboCasa, the inherited cuDNN development layers deserve specific
review. The [current cuDNN supplement](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html)
enumerates shared runtime libraries for distribution; older bundled notices can
differ. Inspect the exact base layers, wheels and their applicable notices.
Removing a file in a later layer does not remove it from the distributed parent
layer. This audit identifies unresolved rights, rather than declaring that a
particular older SDK grant has been conclusively violated. The quarantines stay
in place while that review is outstanding.

Source license evidence is available in the
[RoboCasa v1.0 MIT license](https://raw.githubusercontent.com/robocasa/robocasa/v1.0/LICENSE),
[pinned robosuite MIT license](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/85abee228d1c43ab1939bce33028099945d453b4/LICENSE),
[cuRobo V2 Apache-2.0 license](https://github.com/NVlabs/curobo/blob/8e734f3ced1df898990bcd92de40abce475907db/LICENSE),
and [pinned OpenPI Apache-2.0 license](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/LICENSE).
These grants cover their respective sources, not the whole container closure.

## Runtime access and acceptance are separate gates

The Nano model is fetched anonymously at its pinned revision under
[OpenMDW-1.1](https://openmdw.ai/license/1-1/). Other selected models or guardrail
weights can require upstream account entitlement. An HF/NGC credential proves
authenticated access, while an exact-revision payload probe establishes actual
fetch access. Neither is a redistribution grant.

The Super benchmark requires the run-scoped NVIDIA software acceptance value
documented in its skill. OpenPI separately requires exact
`NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES` before build or checkpoint fetch, covering the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and linked prohibited-use
policy. This acceptance must remain runtime-only. It does not waive image
licensing, security, dataset, or GPU qualification requirements. Optional privacy
and telemetry consent remains separate.

## Workload coverage and qualification limits

The audit's substantive workload coverage uses the existing private Nano
diffusion service: one complete synthetic continuation through the SDK, eight
complete concurrent continuations through the CLI, and full-source structural
augmentation of the newly generated single clip. It uses the supported live
acceptance tests and verifies immutable S3 publication and publication-only
recovery. This directly tests that a restricted runtime can execute useful work
without changing its redistribution classification.

The continuation acceptance completed successfully on the existing 16-replica
B200 deployment:

| Fresh workload | Observed result | Verified artifact objects | Client batch interval |
| --- | --- | --- | --- |
| SDK single continuation | 1/1 complete video | 13 | 185.04 s |
| CLI concurrent continuation | 8/8 complete videos, eight distinct replicas and eight overlapping rollout/chunk intervals | 97 | 190.67 s |

All nine final videos fully decoded to **720 frames, 832×480, 24 fps and
30 seconds**, with their three raw diffusion chunks retained. The live test
passed and both batches verified every published object's hash after reading it
back. The client batch intervals include generation, downloads and validation,
and exclude S3 publication; the complete test invocation took 440.06 seconds.
Models were already resident, and local CPU validation ran alongside other
checks. These are observed task intervals, not isolated or sustained-throughput
benchmarks. They apply to the deployed digest retained in private evidence,
not to an unbuilt image from the latest source.

Full-source augmentation then passed its real SDK generation and CLI recovery
test. Source and augmentation each decoded to **720 frames / 30 seconds**; the
labelled comparison decoded at **1664×480**. All **18 artifact objects** passed
S3 readback. Recovery completed with **zero serving HTTP attempts** and an
unchanged submission marker, proving that it did not generate another video.
Server work took **421.09 seconds**; the full test invocation, including local
validation, publication and recovery, took **635.61 seconds**.

Independent review inspected all 18 continuation join sheets. Augmentation
review inspected source/output comparisons at four positions across the clip
and eight consecutive frames around each of its two joins. No obvious scene
replacement was seen at those sampled joins. The augmentation visibly adds
warm dim lighting and a darker floor while retaining the recognizable robot
and warehouse composition. Small-wheel geometry is distorted, and puddles are
weak or inconsistent later in the clip. These are qualitative artifact
observations, not a human quality score or certification of physical rolling,
slip, contact forces or full-video motion continuity.

The generated source MP4 has SHA-256
`cafa167d9ebe1cf9bcfcbfa6d8a5def9a0f1b3c920999a5acfa662fba0900099`;
the augmentation has SHA-256
`1bc02d5cb3e1cb882c7563f148971f7e55fa5c763444532c6e92f424b6a80980`.
The task created no cloud infrastructure and closed its own local port-forward
after completion, retaining the existing shared service and private artifacts.

Reproduction uses the endpoint, token, distinct output URIs and private evidence
directory described in the [Nano deployment recipe](../../npa/deploy/cosmos3-nano-video/README.md):

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest -q -s \
  npa/tests/e2e/test_cosmos3_nano_video_live.py
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest -q -s \
  npa/tests/e2e/test_cosmos3_nano_video_augment_live.py
```

The second invocation uses the first invocation's newly generated single MP4
as its S3 input. Credentials and concrete destinations stay outside Git.

The other candidates remain unqualified. A complete cuRobo benchmark must retain
every problem in both pinned datasets, in kinematic and 3 kg payload-dynamics
modes, with validated problem journals and decoded RRD. OpenPI's required
eight-node RTX qualification requires eight distinct one-GPU nodes, the `(1,8)`
JAX mesh, byte-verified full DROID preparation, normalization, 100 optimizer
updates and the specified training-journal/RRD evidence. Its full recipe is a
separate 100,000-update training run; neither ran in this audit. Any initial
OpenPI development-image publication must first satisfy its pre-push byte,
licensing and security gates; supported release promotion additionally requires
the accepted qualification evidence.

The fixed Super benchmark sweep also did not run here. Its primary suite would
require all four eight-GPU service arrangements, validated warmups and 24 measured
attempts per cell with complete MP4 decoding and failure-inclusive accounting.
No new Super throughput result is claimed. This audit does not
repeat every historical capability result for the 32 public images.

RoboCasa was tested at its immediate startup boundary. An isolated host process
recreated the Dockerfile's `/app/npa` package layout, copied only
`workbench/robocasa`, and imported the service. It exited **1** with
`ModuleNotFoundError: No module named 'npa.clients'`. The package eagerly imports
`npa.clients.storage`; its schemas also require `npa.cli.path_contract`, which is
likewise outside the COPY closure. This is a reproducible packaging-layout
failure, not a built-image scan or an executed GPU rollout. Repair and validate
the complete service dependency closure before attempting real kitchen
trajectory export, policy training, and held-out evaluation.

The RoboCasa GPU metadata was also stale: the Dockerfile installs Torch 2.12.1
and Torchvision 0.27.1 from PyPI, so the CUDA 12.4 base tag does not establish the
installed Torch CUDA/SM support. The candidate remains blocked; its metadata now
requests actual installed-runtime and hardware evidence instead of claiming
that no Torch is baked or inferring compatibility solely from the base tag.

Exact operational identities, service coordinates, generated media, test logs,
and trajectory delivery receipts are retained in access-controlled evidence.
Only sanitized measurements and artifact hashes belong in this report.
