# Per-image PAIDF Rerun evidence

The PAIDF integration has exactly seven operator-private container recipes: one
DIG image, two IAA-selected images, and five EVG-selected images, with the shared
attribute-search image counted once. The packaging contract and evidence-helper
test independently assert this same seven-image restricted inventory. All seven
have separate Rerun recordings derived from the completed native IAA, EVG, and
DIG workloads.

The protected Agent artifact API independently returned these genuine run IDs
while discovering the seven recording keys. The DIG entry comes from the
post-publication protected response, not from its workflow manifest:

| Workflow | Run ID | Replacement recordings discovered |
| --- | --- | --- |
| Image Attribute Augmentation | `paidf-iaa-efeb0a24d86a` | 1 |
| Event Video Generation | `paidf-evg-e5938e6d2c96` | 5 |
| Defect Image Generation | `paidf-dig-15395d41fe18` | `1` |

The read-only IAA/EVG query followed every selected-source page, returned HTTP
200, and matched all six replacement keys on 2026-09-06. A later independent
DIG query returned the row above after durable publication and readback. The API
reuses the original workflow run IDs while each recording carries its
image-specific Rerun identity. Live browser switching remains unverified.

Each replacement recording and its companion JSON manifest are stored under the
original workflow's `reports/image-evidence/` directory. Object names contain
the image name and full RRD SHA-256. Their manifests identify the superseded RRD
hashes. The earlier recordings remain immutable, and the original workflow spec,
successful ledger, reports, and outputs retain their original bytes.

## Recording inventory

| Image | RRD SHA-256 | Bytes |
| --- | --- | ---: |
| `npa-paidf-image-edit-sky` | `b1b8184a00b8aaaa530098f0f644555320ec27e7081d40f098fff08f80ac6020` | 594248 |
| `npa-paidf-event-video-sky` | `53ba15efd84612c90e02803ba8a5c67509bf8826962ad5ff934cca993bad1f12` | 50124804 |
| `npa-paidf-detection-sky` | `741d8186eeecceb03b45732b9599df5196b8133cc258554ef1fdf14e9d5de2e8` | 50122657 |
| `npa-paidf-captioning-sky` | `042e6f847377319f43232aab5a5987e08aca3d31e9c085df70b04bb409c886c5` | 50122379 |
| `npa-paidf-visual-qa-sky` | `5f289c807a185e800131a2097b9b84baa7ca09bba1dd34be23510cec764a753a` | 50159590 |
| `npa-paidf-attribute-search-sky` | `c319347e21516cdd1c32db7536834fbd5d3a4cbe85f182b5ed867fbf0e576d1f` | 50101234 |
| `npa-paidf-anomalygen-sky` | `fb099f7b670fede587398c1d5374db7cb6a231bad0fc839432c9da49b6870074` | `221486` |

The DIG object's content-addressed filename is
`npa-paidf-anomalygen-sky-fb099f7b670fede587398c1d5374db7cb6a231bad0fc839432c9da49b6870074.rrd`; its companion manifest SHA-256
is `15c1367ce81f0352185035445444da6c6cc5c0fb0cc84e47628ecb1bd153bdcf` and its size is `6792` bytes.
The recording passed CLI and SDK reopening, durable storage readback matched,
and the protected artifact index independently returned the genuine workflow
run ID. A filename alone is not evidence.

This pair supersedes the immutable `dd668be7ce2d1bd60a18b7c1b0e1a491253256f237959b67efcff9a0763e1075`
RRD and `c9fb26016ff5a64aef735176c049527fdbdae458bd5c0c21eb65bef3127ce338`
manifest. Those objects remain retained as historical evidence but are excluded
from the selected inventory because their normalized upstream repository
attribution did not match the pinned source exactly.

Each recording retains the published runnable manifest digest pinned by its
workflow, image-build revision, runtime source revisions, workflow run ID,
stage outcomes and durations, validation checks, and source artifact hashes.
Image-build and runtime revisions are explicitly separate. EVG retains its
first seven successful stages from
`27700b94612d7f8297a4f879c1c3f550bff467f1` and the remaining five from the
verified durable resume at `f466e119f1249de81e2e752ff3092098c5839964`.
IAA executed from `39120bc9b567d6400d4fe955988132ba1f6ce682`.

The original publication receipts identify these as runnable child manifest
digests and record the OCI index and config digests separately. Five images have
runtime image-ID checks recorded by the GPU observer. EVG CPU attribute search
is supported by its successful stage report and exact immutable image reference;
a separate image-ID observation for that run remains unverified.

IAA includes both the actual 896×1184 generated JPEG and the distinct 768×1024
postprocessed JPEG. Each EVG recording includes all 93 decoded video frames.
Labeling recordings use the real EVG scene as visual context and retain their
own service outcomes, measurements, and artifact lineage. GPU observations are
matched to the image and source attempt; the final validation stage is explicitly
identified as workflow-wide evidence from a separate CPU runtime.

The first six replacement recordings passed `rerun rrd verify` and
`rerun rrd print -vv`. Conversion independently reopens the closed recording and
compares provenance, complete scalar/event sequences, every source-frame index,
and every RGB pixel hash. Publication read back all 251,224,912 RRD bytes and
122,732 companion manifest bytes across twelve immutable objects. Before any
replacement write, all six original remote companion manifests matched their
historically verified hashes.

A separate audit merged the six actual replacement files into a private
validation archive and independently decoded six distinct recording stores:
467 embedded frames, 20 stage event rows, and 156 scalar rows. The frame count
includes five copies of the 93-frame EVG context and the two IAA images. Every
recording preserved its original provenance, media, event values and indices,
and scalar values and indices. The merged archive serves only as validation
evidence and is not an additional published recording. This confirms recording
isolation in the archive; live browser switching remains unverified.

Hashes for all 49 referenced image bootstrap, built-byte, security and SBOM
files, plus the six separate image-publication receipts, also matched. These
original image/workload acceptance facts remain unchanged by the RRD identity
correction. Independent Agent API discovery returned all six replacement keys.
Live browser switching remains a separate, unverified presentation check.

The seventh recording was built only from the terminal DIG artifact mirror. It
passed `rerun rrd verify`, `rerun rrd print -vv`, independent SDK reopening,
complete metric/event/media comparison, and content-addressed durable readback.
The final seven-file audit reopened seven distinct recording stores and matched
each recording to its own image digest, source revision, workflow run ID, stage
evidence, and real visual output.
The private merged validation archive is
`paidf-seven-image-evidence-9ebbbd0f7f3f94baa2b032298ef65e9418ab67cc629718073eb6c89ebe852ff3.rrd`
(251,441,994 bytes). Its content-audit JSON is 45,312 bytes with SHA-256
`2d45ee89cb5e732071bf22fa2ac5cf581c2e7a8e7c5967e9bb4f5920a2b87d6b`.
The audit decoded seven stores, 473 media frames, 24 stage events, and 174
scalar rows without semantic crossover; the archive is validation evidence,
not an eighth published recording.

The accepted AnomalyGen child digest is
`sha256:5aff3f4b40a4340ece2594c567ce8e5683a82ddc295c39d80588e228a13a28cf`,
built from `743d87df3a19fc0571d95c1d98b2bc53a2b438e9` and pinned Apache-2.0
upstream source `dbaf7d7d9003f048230f9026da5969e9e5931785`. Its distinct OCI
index and config identities, exact-layer/rootfs inventory, nonempty security
inventory, SPDX SBOM, bootstrap, offline runtime checks, and B200 component
probes are recorded in the [container catalog](../container-image-catalog.md).
The B200 diagnostic passed CUDA/FlashAttention/Triton and four native attention
cases but loaded no model weights and ran no full model forward. It does not
substitute for the independently completed full workflow.

Native DIG run `paidf-dig-15395d41fe18` completed all four logical states. Its
resume lineage retained the valid attempt-1 `record-upstream` result from
`743d87df3a19fc0571d95c1d98b2bc53a2b438e9` (185 seconds), then attempt 8 at
`ff198c6c289ee05ec5953e5968dc2c626ab27eee` completed
`prepare-base-checkpoints` (1,002 seconds), `finetune` (4,710 seconds), and
`anomaly-infer` (1,051 seconds). The unchanged recipe ran all 15,000 iterations
with validation and checkpoint saves every 1,000 iterations, early stopping
disabled, and no early-stop artifact. A separate source-bound native validation
replay returned `passed` after reopening the terminal artifacts. The evaluator
selected checkpoint 13,000 with SHA-256
`f54b720e786229395717f8a6bde9c8a2864735cddcada6f45d147cd5add04f65`,
at score `0.4711651623249054`; the terminal checkpoint score was
`0.4695567297935487`. The retained training trace has 1,500 loss samples.
Fifty-six run-bound B200 observations reached 100% utilization, 40,688 MiB
resident memory, and 945.84 W; these are observed maxima, not capacity claims.

All 30 request indices were accounted: 24 generated RGB images and six
text-guardrail blocks. The generated media total 605,744 bytes and retain 24
native COCO annotations; all 24 image/mask pairs reopened. The media manifest
SHA-256 is
`c8eb3b5da00dbfabf2ff3e4b3fd3f96b7d73a827e26227dbdf847127059fb115`,
and the labels SHA-256 is
`b330be50a83692d05a5bdc692e4d84ff2e7645991d891b1eabe07d78e33d06e8`.
The 1,261-byte timing summary has SHA-256
`9c6738a56d9b2fc7d593b0d9b83e50497b1c601ba7fddbeba35bc046457ed5c3`;
the generation-result and blocked-request CSV hashes are
`d8392b7f946ea13078b25a6fc0687cfffc6afcf00557cd4d12bbb4014c20ed7d`
and `64013a62fc72e1ac992b29ae8d5ca0f61265767005810af2bd9560e2c7a1b404`.
The Qwen text guardrail was enabled and enforcing. The upstream image preset
performs face blurring but has no image-content classifier, so image enforcement
is truthfully recorded as false.

The anomaly-output subtree contains 293 objects and 3,541,820 bytes. The stable
whole-run inventory contains 5,007 objects and 157,040,084,602 bytes, including
runtime-fetched checkpoints; the owner-only sparse validation mirror rehashed
540 selected objects (313,484,469 bytes) rather than republishing that payload.

## Interpretation and attribution

The IAA result is a three-view illustration with the requested clothing; it
does not establish photographic fidelity or single-person composition. EVG
anomaly VQA returned 21 of 21 answers, while person VQA returned 29 of 33.
The recording retains the missing-answer qualification and the limits of visual
review for physical motion and face blurring. IAA GPU observations measured
resident memory but missed active kernels. Zero utilization samples are not
proof that generation performed no GPU work.

The image checks passed the repository's existing fixed-CRITICAL security
policy. The IAA and EVG generation inventories each retain six unfixed CRITICAL
findings; a passing policy result does not mean zero vulnerabilities. See the
[container catalog](../container-image-catalog.md) for accepted digests and the
full inventory interpretation. The DIG inventory separately retains 5 CRITICAL,
186 HIGH, 2,173 MEDIUM, 268 LOW, and 3 UNKNOWN findings. Its one JWT-shaped
scanner match and six embedded PEM blocks remain retained with public-source
classifications; no credential path or populated model-cache path was accepted
as payload. All seven recipes retain their restricted, operator-private
classification. Private image access and successful workload execution do not
grant redistribution rights.

NVIDIA receives attribution in every recording's static provenance: upstream
repository, immutable revision, license, and NPA adaptation. The exact execution
and licensing boundaries remain in
[NOTICE-NVIDIA-PAIDF](../../../skills/NOTICE-NVIDIA-PAIDF). Permitted fixture
sources are documented in
[NOTICE-PAIDF-STARTER-MEDIA](../../../skills/NOTICE-PAIDF-STARTER-MEDIA).
Recordings contain generated output pixels and sanitized facts. Third-party
weights, raw customer inputs, credentials, private endpoints, and concrete
infrastructure identifiers are excluded.

## Converter contract

`npa.workflows.paidf_evidence_viz.build_image_evidence_rrd` consumes a normalized
`npa.paidf.image-evidence.v1` document plus local media paths. The caller must
first establish the original workload, image/source bindings, artifact hashes,
and permission to include the media. The converter validates that strict
contract and the exact media bytes; it does not establish workload truth from
an arbitrary supplied manifest.

Each Rerun recording ID combines the genuine workflow run ID, the literal
`:image-evidence:`, and the full SHA-256 of canonical normalized evidence. The
canonical form uses UTF-8 JSON with sorted keys and compact separators. The
original workflow run ID remains unchanged in provenance. This gives distinct
image evidence its own recording identity when files from one run are loaded
together; a corrected evidence document receives a distinct identity.

The `stage_index` timeline preserves supplied evidence order. The separate
`source_frame` timeline preserves each file's decoded frame indices; it does not
assert synchronization between different media files. Every frame is encoded
losslessly without source-file metadata. The output uses mode 0600, is created
atomically, and cannot overwrite an existing recording. Publishing and protected
Agent discovery verification are separate steps after conversion succeeds.

Validate converter changes with:

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_paidf_evidence_viz.py -q
```
