# Per-image PAIDF Rerun evidence

The PAIDF integration adds seven operator-private container recipes. Six have
separate Rerun recordings derived from completed native IAA and EVG workloads.
The AnomalyGen recording still requires completed DIG image and native workload
acceptance. This inventory does not claim that DIG has passed.

The protected Agent artifact API independently returned these genuine run IDs
while discovering all six replacement recording keys:

| Workflow | Run ID | Replacement recordings discovered |
| --- | --- | --- |
| Image Attribute Augmentation | `paidf-iaa-efeb0a24d86a` | 1 |
| Event Video Generation | `paidf-evg-e5938e6d2c96` | 5 |

The read-only query followed every selected-source page, returned HTTP 200, and
matched all six replacement keys on 2026-09-06. The API reuses the original
workflow run IDs while each recording carries its corrected, image-specific
Rerun identity. Live browser switching remains unverified.

Each replacement recording and its companion JSON manifest are stored under the
original workflow's `reports/image-evidence/` directory. Object names contain
the image name and full RRD SHA-256. Their manifests identify the superseded RRD
hashes. The earlier recordings remain immutable, and the original workflow spec,
successful ledger, reports, and outputs retain their original bytes.

## Verified recordings

| Image | RRD SHA-256 | Bytes |
| --- | --- | ---: |
| `npa-paidf-image-edit-sky` | `b1b8184a00b8aaaa530098f0f644555320ec27e7081d40f098fff08f80ac6020` | 594248 |
| `npa-paidf-event-video-sky` | `53ba15efd84612c90e02803ba8a5c67509bf8826962ad5ff934cca993bad1f12` | 50124804 |
| `npa-paidf-detection-sky` | `741d8186eeecceb03b45732b9599df5196b8133cc258554ef1fdf14e9d5de2e8` | 50122657 |
| `npa-paidf-captioning-sky` | `042e6f847377319f43232aab5a5987e08aca3d31e9c085df70b04bb409c886c5` | 50122379 |
| `npa-paidf-visual-qa-sky` | `5f289c807a185e800131a2097b9b84baa7ca09bba1dd34be23510cec764a753a` | 50159590 |
| `npa-paidf-attribute-search-sky` | `c319347e21516cdd1c32db7536834fbd5d3a4cbe85f182b5ed867fbf0e576d1f` | 50101234 |

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

All six replacement recordings passed `rerun rrd verify` and
`rerun rrd print -vv`. Conversion independently reopens the closed recording and
compares provenance, complete scalar/event sequences, every source-frame index,
and every RGB pixel hash. Publication read back all 251,224,912 RRD bytes and
125,599 companion manifest bytes across twelve immutable objects. Before any
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
full inventory interpretation. All seven recipes retain their restricted,
operator-private classification.

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
