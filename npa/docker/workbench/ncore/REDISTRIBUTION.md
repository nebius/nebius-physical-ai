# NCore COLMAP ingestion image

This is a public-eligible **CPU ingestion** package, quarantined pending accepted
image scans and real capture conversion. No public release availability or full
functional validation is asserted by these packaging files or the golden check.
NRE reconstruction and rendering remain separate proprietary downstream operations.

## Exact baked source

| Component | Source | License and retained notice |
| --- | --- | --- |
| NCore converter and V4 reader | NVIDIA/ncore `59c698d206da92b406a4f72619fce3b3a2c64bfd` | Apache-2.0; `/opt/ncore/src/ncore/LICENSE`, source SPDX headers, upstream NOTICE if present |
| COLMAP model reader | trueprice/pycolmap `fe7a7c45df803b6c391777e349f0d8d65d39d777` | MIT; `/opt/ncore/src/pycolmap/LICENSE.txt`, copyright True Price / UNC Chapel Hill |
| NPA | Exact committed `SOURCE_SHA` exported by `build.sh` | Apache-2.0; `/usr/share/doc/npa-ncore/LICENSE-APACHE-2.0` and package provenance |
| CPython / Debian base | Python 3.12.12 slim Bookworm, digest in Dockerfile | PSF-2.0 and Debian component licenses in `/usr/share/doc/*/copyright`; base distro sources remain available from Debian snapshot archives |

`source-lock.json` pins archive bytes. `stage_upstream.py` extracts source files
and licensing/build metadata only. It applies NVIDIA's unmodified
`deps/pycolmap/fix-python3-map.patch` from that hash-verified NCore archive.
It also marks and applies one NPA compatibility edit to trueprice's unsigned
invalid-point sentinel: `np.uint64(-1)` becomes `np.uint64(np.iinfo(np.uint64).max)`.
This preserves the exact uint64 maximum value without NumPy 2's overflow error.
It also marks one NPA correction in the official converter's downsample loop:
each downsampled camera uses the loop's current camera calibration instead of
the last registered image's camera. Downsampling remains enabled. The patch
requires exactly one match of the pinned source context and fails on source
drift. Original source/license pins remain unchanged.
Every retained file's post-patch SHA-256 is recorded in
`/opt/ncore/src/source-inventory.json`.
The converter is NVIDIA's `//tools/data_converter/colmap:convert` **py_binary**,
executed as its official Python module; NPA does not reimplement conversion.
Both import roots declared by upstream's `deps/pycolmap/pycolmap.BUILD` are
preserved. PyPI's modern `pycolmap` bindings are not installed.

The Bazel wheel target's broad `torch` dependency supports `ncore.sensors`.
The imported converter/V4 reader path needs no Torch. This image makes only
that narrow capability claim; tensor-based sensor simulation is not packaged.
The converter retains the upstream NumPy 1.26.4 pin. NPA's Rerun requires NumPy 2,
so `/opt/venv/bin/python` and `/opt/ncore/converter-venv/bin/python` have separate
locked dependencies. Both load the same official NCore source and patched
trueprice reader; NPA independently reads the source model before conversion.

## Baked dependency licenses

`converter-requirements.lock`, `npa-requirements.lock` and
`build-requirements.lock` enumerate every Python distribution and accepted
archive/wheel hashes. Wheel license and dist-info records are retained in both
environments, including bundled-library notices. The direct converter/reader
closure includes NumPy and SciPy (BSD-3-Clause, built against Debian BLAS/LAPACK),
Pillow (MIT-CMU and bundled codec notices), Click (BSD-3-Clause), tqdm
(MIT/MPL-2.0), debugpy (MIT, with vendored debugger notices), dataclasses-json
(MIT), Zarr (MIT), numcodecs (MIT, with bundled codec notices), cbor2 (MIT),
universal-pathlib (MIT), fsspec (BSD-3-Clause), typing-extensions (PSF-2.0),
asciitree (MIT), fasteners (Apache-2.0), and the locked serialization helpers.
NPA's full declared dependency closure is installed without GPU extras.
Its transitive licenses, Debian GPL/LGPL components and binary-bundled notices
must be checked in the generated SBOM and actual bytes before publication;
the top-level OCI license label describes NPA/NCore/trueprice, not every library.
Public distribution must retain all included notices and satisfy applicable
source-offer obligations for the exact distro and bundled binary components.

PyAV 17.1.0, both NumPy/SciPy pairs, and unmodified official FFmpeg 8.1.1
are source built. FFmpeg disables GPL/nonfree code and external autodetection;
its shared libraries remain replaceable. The readable
`/opt/ncore/native-sources/` annex accompanies these components with exact source
archives, original notices, recipes, locks, configuration and verification
receipts. See [native source provenance](NATIVE-SOURCES.md). This scoped annex
does not establish source completeness for all other Python/OS components or
superseded ancestor-layer bytes; whole-image source and recipient-delivery
verification remain mandatory before publication.

No model weights, runtime capture photographs or datasets,
NRE/Kit/Isaac payloads, CUDA, Torch, credentials, `.git`, or acceptance records
are downloaded or installed by the image build. Operator captures enter only
at runtime; their dataset terms and resulting output attribution remain separate.
The complete OSS source archives retain their upstream software test fixtures;
these are source-delivery material, not packaged NVIDIA capture datasets or
functional workload evidence.

## Integration and build contract

The parent builds the integrated commit after review:

```bash
SOURCE_SHA=<full-integrated-commit>
bash npa/docker/workbench/ncore/build.sh --source-sha "$SOURCE_SHA" \
  --image "ghcr.io/nebius/nebius-physical-ai/npa-ncore:dev-$SOURCE_SHA"
```

This command builds locally; it never pushes. It exports only committed NPA,
packaging and workflow sources and stages that commit's workflow package data.
`SOURCE_SHA` (or publisher-compatible `NPA_SOURCE_SHA`) is mandatory in direct
Docker builds. NPA is installed from that exported tree, never PyPI or a branch.
The final image records it in OCI labels, `NPA_SOURCE_SHA`, and
`/usr/share/doc/npa-ncore/npa-source-sha`. Retained locks plus source inventory
are input provenance; the parent must separately generate/verify OCI/SBOM
attestations when exporting and publishing the exact scanned artifact.

`/opt/ncore/bin/colmap-convert` forwards native argv unchanged:

```bash
/opt/ncore/bin/colmap-convert --root-dir /workspace/capture \
  --output-dir /workspace/ncore colmap-v4 --help
```

`workbench.nurec.convert_colmap` routes to `ncore`. Select its validated digest
explicitly with `--image-override workbench.nurec.convert_colmap=IMAGE@sha256:DIGEST`.
There is no default public release. The inventory's `-unbuilt` source pin is
never resolved into an automatic runtime image. Setup uses `/opt/venv/bin/python`
and does not install floating `nvidia-ncore` or overlay NPA source. The image
exposes `NPA_NCORE_CONVERTER` for the CLI adapter and the actual
`ncore.data.v4.SequenceLoaderV4` to the NPA interpreter.

Conversion destinations are immutable, single-writer prefixes. A provider-atomic
conditional put reserves `.npa-colmap-claim.json` before any sequence members are
uploaded; `sequence.json` is published last. Occupied prefixes are rejected.
Interrupted or ambiguous publication retains its permanent claim: retry with a
fresh output prefix, never by deleting a claim or taking over an earlier writer.
NuRec verifies the downloaded conversion inventory and claim before handoff.
Existing sequences without COLMAP provenance remain supported.

The final user is `ubuntu` (uid 1000). SkyPilot's Kubernetes bootstrap requires
the recorded sudo exemption; no SSH service starts by default. The build proves
fresh-key generation, sudo, SSH service restart/stop, writable home/tmp and
rsync under that uid, removes generated keys in the same layer, and only then
attests `skypilot-0.12.2-v1`. The entrypoint creates per-container host keys and
executes the orchestrator's argv unchanged.

Golden runs only source-hash, import and CLI-schema checks on CPU. Acceptance
requires the parent to convert a complete real capture, independently reopen
and decode all images/calibrations/poses/points with the actual reader, and run
the existing NRE RTX consumer. Before any publication, scan actual ordered
layers, OCI config/history and final filesystem for payloads, vulnerabilities,
secrets and licenses, validate revision/SBOM/bootstrap evidence, and prove
anonymous digest pullability. Imports cannot replace those gates.
