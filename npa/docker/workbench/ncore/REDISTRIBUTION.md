# NCore COLMAP ingestion image

This CPU ingestion image remains **quarantined** pending accepted image scans,
real capture conversion, and the downstream NRE RTX consumer. These files do not
assert a published or validated release. NRE reconstruction and rendering remain
separate proprietary operations.

## Narrow dependency boundary

The image carries CPython/Debian bootstrap tools, the pinned official NCore
converter/V4 reader Python source, trueprice's COLMAP reader, and 19 NPA source
files (including two compatibility aliases). It does not install the NPA Python
distribution. `npa.workflows.ncore` registers the existing
`npa.cli.nurec.convert_colmap_cmd` callback; conversion, path validation, JSON
output, rig derivation, full sequence validation, and immutable S3 publication
continue to use the existing shared implementation. Public CLI and SDK source
outside the image is unchanged.

The explicit `COPY` list is the NPA import closure: package initializers,
`_sdk`, `errors`, `lifecycle_intent`, `clients.storage`, `cli.path_contract`, the
NuRec CLI callback and workbench modules, and the two narrow workflow modules.
The image aliases the adapter as `npa.cli.entry` and `npa.__main__`, supporting
both SkyPilot's current shim and `python -m npa workbench nurec convert-colmap`.
Only `convert-colmap` is registered in this image. NPA's full SDK, unrelated CLI
registry, workflow catalog, examples and visualization modules are not copied.

`runtime-lock.json` pins 43 exact CPython 3.12 Linux x86_64 artifacts by public
URL, filename, version and SHA256. The bootstrap embeds the lock's SHA256.
The human-readable `runtime-requirements.in` describes direct dependencies;
`build-requirements.lock` and `runtime-requirements.lock` mirror the exact
artifact set. Locks are runtime inputs, not build-time package installations.

| Required path | Dependency reason |
| --- | --- |
| Official converter and trueprice reader | NumPy 1.26.4, SciPy 1.15.2, Pillow, Click, tqdm; trueprice's original import roots are preserved |
| NCore V4 stores/types/serialization | dataclasses-json, cbor2, Zarr 2, numcodecs, universal-pathlib, fsspec, typing-extensions and their serialization/path helpers |
| Official converter CLI | `tools.debug` imports debugpy; it remains a real locked dependency even when debugging is unused |
| Existing NPA request/CLI | Pydantic and Typer plus their small declared closures, including Rich for Typer output |
| S3 handoff | boto3/botocore, s3transfer, jmespath, python-dateutil, six, urllib3 |
| Runtime installation | Locked pip, setuptools, wheel and packaging; asciitree 0.3.3 has only a pure-Python sdist, built without isolation from that exact archive with those exact tools |

The import boundary follows the [pinned NVIDIA converter](https://github.com/NVIDIA/ncore/blob/59c698d206da92b406a4f72619fce3b3a2c64bfd/tools/data_converter/colmap/converter.py)
and [V4 reader exports](https://github.com/NVIDIA/ncore/blob/59c698d206da92b406a4f72619fce3b3a2c64bfd/ncore/data/v4/__init__.py).

The converter and V4 reader share one NumPy 1.26.4 environment. The old NumPy 2
environment existed to satisfy Rerun. Neither Rerun, LanceDB, PyArrow, PyAV nor
FFmpeg is imported by this converter/reader closure; none is now fetched or
baked. Torch's sensor-simulation path is outside the packaged capability.
The custom NumPy/SciPy/PyAV/FFmpeg builders, compiler closure, duplicate NumPy
versions and full NPA dependency/source annex have been removed.

## Runtime cache

No application wheel, application sdist, installed application dependency or
populated runtime cache enters an image layer. On first invocation the stdlib
bootstrap downloads the exact public PyPI artifacts directly to operator storage.
It disables ambient pip configuration, indexes, user site and Python paths;
installs offline with `--require-hashes --no-deps --no-build-isolation`; runs
`pip check`; and checks the actual converter, V4 reader and NPA imports.
Native wheels retain their bundled libraries and license records in the cache.
There is no replacement licensing claim for those third-party bundled bytes.

Cache precedence is `NPA_NCORE_RUNTIME_CACHE`, then
`NPA_MODEL_CACHE_DIR/ncore/runtime`, then
`${XDG_CACHE_HOME:-$HOME/.cache}/npa/ncore/runtime`. The selected POSIX directory
must belong to the worker with mode `0700`. Directory traversal rejects symlinks,
foreign owners, and writable ancestors except root-owned sticky directories.
Identity locks must be singly linked regular files owned by the worker with
mode `0600`; ready generations must retain private ownership and a regular,
non-symlinked receipt. Existing permissions are never changed automatically.
This is ephemeral
unless the operator explicitly mounts durable storage; no PVC is provisioned.
A POSIX filesystem that supports `flock` and atomic same-directory rename is
required. Do not use an object-store mount without those semantics.

Identity includes the entire artifact lock, interpreter version, exact NPA
source revision, NCore source inventory, bootstrap and source import roots.
Independent processes lock the same identity, prepare in a unique temporary
directory, verify all downloaded hashes and runtime imports, then atomically
publish a completed generation. Reuse checks the complete file/symlink inventory;
corruption fails closed. Interrupted partial directories are never reused.
Console-script shebangs are relocated before publication. Credentials and
upstream diagnostics are not included in ready receipts or public errors.

A cache contains executable third-party code. Do not publish it as an image or
redistribute it as a source annex. Persistent-cache access must remain within the
operator's trust boundary. New lock or source identities get separate generations.
Runtime fetch does not itself establish functional acceptance or substitute for
reviewing the fetched artifacts.

## Source and notices actually delivered

| Component | Exact source and retained notice |
| --- | --- |
| NCore | NVIDIA/ncore `59c698d206da92b406a4f72619fce3b3a2c64bfd`; Apache-2.0, upstream LICENSE and source headers under `/opt/ncore/src/ncore` |
| COLMAP model reader | trueprice/pycolmap `fe7a7c45df803b6c391777e349f0d8d65d39d777`; MIT, upstream `LICENSE.txt` under `/opt/ncore/src/pycolmap` |
| NPA adapter and shared modules | Exact committed `SOURCE_SHA`, readable `/opt/npa/src/npa`, root Apache license and retained adaptation notices under `/usr/share/doc/npa-ncore/notices` |
| Retained base binaries | Explicit file selection from digest-pinned Python 3.12.12 slim Bookworm and immutable Debian snapshot, copied into one `FROM scratch` filesystem layer. Original notices for permissive files and actual corresponding source/build scripts for covered binaries accompany the image; superseded ancestry is absent. See `base-source-lock.json` and `/opt/ncore/base-sources`. |

The image retains `NOTICE-NVIDIA-NCORE-COLMAP`, `NPA-LICENSE` and
`NOTICE-NVIDIA-SKILLS`; the latter covers the adapted NuRec modules in the NPA
import closure. `NOTICE-NVIDIA-COSMOS-OSS` applies to the separate Cosmos
Evaluator/Curator implementations, which this image does not contain, so it is
not copied into the image. Upstream NCore/pycolmap licenses, source headers and
required base-component notices remain included.

`source-lock.json` binds the upstream archive hashes. The stager retains source
and build/licensing metadata, applies NVIDIA's original
`deps/pycolmap/fix-python3-map.patch`, and marks two narrowly checked NPA changes:
the unsigned maximum point sentinel remains valid under NumPy 1 and 2; and each
downsampled camera uses its own calibration. Patches fail on source-context drift.
Every retained post-patch source file is hashed in `source-inventory.json`.
The actual converter is NVIDIA's `//tools/data_converter/colmap:convert` Python
module, not a replacement conversion implementation or PyPI's unrelated pycolmap.

[BASE-SOURCES.md](BASE-SOURCES.md) documents retained binary source delivery.
The retained Debian and CPython files are bound to actual byte hashes and
original copyright notices. Permissive components have a binary-specific
`delivery: notice` decision; their unnecessary full source archives are absent.
Inherited pip/ensurepip and optional Python build/test/GUI payloads are absent.
Real APT/dpkg/curl and their checked helper/library closure are required by the
pinned SkyPilot 0.12.2 startup and are baked bootstrap utilities. The published
dpkg status is empty because selected loose package files are not configured
Debian installations. The lock inventories those bytes; runtime APT writes real
package status, lists and conffile metadata. No installed-package claims are
fabricated to satisfy SkyPilot's readiness predicates. APT recommendations and
translation indexes are disabled; required dependencies and repository signature
checks remain enabled. See [BASE-SOURCES.md](BASE-SOURCES.md) for the exact
bootstrap boundary and disposable-root upstream-command probe.
The remaining covered binaries receive real preferred source and build/install
scripts. GCC runtime exceptions do not waive standalone library source delivery.
The three reviewed source transformations remove only identified optional
tests/fixtures, retain required build inputs, and record input/output hashes and
library buildability evidence. They do not create scanner exceptions or source
offers. [BASE-SOURCES.md](BASE-SOURCES.md) gives the obligation reasoning and
reproducible preparation/verification commands.

No model weights, capture datasets, NRE/Kit/Isaac payloads, CUDA, Torch, credentials,
`.git`, or acceptance records are added. Any remaining source content is included in the unchanged recursive scans.
Neither an upstream filename nor a source-only classification approves bytes
that a scanner rejects.

## Build and acceptance contract

The parent builds only after integration into a committed SHA:

```sh
bash npa/docker/workbench/ncore/build.sh --source-sha "$SOURCE_SHA" \
  --image "ghcr.io/nebius/nebius-physical-ai/npa-ncore:dev-$SOURCE_SHA"
```

The command exports committed source and builds locally without pushing. Direct
Docker builds require `SOURCE_SHA` or publisher-compatible `NPA_SOURCE_SHA`.
The image records the revision in labels, environment and
`/usr/share/doc/npa-ncore/npa-source-sha`.

`verify-packaging.py --image-only` is an offline build check of the unpopulated
image boundary, source hashes and bootstrap commands. The normal verifier used
by existing setup and golden commands fetches the locked runtime and validates
real imports and CLI schemas. It reports `import-and-cli-schema-only`; it is not
capture validation. The existing `/opt/venv/bin/python` remains the stdlib
bootstrap interpreter. Use the returned runtime interpreter for direct V4 Python
imports; `colmap-convert`, `npa` and `python -m npa` handle this automatically.

The final user remains `ubuntu` (uid 1000). The build repeats the fresh SSH key,
sudo, service restart/stop, writable home/tmp and rsync checks inside the
assembled root, then removes generated host keys and temporary test devices
before the single scratch copy and `skypilot-0.12.2-v1` attestation. The entrypoint creates per-container
keys and executes the orchestrator's argv unchanged. No SSH service starts by
default. The cache also works when SkyPilot replaces the image entrypoint.

The bootstrap test `RUN` that invokes `su ubuntu` has an adjacent
`trivy:ignore:AVD-DS-0010` annotation. It tests the non-root sudo contract
recorded in `packaging-contract.yaml`; removing sudo would stop proving that
contract. The annotation uses Trivy's supported per-instruction scope and does
not exempt installation/assembly instructions or other Dockerfiles. Generated
host keys are removed in the same test instruction.

Before any publication the parent must inventory **every actual final/ancestor
layer**, verify all retained binary/source correspondence and licenses, and run
the existing recursive payload, secrets, vulnerability, OCI/config/history,
SBOM/provenance and bootstrap gates on the exact integrated artifact. The
selected base hash map comes from actual pinned upstream package/base bytes;
it is not evidence for a newly built image;
any mismatch must be resolved against real bytes, never skipped. Runtime
artifacts need their own recorded hashes, license/security review and successful
cold/warm execution. Import checks cannot replace complete real-capture
conversion, independent image/calibration/pose/point decoding, immutable S3
handoff and the existing NRE RTX consumer. Quarantine, immutable image override
and anonymous digest pullability requirements remain unchanged.
