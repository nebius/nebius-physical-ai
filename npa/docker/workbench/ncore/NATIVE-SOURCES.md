# Native sources and build provenance

The image builds PyAV 17.1.0, NumPy 1.26.4 / 2.5.3, and SciPy 1.15.2 / 1.18.1
from the exact official archives in `native-source-lock.json`. The converter
retains NumPy 1.x; the NPA/reader interpreter retains NumPy 2.x. Other runtime
dependency versions remain in the existing requirement locks.

PyAV links dynamically to unmodified official FFmpeg 8.1.1. The archive SHA256 is
`b6863adde98898f42602017462871b5f6333e65aec803fdd7a6308639c52edf3`.
The build verifies its detached release signature with the hash-pinned official
FFmpeg release key (signer `FCF986EA15E6E293A5644F10B4322F04D67658D8`). It does not
use the pyav-ffmpeg recipe or its license-reclassification patch. FFmpeg uses
`--disable-gpl --disable-nonfree --disable-version3 --disable-autodetect` and
explicit Debian zlib support. Native FFmpeg JPEG/PNG decoders remain enabled.
No x264/x265 libraries are linked. This FFmpeg build is LGPL-2.1-or-later; PyAV's
own BSD license and every original source notice remain intact. The configuration,
dependencies, actual decoded images, and original sources are checked together;
the exported license string alone is not the basis for this conclusion.

NumPy and SciPy link to Debian's reference BLAS/LAPACK and GCC runtimes. SciPy
selects Debian's `blas-netlib.pc`; the real CBLAS symbols are exported by
Debian's `libblas.so`. There is
no auditwheel repair or bundled `numpy.libs`, `scipy.libs`, or `av.libs` tree.
`debian-native-lock.json` binds nine exact binary packages to SHA256s extracted
from the actual `.deb` files, whose checksums came from authenticated Debian
snapshot Packages indexes. The image checks installed package versions and
shared-library bytes against that lock and refuses unmapped dependencies.
The corresponding GCC 12, glibc, LAPACK (including BLAS), and zlib source archives,
Debian patches and source-control files are included. GCC's runtime exception
does not remove its runtimes' notices or source duties. Debian copyright files
and common license texts stay under `/usr/share/doc` and `/usr/share/common-licenses`.

The final image carries `/opt/ncore/native-sources/`, containing the exact source
archives, native build-tool sdists, source and binary locks, Dockerfile, actual
build scripts and commands, FFmpeg configuration/logs, compiler package inventory,
built-wheel hashes, and both interpreters' library/codec verification receipts.
The complete original archives retain all upstream notices, including files not
compiled into this configuration. The shared FFmpeg libraries, headers and
pkg-config files remain under `/opt/ncore/ffmpeg`; recipients can rebuild PyAV or
replace compatible shared libraries. No additional restriction on modification,
replacement, or reverse engineering for debugging those modifications is added.

To assemble the identical archive inputs locally using the repository interpreter:

```sh
npa/.venv/bin/python npa/docker/workbench/ncore/assemble_native_sources.py \
  --lock npa/docker/workbench/ncore/native-source-lock.json \
  --output /path/to/native-annex --cache /path/to/read-only-archive-cache
```

The cache is optional and never written. Every fetched or cached artifact is
SHA256 checked. A Docker BuildKit named context `native-source-cache` can point
at that annex; without the override the empty context causes the same pinned
public archives to be fetched. The Dockerfile is the complete native build
recipe; its build-only Python environments use their own hash locks with build
isolation disabled and Meson subproject downloads disabled. No compiler/build environment is copied into the final image.

Build-tool lock maintenance uses the matching `native-*-build.in` with
`uv pip compile --excludes native-build-excludes.in --python-version 3.12
--generate-hashes --no-header --no-emit-index-url`. The explicit NumPy exclusion
keeps Pythran's dependency from introducing an opaque NumPy wheel into the build
environment; `build_native.py` installs our source-built NumPy before SciPy.
Update the exact build-tool source entries in `native-source-lock.json` together
with any intentional build-tool lock change.

This remains a **native-component annex**. The companion
`/opt/ncore/whole-sources/` annex reuses its archive identities and supplies the
remaining locked source distribution, including superseded Debian/base bytes,
Python copyleft, CPython and complete image recipes. See
[whole-image source delivery](WHOLE-IMAGE-SOURCES.md) for license boundaries,
recipient extraction/rebuilding and the offline layer verifier. Both directories
accompany the image; private downloads alone are not recipient delivery.
The publisher must still verify its exact integrated final digest and every
ancestor layer and complete all publication and workload gates. Neither annex
attests publication or GPU acceptance, and an older artifact's audit does not
clear a newly built image.
