# Accompanying sources for the NCore CPU image

The image delivers source bytes with its binaries. `/opt/ncore/whole-sources/`
contains complete source archives, Debian source control and patch archives,
authenticated repository metadata, the immutable source lock, and packaging
recipes. `/opt/ncore/native-sources/` supplies the previously documented native
sources and actual native build records. The whole-image lock references those
same archives by hash; it does not duplicate them. Installed preferred-form
source lives in `/opt/npa/src/npa`, `/opt/ncore/src`, the Python standard library,
and the Python environments. Preserve these directories together.

This is the accompanying-source delivery route, not a promise to fulfill a
future postal offer. The source annex is readable by the final non-root user
and is delivered in the same image pull, without a second account, payment,
credential or external source service. Its public URLs document provenance and
allow a builder to retrieve the exact inputs; a URL alone is not fulfillment.
Recipients may copy, modify and redistribute each covered source under its
own retained license. No additional restriction on replacement, relinking or
reverse engineering for debugging modifications is imposed.

## Coverage and license boundaries

| Material | Delivery and conditions |
| --- | --- |
| Debian binaries in every layer | Exact `.dsc`, original archives and Debian patches for **111 source versions**, including older versions hidden by later updates. **159 binary package versions** map through signed Packages and Sources indexes; epochs and binary rebuild suffixes remain distinct. The source archives include Debian build/install rules and their complete notices. The GPL/LGPL obligations attach to the applicable components, not to unrelated aggregated NPA code. |
| FFmpeg, PyAV, NumPy and SciPy | The native annex supplies the actual source, build tools, commands, patches/configuration and library replacement materials described in `NATIVE-SOURCES.md`. GCC runtime exceptions are retained with their source; no compiler origin is inferred from a version string. |
| Paramiko and vendored autocommand | LGPL-covered preferred-form `.py` files remain installed and replaceable. Complete Paramiko and setuptools sdists accompany them, including vendored autocommand source and its LGPL-3.0 notice. The standalone autocommand source is also included. |
| certifi, pathspec, tqdm, pip's vendored certifi | Complete exact sdists plus installed source/CA bundle and notices accompany the MPL-covered files. The MPL applies to those files; tqdm's separate MIT-covered contributions retain their notices. Both inherited pip 25.0.1 and the environment pip 26.1.2 are covered. |
| debugpy / PyDev.Debugger | The debugpy 1.8.20 sdist includes its vendored debugger, Cython preferred source, generated C, C++ attach source, headers, setup scripts and platform compile scripts. EPL-1.0 debugger source, BSD-licensed WinAppDbg source and the complete `ThirdPartyNotices.txt` remain available. The native Linux wheel and its three ELF files are hash mapped to the actual PyPI artifact. Installed source differs from the sdist only in permitted line-ending normalization. EPL section 3's source distribution and source-location conditions apply; commercial distributors must also observe section 4's responsibilities. NPA does not adopt Microsoft's separate postal source offer. |
| Pillow 12.3.0 | Both environments retain the full wheel license, including bundled codec notices, and the complete Pillow sdist. The actual Linux wheel's FreeType, JPEG, TIFF, WebP, AVIF, XCB and other bundled notices apply. FreeType uses its offered FreeType License route with the acknowledgement below. The inherited XZ/liblzma notice describes separate GPL build tools and LGPL getopt code; that notice alone does not establish their presence in liblzma. The inspected wheel has no FriBidi or libimagequant library; RAQM and libimagequant are unavailable in the runtime feature probe. No unused-source notice is deleted or reclassified. |
| Other installed Python distributions | Exact sdists accompany the installed closure where PyPI supplies them, including permissively licensed native packages and their retained bundled notices. Lancedb 0.30.2 (Apache-2.0) and rerun-sdk 0.31.4 (MIT OR Apache-2.0) have no PyPI sdist; those grants require retained notices, not delivery of corresponding source. These two entries are explicitly `notice-only`, not falsely marked source delivered. The Rerun wheel omits license files: both original licenses from its exact upstream revision are therefore supplied as `notices/rerun-0.31.4-LICENSE-APACHE` and `notices/rerun-0.31.4-LICENSE-MIT` in the annex. Their actual wheel ELF bytes and bundled components still require the publisher's complete license review. |
| CPython 3.12.12 | Complete official source archive, bound to the SHA256 recorded in the digest-pinned base; the exact OCI-history build command is retained as `CPYTHON-BASE-RECIPE.txt`. The standard library license remains `/usr/local/lib/python3.12/LICENSE.txt`. PSF and bundled notices are preserved; Debian library dependencies have their own source packages. |
| NPA, NCore and trueprice COLMAP | Installed preferred-form source, build metadata, exact revision/source inventories, and the complete image recipes accompany these Apache/MIT components. NCore/trueprice originals and modification markers remain in `/opt/ncore/src`. NPA's own root Apache license is separately retained as `/usr/share/doc/npa-ncore/notices/NPA-LICENSE`; the NVIDIA NCore license is not presented as an NPA-specific notice. Retained third-party adaptation notices are in that same notices directory. |

The source route follows the accompanying-source provisions of
[GPLv2 section 3(a)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html),
[GPLv3 section 6](https://www.gnu.org/licenses/gpl-3.0.html), and the applicable
LGPL terms. The covered-source location is supplied here for
[MPL 2.0 section 3](https://www.mozilla.org/en-US/MPL/2.0/) and
[EPL 1.0 section 3](https://www.eclipse.org/legal/epl/epl-v10.html).
The original component notices and license choices, not this table, define the
rights and conditions. Debian copyright/common-license files, Python dist-info
licenses, source headers and full archive notices must remain with redistribution.
The copied repository adaptation notices are preserved verbatim and discuss
broader NPA capabilities. Their historical runtime-fetch descriptions do not
change the explicitly documented delivery of Apache-licensed NCore source here.

This software is based in part on the work of the Independent JPEG Group.
Portions of this software are copyright © The FreeType Project
(https://www.freetype.org). All rights reserved. These acknowledgements accompany
the included full FreeType and JPEG notices; they do not replace them.

## Extract without executing the image

Use the publisher's immutable image digest. `docker create` does not execute
the entrypoint. Keep all extracted directories together:

```sh
IMAGE=registry.example/npa-ncore@sha256:REPLACE_WITH_PUBLISHED_DIGEST
mkdir -p ncore-source
container=$(docker create "$IMAGE")
docker cp "$container:/opt/ncore/whole-sources" ncore-source/whole-sources
docker cp "$container:/opt/ncore/native-sources" ncore-source/native-sources
docker cp "$container:/opt/ncore/src" ncore-source/ncore-src
docker cp "$container:/opt/npa" ncore-source/npa
docker cp "$container:/usr/share/doc/npa-ncore" ncore-source/notices
docker cp "$container:/usr/share/keyrings/debian-archive-keyring.gpg" ncore-source/debian-archive-keyring.gpg
docker rm "$container"
docker image save "$IMAGE" -o ncore-source/image.tar
```

In a repository checkout, use its `npa/.venv/bin/python` to run the extracted
script. Python 3.11+ and `gpgv` are sufficient; verification uses no network:

```sh
npa/.venv/bin/python ncore-source/whole-sources/recipes/whole_image_sources.py \
  inventory --image-archive ncore-source/image.tar --output ncore-source/inventory.json
npa/.venv/bin/python ncore-source/whole-sources/recipes/whole_image_sources.py \
  verify --lock ncore-source/whole-sources/recipes/whole-source-lock.json \
  --annex ncore-source/whole-sources --native ncore-source/native-sources \
  --keyring ncore-source/debian-archive-keyring.gpg --inventory ncore-source/inventory.json
```

The verifier checks all archive hashes, signed metadata and exact source
mappings, the pinned base layer chain, every layer's dpkg/Python identities and
ELF hashes, and installed copyleft preferred-form sources. Native ELF hashes
come from the build's existing native-verifier receipts; the publisher must
independently rerun both native verifiers on its exact build. The inventory
retains deleted/overwritten entries and binds the saved image, config, compressed
layers and uncompressed diff IDs. It is not a secret/payload scanner and does
not replace recursive inspection of archives or other binary formats.

To reassemble the downloadable inputs, run the same script's `assemble`
command with `--lock`, `--annex`, and `--native`. Optional repeated `--cache`
arguments are read-only; `--offline` refuses any missing input. In Docker builds,
the optional named context `whole-source-cache` accepts this annex, just as
`native-source-cache` accepts the native annex. Defaults fetch the identical
hash-pinned public bytes. Neither cache changes the lock or runtime pins.

## Rebuild and replace

For a Debian component, find its `debian:NAME@VERSION` entry, collect **every**
referenced artifact into one writable directory (some originals are reused from
the native annex), and run `dpkg-source -x NAME_VERSION.dsc`. The resulting
`debian/control`, `debian/rules`, patches and upstream build scripts describe
the build. Install the declared build dependencies from the same locked
snapshot in an isolated Debian builder, then use `dpkg-buildpackage -us -uc`.
Binary rebuild suffixes such as `+b1` are not source revisions. The binary lock
records both identities. Toolchains must be provisioned for rebuilding; the
runtime image intentionally does not carry a compiler or an offline apt mirror.

For Python packages, extract the corresponding sdist into a writable directory
and follow its `pyproject.toml`/`setup.py`. The image recipes retain the exact
environment and build locks. For PyDev's Cython extensions use the included
`setup_pydevd_cython.py`; the attach library's source tree includes
`pydevd_attach_to_process/linux_and_mac/compile_linux.sh`. Installed Python
source is already editable in a derived image or replacement environment.
The original wheel's build-platform/compiler is not claimed reproducible
bit for bit merely because its source version matches.

For FFmpeg/PyAV and both NumPy/SciPy environments, follow `NATIVE-SOURCES.md`
and the actual `build_ffmpeg.py`, `build_native.py`, and recorded commands.
Keep FFmpeg dynamically replaceable under `/opt/ncore/ffmpeg`; rebuild PyAV
against a changed ABI when necessary. For CPython, extract its archive and
follow the retained base build command's configure/compiler/linker flags.
For NPA, `/opt/npa` includes `pyproject.toml` and the installed build source;
build it with the retained hatchling/build requirements. NCore is installed as
preferred-form Python source; its stager and exact original/modified file hashes
are included. No capture data is needed to rebuild source packages.

These instructions support modification/rebuilding. They do not claim a
bit-identical rebuild of every upstream binary or an offline mirror of all
toolchain dependencies. The publisher's exact committed SHA is still required
to reproduce the integrated image through `build.sh`.

## Original source fixtures and acceptance limits

Complete source archives retain upstream software tests, license texts, public
test keys/certificates and small format fixtures. Public test-key bytes are
software fixtures only when their exact source archive/member/hash and test
use establish that fact. A filename or a public URL alone is insufficient.
No runtime host keys, operator credentials, customer captures, model weights,
NVIDIA runtime capture archives, proprietary NRE/Kit/Isaac payload or acceptance
record is added by this source annex. Scanner gates are unchanged. If a gate
cannot distinguish a proven upstream test fixture under its existing policy,
publication remains blocked; deleting required source or suppressing its notice
is not a remedy.

The lock was grounded against an **unpublished, unaccepted local experiment**.
Source assembly and offline correspondence are separate from publication
acceptance. The parent publisher must integrate and commit, rebuild the exact
SHA, repeat this inventory and verification on **every final layer**, validate
notices/source and secret/payload/full-byte/security/SBOM/provenance/bootstrap
gates, establish recipient access and anonymous digest pullability, and complete
the real capture/NRE RTX workflow. No such parent gate is asserted complete here.
