# Selected base filesystem, notices and corresponding source

The published stage is `FROM scratch`. It contains one copy of an explicitly
assembled filesystem, with one retained version of each Debian package. The
Python base and the APT/source builders are build inputs, not published ancestry.
The bootstrap includes real APT, dpkg and curl, bash/sh, sudo, SSH, rsync,
service and their required helpers and libraries. Inherited pip/ensurepip,
Python headers/static libraries, optional GUI/DB/interactive extensions, build
trees, populated caches and test extensions are absent. The converter and its
hash-locked runtime dependency environment are unchanged.

The OCI exporter may retain one additional `WORKDIR /workspace` no-op layer.
The source verifier permits only its exact canonical 1,024 zero decoded bytes
and exact stored gzip bytes (or those same uncompressed zeros in a Docker save).
It rejects extra entries, altered gzip headers, padding and further layers.
Both physical layers remain in the graph, inventory counts and complete-byte
scan; this does not permit another filesystem payload or builder ancestry.

`base-source-lock.json` identifies the selected Debian files and CPython files by
path, content hash or exact symlink, together with the .deb hash, signed package
index and source identity. The list was derived from the actual pinned .deb
metadata and copyright files, and `readelf` dependencies, including the pinned
CPython ELF files. `assemble-root` copies only this selection, original notices,
and enumerated clean-builder configuration. It writes static localhost entries;
builder hostnames, resolver configuration, host keys and arbitrary `/etc` trees
are never copied. The lock is the SBOM/source inventory of **selected loose
files**, not a Debian installation database. The published dpkg database is
empty: no selected partial package is falsely marked `install ok installed`.
There are no invented package lists, conffile records or maintainer-script
receipts. Actual APT/dpkg execution during worker initialization installs the
required packages and writes their genuine status, lists and configuration.
The publication verifier rejects nonempty dpkg status and inherited installation
metadata. Do not copy the builder's status or `/var/lib/dpkg/info` into scratch.
SBOM and vulnerability coverage must use the explicit lock and file inventory;
an empty SBOM derived only from dpkg is not acceptable evidence.

The final copy preserves sudo's mode and dynamic library replacement. Uid 1000
can read the sources, notices and recipes. No term restricts modification,
replacement, relinking or debugging of the covered libraries.

## SkyPilot 0.12.2 bootstrap

The [pinned Kubernetes template](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/templates/kubernetes-ray.yml.j2)
queries `dpkg -l`, always updates APT indexes, and installs its missing packages.
Its runtime branch waits for real installed-package records for curl and patch;
workers also wait for netcat. The
[pinned provisioner](https://github.com/skypilot-org/skypilot/blob/v0.12.2/sky/provision/kubernetes/instance.py)
then unconditionally runs `apt install openssh-server rsync -y`. An SSH-only
smoke cannot validate these prerequisites. Both upstream files are hash-bound
under `bootstrap.upstream` in the lock.

| Baked selection | Bootstrap requirement |
| --- | --- |
| `apt`, `apt-get`, `apt-config`, `apt-key`, APT acquisition helpers, libapt-pkg/libapt-private, `gpgv` | Real package resolution, download, signed repository verification and installation |
| dpkg/query/deb/split/divert/statoverride/trigger/maintscript helpers, update-alternatives, dpkg architecture tables, `diff`, `cmp`, tar | Real package queries, unpacking, conffile handling and maintainer scripts |
| curl and its actual Debian library closure | HTTPS runtime download; linked protocol libraries remain necessary even for an HTTPS request |
| `seq`, `whoami`, `tty`, `stty`, `uniq`, `chgrp`, `md5sum`, Debian `which`, `getent`, `locale`, `ldconfig`, `mountpoint`, `getopt` | Upstream shell operations and actual Debian package configuration/pre-installation checks |
| Debian init helpers, one Perl interpreter and 35 core module/XS files | `deb-systemd-helper` during initial dpkg/APT configuration; selected from actual module and shared-object loading, without full perl-base modules or its duplicate versioned interpreter |
| Debian profile, bash defaults, writable ubuntu `.profile`/`.bashrc`, `/etc/profile.d`, `/var/run` | Provisioner environment propagation and login-shell setup |

The additional linked libraries are libapt-pkg/libapt-private, libcurl,
libbrotlidec/libbrotlicommon, libhogweed/libnettle, libidn2, libldap/liblber,
libnghttp2, libp11-kit, libpsl, librtmp, libsasl2, libseccomp, libtasn1,
libunistring, libmount and libblkid. Libgnutls and libssh2 are delivered at
runtime as described below. Existing selected libraries satisfy the remaining
ELF dependencies. Each is bound to actual pinned Debian bytes.

APT retains only the clean builder's pinned Debian snapshot sources and validity
configuration plus the public Debian signing keyring and CA trust. Empty writable
APT/dpkg state directories are created. `90npa-bootstrap` disables recommended
packages and translation indexes; required dependencies and signature checking
remain enabled. This avoids unnecessary recommended services and certificate
reconfiguration during SkyPilot's bounded upstream initialization. No NPA timeout
or retry policy is added. Package indexes, downloaded `.deb` files and installed
package state are runtime data and are never copied back into the public layer.
The template still installs its own platform tools, including compiler/patch/FUSE
packages, at runtime; they are not added to the baked NCore application closure.

The opt-in `test_ncore_skypilot_bootstrap.py` probe executes the original template
APT block and provisioner APT command as uid 1000 against a disposable copy of the
selected root. It checks real readiness predicates, package lists, `apt-get check`
and `dpkg --audit`. Supply `NPA_NCORE_BOOTSTRAP_ROOT` and
`NPA_NCORE_SKYPILOT_SOURCE` (a directory containing the two hash-matching upstream
files). The disposable root needs runtime device/DNS inputs and the standard
`policy-rc.d` exit-101 policy to prevent automatic daemon starts. The separate SSH
and full SkyPilot runtime checks remain necessary on the parent's exact image.
The probe explicitly sets the same native `BASH_ENV` hook inside the disposable
root because its `env -i` commands do not inherit OCI environment. Never run the
mutating APT probe against `/public-root` before its scratch copy.

## Native startup delivery

The public selection excludes exactly these original Debian libraries and SONAME
links. Their complete package identities, byte hashes, modes and member sizes
are in `native-bootstrap-lock.json`; the bootstrap pins that entire lock hash.

| Runtime package | Library | SONAME link |
| --- | --- | --- |
| `libgnutls30` `3.7.9-2+deb12u7` | `libgnutls.so.30.34.3` | `libgnutls.so.30` |
| `libssh2-1` `1.10.0-3+deb12u1` | `libssh2.so.1.0.1` | `libssh2.so.1` |

The libraries retain their original bytes at `/usr/lib/x86_64-linux-gnu`.
GnuTLS is required by APT's HTTPS transport, and libssh2 by curl's linked protocol
closure. Bash, APT, dpkg, sudo, SSH, rsync and all other selected base files remain
the original programs. No package wrapper or synthetic dpkg status substitutes
for them.

The final image alone sets `BASH_ENV=/opt/ncore/bin/native-bootstrap.sh`. The
pinned SkyPilot Kubernetes template overrides ENTRYPOINT with `/bin/bash -c`, so
this hook must establish the libraries before the actual upstream APT commands.
The normal entrypoint also sources the same hook before generating SSH keys.
Failure exits the shell before its command, including when `errexit` is off.
There is no environment-variable success marker that bypasses verification.

The unprivileged bootstrap uses the existing stdlib/OpenSSL HTTPS transport with
exact destination and redirect-host validation. It downloads only the two pinned
public Debian artifacts and checks complete archive and selected member hashes.
The parser accepts their exact three-member ar layout and safe tar paths/types;
it never extracts an archive tree or runs package scripts. See Debian's
[package format](https://manpages.debian.org/bookworm/dpkg-dev/deb.5.en.html).

`NPA_NCORE_NATIVE_CACHE` selects an absolute per-user private directory, defaulting
to `${XDG_CACHE_HOME:-<account-home>/.cache}/npa/ncore/native` (normally
`/home/ubuntu/.cache/npa/ncore/native`). It needs POSIX `flock` and atomic rename;
it is ephemeral unless the operator mounts durable storage. Directories reject
symlinks, foreign ownership and writable ancestors (except a root-owned sticky
ancestor for an unprivileged user). The cache root is `0700`; singly linked
regular artifacts and its lock are `0600`. Existing modes are not repaired.
Every warm startup rechecks the full `.deb` and member hashes. An interrupted
download publishes no artifact; a later invocation can retry. Corrupt or unsafe
existing cache entries fail closed and need operator removal before retry.

The downloader passes complete verified artifacts through stdin to a fixed
`sudo` invocation of root-owned `/usr/local/bin/python3.12 -I -S -B` and the
root-owned script. Its environment is cleared. The installer accepts no cache,
lock or destination argument and interprets no environment variable as code.
It rehashes both artifacts and their selected members into memory before opening
any installation path. It checks root ownership and permissions of the fixed
script, interpreter, lock and ancestor directories. A separate private root
flock serializes installs; staged files are checked before atomic replacement,
then all installed bytes, modes, ownership and links are checked again. Missing
files from an interrupted installation can be completed; altered existing
libraries or SONAMEs fail closed. There is no `LD_LIBRARY_PATH` dependency.

`base-source-lock.json` retains every unaffected binary/source record and all six
signed repository metadata artifacts. Only the absent packages' two components,
two copyright notices and four unshared GnuTLS source/signature artifacts are
removed from the delivered selection. Their identities and source/license
provenance remain metadata in the native lock; these records are not a delivered
source annex or a source waiver for any retained binary. GnuTLS's selected
library remains LGPL-2.1-or-later, and libssh2 BSD-3-Clause. Runtime cache contents
must not be copied into a final build input or published as a derived image.

Build probes only import the stdlib bootstrap and verify its lock while the
selected native bytes are absent. The final copy is guarded by absence checks
for the four paths, cache and installation state. Do not invoke either startup
hook against `/public-root`. Exact-image cold/warm startup, the explicit Bash
entrypoint override, real APT/SSH bootstrap, complete-byte scans and application
checks remain acceptance gates; source tests do not establish image acceptance.

## Why source is or is not delivered

Each component has a binary-specific `license` and `license_reason`; every
selected Debian package keeps its original copyright file. Common license texts
and CPython's original `LICENSE.txt` are included. A permissive component is
`delivery: notice`, with no full source archive. This classification is scoped to
the files listed for that component, not the entire upstream project.

Examples established from actual Debian copyright/build records:

- OpenSSL 3 is Apache-2.0; OpenSSH and sudo retain their component-specific
  BSD/ISC/MIT and other original notices. Their private test-key archives are
  unnecessary and absent.
- The selected zstd and libcap libraries use their offered BSD alternatives;
  lz4's library is BSD. GPL command-line programs in those projects are absent.
- Debian explicitly identifies liblzma and libselinux as public-domain
  libraries. Their differently licensed tools and source trees are absent.
- PAM's selected modules use the BSD alternative and their specific notices.
  The LGPL `mkhomedir_helper` and GPL `pam-auth-update` are absent. Separately
  linked LGPL libaudit/libcap-ng/libcrypt libraries still carry source.
- `libcom_err`'s actual `lib/et/Makefile.in` object list uses MIT SIPB and BSD
  source; the GPL filesystem tools are absent. `libuuid`'s actual
  `libuuid/src/Makemodule.am` selects BSD library/randutils code and public-domain
  MD5/SHA1/common inline helpers. Their original source headers and permission
  text are delivered as `UPSTREAM-NOTICE`, with archive/member hashes and exact
  excerpt lengths in the lock. e2fsprogs source remains unnecessary. util-linux
  now also supplies the required `mountpoint` check and libmount/libblkid, so its
  actual GPL/LGPL source is delivered as well.
- CPython's interpreter/stdlib are covered by its PSF and bundled notices.
  Inherited pip and ensurepip are removed before publication, so neither pip's
  source nor its vendored fixture corpus is delivered.

GPL/LGPL source is actually delivered for the remaining covered binaries,
including bash/coreutils, glibc, GCC runtime libraries, and the selected audit,
cryptography, process and bootstrap tools. The standalone libgcc/libstdc++
shared libraries still require source; the GCC runtime exception is **not** a
source waiver for distributing those libraries independently. See the
[FSF exception FAQ](https://www.gnu.org/licenses/gcc-exception-3.1-faq.html.en).
Public CA trust data and the public Debian verification keyring retain their
small source packages and notices. Public certificates/verification keys are
intentional trust inputs; no private key is selected.

## Reviewed source transformations

Nebius Physical AI modified these source distributions on 2026-09-07 as described
below. The `.npa.tar.xz` suffix distinguishes them from the upstream archives.

The three transformation recipes change the delivered bytes themselves. They
are not scanner exceptions. Original URL/hash and signed Debian Sources/.dsc
metadata identify the inputs; `transformation.output_sha256` identifies the
source actually delivered. All original license/build/install material remains.
The lock carries exact removed paths, retained build inputs, rationale and local
library build/install evidence. Assemblies fail if an input, output or expected
removal changes, or if the build proof is absent.

- GCC: remove the enumerated optional DejaGNU testsuite directories inside its
  nested source archive. Keep configure/Makefile templates and the compiler's
  **required** `gcc/testsuite/selftests` inputs. A build check caught that
  requirement; they are not discarded. Production sources, generators,
  architecture headers, original Debian patch/build rules and runtime-library
  sources remain. The helper below filters only Debian patch sections targeting
  the removed paths, applying production and retained-template hunks normally.
- Libgcrypt: remove optional `tests/` programs, vectors and benchmark inputs;
  retain configure templates. Production cipher self-tests and their vectors
  remain in the library sources. Build/install with
  `make SUBDIRS='compat mpi cipher random src'` and the same override for
  `make install`; this avoids compiling the removed optional tests.
- Systemd: remove only `test/test-network`, `test/test-resolve` and
  `test/dmidecode-dumps`. These integration fixtures and machine dumps are not
  libsystemd/libudev build or installation inputs. Use `-Dtests=false
  -Dinstall-tests=false`; the remaining production sources and build scripts
  are unchanged.

These are buildability checks, not bit-identical rebuild claims or new Debian
binary provenance. The image still copies the **original pinned .deb binary
bytes**, checked against their exact hashes. Other required source archives are
unmodified and remain subject to recursive scanning. No source fixture path is
excluded from those scans.

## Verify and rebuild

Recipients receive `/opt/ncore/base-sources` in the same image pull, plus the
selected runtime notices under `/usr/share/doc`. URLs identify original inputs;
they are not substitutes for corresponding source or invented written offers.
The annex contains source archives, original Debian patches/.dsc files, the lock,
this document and `recipes/base_sources.py`. Exactly six signed repository inputs
(the two snapshots' InRelease, Packages.xz and Sources.xz files) have
`delivery: build-only`. They authenticate the selection during the build in
`/build/base-metadata`, outside the delivered annex. The final root check mounts
that directory read-only; the scratch copy never includes it. Classification
must equal the signed repository input population and cannot remove any required
corresponding source, including original patches and transformed source outputs.

Extract the annex, `/usr/share/doc`, and
`/usr/share/keyrings/debian-archive-keyring.gpg` from the image, and save that
exact image as `image.tar`. Using Python 3.12+ and gpgv outside the image, first
fetch the hash-locked build metadata into a separate directory. `assemble` reuses
and verifies the delivered sources; it downloads missing inputs over anonymous
HTTPS. A fresh cache may instead supply these inputs with `--cache` and
`--offline`. The metadata directory must neither contain nor be contained in the
source annex, and must not resolve to the same directory.

```sh
npa/.venv/bin/python base-sources/recipes/base_sources.py assemble \
  --lock base-sources/recipes/base-source-lock.json \
  --annex base-sources --native base-sources --metadata build-metadata
npa/.venv/bin/python base-sources/recipes/base_sources.py inventory \
  --image-archive image.tar --output inventory.json
npa/.venv/bin/python base-sources/recipes/base_sources.py verify \
  --lock base-sources/recipes/base-source-lock.json \
  --annex base-sources --native base-sources --metadata build-metadata \
  --keyring debian-archive-keyring.gpg --inventory inventory.json
```

`--native` remains only an interface compatibility parameter; there is no native
application annex. The verifier authenticates binary/source metadata, delivered
archive hashes, every selected base file/notice, and every ELF in every layer.
Every delivered archive must also exist at its published annex path with its
exact output hash in the final filesystem inventory; a separate valid annex
cannot supply missing image bytes. Build-only metadata is rejected by path and
by exact hash anywhere in any published layer, even under a different name.
A second filesystem layer, unknown ELF, omitted source or changed notice fails.
The same source/selected-file check runs on `/public-root` before the scratch copy.
It does not replace security, license or complete-byte scanning.

For unchanged Debian sources, use `dpkg-source -x` on the delivered .dsc with its
adjacent archives and install the build dependencies declared in that .dsc.
For one of the three transformed packages, use its exact `debian:NAME@VERSION`
component ID from the lock:

```sh
npa/.venv/bin/python base-sources/recipes/base_sources.py prepare-source \
  --lock base-sources/recipes/base-source-lock.json \
  --annex base-sources --component "$COMPONENT_ID" --output ./source-work
```

This unpacks the actual delivered archives, overlays the original Debian build
metadata and applies the patches, recording original/applied patch hashes in
`NPA-SOURCE-PREPARATION.json`. `prepare-source` verifies the delivered source
hashes without requiring build-only repository indexes. GCC preparation needs
make, dpkg-dev, lsb-release, Perl and patch. The upstream .dsc checksums intentionally describe the original
inputs, so do not claim that a transformed archive matches its original .dsc.
Use the locked transformed hash and the preparation helper instead of disabling
source verification. The corresponding build commands are recorded in the lock;
external compiler/build dependencies are not mirrored into the runtime image.

The parent must still build the integrated commit and verify the resulting
saved image: exact layer/config/history bytes, recursive source contents,
secrets, licenses, payload, SBOM, vulnerabilities, non-root bootstrap and source
revision. Then run cold/warm runtime fetch, complete real-capture conversion and
independent V4 readback, followed by the existing NRE RTX consumer. Source and
unit checks alone do not remove quarantine or establish publication acceptance.
