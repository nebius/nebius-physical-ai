# Selected base filesystem, notices and corresponding source

The published stage is `FROM scratch`. It contains one copy of an explicitly
assembled filesystem, with one retained version of each Debian package. The
Python base and the APT/source builders are build inputs, not published ancestry.
The bootstrap includes real APT, dpkg and curl, bash/sh, sudo, SSH, rsync,
service and their required helpers and libraries. Inherited pip/ensurepip,
Python headers/static libraries, optional GUI/DB/interactive extensions, build
trees, populated caches and test extensions are absent. The converter and its
hash-locked runtime dependency environment are unchanged.

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
libbrotlidec/libbrotlicommon, libgnutls, libhogweed/libnettle, libidn2, libldap/liblber,
libnghttp2, libp11-kit, libpsl, librtmp, libsasl2, libseccomp, libssh2, libtasn1,
libunistring, libmount and libblkid. Existing selected libraries satisfy their
remaining ELF dependencies. Each is bound to actual pinned Debian bytes.

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
Never run the mutating APT probe against `/public-root` before its scratch copy.

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
The annex contains source archives, original Debian patches/.dsc files, signed
repository indexes, the lock, this document and `recipes/base_sources.py`.

Extract the annex, `/usr/share/doc`, and
`/usr/share/keyrings/debian-archive-keyring.gpg` from the image, and save that
exact image as `image.tar`. Using Python 3.12+ and gpgv outside the image:

```sh
npa/.venv/bin/python base-sources/recipes/base_sources.py inventory \
  --image-archive image.tar --output inventory.json
npa/.venv/bin/python base-sources/recipes/base_sources.py verify \
  --lock base-sources/recipes/base-source-lock.json \
  --annex base-sources --native base-sources \
  --keyring debian-archive-keyring.gpg --inventory inventory.json
```

`--native` remains only an interface compatibility parameter; there is no native
application annex. The verifier authenticates binary/source metadata, delivered
archive hashes, every selected base file/notice, and every ELF in every layer.
A second filesystem layer, unknown ELF, omitted source or changed notice fails.
The same selected-file check runs on `/public-root` before the scratch copy.
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
`NPA-SOURCE-PREPARATION.json`. GCC preparation needs make, dpkg-dev, lsb-release,
Perl and patch. The upstream .dsc checksums intentionally describe the original
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
