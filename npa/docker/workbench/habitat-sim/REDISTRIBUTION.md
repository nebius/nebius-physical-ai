# Habitat-Sim image redistribution boundary

This directory defines an unbuilt, quarantined candidate targeting public delivery. It
does not record a built image, a registry publication, or functional acceptance.

The image may contain only the exact Habitat-Sim source projection and
redistributable source, Ubuntu, and Python bootstrap closure recorded by the
adjacent manifests. The optional scientific/visualisation stack (SciPy,
NumPy-quaternion, SciPy, Numba/llvmlite, Matplotlib, FontTools, contourpy, and
kiwisolver) is deliberately
absent from the bootstrap image because its ELF closure exceeds the retained
512 MiB guard; its exact hash-locked rows are runtime-fetched into a private,
run-owned cache only when the smoke workload is invoked. This is a delivery
mechanism, not a redistribution or licensing qualification. The Habitat code and PBR configuration are MIT. The PBR resource
projection also contains the upstream BRDF LUT under its MIT grant and five
Poly Haven environment maps under CC0 1.0, together with the exact upstream
`data/pbr/license.txt` notice. The image carries the corresponding notice at
`/usr/share/doc/npa-habitat-sim/THIRD_PARTY_NOTICES.md`. That closure includes
the exact BSD-3-Clause Imath v3.1.9 source required by vendored OpenEXR; CMake
is forced to use the local projection with FetchContent network access
disabled. The official Meta Habitat test-scene archive is never an image input.
At runtime the smoke downloads that archive from its official HTTPS locator,
checks its complete SHA-256, extracts only the pinned Skokloster Castle GLB and
navmesh members after their metadata and SHA-256 checks, and deletes the archive.
The scene is The King's Hall at Skokloster Castle, scanned by Erik Lernestål and
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en).
The original asset record is on
[Sketchfab](https://sketchfab.com/3d-models/the-kings-hall-d18155613363445b9b68c0c67196d98d).

The GPL-3.0+ `rsync` binary needed by SkyPilot is accompanied under
`/usr/share/doc/npa-habitat-sim/ubuntu-sources/rsync` by its exact upstream tar,
signature, Ubuntu packaging delta, and `.dsc`. The same signed immutable Ubuntu
snapshot supplies those four source artifacts; `apt-runtime.lock` pins their
source version, signed metadata index, sizes, and SHA-256 values. The compressed
index is size/hash verified and parsed: exactly one matching source/version and
directory must contain the complete locked Files and Checksums-Sha256 population.
Downloaded source sizes and both index checksums must match before acceptance. The
final OCI verifier requires each exact source byte. Every direct Ubuntu package
is likewise downloaded and lock-hash verified before its local `.deb` is passed
to APT; transitive dependencies remain governed by the signed snapshot and final
package inventory. This is accompanying corresponding source, not a mutable
third-party link or an invented written offer.

This rsync delivery does **not** close the other obligations of the Ubuntu base,
installed packages, bundled native wheel components, or superseded ancestor-layer
bytes. Complete corresponding-source closure remains pending. The OCI verifier
retains every layer's distinct dpkg source/version/copyright identity and Python
metadata/RECORD identity, including packages replaced or removed later. It refuses
an absent, partial, or changed reviewed inventory and missing or modified accompanying
source files. Conservatively every inventoried package requires source delivery;
there is no guessed non-copyleft exemption or invented written offer. The checked-in
contract is deliberately pending, so a source-only test cannot qualify real bytes.
An authorized later inventory must also reconcile all native/bundled components
and license obligations against SBOM and full-layer bytes before that contract can
be completed. Hash agreement verifies delivery, not legal sufficiency of its content.

Before HTTPS is available, one isolated trust stage raw-extracts checksum-pinned
CA, OpenSSL CLI, and libssl packages from that same timestamped snapshot. It
copies only the exact generated CA configuration and parsed PEM bundle into the
build and runtime stages; none of those bootstrap package archives or parser
binaries can flow through this stage into the final image.

Source, baked runtime, weights, runtime data, cache, and outputs are independent
boundaries. This candidate has no weights. Its scene cache is run-owned and
ephemeral. Saved RGB/depth observations and proof JSON are operator-owned derived
outputs and retain title, creator, scan credit, original URL, CC BY link,
modification notice, scene hashes, source revision, and exact image digest.

The aggregate Hugging Face collection is neither a runtime source nor permission.
Credentials do not grant data-use rights. No Habitat-Lab training, gated datasets,
semantic annotations, HM3D, Matterport, Replica, or other scene data is supported.
Meta also warns that its internal teams do not officially maintain Habitat-Sim
releases beyond v0.3.4.

The candidate remains in `UNVALIDATED_PUBLICATION_TOOLS`. A later trusted public
workflow must rebuild an exact reviewed full Git SHA; that rebuild is a new digest
and must repeat complete OCI/layer, SBOM, provenance, vulnerability, secret,
license, and payload scans plus the real exact-digest RTX capability gate. No
private build evidence is inherited by a public rebuild.
