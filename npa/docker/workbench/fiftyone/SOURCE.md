# MongoDB source delivered with this image

The bundled, unmodified MongoDB Community Server 7.0.40 executable is governed
by SSPL v1 and its included additional permissions and third-party notices.
SSPL is a source-available license; it is not OSI approved. The original
`LICENSE-Community.txt`, `MPL-2`, and `THIRD-PARTY-NOTICES` are retained in
`/opt/fiftyone/mongodb-notices/`.

The same directory contains the complete public source archive
`mongodb-source.tar.gz`, its provenance and SHA256 in `source.json`, and this
document. These files travel in the same image as the executable and require
no credential or additional download to copy or unpack. A copy of these
directions is also installed beside `mongod` as `MONGODB_SOURCE.md`.

The binary reports origin revision
`37741b538c02da076d1e34d66113ac0230d4523a`. MongoDB's public commit
[`d18b92bf3d150c32f129aba9317efb627d3f5ab1`](https://github.com/mongodb/mongo/commit/d18b92bf3d150c32f129aba9317efb627d3f5ab1)
records that revision in its `GitOrigin-RevId` trailer. The archive is pinned to
this exact public commit. Its original license files and build/dependency
sources remain intact.

To inspect the source, copy the archive and its neighboring files from the
image, then unpack it with `tar -xzf mongodb-source.tar.gz`. Read
`mongo-d18b92bf3d150c32f129aba9317efb627d3f5ab1/docs/building.md` for the upstream
compiler, system-library and Python build requirements. The GitHub archive
contains no `.git` history or generated root `version.json`. Copy the separately
supplied `version.json` into that extracted source root before following the
upstream build instructions. NPA supplies this small build metadata file; it
identifies the public source revision and does not modify the vendor archive.

The source includes its SCons build scripts, vendored dependencies and their
notices. General-purpose build tools and system prerequisites remain external.
The vendored variant project retains a `.gitmodules` reference to
googletest sources that are absent from the public tree; the exported commit
contains no Git submodule entries. Delivery of this source is not evidence of
a byte-identical rebuild of the vendor executable.

Source delivery addresses distribution of the included program. MongoDB's
[SSPL terms](https://www.mongodb.com/legal/licensing/server-side-public-license)
separately govern offering its functionality as a service; those terms still
apply to operator use. This image runs MongoDB as FiftyOne's local metadata
database and does not publish a database port.
