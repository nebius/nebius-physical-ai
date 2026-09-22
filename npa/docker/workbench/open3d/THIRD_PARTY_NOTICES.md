# npa-open3d third-party notices

Most of what this image bakes carries its own licence text where a recipient can
find it: the Debian packages keep their `/usr/share/doc/*/copyright` files, the
Python distributions keep their `dist-info` licence files, and components under
Apache-2.0 can reference the full text at
`/usr/share/common-licenses/Apache-2.0`. Those need no copy here.

`notices/` holds the exceptions — upstream notices that the built distribution
does not deliver, retained verbatim so the image ships the grant it relies on.

| Component | Version | Notice | Upstream source |
| --- | --- | --- | --- |
| `mcap` (Foxglove) | 1.4.0 | `notices/mcap-LICENSE.txt` | [`foxglove/mcap@b33fa68`](https://github.com/foxglove/mcap/blob/b33fa682a5c517b1d213faeabd118e0b4f9d9d93/LICENSE) |

## Why mcap is listed

`mcap` is MIT, and says so in its Trove classifier, but neither its built
distribution nor its PyPI source distribution contains the licence file. A
classifier is metadata about the grant, not the grant, so an image that ships
only the classifier ships no notice at all. The retained copy is the exact
1,077-byte `LICENSE` from the upstream release tag
`releases/python/mcap/v1.4.0`, sha256
`da11235665c17d4c1634072dae92b8ba1b38d6fdde2ccf19a6bbede33253f58d`.

The Dockerfile checks that hash at build time, and checks that the `mcap` it
installed is still the version this notice was taken from. Both are deliberate
build failures rather than warnings: a notice that has drifted from the package
it describes is worse than a missing one, because it reads as though someone
checked.

`mcap` reaches this image transitively, through `rerun-sdk` in the npa install,
rather than from `open3d-requirements.txt`.

## Scope

This file covers the gap named above. It is not a complete third-party
inventory for the image; the release SBOM is authoritative for every byte in a
published digest.
