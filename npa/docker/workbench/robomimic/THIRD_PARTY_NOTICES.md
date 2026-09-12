# Third-party notices for the neutral robomimic candidate

The intended image contains `ARISE-Initiative/robomimic` at commit
`d309eaecc18acf4152a830a895a6984b8ac71b05`, licensed under MIT. Its exact
`LICENSE` SHA-256 is
`7cdbfab482b23a4d925d59ff169ab0bc5f8c97ceb0db79f9fd5bf46ef8aa1556`.
`source-manifest.json` also binds Git tree
`4c8ebe35dbef16126dadf59cf8b771b9203753ab` to a canonical 57,907,200-byte
`git archive` tar whose SHA-256 is
`8dd695200bba3ca6043693a7db4b15713a740d5d6a91787984d4c0053f77fd8b`.
The build helper creates that archive only from the fetched exact commit; both
the helper and Docker build reject any archive, tree, member inventory, or
license mismatch.

The base is the digest-pinned official Python 3.11 slim Bookworm image. The
`debian-packages.lock` file binds the five required Debian roots and their full
`Pre-Depends`/`Depends` closure to 78 exact Bookworm amd64/all package artifacts
from the official 2026-09-06 Debian snapshot: 29,807,672 bytes in total. It also
records each exact source version, the license labels from its authoritative
Debian copyright metadata, a license reference URL, and the installed
`/usr/share/doc/<package>/copyright` notice
path. Recommends are excluded. The Dockerfile consumes only the preverified
artifacts through a read-only build-context mount, invokes no APT repository,
and verifies all 78 installed versions and notices. Applicable copyleft source
delivery remains a separate public-publication gate; the lock does not claim
that an upstream URL alone discharges NPA's source-conveyance obligations.

`baked-requirements.lock` remains byte-identical and identifies all 40 Python
distributions intended to be added through 214 accepted artifact hashes. A
future authorized build must generate and review the complete base, selected
wheel, package, and license inventory. These source records do not themselves
establish redistribution eligibility for resulting built bytes.

PyTorch, torchvision, Triton, and the NVIDIA CUDA/cuDNN/NCCL distributions named
by `runtime-requirements.lock` are not included in the candidate. That lock is a
compatibility declaration for an independently prepared external runtime. Its
artifact hashes and installed-file inventory must be supplied and verified at
the operator boundary after the applicable rights decision. PyTorch source
licensing is not evidence that wheel or bundled binary dependencies may be
redistributed or used as a hosted service.

The official Lift proficient-human low-dimensional dataset is also excluded. A
later authorized live gate fetches only revision
`74fa018461f479cd9fd15b924a16103012096203`, path
`v1.5/lift/ph/low_dim_v15.hdf5`, and accepts only SHA-256
`2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540`
with size 21,084,088 bytes. Dataset access and use remain the operator's separate
responsibility.
