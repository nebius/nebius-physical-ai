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
delivery is enforced by `source_delivery.py` against every image layer.
`corresponding-source.lock.json` includes inherited and replaced parent package
versions. All matching source archives, Debian patches/build rules and the exact
CPython source archive are shipped at
`/usr/share/npa/robomimic/corresponding-source`, alongside the source lock; an
upstream URL alone is not used as delivery evidence.

`baked-requirements.lock` remains byte-identical historical source evidence; it
is not copied into the image and its 40 distributions are not installed in the
public bootstrap. The complete 62-entry public/CUDA/PyTorch/vendor package map
is declared by `runtime-requirements.lock` and is fetched only into a
customer-owned runtime volume. The customer supplies the exact official wheel
URLs, credential and independently selected inventory; the bootstrap downloads
into a private temporary wheelhouse, sends credentials only to their bound
vendor origin, revalidates entitlement before every request/install/publish,
captures the bound credential only for that transaction and never logs it,
verifies each hash and size, installs with no dependency resolution, and checks
every installed `RECORD` before publication into the runtime cache. The public
image therefore contains no Python wheel bytes; a future authorized runtime
operation must still generate and review the complete selected-wheel, package,
and license inventory. These source records do not themselves establish
redistribution eligibility for customer runtime bytes.

PyTorch, torchvision, Triton, and the NVIDIA CUDA/cuDNN/NCCL distributions named
by `runtime-requirements.lock` are not included in the candidate. That lock is a
compatibility declaration for an independently prepared external runtime. Its
artifact hashes and installed-file inventory must be supplied and verified at
the operator boundary. Before access, the customer must create an unexpired
record bound to the exact run, runtime lock and inventory after reviewing the
official CUDA Toolkit EULA, NVIDIA Software License Agreement, and cuDNN
Software License Agreement recorded in that lock. PyTorch source
licensing is not evidence that wheel or bundled binary dependencies may be
redistributed or used as a hosted service.

The official Lift proficient-human low-dimensional dataset is also excluded. A
later authorized live gate fetches only revision
`74fa018461f479cd9fd15b924a16103012096203`, path
`v1.5/lift/ph/low_dim_v15.hdf5`, and accepts only SHA-256
`2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540`
with size 21,084,088 bytes. The official exact-revision repository metadata is
public and ungated and declares license identifier `mit`; the exact card is Git
object `736f9c17ae642026c84d2b534119cc4dfea1548a`, 1,065 bytes, SHA-256
`e09a24720408bac08425dbaa0b7b55615e4440f1af0a61133303e4b5d5d6b09a`,
at <https://huggingface.co/datasets/robomimic/robomimic_datasets/raw/74fa018461f479cd9fd15b924a16103012096203/README.md>.
The selected file's tree object is
`70f345df97259439b111f93803e4007da01ad7a8`; its LFS OID is the SHA-256
above. The exact revision has no standalone `LICENSE` file (the official raw
endpoint returned 404), so the MIT card must remain attached to this identity.

The current official provider terms at
<https://huggingface.co/terms-of-service> were retrieved anonymously on
2026-09-16 at 12:41:42 UTC as HTTP 200 `text/html`, 114,631 bytes, SHA-256
`42020fcaac52b7b036bf7e816910ca45485ad04c2d31faf09e13636a5b48a36b`;
the page reported an effective date of 2022-09-15. That provider page is
mutable and must be rechecked for a future transaction. The exact card and
observed provider terms disclosed no dataset-specific field-of-use,
service-use, or training-output restriction. MIT notice/license obligations
still apply to copies or substantial portions. Public, anonymous reachability
is access evidence, not a grant of rights, and does not close the independent
CUDA, cuDNN, PyTorch, service, output, image, or live-execution boundaries.
