# Third-party notices and runtime references

The unbuilt neutral recipe selects these public operating-system inputs:

- Docker Official Image `ubuntu:22.04`, linux/amd64 manifest
  `sha256:281c5745f657873d78e5531fc5ba8575f46ab7769b94550ac99543f122679986`.
  The image is maintained by Canonical and built from Canonical's official
  Ubuntu rootfs. Exact config/layer identities and immutable official metadata
  are recorded in `runtime-lock.json`.
- Ubuntu Snapshot Service timestamp `20260912T000000Z`, suites `jammy`,
  `jammy-updates`, and `jammy-security`, component `main`, architecture `amd64`.
  `apt-packages.lock` records 75 exact binary packages, 57 exact source package
  versions, the signed InRelease hashes, archive hashes/sizes, and each installed
  copyright-file hash.
- No third-party Python application distribution is selected;
  `runtime-requirements.lock` has an artifact count of zero.

The Ubuntu package archive classifies `main` as open-source software supported
by Canonical. Ubuntu's package review requires licenses to allow redistribution
and requires `debian/copyright` to report the licenses and copyright holders.
The build recipe preserves and verifies each selected package's corresponding
`/usr/share/doc/<package>/copyright` file. The exact source versions remain
available from the same signed immutable snapshot. Ubuntu is an aggregate work;
the package-specific licenses and Canonical's intellectual-property/trademark
policy continue to apply. This project claims no Canonical endorsement.

Authoritative references:

- https://hub.docker.com/_/ubuntu
- https://github.com/docker-library/repo-info/blob/c435ffb46ce70e116a1608c5f4f61336670ffdd5/repos/ubuntu/remote/22.04.md
- https://snapshot.ubuntu.com/
- https://documentation.ubuntu.com/project/how-ubuntu-is-made/concepts/package-archive/
- https://documentation.ubuntu.com/project/maintainers/AA/aa-new-review/
- https://ubuntu.com/legal/intellectual-property-policy

Runtime-only identities (not included in image layers):

- RoboTwin commit `96c1feab536306b50c26af200044fcdf126e8904`, MIT license.
- CuRobo v0.7.8 commit `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, NVIDIA Source Code License for cuRobo, including its noncommercial research/evaluation restriction.
- RoboTwin2.0 asset revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`; aggregate component/output rights require human/legal approval.
- CUDA and cuDNN runtime delivery and use require their exact applicable NVIDIA terms.

These references document boundaries; they neither fetch bytes nor record
acceptance. No RoboTwin/NVIDIA source, runtime, model, asset, cache, or output
payload was used to resolve the neutral inputs. Built-byte notices and source
availability still require exact-image verification before publication.
