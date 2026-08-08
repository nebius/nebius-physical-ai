# LeIsaac container redistribution

`npa-leisaac:0.4.0` is an additive image over NPA's public, runtime-fetch
`npa-isaac-lab:2.3.2.post1` image. It bakes the Apache-2.0 LeIsaac source at
commit `1651c321e9b0c1bb54233211fc7b3cd70d8373d5` and its OSS Python
dependencies. It does not bake Isaac Sim, Isaac Lab, Omniverse Kit, the NVIDIA
WebRTC browser client, or LeIsaac task assets.

The collector adds distro FFmpeg plus PyArrow and boto3 from PyPI to encode the
real viewport and publish LeRobot v3 Parquet/video objects. These are OSS
runtime dependencies and do not change the `redistribution: public`
classification. Demonstration data and operator S3 credentials are runtime
inputs and are never image layers.

The current Kubernetes launcher and functional smoke support only the
hard-selected RTX PRO 6000 Blackwell pool. L40S has RT cores in general, but is
not an advertised LeIsaac route until this exact image and launcher are tested
there.

The agent-relay Kubernetes deployment references coturn `4.6.3-r3` by its
immutable linux/amd64 image digest. Coturn and its image-bundled runtime
dependencies permit binary redistribution under their included OSS licenses;
the coturn image includes its BSD-3-Clause license at
`/usr/share/licenses/coturn/LICENSE`. It contains no NVIDIA runtime, weights,
or task assets and does not change this image's `redistribution: public`
classification.

At container startup, the operator must explicitly provide both
`OMNI_KIT_ACCEPT_EULA=YES` and `ISAACSIM_ACCEPT_EULA=YES`. Only then does the
shared NPA bootstrap fetch the pinned Isaac Sim 5.1.0.0 / Isaac Lab
2.3.2.post1 runtime. The service also fetches the pinned SO101 asset, the
v0.1.0 kitchen scene, the v0.1.2 table-with-cube scene, and NVIDIA WebRTC
client 5.6.0 into its mounted cache, verifies their
cryptographic hashes, and writes `provenance.json`. The pristine client source
hash is also the served hash: NPA does not modify NVIDIA's JavaScript bytes.
An NPA-owned browser adapter selects WSS for the exact same-origin signaling
path on port 443, matching NVIDIA's documented TLS proxy topology; provenance
records that adapter decision.
EULA acceptance is never a Docker `ARG` or `ENV` and is not persisted in an
image layer.

Vendor-source review (accessed 2026-08-06): NVIDIA's
[Traffic Encryption](https://docs.omniverse.nvidia.com/ovas/latest/configuration/traffic-encryption.html)
documents WSS signaling and TLS termination at a proxy, and NVIDIA's
[Web Viewer Sample](https://github.com/NVIDIA-Omniverse/web-viewer-sample)
documents configuring the library rather than editing its distributed bundle.
Those sources support the adapter architecture, but this engineering record is
not legal advice. The NVIDIA client remains proprietary and operator-fetched;
the operator must confirm that its NVIDIA agreement permits the intended use.

The browser service uses upstream LeIsaac's software keyboard device. The
unlicensed `feetech-servo-sdk` package used only by physical SO101 leader
hardware is intentionally neither installed nor redistributed. NPA applies one
packaging-only patch that removes that mandatory dependency edge from upstream's
`pyproject.toml`; the lazy-loaded hardware implementation and the real
`SO101Keyboard`/task source are otherwise unchanged. The build runs `pip check`
after installation.

Before publication, run the full-filesystem (not `--history-only`) scan of the
exact published digest with `npa/scripts/scan_image_omniverse_payload.py`. Its
version-agnostic classifier rejects Omniverse/Isaac payload, any versioned
`leisaac-cache/client/` tree or WebRTC client archive, and runtime/source-tree
LeIsaac USD task assets. A clean digest-bound report is the mechanical evidence;
the Dockerfile's absence checks are only an earlier defense-in-depth gate.
