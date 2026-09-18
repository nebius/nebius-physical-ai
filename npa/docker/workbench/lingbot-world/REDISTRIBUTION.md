# Redistribution: npa-lingbot-world

This public job image packages LingBot World v1 authored-camera video generation using the upstream
Apache-2.0 source at [`a43bec7f8091c83e9b30b16b912f6fc906236fa6`](https://github.com/Robbyant/lingbot-world/tree/a43bec7f8091c83e9b30b16b912f6fc906236fa6).
The upstream license and included notices remain in `/opt/byof`.
NPA adapters are Apache-2.0. The inherited digest-pinned Wan runtime's
redistribution records remain under `/usr/share/doc/npa-wan2-2`.

The image contains source and open-source CPU dependencies. It does not contain
model weights, CUDA Python distributions, user media, credentials, or terms
acceptance. CUDA dependencies are installed into a writable runtime volume by
`model-runtime ensure`. Models are fetched under the operator's own access and
their respective model licenses when a GPU capability runs. Source licensing
does not grant rights to model weights, input data, or outputs.

Passwordless sudo is inherited solely for SkyPilot bootstrap. The final user is
`ubuntu`; the image does not start sshd. Bootstrap capability is probed against
the exact digest rather than claimed through an unverified image label.
Publication requires complete image/layer inspection and secret, vulnerability,
and license scans. Supported release promotion additionally requires native GPU
capability evidence for the exact development digest.
