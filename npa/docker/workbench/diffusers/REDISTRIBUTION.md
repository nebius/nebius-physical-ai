# Redistribution: npa-diffusers

This public job image packages Mochi 1, CogVideoX-2B, Wan 2.1 14B, and Depth Anything V2 using the upstream
Apache-2.0 source at [`275869dcae4ebcfee6a80253fdabc56033335020`](https://github.com/huggingface/diffusers/tree/275869dcae4ebcfee6a80253fdabc56033335020).
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
