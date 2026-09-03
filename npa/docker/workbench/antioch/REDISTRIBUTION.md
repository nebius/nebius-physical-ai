# Antioch adapter redistribution

This image is an Apache-2.0 NPA control-plane client and is eligible for public
redistribution. It contains no Antioch engine, `antioch-sim` distribution,
operator configuration, credentials, projects, or customer data.

At runtime, the operator supplies an existing Antioch configuration as a
read-only mount. The adapter downloads the exact `antioch-sim` 0.3.63 wheel
directly from the vendor's PyPI delivery URL into a writable cache and verifies
SHA-256 `9037118b94d1b8241ca7e693b34a7a6ccffa7f47492b929b56c6f3b813032e8c`
before installation. That direct delivery is governed by the operator's Antioch
terms and entitlement. Runtime fetch and use require the exact, scoped
`NPA_ANTIOCH_ACCEPT_TERMS=YES` attestation for the Antioch Terms of Service at
`https://antioch.com/terms` (2026-02-28), `antioch-sim==0.3.63`, and Antioch
Service use. The attestation is injected only at runtime and is never stored in
this image or cache. A cold cache cannot be used in offline mode, another
version is rejected, and no runtime cache is copied into an image layer.

Release validation must inspect the built image, not just this Dockerfile, and
must fail if an `antioch-sim` distribution or credential-shaped file is present.
