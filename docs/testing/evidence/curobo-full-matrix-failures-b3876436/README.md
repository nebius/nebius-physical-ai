# cuRobo full benchmark: both GPU attempts failed

**The B200 and RTX PRO 6000 full benchmarks failed before the first complete problem row.** This is failed benchmark evidence, not benchmark acceptance or a usable robot-planning result.

| Retained attempt | Prepare | Benchmark | Complete rows | Verified durable objects | Partial journal |
| --- | --- | --- | ---: | ---: | ---: |
| B200 | SUCCEEDED | FAILED | 0 / 5,200 planned | 19 | 0 bytes |
| RTX PRO 6000 | SUCCEEDED | FAILED | 0 / 5,200 planned | 20 | 0 bytes |

Both retained runtime logs report:

> inverse-dynamics evidence shape or values are invalid

Both failures use schema `npa.curobo.failure.v1`, type `subprocess_exit`, and benchmark exit code 1. Every retained object size and SHA256 was independently rechecked: 39 objects across the two attempts. Both terminal stdout/stderr hashes match. [The numerical audit and original receipt hashes](failure-summary.json) preserve those bindings without publishing private logs or infrastructure details.

The submitted immutable image is `sha256:73d4ba23af61b3762200ee29fb2cdd153d6df6b75b4ada481ccb75b6c8ed2955`, produced from commit [`b38764368`](https://github.com/nebius/nebius-physical-ai/commit/b387643682271d867c816fba1a598295ffe57f37). Both failed-run plan records match that digest; producer source binding comes from the reviewed exact-image control evidence.

Separate [B200 controls](https://github.com/nebius/nebius-physical-ai/blob/be95e537d8492d7ae38f737349f04982309639d2/docs/testing/evidence/curobo-b200-controls-b3876436/README.md) and [RTX controls](https://github.com/nebius/nebius-physical-ai/blob/3d63acfc0dd2ee2e496e2fe2994269b5696f6a11/docs/testing/evidence/curobo-rtx-controls-b3876436/README.md) passed positive, blocked-goal and malformed-input checks. Those successes do not repair either failed full matrix. The separate preliminary RTX control run used a disclosed ephemeral bootstrap repair. This full-matrix artifact audit does not itself establish the bootstrap repair history of the later full RTX run.

The root cause is unresolved. Actual live joint dimensions were not retained in this evidence, so the proposed joint-mapping explanation is not established. The next step is a reviewed source repair and a newly qualified image before another full benchmark attempt; thresholds and denominators remain unchanged.

This review rehashed retained evidence only and made no new GPU, provider or object-store calls. Runtime operators retained the accelerator execution attribution; this audit did not independently re-query hardware. No completed-matrix success rate, final image qualification, collision certification, physical usefulness, or robot-safety claim is made. Evidence reviewed at 2026-09-21T10:10:43.406122+00:00.
