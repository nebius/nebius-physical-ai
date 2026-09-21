# Evo: fresh evaluation of the rebuilt image

**PASS in native CPU scope:** 62 functional outputs and 2 logs, 10 native result archives, and 690 error values independently recomputed from the input poses. All four frozen controls classify correctly: two positive and two negative; the malformed input fails without a result archive.

The immutable image index is `2d1e1c074e2b6f066bb49aaee1ac6bf5252ca48689312ee9c368941ca42f15a9`. It was built from recipe `841ceb59e051d7e7216992df56318c5cbbf7e7f5` and pinned Evo source `8dd6cfe0ec1747f9e1b5b569edd82c54d1a3f422`. Runtime bytes changed, so the earlier image's evaluation was not reused.

The real native commands ran as the configured non-root user, with networking disabled and no source overlay. The evaluator retained its 0.05 m APE and 0.02 m RPE limits and minimum 100 matched poses. All 19 plots decode; the actual synthetic trajectory plot appears below, unchanged. Owned-container cleanup was verified.

![Actual synthetic trajectory output](synthetic-controls.png)

[Measurements, image identities and scope](review.json) · [690 recomputed errors](synthetic-errors.json) · [Verification hashes](SHA256SUMS)

These are synthetic metric controls and pinned upstream example compatibility checks. No new GPU or VLM result, managed Sky run, real-world SLAM accuracy or robot-safety claim is made. Complete image security qualification and current hosted CI remain pending; the image was not pushed publicly. Earlier failures remain retained.
