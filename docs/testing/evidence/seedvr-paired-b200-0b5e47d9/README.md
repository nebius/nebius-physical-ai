# SeedVR2 paired VAE diagnostic: Gate A failed

This is a private 7B/B200 diagnostic. The shipped PR #593 interface remains 3B/H100; this experiment does not qualify that product interface or claim that 7B/B200 support has landed. The shipped 3B path also retains its earlier failed quality measurements.

The actual B200 experiment compared official sliced VAE decoding (split size 4) with unsliced decoding of the same immutable latent. Both official clips exactly reproduced the independently accepted controls. This isolates the decoder diagnostic from diffusion randomness; adding SM100 support is a separate architecture change.

**The candidate failed Gate A.** Main median LPIPS improved against bicubic, but temporal error exceeded the unchanged maximum. The small encoded temporal improvement was not supported before encoding and failed the codec-attribution control. The source-resolution light control also regressed in LPIPS and temporal error.

| Measurement | Bicubic | Official sliced | Unsliced candidate | Result |
|---|---:|---:|---:|---|
| Main median LPIPS, lower is better | 0.114428159 | 0.097947191 | 0.097329594 | Main LPIPS gate passed |
| Main temporal error, lower is better | 0.001565149 | 0.002472094 | 0.002425673 | Failed maximum 0.001799921 |
| Main temporal error before encoding | — | 0.003204036 | 0.003204919 | Candidate regressed |
| Light median LPIPS | — | 0.153703660 | 0.153787404 | No-regression control failed |
| Light temporal error | — | 0.003216324 | 0.003275854 | No-regression control failed |

Encoded temporal improvement was 0.000046420565, below the paired codec perturbation of 0.000047303415; pre-encoding improvement was negative (−0.000000882850). Detail retention and frame integrity passed, but cannot override the failed gates. Full numeric aggregates, per-frame measurements, paired controls and hashes are in [measurements.json](measurements.json).

The unedited artifacts are [main official](main-official.mp4), [main candidate](main-candidate.mp4), [light official](light-official.mp4), and [light candidate](light-candidate.mp4). Main clips contain 100 frames over 2 seconds; light clips contain 25 frames over 0.5 seconds. All are 640×480 at 50 fps. See [attribution.md](attribution.md).

Source `0b5e47d97414a2fe46d6184efe9b4423b2b52198` and the exact private image digest are bound in the measurements. The evaluator required explicit corrections to producer identity assumptions: native Job labels, the interpreter alias, and the already accepted float32 latent metadata. Original failures remain retained; numerical functions, thresholds and GPU artifacts were unchanged.

The frozen plan skips annotation and VLM review after this objective/control failure. The held-out episode remains sealed, the candidate is not integrated, and no further candidate was run. All resources created for this experiment were removed with UID checks. Generated pixels remain review aids, not recovered sensor truth. Mandatory private runtime gates passed; complete-byte image scanning and final image publication/readiness qualification remain pending.
