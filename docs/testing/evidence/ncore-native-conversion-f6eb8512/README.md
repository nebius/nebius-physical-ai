# Actual NCore conversion and independent readback

**CPU qualification passed; RTX reconstruction and rendered-output quality remain pending.** This is a full reference-capture conversion, not an import-only smoke test.

| Measurement | Actual result |
| --- | --- |
| Input | Complete 1,535,277,338-byte public reference archive |
| Source / converted images | 518 / 518 |
| Cameras | 3 |
| Camera poses | 518 |
| Retained SfM points | 163,453 |
| Independently read-back objects | 10 |
| Independently read-back bytes | 1,002,432,577 |
| Wrong-source negative control | Rejected before native execution; zero objects before and after |
| Actual container exits | Wrong-source control: 0; native conversion: 0; independent audit: 0 |

The actual image producer was `f6eb851255ecf219e94cdc8ca937875509bc0a48`; image manifest `sha256:cdd4e00174f74a25fb0b6ac6dafa3d9d3f5f351deba3e4b822e87f78228f3de6`. Each command ran in its own inspected, pull-disabled container bound to the checked local image. All three owned containers were subsequently confirmed absent.

The native audit independently reopened the original source and the uploaded NCore V4 generation. It checked every converted member, camera calibration, poses, finite geometry, per-camera frame inventory and source/conversion counts. Root then downloaded all 10 objects again, verified every member hash and the immutable publication claim, and checked that the complete object listing and identities matched the native audit.

The original host receipt collector **exited 1** after the successful containers: it checked the obsolete `output_objects` key instead of the producer's separate before/after counts. That failure remains retained. A separate post-run review validated the corrected contract, actual command/image bindings and complete output readback; no conversion was repeated and the original wrapper was not relabeled as passed. The source repair is undergoing its full validation before publication.

This does not establish NRE training, a successful novel-view render, GPU quality, VLM acceptance, public-image release or merge readiness. Those are separate remaining gates for [PR #646](https://github.com/nebius/nebius-physical-ai/pull/646).

The [machine-readable measurements and hashes](results.json) and [checksums](SHA256SUMS) open directly without private storage access. Infrastructure identifiers, credentials and raw operational logs remain private.
