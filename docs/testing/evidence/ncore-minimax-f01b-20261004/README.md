# NCore fixed-input MiniMax comparison

One separately authorized model-plus-provider-profile comparison passed its four
frozen controls, then its conditional final judgment. This does not erase the
[original MiniCPM calibration failure](https://github.com/nebius/nebius-physical-ai/blob/aec8ba73668a8b7fd1122144e2a12fbc554669bc/docs/testing/evidence/ncore-vlm-failed-aa395-20261004/README.md), establish general judge
quality, or authorize image publication.

Actual inference source: `f01b7749e194b1929d65dde553f6ce457d3767e1`.
The only request changes were the selected MiniMax-M3 model, its required
disabled-thinking profile, and omission of unsupported provider JSON mode.
The local strict parser, exact returned-model check, images, prompts, ordering,
hidden expected labels, threshold and one-shot gates were unchanged.

| Frozen request order | Score | Expected | Result |
| --- | ---: | --- | --- |
| Synthetic local duplicate patch | 0.2 | Negative | TN |
| Authentic camera2 photographs | 0.8 | Positive | TP |
| Synthetic global block permutation | 0.0 | Negative | TN |
| Authentic camera1 photographs | 0.8 | Positive | TP |
| Conditional final camera1 render sample | 0.8 | No calibration label | PASS |

Calibration: **2 TP, 2 TN, 0 FP, 0 FN**. Only after that result did the final
request execute. All five requests returned HTTP 200 and completed normally;
there were no retries, model cycling, threshold/label changes or response
substitutions. The unchanged final threshold was 0.8.

[Complete numerical results, all five unabridged rationales, request/response
hashes and twenty exact image links](summary.json) retain the result. The images
and prompts are byte-identical to the previously published frozen population:
[control gallery and prompts](https://github.com/nebius/nebius-physical-ai/blob/aec8ba73668a8b7fd1122144e2a12fbc554669bc/docs/testing/evidence/ncore-vlm-failed-aa395-20261004/README.md), and
[four final frames](https://github.com/nebius/nebius-physical-ai/blob/aec8ba73668a8b7fd1122144e2a12fbc554669bc/docs/testing/evidence/ncore-vlm-failed-aa395-20261004/final). That earlier page correctly states those final
images were not sent in the failed MiniCPM schedule; they were sent once in this
separate MiniMax schedule. No new GPU training/rendering or image edit occurred.

The final rationale explicitly reports softness/blur on metallic surfaces and
slight glass-facade smearing. The model judged only camera1 source indices
0, 88, 177, 266—not all cameras, all frames, held-out data, physical safety or
generalized model quality. Passing four controls is a bounded diagnostic,
not calibration over a representative population. The returned provider model
identifier is not independent proof of weights identity.

Provider usage: 13,579 prompt + 855 completion = 14,434 tokens. Billed cost is
unmeasured. Exact raw responses, all five conditional-write/read markers,
raw ETags and complete prefix listings, and raw before/after 142-package inventories
are retained privately. Actual complete-evidence CPU replay passes with no new
inference. Source and environment remained unchanged throughout execution.

The source/profile and fixed-input protocol were independently accepted before
execution. Terminal response and narrow consumer review were pending when this
proof was issued; consult the PR for later receipts. The original aggregate
consumer still expected MiniCPM and correctly rejected these model records.
Successor `6fcead42cee41bd23d8d5bdb3d84286cd1789e8b` reconciles only the exact
expected model with strict mismatch controls. Its synthetic nonvisual aggregate
fixture replay is not actual image, cleanup or publication qualification.

Original MiniCPM failure and all earlier scientific failures remain immutable.
The [native photometry experiment](https://github.com/nebius/nebius-physical-ai/blob/8e5c4a1d07f6f04e7339ad72615b45f32afc5345/docs/testing/evidence/ncore-photometry-10b-20261004/README.md)
has separate fitted-population numeric and workflow evidence. Local narrow
infrastructure-literal scan/adjudication and degraded-local-metadata cleanup
limits remain; no private image or vendor model was published.
