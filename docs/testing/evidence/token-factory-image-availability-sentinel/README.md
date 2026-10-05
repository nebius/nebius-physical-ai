# Token Factory image-availability sentinel evidence

This evidence pack covers the exact Token Factory caption candidate at commit
`3df75c244e173ec1c3a134eaad20e6ed0b26db8e` (tree
`40d54e519db8b25717cc378056eb5f23be7ef197`). The candidate treats the complete
visible answer `NO IMAGE RECEIVED.` as an unavailable-image result after
stripping surrounding whitespace and comparing case-insensitively. It does not
use substring matching.

The live check used one clear 256x256 image with a red square, blue circle, and
green triangle. The exact normalized PNG submitted by the candidate is
[three-shapes-submitted.png](three-shapes-submitted.png), SHA-256
`ce89995a92f4277d6cd943575a4d839f783bb337a42b46f71015cf60b021b943`.
An independent pixel-first review found the image clear, text-free, and
pixel-equivalent to its source PNG.

## Results

| Control | Hosted requests | Visible answer | Result |
| --- | ---: | --- | --- |
| Candidate caption with the submitted PNG | 1 | `A red square, a blue circle, and a green triangle are arranged in a horizontal row against a white background.` | Passed: all six frozen color/shape terms were present |
| Direct request with the same instruction and no image part | 1 | `NO IMAGE RECEIVED.` | Passed: exact cooperative sentinel |
| Retained negative response replayed through the candidate and Typer CLI | 0 | `NO IMAGE RECEIVED.` | Passed: failed result, one `image_unavailable` item, artifact written before exit 1 |
| Dry-run replay through the Typer CLI | 0 | `NO IMAGE RECEIVED.` | Passed: exit 1 and no artifact |

Both hosted completions returned the requested model with HTTP 200 and
`finish_reason=stop`. Each retained inference transport recorded one attempt,
zero retries, and no semantic replay. The exact prompt, sanitized provenance,
hashes, counters, and usage are in [evidence.json](evidence.json).

The CLI's emitted write result contains one convenience field,
`written_uri`, that is not stored in the artifact. Independent recomputation
verified that removing only that field makes the parsed emitted object equal to
the parsed artifact. The raw streams are therefore intentionally reported as
different bytes rather than mislabeled as byte-identical.

## Request and call accounting

`request_body_sha256` in the private transport evidence is a SHA-256 identity of
the sorted, compact logical request JSON. It is not a captured HTTP wire-body
hash. The public machine-readable evidence uses the clearer name
`canonical_logical_json_sha256`.

The live sequence consisted of:

- one successful Token Factory health command performing one semantic model
  listing; the health client's internal retry count was not exposed, so this is
  bounded at 1–4 physical attempts;
- one separately retained model-list request;
- one positive image inference request;
- one direct no-image inference request.

Thus the exact inference count is two and the supportable total live HTTP
attempt bound is 4–7. Two offline candidate/CLI passes used six MockTransport
requests in total and made zero provider requests.

## Execution applicability

| Stage | Customer-side hardware | Hosted hardware | Applicability |
| --- | --- | --- | --- |
| Input normalization, candidate classification, CLI replay, and review | CPU; no local GPU required | Not applicable | Reproducible from the exact source and retained bytes |
| Vision completion | No customer cluster or GPU | Provider-managed; accelerator details not exposed | Evidence applies to `MiniMaxAI/MiniMax-M3` through Token Factory for these two bounded controls |

This is hosted API evidence, not a benchmark of a customer GPU type.

## Provenance and review

- Frozen plan SHA-256:
  `8d9bb860ef32c69534cbb821fbdbad125c95c70578ddc33d7226bd8455a2d6b9`
- Reviewed harness SHA-256:
  `ab6a04cec95616374baa367b3578b8df7aa5a6ee6b229da3223e3b8b1fc5ceed`
- Private run summary SHA-256:
  `60de7a4efeafc30e4ed0682ea5b51f83c740ff095712410acc9488840013a8fa`
- Positive raw response SHA-256:
  `49a2524588ceb2661a9d3cd6ba721f271129913a1166fbbd906137cdd0cdbee1`
- Negative raw response SHA-256:
  `7d72a0615052a65c676da8b452d50580aa2ef5ec64358b5c54b26ed7e94afcb9`
- Independent post-run verdict: `APPROVED_EVIDENCE`
- Independent post-run review SHA-256:
  `3de7f1b6a7d2dbe8c6f9521cb811921cbb2689926fb911d9e4a92de6552a5a7a`

The private evidence retains provider request identities, the key-scoped model
inventory, raw responses, exact timestamps, and operational paths. Those values
are intentionally omitted here. No credential, authorization header, hidden
reasoning, private endpoint, or live infrastructure identifier is published.

## Limits

These two controls do not estimate an operational image-delivery failure rate
or establish general vision quality. The cooperative sentinel is prompt
sensitive: operator text or text inside an image can induce or suppress it, and
a legitimate complete caption exactly equal to the sentinel would false-fail.
The evidence does not qualify the model as a visual judge and does not prove
physical correctness, usefulness, or robot safety.
