# Token Factory no-image format normalization evidence

This evidence pack covers source commit
`15ebe402794d98793f297655f5d48713ce92233c` (tree
`72c309b4878a4c9f6f91563abc8e8b207a6fb861`), stacked on the exact reviewed
image-availability sentinel head. Classification accepts only a complete
case-insensitive `NO IMAGE RECEIVED` answer with one optional final ASCII period
and, at most, one matching pair of bold markers, ASCII quotes, or smart quotes.
The visible answer retained in the result is not rewritten.

## Evidence separation

The inherited hosted run observed one exact punctuated response,
`NO IMAGE RECEIVED.`, and one positive image response grounded in a submitted
red-square, blue-circle, green-triangle PNG. It did **not** observe a periodless,
bold, quoted, inner-whitespace, or case-varied sentinel response.

All formatting, rejection, request-boundary, empty-output, and mutation results
in this follow-up are deterministic local controls. This follow-up made zero
provider, health, or model-list calls and accessed no credentials. The inherited
hosted run was not replayed.

## Closed normalization matrix

The local matrix has 47 controls:

| Control class | Count | Result |
| --- | ---: | --- |
| Two punctuation cores across unwrapped plus six matching wrapper pairs | 14 | `image_unavailable` |
| Inner-whitespace cases for all six wrapper pairs | 6 | `image_unavailable` |
| Case variation across unwrapped plus six wrapper pairs | 7 | `image_unavailable` |
| Outer-whitespace case | 1 | `image_unavailable` |
| Nonempty longer, mismatched, nested, other-punctuation, code, and paraphrase answers | 18 | `completed` |
| Empty classifier-only input | 1 | Not an unavailable sentinel |

All 46 end-to-end rows made one synthetic completion call, produced the expected
aggregate status, and retained the answer exactly after the pre-existing outer
whitespace strip. The empty string is deliberately separate: the real
`TokenFactoryClient` rejects empty visible content before caption
classification. A one-request `httpx.MockTransport` control confirmed
`TokenFactoryToolError`, no emitted caption item, and no result artifact.

The rejected set includes longer answers, doubled or Unicode punctuation,
punctuation outside a wrapper, backticks and code fences, both two-wrapper
nesting orders, four smart/ASCII quote mismatches, a natural-language
paraphrase, and the original substring control.

## Request and mutation proof

At the actual client request boundary, the captured message contained exactly
one `image_url` part. Its `data:image/png;base64,` suffix strictly decoded to a
nonempty 32x24 RGB PNG.

Thirteen source mutations were tested and all were killed. They restored exact
punctuation-only equality, removed each wrapper family independently, enabled
substring or mismatched-quote matching, stripped unbounded periods, moved
period stripping before wrapper recognition, recursively unwrapped nested
wrappers, or rewrote retained captions. The recursive mutant restarted wrapper
recognition after every unwrap and was killed by both nesting orders. Exact
source bytes were restored and rehashed after mutation testing.

## Validation and independent review

- 62 focused caption-tool tests passed.
- 155 Token Factory workbench, batch, client, CLI, and SDK tests passed.
- 163 workbench and skill contract tests passed.
- Ruff lint and formatting passed across `npa`.
- Independent exact-source verdict: `APPROVED_SOURCE`, with no findings.

The machine-readable record in [evidence.json](evidence.json) binds the exact
source, plan, private result identities, mutation outcomes, applicability, and
review.

## Hardware applicability

The normalization, request capture, mutation testing, and review are CPU-only;
no customer GPU or cluster is required. The inherited vision completions ran on
provider-managed hardware whose accelerator details were not exposed. This
follow-up does not add hosted or customer-GPU evidence.

## Limits

Whole-answer matching limits false positives but does not eliminate them. A
legitimate image whose only salient text is a newly accepted sentinel format
can false-fail, and operator or in-image text can induce or suppress the
cooperative response. These controls do not estimate a provider failure rate,
prove that pixels were consumed, qualify a visual judge, or establish physical
correctness, usefulness, or robot safety.
