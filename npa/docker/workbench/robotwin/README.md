# RoboTwin publication helpers

`byte_gate.sh` checks the saved image before publication and after its exact
digest is pulled. A failure continues to block publication.

For private investigation, an operator may temporarily set the repository
Actions secret `ROBOTWIN_PRIVATE_FAILURE_UPLOAD_URL` to a presigned HTTPS PUT
for one authorized private object. It is optional and used only after the byte
scan fails. The upload contains exactly `image.tar`, `scan/report.json`, and
`scan/records.jsonl`; policy, authorization, and credentials are excluded.

The upload validates TLS, refuses redirects, and exposes only a fixed completion
or failure message. Its result never replaces the scan's original failure.
The operator controls private storage access and removes the temporary secret
after collecting the evidence. This is not a public Actions artifact upload.

For an independently reviewed exact finding population, the optional private
review handoff uses four separate Actions secrets:

- `ROBOTWIN_PRIVATE_REVIEW_PRE_REQUEST_URL` and
  `ROBOTWIN_PRIVATE_REVIEW_POST_REQUEST_URL`: HTTPS PUT for one private request
  object per phase.
- `ROBOTWIN_PRIVATE_REVIEW_PRE_RESPONSE_URL` and
  `ROBOTWIN_PRIVATE_REVIEW_POST_RESPONSE_URL`: HTTPS GET for separate objects
  written only by the independent reviewer. The runner has no response-write
  capability. Each step receives only its phase pair as
  `ROBOTWIN_PRIVATE_REVIEW_REQUEST_URL` and `ROBOTWIN_PRIVATE_REVIEW_RESPONSE_URL`.

The request retains exact archive, raw scan, graph, authorization, and narrowly
bound scanner policy/tool receipts in private storage. It includes a `request.json`
context and fixed review directory; it excludes environment dumps, registry or
storage credentials, customer acceptance, and signing material. Original scanner
inputs remain unchanged on the runner throughout review.

Before dispatch, initialize each response object as a tar containing only
`response.json` with `schema_version: npa.robotwin.private-byte-review-response.v1`,
`status: pending`, and its `phase` (`pre` or `post`). Only this explicit pending
state is retried. HTTP errors, redirects, changed inputs, or malformed responses
fail closed. No new review deadline or attempt limit is imposed.

After reviewing every exact occurrence, the independent reviewer writes a tar
containing `response.json`, `manifest.json`, `review.json`,
`proofs/<64hex>.json`, and `evidence/<64hex>.json`, `.txt`, or `.bin` as required
by the existing adjudicator. All members must be unique regular files; directory
entries, links, extensions, unsafe names and nonzero trailers are refused. The
completed envelope adds `status: reviewed`, SHA-256 of the exact `request.json`
as `request_sha256`, and independently approved `manifest_sha256` and
`review_sha256`. Its phase and original schema remain mandatory. Proof paths
must point into the request's private review directory.

The existing adjudicator verifies the external pins, exact context and complete
occurrence population, retaining the raw invalid report and ledger. Only its
separate successful receipt can pass that RoboTwin phase. Missing review,
unsupported roles, real credentials, incomplete scans or any changed binding
still fail. A pre-publication decision cannot approve the later post-pull scan.
The operator removes all four temporary secrets after the run. Public logs
contain only fixed status messages, never matches, policies or transport URLs.
