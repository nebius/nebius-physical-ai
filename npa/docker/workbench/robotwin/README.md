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
