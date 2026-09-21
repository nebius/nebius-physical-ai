# Live CPU agent proof for PR #654

Source commit: `0666db15f7c5a715ceba74b9dbfa57fe93054e26`; tree: `c9780675722de9fb1ac99547685f4e8529d9cbc4`.

A fresh CPU agent completed the normal deployment lifecycle. Authenticated `/api/access?refresh=true` returned HTTP 200 in 3.636991 seconds with the staged metadata credential source; `/api/resources` returned HTTP 200 in 5.658939 seconds. The retained status shows authenticated UI/Rerun HTTP 200 and unauthenticated UI HTTP 401. The auth negative control is the UI probe; separate unauthenticated probes of the two affected API routes were not retained.

The access report remains partial access, not unrestricted tenant access. No GPU or VLM claim is made. Request success does not replace timeout fault injection.

The exact source producer is the clean pinned checkout and its own installed NPA entry point. The production deployment path verifies the live `/deployment` identity before recording healthy setup. Its separately returned manifest body and deployed Python-file hashes were not retained before teardown. The service fingerprint covers only deployment metadata and the systemd unit; it is not a backend-code byte attestation. These historical capture limits remain explicit.

The retained destruction receipt reports `verified_deleted`, `verified=true`, and `infrastructure_absent=true`. Independent review checked the source path that verifies the exact provider instance after successful Terraform destruction, both terminal operation journals, and a separate local-record absence readback. The reviewer made no new provider queries or mutations. Shared IAM was deliberately retained with `--keep-iam`; this proof does not claim IAM deletion.

`audit-receipt.json` binds retained private outputs and source files by SHA-256. Raw responses contain private infrastructure inventory and remain private; they are not needed to access this public summary. No endpoint, credentials, customer data, inventory, or live resource identity is included here.

`independent-real-process-control.json` records a separate unmocked Linux child-and-descendant timeout check against the exact source, including actual ownership checks, group cleanup, breaker recovery, and a successful subsequent command. This is a CPU process-control measurement, not a cloud workload or GPU evaluation.

Current-head hosted CI and draft promotion are separate readiness gates. This proof pack does not itself assert that CI has completed or that the PR has merged.

To reproduce the CPU failure control on Linux, check out the exact source commit, install the repository development environment, and run `npa/.venv/bin/python /path/to/verify_process_cleanup.py`. The script verifies the imported source commit and launches only its own temporary test processes.

Independent exact-source review accepted the scoped CPU proof with zero blockers: **74 resource/access tests and 132 rendered backend tests passed**, alongside the unmocked process-control check. [Review measurements and limits](independent-review.json). Hosted current-head CI was still incomplete at publication preparation; no merge is claimed.
