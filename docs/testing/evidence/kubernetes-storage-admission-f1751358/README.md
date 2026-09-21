# Kubernetes durable-storage admission proof

[PR #659](https://github.com/nebius/nebius-physical-ai/pull/659), exact source `f1751358cd21bf48384fd597550da4a29fbfebb4`, successfully admitted a real B200 workload with declared Nebius durable storage while keeping compute on the selected Kubernetes context. The reviewed configuration enables Nebius only for `storage`. The source and live evidence are bound in [review.json](review.json).

The job produced planning artifacts and eight durable objects. It then **failed** a separate strict cuRobo metric check. That failure remains recorded; it does not invalidate the successful storage admission, and this evidence does not claim a completed cuRobo benchmark.

The retained upload receipt reports successful object readbacks and no upload errors. Independent review verified the launcher, client/server configuration and job binding. A later root audit checked the published source bytes and four retained downloaded originals against the eight-object manifest. It did not make a new object-store query or independently download the remaining four objects.

Supported terminal cleanup retained the failed outputs and recorded zero run-owned Kubernetes objects, zero exact-image pods and an absent local API process. Earlier manual reconciliation of an unrelated no-launch journal is explicitly excluded from product recovery acceptance.

Source review passed with 301 focused workflow, storage and execution-preflight tests; the original focused suite also passed 151 tests. The broad local run recorded 24,949 passes and four failures caused by missing Torch. The exact four passed after installing that dependency; the original failed log remains retained. Required hosted CI was pending when this pack was written, so it does not declare the PR merge-ready.

A separate visual evaluation is not applicable to this host configuration change. The live GPU job is evidence of actual submission, execution and durable transport; GPU algorithm correctness belongs to the cuRobo PR. Operational identifiers, endpoints, credentials and raw logs remain private.
