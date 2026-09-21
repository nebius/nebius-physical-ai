# PR654 test-only repair: production source equivalence

The independent review accepted commit `09d0fa9f3693db57142b82d7d7caf192b72257e0`, a direct child of live CPU producer `0666db15f7c5a715ceba74b9dbfa57fe93054e26`. Only two test files change. All other tracked files, including production and deployment code, are identical. The accompanying JSON supplies explicit SHA256 comparisons for the four relevant production modules.

The updated tests exercise subprocess timeout translation, owned-process abandonment, breaker/reaper state, closed pipes, and both rendered HTTP502 timeout responses. Independent validation passed 23 scoped tests, Ruff check, Ruff format check, and diff check. Two existing FastAPI lifecycle deprecation warnings remain.

The original CPU evidence remains applicable by source equivalence. This is not a new live deployment run of the repaired commit. Fresh exact-head CI and mergeability must still pass before readiness.

Original immutable live proof: [CPU proof and limitations](https://github.com/nebius/nebius-physical-ai/blob/faf4e8402832d1803eb746c00db02ff1cee91d84/docs/testing/evidence/agent-metadata-cpu-0666db15/README.md).

Retained limitations: deployment source identity was bound, but separate deployed Python byte hashes were not retained. The anonymous negative probe covers the UI root. Cloud absence relies on retained production teardown evidence; the independent reviewer did not query the provider. Live happy-path CPU transport and separate real Linux process fault controls prove different boundaries. GPU/VLM testing does not apply to this maintenance change.

A reviewer with a repository checkout can verify the two-file delta with:

```sh
git diff --name-status 0666db15f7c5a715ceba74b9dbfa57fe93054e26 09d0fa9f3693db57142b82d7d7caf192b72257e0
git diff --exit-code 0666db15f7c5a715ceba74b9dbfa57fe93054e26 09d0fa9f3693db57142b82d7d7caf192b72257e0 -- . ':(exclude)npa/tests/**'
```

Independent public review also identified two nonblocking test gaps: explicit `WNOWAIT` ownership flags and conservative handling of `/proc` read errors are not directly asserted. Those behaviors are not newly proved by these 23 tests. See [review](https://github.com/nebius/nebius-physical-ai/pull/654#issuecomment-5755227400) and [amended-delta review](https://github.com/nebius/nebius-physical-ai/pull/654#issuecomment-5755287940).
