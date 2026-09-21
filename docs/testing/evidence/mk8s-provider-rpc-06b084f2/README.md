# Real CPU-only MK8s lifecycle with provider RPC deadlines

Exact source: `06b084f2db1870c6c258ce260e39e1e8379fa56b` ([PR #692](https://github.com/nebius/nebius-physical-ai/pull/692)).

One newly owned CPU-only cluster was provisioned through `npa cluster up`, passed the normal Ready-node/default-storage checks, passed the committed live verifier, and was destroyed through `npa cluster down`. Fresh independent reads then confirmed all four recorded resources absent, complete project/child inventories, removal of the local context, and release of the project lease. No GPU, model, or SkyPilot workload ran in this experiment.

The actual materialized provider contained `timeout`, `per_retry_timeout`, and `auth_timeout` at `120m`, derived from the existing apply deadline. Their bytes, the producing source/command/result, Terraform state, and credential authority were bound before the read-only live check. Both committed live tests passed, one before and one after destroy. The JSON records the observed elapsed times and retained evidence hashes.

This is a real supported lifecycle, not a forced-timeout benchmark or an A/B comparison with the previous source. It does not prove a throughput improvement, GPU capability, or which individual deadline was necessary. Raw operational identifiers, credential selectors, private configuration, and provider responses remain in the access-controlled evidence packet.
