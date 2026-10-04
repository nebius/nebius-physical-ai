# Model usage and estimated cost

`../workflow_usage.py` reads a completed experiment's `usage.json` and
`execution.json`. It counts every retained coordinator turn and specialist
response, including rejected generations and fallback models. It does not call
models, inspect live tasks, or change the experiment receipts.

```bash
npa/.venv/bin/python npa/examples/specialists/workflow_usage.py \
  --evidence "$COMPLETED_EXPERIMENT" --prices "$OPERATOR_PRICES" \
  --output "$NEW_USAGE_REPORT"
```

Omit `--output` to print JSON. An existing output file is never overwritten.
`--coordinator-model` defaults to `gpt-6-astra`; set it to the actual identifier
when using the function with other coordinator evidence. The runner currently
uses Astra. Do not summarize an active experiment as if its receipts were final.

Python callers can import the standalone module and call
`summarize_usage(usage, execution, prices, coordinator_model=...)` with the
three decoded JSON objects. It returns the same report and raises `ValueError`
for invalid token counters or price configuration.

## Supply the applicable prices

The operator owns the price table. Check provider terms for the exact endpoint,
model, service tier and date, then retain the sources with the private evidence.
No provider prices or cache discounts are built into the script. Its arithmetic
follows the simulation scorer's input/cache-read/cache-write/output accounting;
the independent script needs none of that scorer's physics dependencies.

This example uses **illustrative rates**, not quotes for a real model. Replace
the model keys and rates with the actual model IDs and applicable prices:

```json
{
  "models": {
    "coordinator-model": {
      "cache_policy": "itemized",
      "rate_options": [
        {
          "input_price_per_million_tokens": 2,
          "output_price_per_million_tokens": 4,
          "cached_input_price_per_million_tokens": 1,
          "cache_write_price_per_million_tokens": 3
        },
        {
          "input_price_per_million_tokens": 4,
          "output_price_per_million_tokens": 8,
          "cached_input_price_per_million_tokens": 2,
          "cache_write_price_per_million_tokens": 6
        }
      ]
    },
    "specialist-model": {
      "cache_policy": "full-input",
      "rate_options": [
        {
          "input_price_per_million_tokens": 1,
          "output_price_per_million_tokens": 2
        }
      ]
    }
  }
}
```

Each model must explicitly select a cache policy:

- `full-input` charges every input token at the input rate. Use this when the
  endpoint's applicable tariff does not discount cached tokens. A reported
  cache hit alone does not justify a discount. Cache counters remain visible.
- `itemized` subtracts cache-read and cache-write tokens from ordinary input,
  then prices each category separately. Missing cache counters remain unknown;
  the estimator considers every feasible partition of the known input count.
  A needed but missing cache rate makes that record's price unknown. A known
  zero count does not need a corresponding rate.

`rate_options` must list **all possible applicable per-request tariffs**. When
the evidence proves a single tariff applies, supply just that one. When an
aggregate Codex turn does not reveal individual request context lengths,
supply both ordinary and long-context tariffs if either could apply. The
script never uses the aggregate turn token count as one request's length.
It returns bounds across the supplied rates; it cannot prove that the operator
included every applicable tariff. Reasoning tokens are a subset of output
tokens and are never charged twice.

## Read the report

`models` provides per-model record counts, accepted/rejected specialist response
counts and normalized input/output/cache/reasoning counters. A Codex turn may
contain multiple API requests, so its `recorded_api_calls` is `null`.
Specialist records each represent one observed provider response. Attempts
that never return provider usage cannot be reconstructed or assigned a cost.

`tokens` is `null` for any counter missing from one or more records.
`known_tokens` adds only reported values; `missing_counter_records` explains
those partial sums. An unknown cache count is never presented as a measured
zero. The recorder's incomplete flags, required missing counters, interrupted
execution and snapshot errors prevent a complete-run cost estimate.

The explicit `specialists-first` mode can complete with zero coordinator turns.
That is accepted only when the execution mode, protocol and invocation journal
agree, the recorder confirms no contradictory Codex events, and the usage
snapshot is complete. Missing coordinator logs in other modes remain unknown.
Every specialist response and any actual Astra escalation still enters the
estimate; zero coordinator usage does not establish task success by itself.

`priced_recorded_cost_usd` gives bounds for the subset of records with enough
usage and pricing information; inspect `unpriced_records` beside it.
`estimated_cost_range_usd` is populated only when the recorded usage is complete
and every record can be priced. `estimated_cost_usd` is populated only when
those lower and upper bounds coincide. These are model-inference estimates,
not invoices or total workload costs. GPU jobs, host compute, storage, network
and human work are excluded and need independent accounting.
`price_table_sha256` identifies the canonical JSON price table; retain that
table and its provider sources alongside the report.

The report preserves the matched-tool-scope flag, unfinished-task count and
coordinator/full local shutdown timing. It never grades workflow success,
artifact quality, reliability, or whether either experiment arm is better.

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/agent_eval/test_specialist_workflow_usage.py -q
```
