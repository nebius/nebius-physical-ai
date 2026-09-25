# Jev model routing with Token Factory

To turn model routing into a synthetic-data pipeline, use
[Token Factory SDG](token-factory-sdg.md). It adds seed classification, generation,
review, and dataset export. Its default router also runs on Token Factory;
`--router jev` retains the TypeSafe option described here.

The agent can use TypeSafe Jev to choose between the eligible Token Factory
text models before generating an answer. This is an opt-in experimental path.
It keeps grounded answers, explicit model choices, vision requests, and tool
permissions under their existing controls.

The initial candidates are `nvidia/Nemotron-3_5-Lightning` for straightforward
text tasks and `MiniMaxAI/MiniMax-M3` for reasoning. Both must survive the
operator's model configuration and account-scoped availability checks. Custom
models retain the existing routing behavior. This integration calls TypeSafe's
HTTP API directly and adds no LangChain dependency.

## Enable the agent integration

Provide `TYPESAFE_API_KEY` through your private environment, or save it under
`tokens.TYPESAFE_API_KEY` in the existing owner-only NPA credential file.
`npa configure --save-env-credentials` can persist the environment value.
The key is required to enable deployment configuration and has no default.
Keep the normal `NEBIUS_TOKEN_FACTORY_KEY` configuration for generation.

Set `NPA_AGENT_MODEL_ROUTER=jev` in the environment when running the normal
agent deploy or bootstrap command for your selected agent. Its default is
unset, which keeps deterministic model-tier routing. The optional router
supports the Token Factory provider only. The bootstrap transfers the key and
switch through the existing private SFTP staging path into the service's
mode-0600 `llm.env`; neither belongs in command arguments or source control.
Repeat the setting when re-bootstrapping to retain the opt-in. To disable it,
unset the switch and bootstrap the same agent through its normal procedure.
See [agent deployment](../agent.md) for selecting and managing that agent.

Enabling the router permits sending the latest user text to TypeSafe. The
existing redactor removes known credentials and infrastructure references first;
it is not a guarantee that arbitrary proprietary prose is safe to share. System
prompts, tool results, and the full conversation are not sent to the classifier.
Use this feature only with inputs permitted for that provider.

The classifier uses pinned `jev-1.13.0` and question revision
`npa-generation-model-v1`. A `Choice` answer must name an eligible model, contain
a complete finite probability distribution, and meet the experimental confidence
threshold of 0.8. That threshold is a starting configuration, not a measured
NPA accuracy guarantee. The Python helper exposes `min_confidence` for evaluation.
Low confidence, `none`, missing credentials at runtime, invalid responses, and
provider failures preserve the existing generation ladder. If the selected
model's generation fails, the existing provider/model fallback still runs.

The final text-generation path performs this routing once. The earlier semantic
intent classifier and the internal action loop retain their existing behavior.
Choosing a reasoning model enables its existing reasoning-tier parameters. The
original generation messages are passed through unchanged, preserving their
shared prefixes for provider caching.

## Inspect routing and cache evidence

The agent's chat response includes `model_routing` when classification is
attempted. It separates the selected model from the model that actually served
the response, and retains classifier status, probabilities, confidence, usage,
latency, and question revision. Classifier tokens remain separate from Token
Factory usage because they belong to different providers.

The existing `usage` object now retains `cached_tokens` when the provider reports
a valid count, including a reported zero. It also retains Token Factory's
`prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` fields. Missing counters
remain missing. A latency improvement alone does not prove a cache hit.

Token Factory documents the nested cached-token field in its
[chat API](https://docs.tokenfactory.nebius.com/api-reference/inference/create-chat-completion)
and KV-cache hit rate in its
[observability metrics](https://docs.tokenfactory.nebius.com/ai-models-inference/observability).
The counters are provider-reported evidence of reused prompt computation; this
client cannot inspect the provider's internal KV tensors. There is no local
response cache in the router or the evaluation harness. Jev's API does not expose
a KV-cache counter in the inspected contract, so this evidence concerns Token
Factory generation, not Jev's inference internals.

## Run the live evaluation

From a contributor checkout with its own `npa/.venv`, first verify the Token
Factory credential and current model access:

```bash
npa workbench health preflight --checks token_factory --json
npa workbench token-factory models
```

The developer harness uses public synthetic text only. It interleaves requests
to both models with one shared prefix, changing the user suffix each time.
Every request reaches the provider. It records model identities, input/output
usage, cache counters, transport attempts, latency, request/response hashes,
and whether the requested output marker appeared. A fresh random prefix
namespace distinguishes each experiment. Output hashes and distinct suffixes
separate prefix reuse from reusing a previous answer.

For the full Jev-to-Token-Factory experiment, configure both credentials and run:

```bash
npa/.venv/bin/python npa/scripts/evaluate_jev_routing.py \
  --output-path /tmp/jev-routing-evidence.json
```

Missing Jev credentials fail before inference. A pass requires accepted Jev
routes matching both scenario labels, valid generation responses, and reported
cache hits on later requests to each model. This small contract experiment
proves an exercised path, not comparative routing quality or calibration.

The independently useful Token Factory-only probe is explicit:

```bash
npa/.venv/bin/python npa/scripts/evaluate_jev_routing.py \
  --token-factory-only --output-path /tmp/token-factory-cache-evidence.json
```

That mode does not claim Jev routing evidence. The output contains
`mode: token_factory_only`. The harness exits nonzero if the required counters
are missing or positive warm hits are not observed; the report is still written
for inspection. It does not retry the experiment until a favorable result appears.
No GPU resources are allocated. Remove the local report when no longer needed.

The same experiment is registered as opt-in live pytest coverage:

```bash
NPA_JEV_ROUTING_LIVE=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_jev_token_factory_live.py \
  --basetemp /tmp/jev-routing-live-tests -q
```

`NPA_JEV_ROUTING_LIVE` defaults to unset, so CI does not spend provider tokens.
Setting it to `1` requires real credentials and makes missing access a failure.
Use `-k token_factory_cache` to run just the independent Token Factory proof.
Use `-k token_factory_rendered_fallback` to exercise the real rendered `/chat`
service with live Token Factory and an intentionally absent Jev key. The full
suite also requires real accepted Jev choices through that service. It starts an
isolated loopback backend with synthetic state and does not deploy a VM.

## Recorded Token Factory proof

The [sanitized JSON evidence](../architecture/evidence/jev-token-factory-cache.json)
records a real hosted run on 2026-09-20 UTC:

| Model | Prompt tokens across three requests | Reported cached tokens | Total request seconds |
| --- | --- | --- | --- |
| Nemotron-3.5-Lightning | 5,484 / 5,484 / 5,485 | 0 / 2,096 / 2,096 | 0.872 / 0.821 / 0.850 |
| MiniMax-M3 | 5,259 / 5,259 / 5,260 | 0 / 5,120 / 5,120 | 2.404 / 2.216 / 2.415 |

All six requests returned the requested model, a nontruncated answer, and the
requested marker. These are individual measurements, not latency percentiles
or an estimate of billed cache savings. The public catalog at measurement time
listed input/output USD per million tokens of 0.06/0.24 for Lightning and
0.30/1.20 for MiniMax. The report retains that source and price context; it does
not infer an undocumented cache discount.

The [rendered-agent evidence](../architecture/evidence/jev-token-factory-agent.json)
also records successful live `/chat` answers from both models. With the router
enabled and its key absent, both turns reported `missing_credential` and used
the existing tier selection. The MiniMax response exposed 3,328 cached prompt
tokens through the agent API. That measurement validates telemetry propagation;
the isolated-prefix experiment above is the deliberate cache-reuse test.

Live Jev inference remains unverified because the evaluation environment had no
TypeSafe key. Jev transport, malformed-response handling, privacy redaction,
and rendered-agent wiring have hermetic coverage. No remote VM was destroyed
or redeployed for this evidence. See the
[integration assessment](../architecture/jev-workbench-evaluation.md) for the
remaining quality evaluation and related developer uses.
