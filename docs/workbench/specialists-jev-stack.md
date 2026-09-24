# Jev + LangGraph + Token Factory specialists

Jev selects a model from an operator-defined endpoint list. LangGraph runs the
specialist's durable edit, command and verification loop. Token Factory hosts
the selected model, and Workbench supplies scoped files, named operations and
completion receipts. Astra can handle recovery after an accepted route when a
specialist cannot complete the assignment.

## Configure required routing

Install `npa[agent-specialists]` and follow the
[specialist configuration guide](specialists.md) for workspaces and operations.
Configure `NEBIUS_TOKEN_FACTORY_KEY` and `TYPESAFE_API_KEY` in the execution
host's private NPA credential store or environment. Jev uses its public TypeSafe
API; the LangGraph runtime and Workbench run on your host. Credentials are not
part of the team configuration.

Merge this fragment into each operator-owned profile, retaining its existing
workspace, instructions, operations and required checks:

```json
{
  "model": "zai-org/GLM-5.3-Flash",
  "model_options": {
    "chat_template_kwargs": {"reasoning_effort": "low"}
  },
  "fallback_models": [
    {
      "model": "zai-org/GLM-5.3",
      "model_options": {
        "chat_template_kwargs": {"reasoning_effort": "low"}
      }
    }
  ],
  "model_router": "jev",
  "require_model_route": true,
  "model_criteria": {
    "zai-org/GLM-5.3-Flash": "Localized workflow repairs with explicit validation and a clear defect",
    "zai-org/GLM-5.3": "Ambiguous cross-component diagnosis requiring deeper reasoning"
  },
  "compact_context": true
}
```

These are candidate routing criteria, not a measured quality guarantee. Confirm
model availability through Token Factory's authenticated model catalog. You can
substitute other endpoints and criteria, including DeepSeek, without changing
the worker grants. An explicit workspace assignment still invokes this model
router. Team-level `router: "jev"` alone does not enforce endpoint selection.

For predefined assignments with required operations, run:

```bash
umask 077
npa/.venv/bin/python npa/examples/specialists/workflows/experiment.py --live \
  --team-config "$HYBRID_TEAM_CONFIG" --prompt "$COMMON_TASK_PROMPT" \
  --output "$HYBRID_EVIDENCE" --arm astra-tofa \
  --coordination specialists-first --effort medium
```

Use fresh state, evidence and workspaces. A missing key stops preflight before
inference. Rejected or unavailable Jev routing stops the experiment before
specialist generation or Astra recovery. An accepted choice and every later
fallback remain in `route.model_selection` and the model events. Successful
required operations finish without invoking Astra; worker failures after an
accepted route can request Astra recovery. The default configuration remains
compatible with explicit endpoints and advisory Jev routing.

## Prove the complete path

The live check injects a bad transition into a copied public PAIDF Cosmos3
workflow. A Jev-selected Token Factory model must edit it back to the original
and pass real Workbench `validate-spec` and `plan-spec` operations through
LangGraph. It uses no GPU and does not submit the data-generation pipeline.

```bash
umask 077
NPA_JEV_ROUTING_LIVE=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_jev_token_factory_live.py \
  -k jev_langgraph_token_factory_repairs_workflow \
  --basetemp "$JEV_TEST_EVIDENCE" -q
```

Choose a new private `JEV_TEST_EVIDENCE` directory: pytest replaces an existing
base directory. The task receipt under that directory must show an accepted,
attempted Jev decision, positive routing usage, an actual generation from the
selected model, a recorded file edit and passing required operations. Hermetic
tests exercise this contract but do not establish live provider access.

Live Jev remains unverified in the current evaluation environment because its
TypeSafe credential is absent. The earlier
[model-selection experiment](specialists-model-selection-experiment.md) used
LangGraph and Token Factory with explicit endpoints. Its savings do not include
Jev. A new comparison must count Jev latency and usage, all generation and Astra
recovery calls, and unsuccessful attempts. TypeSafe currently lists Jev at
[$0.042 per million input tokens, with output free](https://docs.typesafe.ai/models);
retain the pricing date and actual reported usage in any cost estimate. Provider
cache counters establish reuse; they do not establish an undocumented billing
discount. This stack requires no private orchestration repositories.
