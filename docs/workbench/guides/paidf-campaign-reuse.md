<!-- register: operator runbook | reader: PAIDF workshop authors and provider-integration contributors | consumed: workshop preparation and derivative-run execution -->
# Reuse one PAIDF campaign across provider workshops

Build the expensive source, Cosmos generation, and evaluation stages once. Freeze their exact manifests as a base campaign. Each Encord, FiftyOne, Foxglove, Rerun, or simulation-provider workshop then becomes a derivative run that reads the base and writes to its own prefix.

This is a small contract layer, not a provider plugin system. Provider execution remains in the relevant SDK, CLI, or workflow code.

## The two run types

A base campaign contains three required artifact references:

- `source`: authoritative trajectories and their robot, task, timing, action, and state contracts
- `generation`: Cosmos outputs, prompts, seeds, model identity, and source lineage
- `evaluation`: deterministic and model-assisted quality evidence

Every reference includes an exact S3 object URI, its SHA-256 digest, and its payload schema. `stage_fingerprints` provide one content identity per base stage, and every artifact digest must equal its stage fingerprint. A campaign with an extra, missing, or divergent base stage fails validation. Regeneration creates a new base campaign instead of silently changing an existing one.

A partner run contains:

- the base campaign ID, manifest URI, and canonical digest
- the provider name and one role: `curation`, `observability`, or `simulation`
- exact base stages and artifacts reused read-only
- overlay stages that the partner run will execute
- a hashed provider configuration artifact
- credential references such as `secret://antioch/workshop`, never credential values
- a separate S3 output prefix

The contract rejects partner inputs that execute `source`, `generation`, or `evaluation`.

## Common envelopes

All partner integrations use the same three envelopes:

| Envelope | Purpose |
|---|---|
| `npa.paidf.partner-input.v1` | Pins the base campaign, provider configuration, read set, planned overlay stages, and write prefix. |
| `npa.paidf.execution-receipt.v1` | Accounts for every declared overlay stage and records a terminal status without claiming that the base changed. |
| `npa.paidf.partner-result.v1` | Links terminal artifacts to the exact input and receipt digests and requires every output to remain under the derivative prefix. |

The implementation is in `npa.workflows.paidf_campaign`. It is dependency-light and does not create cloud or provider resources.

## Decision contracts

Curation outputs flow through three provider-neutral artifacts, so no dataset identity couples to a specific provider:

- `npa.paidf.candidates.v1`: one row per generated variant, each pinned by `candidate_id`, source episode and camera identity, and the exact output/source SHA-256 digests.
- `npa.paidf.decisions.v1`: one `accept`, `reject`, or `review` per candidate. The provider object is exactly `name` plus a `curation` or `evaluation` role, and each decision carries exactly `candidate_id`, `decision`, `evidence`, `reason`, and a caller-supplied `decided_at` timestamp in canonical UTC-second form (`YYYY-MM-DDTHH:MM:SSZ`). Canonical builders never read the wall clock.
- `npa.paidf.reconciliation.v1`: the authoritative keep/drop manifest. `accept` enters `keep`; `reject` and `review` enter `drop`. `review` is a valid but unresolved state: reconciliation reports its count and IDs explicitly, and it never enters accepted replacements.

Provider exports reconcile to the candidate manifest by exact external ID, and every terminal export row must carry a non-null `accept`, `reject`, or `review` decision. The accepted-replacement manifest is derived only after the reconciliation is proven to cover every candidate exactly once with consistent states.

## Object-storage layout

```text
paidf/
  campaigns/
    <campaign-id>/
      campaign.json
      source/
      generation/
      evaluation/
  derivatives/
    encord/<run-id>/
    fiftyone/<run-id>/
    foxglove/<run-id>/
    rerun/<run-id>/
    antioch/<run-id>/
```

The directories illustrate ownership. The manifests contain exact object references, so a consumer does not infer authority from path names.

## Antioch derivative run

Antioch is a simulation overlay, not a curation adapter. Its first workshop should reuse the base robot, task, scenario, and evaluation contracts while producing new provider-specific simulation evidence.

```python
from npa.workflows.paidf_campaign import build_antioch_input

partner_input = build_antioch_input(
    campaign,
    campaign_manifest_uri="s3://<bucket>/paidf/campaigns/<campaign-id>/campaign.json",
    run_id="<antioch-run-id>",
    provider_config={
        "uri": "s3://<bucket>/paidf/config/antioch.json",
        "sha256": "<64-lowercase-hex-characters>",
        "schema": "npa.paidf.antioch-config.v1",
    },
    credential_refs=("secret://antioch/workshop",),
    output_prefix="s3://<bucket>/paidf/derivatives/antioch/<antioch-run-id>/",
)
```

The generated plan runs only:

1. `partner-simulation`
2. `result-normalization`
3. `comparison`

It does not invoke Cosmos or replace the base source, generation, or evaluation manifests. Provider outputs should normalize into JSON, RRD, MCAP, or object references that Workbench can inspect without hidden provider state.

Antioch's exact production API, authentication names, artifact schemas, and training-ready episode export remain discovery items. Keep those details inside the Antioch configuration and executor until live evidence proves a stable reusable boundary.

## Workshop execution rule

Start a derivative workshop with:

1. one base campaign manifest URI
2. one provider configuration artifact
3. any required secret references
4. one new derivative output prefix

The workshop is repeatable when a clean run produces validated input, receipt, and result envelopes; reads the same base hashes; accounts for every declared stage; and writes nothing outside its derivative prefix.

For Encord and FiftyOne, the overlay payload contains curation decisions. For Foxglove and Rerun, it contains observability indexes and recordings. For Antioch or another simulator, it contains rollout, telemetry, and comparison evidence. The envelope stays stable while the role-specific payload changes.

## Deliberate limits

Do not add a provider registry, plugin loader, cache service, data-versioning platform, or campaign-management daemon for the first two integrations. Implement the second provider against this envelope, compare the two real payloads, and extract another abstraction only when both require the same behavior.

Do not add a workflow specification until its provider path can satisfy the repository's live execution and artifact proof requirements. The contract module and runbook support preparatory and fixture-based work without presenting it as a live provider integration.
