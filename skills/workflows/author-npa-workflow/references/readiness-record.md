# Workflow readiness record

Use this record when saving or reviewing a workflow. It records authoring results
and execution prerequisites separately. It is a documentation convention, not an
NPA schema or an authorization to execute. Do not add these fields to workflow YAML.

For a saved workflow, copy the JSON below to `<workflow-stem>.readiness.json`.
Set `workflow_sha256` to the SHA-256 of the final workflow file and refresh it
after any edit. Reassess affected statuses and evidence too; updating the hash
alone does not revalidate the workflow. For read-only review, include the record
in the response instead.
Do not create a file outside the user's authorized write scope.

For each entry, supply a status, a reason, and evidence references:

- `verified`: an actual check supports the claim; cite its result in `evidence`.
- `unverified`: no sufficient check was performed; explain what remains unknown.
- `blocked`: a known obstacle prevents the operation; explain the obstacle.
- `not_applicable`: explain why the prerequisite does not apply. A prohibited or
  unperformed check is `unverified`, not `not_applicable`.

Apply the same statuses to planning. `validation` covers schema validation and
planning commands; `task_fidelity` covers whether resolved operations, inputs,
and outputs match the requested work. A valid plan can still perform the wrong task.
Planning checks cannot be `not_applicable`; leave unperformed checks `unverified`.

Keep every entry. `source_image` covers the worker's code/image selection;
`target_runtime` covers the worker environment and required runtime, not the
authoring machine's CLI version. `credentials` includes applicable model access.
Use non-secret evidence references; never include tokens or credential contents.
Empty reasons and missing evidence for `verified` claims are incomplete records.
Missing checks remain unverified without launching extra checks or recovering access.

```json
{
  "schema_version": "workflow-readiness/v1",
  "workflow_sha256": "",
  "planning": {
    "validation": {"status": "unverified", "reason": "", "evidence": []},
    "task_fidelity": {"status": "unverified", "reason": "", "evidence": []}
  },
  "prerequisites": {
    "output_storage": {"status": "unverified", "reason": "", "evidence": []},
    "worker_input": {"status": "unverified", "reason": "", "evidence": []},
    "credentials": {"status": "unverified", "reason": "", "evidence": []},
    "source_image": {"status": "unverified", "reason": "", "evidence": []},
    "target_runtime": {"status": "unverified", "reason": "", "evidence": []}
  }
}
```

A complete record can contain unverified or blocked prerequisites. It does not
prove the evidence is accurate or current, authorize submission, or replace
independent execution checks. Keep the final response concise: link the saved
record, distinguish planning completion from execution readiness, and summarize
the unresolved prerequisites. Do not claim a generated result from planning alone.
