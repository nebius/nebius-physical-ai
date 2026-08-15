---
name: audit-container-docs
description: Audit and update NPA container-image catalogs and related container documentation when images are added, removed, renamed, retagged, republished, reclassified, or materially changed. Use for docs/workbench/container-image-catalog.md drift, public-mirror inventory checks, and container documentation reviews.
---

# Audit Container Documentation

Separate repository intent from registry state. An image can have a Dockerfile,
be eligible for redistribution, or exist in a private registry without being in
the public mirror inventory. State only the layer that the evidence proves.

## Read The Authoritative Sources

Read `AGENTS.md`, `skills/index.yaml`, and the applicable image/tool skills first.
Use these sources for distinct facts:

- `npa/docker/workbench/packaging-contract.yaml`: built-image inventory,
  packaging tier, ports, security exceptions, and redistribution eligibility.
- `npa/src/npa/deploy/images.py`: canonical tool-to-image names, restricted
  tools, public-mirror membership, and tag-resolution exceptions.
- `npa/pyproject.toml` `[tool.npa.supported-tools]`: default immutable image pins.
- `npa/src/npa/deploy/*_image_manifest.json`: variant-specific pins and evidence
  for SONIC, Wan, LeRobot, and any future manifest-backed tool.
- `npa/docker/workbench/<image>/Dockerfile*`, build scripts, lock files, and
  redistribution records: what an image actually contains and does.
- `npa/src/npa/deploy/publish_public.py`: exact public source-to-target plan.
- Golden evaluations, CLI/SDK code, and workflow specs: capability and runtime
  integration. Do not infer capabilities from an image name alone.
- Anonymous registry manifest/config inspection: what is publicly pullable now.
  Treat this as volatile observed state, not a replacement for repository intent.

Do not treat every Dockerfile or every `redistribution: public` entry as a public
catalog row. `public` means eligible to redistribute; public-mirror membership is
the intersection selected by `publicly_publishable_tools()` and its resolved pin.

## Run The Audit

1. Preserve unrelated dirty-tree changes and inspect the diff from the relevant
   base. Search recent changes to Dockerfiles, packaging contracts, image maps,
   manifests, supported-tool pins, workflows, and container docs.
2. Print the repository's current public plan from the installed checkout:

   ```bash
   npa/.venv/bin/python - <<'PY'
   from npa.deploy.images import CONTAINER_IMAGE_NAMES, public_mirror_tag_for_tool, publicly_publishable_tools

   for tool in publicly_publishable_tools():
       print(tool, CONTAINER_IMAGE_NAMES[tool], public_mirror_tag_for_tool(tool), sep="\t")
   PY
   ```

3. Compare that output with the public catalog. Require one row per public-plan
   image and require each row to include its current resolved pin. Keep additional
   aliases or historical tags only when the registry still resolves them and the
   text explains why they remain useful.
4. Compare the packaging-contract keys with Dockerfile directories and the public
   plan. Classify differences: internal/derived variants, restricted build-your-own
   images, unregistered artifacts, or actual drift. Never silently force the sets
   to match.
5. Verify every documented public reference anonymously. Prefer the complete
   publisher check for current pins, then inspect any extra documented tags:

   ```bash
   npa/.venv/bin/python -m npa.deploy.publish_public \
     --target ghcr.io/nebius/nebius-physical-ai --verify-public
   docker buildx imagetools inspect \
     ghcr.io/nebius/nebius-physical-ai/<image>:<tag>
   ```

   Record the UTC verification date only after all retained rows resolve. A 401 or
   403 is not proof that a tag is absent; it proves anonymous pull was not verified.
   Describe that limitation precisely.
6. Inspect OCI config and labels before changing build dates, users, entrypoints,
   ports, digests, or architecture claims. Use an immutable tag timestamp or
   `npa.build_ts` only when the reproducible image intentionally clears `created`.
7. Update affected documentation together: the public catalog, container
   packaging/security docs, tool skills, workflow docs, and release records only
   where the changed source or observed registry state supplies evidence. Preserve
   the distinction among built, supported, publishable, published, and validated.

## Validate The Result

Run the skill validator, catalog/index guardrails, and documentation drift check:

```bash
npa/.venv/bin/python \
  ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/atomic/audit-container-docs
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_audit_container_docs_skill.py \
  npa/tests/guardrails/test_skills_index.py -q
PATH="$PWD/npa/.venv/bin:$PATH" scripts/build_docs.sh --check
```

When container sources or resolver code changed, also run the affected Docker,
deploy, golden-eval, and workflow tests plus the full non-E2E suite required by
`skills/atomic/testing-conventions/SKILL.md`. Documentation must report registry
checks numerically and must not claim live capability validation from unit tests.
