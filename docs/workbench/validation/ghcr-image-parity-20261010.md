# GHCR image parity audit — 2026-10-10

[Public container catalog](../container-image-catalog.md) · [Machine-readable observations](ghcr-image-parity-20261010.json)

Audited `main` at `e0819bd13c8081d77cea4918367d4d0f10487e7f` in an isolated checkout. Observations were completed at `2026-10-10T13:38:40.307213+00:00`. All registry requests used anonymous GHCR pull tokens without Docker, GitHub, or operator credentials.

The supported release inventory has registry parity: all 27 selected tags match their accepted digests. Full source parity with the audited commit is incomplete. Eighteen accepted images have changed declared build inputs, and nine do not record an NPA producer revision. Fourteen of the 52 eligible development references at the audited commit resolve anonymously and carry that exact OCI revision label; 37 return manifest HTTP 404, and NCore returns anonymous-token HTTP 403. Development availability does not promote an image to a supported release.

## Registry and chart results

| Population | Checked | Result |
| --- | ---: | --- |
| Current supported releases | 27 | All anonymous manifest/config reads passed; all digests match acceptance |
| Additional aliases in the public table | 3 | All resolve anonymously; aliases retain their historical scope |
| Quarantined historical releases | 12 | All resolve and match their earlier recorded digests; quarantine remains in force |
| Repaired PAIDF workflow candidates | 3 | All resolve and match their recorded candidate digests; no supported-release promotion |
| Other explicit documented historical/development references | 10 | All resolve anonymously; no new capability claim |
| Development images at audited main | 52 | 14 verified, 37 manifest HTTP 404, one anonymous-token HTTP 403 |
| Accepted-image architecture | 27 | Every inspected runtime is `linux/amd64` |
| GPU coverage chart | 27 images / 135 cells | Matches the current release plan and compatibility matrix |

The existing SVG is current: 15 GPU images have no known blocked platform, six have at least one blocked platform, and six are CPU-only. The renderer comparison ignores its display-only rendering date. No compatibility verdict or SVG regeneration was necessary. The catalog verification date and build-date ordering were refreshed.

## Accepted releases compared with source

The comparison uses the accepted release manifest, the accepted Content Agents implementation manifest, and the exact canonical Dockerfile at the producer and audited commits. It compares tracked COPY/ADD input paths and files in the recipe directory. Counts describe changed repository files, not changed image layers. Content Agents uses its NPA implementation revision rather than its upstream NVIDIA source revision.

| Image | NPA producer | Changed copied inputs | Changed recipe files | Development image at audited main |
| --- | --- | ---: | ---: | --- |
| `npa-alpamayo2-super` | `5b693476c113` | 1178 | 2 | Manifest HTTP 404 |
| `npa-content-agents` | `36dd0578a9b1` | 3 | 0 | Manifest HTTP 404 |
| `npa-cosmos` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-cosmos2-transfer` | `c164fd3480f8` | 1317 | 4 | Manifest HTTP 404 |
| `npa-cosmos3-reason` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-cosmos3-serving` | `d854f6a76cd8` | 6 | 8 | Manifest HTTP 404 |
| `npa-detection-training` | `408700158b2e` | 14 | 1 | Verified |
| `npa-diffusers` | `d54eec137d3b` | 4 | 1 | Manifest HTTP 404 |
| `npa-envgen` | `c164fd3480f8` | 1319 | 1 | Manifest HTTP 404 |
| `npa-fiftyone` | `17e3c9e82cee` | 4 | 5 | Manifest HTTP 404 |
| `npa-flex-pi` | `8904daf36d0c` | 1170 | 2 | Manifest HTTP 404 |
| `npa-foxglove-embed` | Unrecorded | Unproven | Unproven | Verified |
| `npa-groot` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-lancedb` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-leisaac` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-lerobot-policy` | Unrecorded | Unproven | Unproven | Manifest HTTP 404 |
| `npa-lichtblick` | Unrecorded | Unproven | Unproven | Verified |
| `npa-lingbot-world` | `d54eec137d3b` | 4 | 1 | Manifest HTTP 404 |
| `npa-ltx2` | `072952df62d2` | 3 | 4 | Verified |
| `npa-lyra2` | `51be9e814502` | 51 | 1 | Manifest HTTP 404 |
| `npa-openarm` | `01fbf3a554cb` | 1246 | 5 | Manifest HTTP 404 |
| `npa-rerun-viewer` | `c164fd3480f8` | 1320 | 1 | Verified |
| `npa-retargeting` | Unrecorded | Unproven | Unproven | Verified |
| `npa-sam2` | `d54eec137d3b` | 4 | 1 | Manifest HTTP 404 |
| `npa-sim2real-control` | `c164fd3480f8` | 1319 | 2 | Verified |
| `npa-sonic-mujoco` | `5b5b5e69e9e6` | 7 | 11 | Manifest HTTP 404 |
| `npa-wan2-2` | `bc18684d35fa` | 3 | 3 | Verified |

No accepted image with a recorded NPA producer has unchanged declared inputs relative to this commit. In particular, the accepted Lyra rtfetch2 image remains tied to its October 9 producer and RTX qualification; later source changes do not inherit that qualification. The nine unrecorded producers require new provenance or direct artifact/source comparison before source parity can be asserted.

## Development inventory

The 52-reference population is the trusted builder’s scheduled inventory: redistribution-eligible packaging entries that have a mapped tool, plus the separately quarantined Gymnasium-Robotics bootstrap. It excludes restricted images and the duplicate `sim2real-eval` packaging role. The development tag checked for every member was `dev-e0819bd13c8081d77cea4918367d4d0f10487e7f`.

| Anonymous result | Images |
| --- | --- |
| Verified exact main revision | `npa-cosmos-curate`, `npa-cosmos-evaluator`, `npa-detection-training`, `npa-foxglove-embed`, `npa-gymnasium-robotics`, `npa-habitat-sim`, `npa-lichtblick`, `npa-ltx2`, `npa-open3d`, `npa-rerun-viewer`, `npa-retargeting`, `npa-sam3`, `npa-sim2real-control`, `npa-wan2-2` |
| Manifest HTTP 404 | `npa-alpamayo2-super`, `npa-antioch`, `npa-content-agents`, `npa-cosmos`, `npa-cosmos2-transfer`, `npa-cosmos3`, `npa-cosmos3-ray-serve`, `npa-cosmos3-reason`, `npa-cosmos3-serving`, `npa-curobo`, `npa-diffusers`, `npa-envgen`, `npa-fiftyone`, `npa-flex-pi`, `npa-genesis`, `npa-groot`, `npa-isaac-arena`, `npa-isaac-lab`, `npa-lancedb`, `npa-leisaac`, `npa-lerobot`, `npa-lerobot-policy`, `npa-lerobot-vlm-rl`, `npa-libero`, `npa-lingbot-world`, `npa-loop-eval`, `npa-lyra2`, `npa-mjlab`, `npa-openarm`, `npa-openpi`, `npa-reference-policy`, `npa-robocasa`, `npa-robomimic`, `npa-robotwin`, `npa-sam2`, `npa-sonic`, `npa-sonic-mujoco` |
| Anonymous-token HTTP 403 | `npa-ncore` |

A manifest HTTP 404 records that the exact tag was unavailable to the anonymous pull identity. HTTP 403 records that anonymous token access was not verified and does not prove the package or tag is absent. Eight verified development images belong to currently supported release tools; the other six are candidates outside that release inventory.

## Repository inventory reconciliation

The packaging contract contains 62 entries: 53 redistribution-eligible and nine restricted. The tool map contains 53 tools: 27 supported public-release members, 24 quarantined tools, and two restricted tools. Every packaging entry has its canonical Dockerfile.

The nine packaging entries outside the tool map are seven restricted PAIDF wrappers, the independently quarantined Gymnasium-Robotics development bootstrap, and `sim2real-eval`, whose Dockerfile also supplies mapped `loop-eval`. These exclusions are deliberate rather than missing public release rows.

Twelve additional Dockerfiles are outside the canonical packaging paths: two foundation recipes, five Kubernetes-bootstrap derivatives, two hardware variants, the legacy Habitat-Sim baked recipe, and the unmapped `sim2real-explore-policy` and `sonic-export` artifacts. Their existence does not establish an independent supported public release.

## Required work to establish current-source release parity

Build and qualify replacements for the 18 accepted images with changed declared inputs, retaining immutable development tags and exact digests. Where a main development image already exists, its availability alone does not prove full byte scanning, source delivery, supply-chain, or live capability acceptance. Establish NPA producer provenance for the nine older releases that lack it. Keep the 24 quarantined tools and nine restricted packaging entries outside the supported public table until their own acceptance or redistribution requirements are satisfied.

This audit does not rebuild or publish images, change release pins, pull all filesystem layers, scan vulnerabilities or secrets inside images, or repeat GPU qualification. Canonical COPY/ADD and recipe comparisons do not cover every transitive parent, network download, or build-time input. The per-image JSON preserves the observed digest, selected config metadata, source comparison, and failed-access phase.

## Repeat the accepted-release and chart checks

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai --verify-accepted-releases
npa/.venv/bin/python npa/scripts/render_gpu_coverage_chart.py --check
npa/.venv/bin/python -m pytest \
  npa/tests/guardrails/test_audit_container_docs_skill.py \
  npa/tests/guardrails/test_skills_index.py \
  npa/tests/scripts/test_render_gpu_coverage_chart.py -q
```

Verification completed with 16 catalog, skill-index, and chart tests passing; all ten catalog/chart tests passed again after the documentation edits. The collection smoke check collected 44,091 tests without errors. The skill validator, repository Ruff check, generated-documentation drift check, staged confidentiality scan, and staged Gitleaks scan also passed. These checks establish documentation consistency, not live image capability.
