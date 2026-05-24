*Source: distilled output from a Claude Code session, May 2026. Authoritative for rationale; concrete claims verified against repo state in CONTRIBUTING.md.*

> Commit this verbatim to `docs/architecture/contributor-context.md` in Phase 0.5. Treat as authoritative for rationale and intent; treat concrete claims (paths, conventions, counts) as hypotheses to verify against code in Phase 1.

# Context for generating CONTRIBUTING.md

## What this repo is

Nebius Physical AI Workbench (`nebius-physical-ai`) is an open-source platform layer for building Physical AI pipelines — robotics, autonomous vehicles, and industrial AI. It provides a curated marketplace of pre-validated tools (LanceDB, FiftyOne, LeRobot, Genesis, Isaac Lab, Cosmos, GR00T, SONIC), a shared S3 data layer, a CLI/SDK, and SkyPilot-orchestrated YAML workflows. Everything runs on Nebius GPU infrastructure but is designed to be usable by any contributor with their own Nebius account.

The repo is in active development. It was open-sourced in May 2026. Contributors range from Nebius engineers to design partners and external robotics/AV teams.

## Repository structure

```
nebius-physical-ai/
├── npa/                          # Main Python package
│   ├── src/npa/
│   │   ├── cli/                  # Typer CLI (npa workbench <tool>)
│   │   ├── sdk/                  # Python SDK (npa.sdk.workbench.<tool>)
│   │   ├── workbench/            # Per-tool FastAPI services
│   │   ├── orchestration/        # SkyPilot orchestration wrapper
│   │   └── clients/              # Nebius API clients, credentials
│   ├── tests/                    # pytest test suite
│   └── docker/                   # Per-tool Dockerfiles
├── docs/
│   ├── getting-started.md        # First-time setup guide
│   ├── workbench-yaml-guide.md   # SkyPilot YAML pipeline guide (living doc)
│   └── demos/                    # Demo runbooks and assets
├── npa/workflows/skypilot/        # Reference pipeline YAMLs
├── npa/scripts/                  # Pipeline runner scripts
├── .agents/skills/               # Codex agent skill files (per tool + infrastructure)
├── .claude/skills/               # Claude Code agent skill files
├── AGENTS.md                     # Codex root index
├── CLAUDE.md                     # Claude Code root index
└── SECURITY.md
```

## Core architectural pattern (every contributor must understand)

**Three-access pattern**: every workbench tool exposes exactly one capability through three equivalent access modes:
- HTTP API (`POST /train`, etc.) — the source of truth
- CLI (`npa workbench <tool> train`) — a client of the API
- Python SDK (`npa.sdk.workbench.<tool>.train()`) — a client of the API

Never duplicate logic across layers. If you add a new capability, it goes in the FastAPI service; the CLI and SDK call it. This is enforced in code review.

**SkyPilot = sole orchestrator**: workflow orchestration uses SkyPilot managed jobs. Argo is deprecated. New pipeline YAMLs live in `npa/workflows/skypilot/`. SkyPilot runs in an isolated venv (not the NPA venv) accessed via `NPA_SKYPILOT_BIN`.

**S3 as the data bus**: tools communicate via S3 object storage, never directly. Every tool accepts `--input-path` and `--output-path` pointing to S3 URIs.

## Development setup

Reference: `docs/workbench/getting-started.md` for the full setup guide. Key points:

- Python 3.10+
- Install: `pip install -e npa/[dev]` from repo root
- Run tests: `npa/.venv/bin/python -m pytest npa/tests/ --ignore=npa/tests/e2e --timeout=120 -q`
- Linting: `npa/.venv/bin/python -m ruff check npa/src/`
- Credentials: `~/.npa/credentials.yaml` — see `docs/credentials.yaml.example`
- Storage endpoint: always use `storage.eu-north1.nebius.cloud` — the CLI default (`uk-south1`) is wrong for the primary cluster

## Adding a new workbench tool

A new tool follows the established pattern exactly. Use an existing tool (e.g. `npa/src/npa/workbench/detection_training/`) as the reference implementation:

1. Create `npa/src/npa/workbench/<tool>/` with `service.py` (FastAPI), `schemas.py`, and supporting modules
2. Create `npa/src/npa/cli/workbench/<tool>.py` (Typer CLI subcommand)
3. Create `npa/src/npa/sdk/workbench/<tool>.py` (SDK client)
4. Create `npa/docker/<tool>/Dockerfile`
5. Register in `npa/src/npa/workbench/__init__.py`, `npa/src/npa/cli/workbench/__init__.py`, `npa/src/npa/sdk/workbench/__init__.py`
6. Add tests in `npa/tests/workbench/test_<tool>.py` and `npa/tests/cli/test_<tool>_cli.py`
7. Add an agent skill file at `.agents/skills/<tool>/SKILL.md`

The tool must expose at minimum: `/health`, `/status`, `/system-info`, `/list`.

## GPU routing rules

Not all tools run on all GPU types. Document in your tool's SKILL.md:
- **H100**: general training (LanceDB CLIP, detection training, LeRobot, GR00T, SONIC)
- **L40S or RTX Pro 6000**: Isaac Lab and anything requiring RT cores
- **Do NOT** route SONIC to L40S — on-demand availability is effectively zero for the required preset
- **B300/Blackwell**: not yet prioritised; blocked on upstream library support

## Workflow YAML conventions

Reference: `docs/workbench-yaml-guide.md` and `npa/workflows/skypilot/bdd100k-pipeline.yaml`.

Key rules:
- SkyPilot 0.12.2 `envs` block does NOT support self-referencing variable interpolation
- Use `<your-value>` placeholder comments for values that vary by deployment (registry ID, bucket, image tags)
- Parameterise task name, iteration count, and output prefix via env vars
- Training YAMLs must invoke headless mode — never trigger a rendering path in a batch job
- Always use the `cloud: kubernetes` resource block for this cluster

## Testing requirements

- Every new tool must have unit tests covering: deploy, train/run, eval, status, system-info, list
- Every new endpoint must have a test for the success path and at least one failure path
- Pipeline YAMLs must have a snapshot test and a dependency order test
- E2e tests live in `npa/tests/e2e/` and are gated by `NPA_INTEGRATION_E2E=1`
- Smoke tests live in `npa/tests/smoke/` and are gated by tool-specific env vars

Expected passing baseline before any PR: **1242+ passed, 0 failures** (excluding gated smoke/e2e).

## Credentials and secrets policy

- Never hardcode credentials, project IDs, tenant IDs, bucket names, or registry paths in source
- Use `${NPA_S3_BUCKET}`, `${NPA_REGISTRY_ID}`, `${NEBIUS_PROJECT_ID}` as documented placeholders
- See `SECURITY.md` for the full policy and how to report issues
- The repo runs `gitleaks` / secret scanning — commits with hardcoded values will be rejected

## Commit and PR conventions

- Commit messages: imperative, ≤72 chars, scoped to what changed (`Add <tool> workbench service`, `Fix CLIP GPU dispatch batch size`, `Update BDD100K label map for real data`)
- One logical change per commit — don't mix tool additions with infrastructure changes
- PRs must pass the full non-e2e test suite before review
- New tools: include a brief description of the tool's role and a link to upstream docs in the PR description
- Agent skill files (`.agents/skills/<tool>/SKILL.md`) are required for new tools — reviewers will ask for them if missing

## Design partner and customer context

The workbench is being co-developed with robotics and AV design partners. Key patterns emerging from customer feedback:
- Teams want to bring their own tool containers (e.g. a custom Isaac Lab fork) and have the orchestration work without changes — the `image_id` override in SkyPilot YAMLs is the hook
- Domain-specific metadata and schema governance are more important than generic flexibility — avoid making APIs too generic if the cost is customers having to write mapping code
- "Remove glue code" is the core promise — if a contribution adds glue code that customers have to maintain, reconsider the design

## Agent skill files

The `.agents/skills/` and `.claude/skills/` directories contain structured knowledge that AI agents (Codex, Claude Code) read when working on this repo. When you add a new tool or workflow:

- Add `.agents/skills/<tool>/SKILL.md` — covers API contract, GPU routing, known issues, integration patterns
- Add to `.claude/skills/architecture/SKILL.md` if the tool changes the platform architecture

These files are documentation for agents, not for humans. Write them as instructions, not prose.

---

## Updates Since May 2026 Snapshot

This section records architectural decisions that landed after the
original Appendix A snapshot. Each subsection names the W14 condensed
milestone that anchors it. The original snapshot above is preserved as
historical baseline.

### SONIC Routing Reconciled (W12 condensed commit)

The prior code-vs-skill conflict where SONIC's CLI defaulted to L40S
while skills documented H100 has been resolved. Code now defaults to
H100; explicit L40S requests emit an availability warning rather than
silently routing to a starved capacity class. The corresponding entry
in `CONTRIBUTING.md` Known Deviations was removed.

### SkyPilot Bootstrap As CLI Capability (W11 condensed commit)

The SkyPilot install pattern — isolated venv outside the NPA Python
environment, version pinned to 0.12.2, invoked via `NPA_SKYPILOT_BIN` —
was prior operator tribal knowledge. It is now a permanent CLI
capability: `npa skypilot bootstrap` installs idempotently,
`npa skypilot status` reports state, `npa skypilot verify` runs
`sky check`. The bootstrap is the canonical setup step referenced in
`docs/workbench/getting-started.md`.

### BYOF Mechanism For Workbench Tools (W10 condensed commit)

Partners with custom forks of a Workbench tool — exemplified by
Flexion's forked Isaac Lab plus custom rsl_rl framework — can run on
Workbench without sharing their code by overriding two surfaces:

- **Image override**: `--image` on the runner (or `image_id` in the
  SkyPilot YAML) points at a partner-pushed image in the Nebius
  registry. The Workbench provides the orchestration; the partner
  provides the container.
- **Command override**: a YAML `run:` block variant invokes a partner's
  custom training entrypoint inside the partner's image. Validated
  end-to-end via a sentinel file pattern; see
  `docs/workbench/cookbooks/byof-isaac-lab/`.

Both surfaces are validated with worked example. The cookbook's
"Platform Guarantees And Image Responsibilities" section codifies the
contract between Workbench and the partner's image.

### Onboarding Doctrine (W11 condensed commit)

`docs/workbench/getting-started.md` is the canonical day-zero entry point for
contributors and partners. Cookbooks, skill files, and tool READMEs
reference it rather than duplicating setup steps. Verification gates
(`aws s3 ls`, `sky check`, `npa skypilot status`) appear in the doc as
copy-pasteable commands so a partner can confirm their environment
before attempting their first run.

### Troubleshooting Framework (W11 condensed commit)

`docs/workbench/troubleshooting/known-footguns.md` captures operational rough
edges surfaced during validation work with: symptom, root cause,
current workaround, and category for follow-up (capacity / platform /
security / docs). The intent is that partners read footguns before
encountering them rather than after.
