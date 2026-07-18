# Dev VM as a Test Environment

The Nebius dev VM (`nebius-dev` SSH alias) is used as a live execution
environment for verifying branch changes that a sandboxed session cannot run
locally (GPU workloads, `npa` workflow validate/plan/submit, agent
deploy/bootstrap, `.rrd` builds).

Workflow:

1. All repo edits are authored, committed, and pushed from the Claude Code
   session — the single source of truth for the branch.
2. The dev VM fast-forward-pulls the branch (`git pull --ff-only`) and runs
   verification using its real venv, credentials, GPU cluster, and live agent.
   Scratch output stays in `/tmp`, outside the repo.
3. The dev VM never commits or pushes; it is an execution environment only.
   This keeps branch history clean and attributable to the agent session.
