# Agent CLI Split Plan (Issue #491)

`npa/src/npa/cli/agent.py` (11,944 lines) is a god-object that should be split
by command group.

## Intended structure

- `npa/cli/agent/__init__.py` - Main CLI entry, command registration
- `npa/cli/agent/deploy.py` - `deploy` command group
- `npa/cli/agent/status.py` - `status`, `logs` commands
- `npa/cli/agent/ssh.py` - `ssh` command
- `npa/cli/agent/storage.py` - Storage credential helpers
- `npa/cli/agent/auth.py` - Auth secret helpers

## Status

Scaffolding created. Full split requires manual refactoring with integration
test coverage - the 12k-line module has complex interdependencies that automated
splitting would break.
