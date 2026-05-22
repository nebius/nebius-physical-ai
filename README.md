# Nebius AI Agents

Standalone Python package for a 24/7 autonomous multi-agent software engineering pipeline.

The system receives GitHub webhook events, routes them through a 13-agent pipeline, and writes
approved changes back to GitHub through a dry-run-gated client. `DRY_RUN` defaults to `true` and
live push / PR creation is suppressed unless a developer explicitly changes the runtime config.
Stage 9 control smoke runs are docs-only checks that verify task-steering and control-flow logic without modifying runtime behavior.
