# Solutions Framework Validation

## Overview

The solutions framework is the Nebius Physical AI organization model for
separating shared platform behavior from named, user-facing product surfaces.
It exists so each solution can expose a stable CLI, SDK, documentation, and
validation contract while reusing the same repository conventions and platform
architecture.

The framework keeps platform decisions, such as command naming, shared
configuration, and architecture docs, separate from solution decisions, such as
tool behavior, runtime dependencies, workload validation, and runbooks.

## Current Solutions

Workbench is the current solution and the reference implementation of the
framework. It groups robotics and physical AI tools under the `npa workbench`
CLI namespace and the `npa.sdk.workbench` SDK namespace.

Workbench currently covers tools such as Isaac Lab, FiftyOne, LeRobot, LanceDB,
Cosmos, GR00T, Genesis, and SONIC. These tools follow a consistent CLI pattern:

```bash
npa workbench <tool> <verb> [options]
```

Common verbs include `deploy`, `status`, `list`, `system-info`, `run`, and
`train` where they apply. The implementation pattern is that services,
workflows, and shared implementation code own the behavior, while CLI and SDK
surfaces invoke that behavior through stable client paths.

## Platform Architecture

The platform owns cross-solution contracts:

- top-level CLI shape, including `npa <solution> ...`;
- shared authentication and configuration conventions;
- shared error and SDK expectations;
- repository-wide architecture docs and contribution guidance;
- common documentation index expectations.

Each solution owns the behavior behind its namespace:

- tool-specific commands, services, workflows, and jobs;
- container image and runtime assumptions;
- solution SDK modules;
- solution docs, cookbooks, and troubleshooting runbooks;
- validation evidence for its workloads.

This boundary lets the platform add more solutions without mixing their runtime
contracts into Workbench-specific implementation details.

## Adding A Solution

Use [solutions-model.md](../architecture/solutions-model.md) as the canonical
reference for adding another solution. In brief, define the user-facing scope,
choose the `npa <solution>` namespace, document the first tools and contracts,
add implementation behind one source of truth, and publish validation evidence
with the solution docs.

## Validation Status

The solutions framework was validated as part of PR 6. That validation confirmed
the current platform-versus-solution split, with Workbench as the reference
solution, is the supported model for the current repository state.
