# NPA Cluster Scope

`npa cluster` is the NPA Workbench target/profile layer over Nebius Managed Kubernetes, not a replacement for
`nebius mk8s` administration. It may create, inspect, cache, and clean up NPA-managed execution targets,
including local state under `~/.npa/clusters/<name>/`, named kubeconfig context caching, project/subnet defaults,
readiness polling, and GPU node-group aliases for Workbench compositions; raw MK8s administration such as edit,
update, upgrade, operation inspection, version listing, and compatibility-matrix discovery belongs in `nebius mk8s`.
The scope rationale is captured in [MK8s CLI Audit: nebius mk8s vs npa cluster](mk8s-cli-audit-20260515T002328Z.md).
