# `npa provision-if-absent`

## Command Tree

```text
Usage: npa provision-if-absent [OPTIONS] COMMAND [ARGS]...

Ensure configured Kubernetes and S3 runtime resources exist.

Options
--project  <str>  Project alias from ~/.npa/config.yaml.
--cluster-name  <str>  Cluster profile/context name. [default: npa-cluster]
--terraform-dir  <path>  Terraform cluster directory.
--kubeconfig  <path>  Dedicated kubeconfig path.
--context  <str>  Kubeconfig context name.
--skip-k8s  Do not ensure Kubernetes.
--skip-s3  Do not ensure S3.
--validate  --skip-validate  Run post-apply Kubernetes validation. [default: validate]
--sky-smoke  --skip-sky-smoke  Run a SkyPilot GPU smoke task. [default: skip-sky-smoke]
--dry-run  Resolve settings and print intended actions only.
--timeout  <int>  Terraform apply timeout in minutes. [default: 120]
--output-format  <text|json>  Output format. [default: text]
--help  Show this message and exit.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | <str>  Project alias from ~/.npa/config.yaml. |
| `--cluster-name` | <str>  Cluster profile/context name. [default: npa-cluster] |
| `--terraform-dir` | <path>  Terraform cluster directory. |
| `--kubeconfig` | <path>  Dedicated kubeconfig path. |
| `--context` | <str>  Kubeconfig context name. |
| `--skip-k8s` | Do not ensure Kubernetes. |
| `--skip-s3` | Do not ensure S3. |
| `--validate` | --skip-validate  Run post-apply Kubernetes validation. [default: validate] |
| `--sky-smoke` | --skip-sky-smoke  Run a SkyPilot GPU smoke task. [default: skip-sky-smoke] |
| `--dry-run` | Resolve settings and print intended actions only. |
| `--timeout` | <int>  Terraform apply timeout in minutes. [default: 120] |
| `--output-format` | <text\|json>  Output format. [default: text] |
| `--help` | Show this message and exit. |

## Subcommands

No subcommands are listed by `--help`.

## Examples

```bash
npa provision-if-absent --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `provision-if-absent`.
