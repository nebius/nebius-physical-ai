terraform {
  # The vendored nebius-solutions-library modules require >= 1.12.0
  # (k8s-rbac-bindings) and use `ephemeral` blocks (o11y, Terraform 1.10+).
  # Terraform loads every referenced module during `init`, including the ones this
  # config disables, so an older binary cannot even initialise this directory.
  required_version = ">= 1.12.0"

  required_providers {
    nebius = {
      source  = "nebius/nebius"
      version = ">= 0.6.23"
    }
  }
}

provider "nebius" {
  domain = "api.eu.nebius.cloud:443"
  token  = var.iam_token
}
