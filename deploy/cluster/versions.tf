terraform {
  required_version = ">= 1.3"

  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = "~> 0.5.201"
    }
  }
}

provider "nebius" {
  domain = "api.eu.nebius.cloud:443"
  token  = var.iam_token
}
