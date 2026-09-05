terraform {
  required_providers {
    nebius = {
      source = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
    }
    kubernetes = {
      source = "hashicorp/kubernetes"
    }
    units = {
      source  = "dstaroff/units"
      version = ">=1.1.1"
    }

    helm = {
      source  = "hashicorp/helm"
      version = "~> 3.0"
    }
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = ">=1.19.0"
    }
  }
}

provider "nebius" {
  domain  = "api.eu.nebius.cloud:443"
  profile = var.nebius_profile == "" ? null : { name = var.nebius_profile }
}

provider "helm" {
  kubernetes = {
    host                   = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints.public_endpoint
    cluster_ca_certificate = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.auth.cluster_ca_certificate
    token                  = var.nebius_profile == "" ? var.iam_token : null
    exec = var.nebius_profile == "" ? null : {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = var.nebius_cli
      args        = ["--profile", var.nebius_profile, "mk8s", "v1", "cluster", "get-token", "--format", "json"]
    }
  }
}

provider "kubernetes" {
  host                   = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints.public_endpoint
  cluster_ca_certificate = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.auth.cluster_ca_certificate
  token                  = var.nebius_profile == "" ? var.iam_token : null
  dynamic "exec" {
    for_each = var.nebius_profile == "" ? [] : [1]
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = var.nebius_cli
      args        = ["--profile", var.nebius_profile, "mk8s", "v1", "cluster", "get-token", "--format", "json"]
    }
  }
}

provider "kubectl" {
  apply_retry_count      = 5
  host                   = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.endpoints.public_endpoint
  cluster_ca_certificate = nebius_mk8s_v1_cluster.k8s-cluster.status.control_plane.auth.cluster_ca_certificate
  token                  = var.nebius_profile == "" ? var.iam_token : null
  load_config_file       = false
  dynamic "exec" {
    for_each = var.nebius_profile == "" ? [] : [1]
    content {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = var.nebius_cli
      args        = ["--profile", var.nebius_profile, "mk8s", "v1", "cluster", "get-token", "--format", "json"]
    }
  }
}
