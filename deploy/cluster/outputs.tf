output "kube_cluster" {
  description = "Managed Kubernetes cluster information."
  value       = module.k8s_training.kube_cluster
}

output "shared_filesystem" {
  description = "Shared filesystem attached to the cluster."
  value       = module.k8s_training.shared-filesystem
}

output "filesystem_csi" {
  description = "Shared Filesystem CSI installation details."
  value       = module.k8s_training.filesystem_csi
}

output "created_subnet_id" {
  description = "Subnet created by this wrapper when subnet_id was empty."
  value       = local.create_subnet ? nebius_vpc_v1_subnet.cluster[0].id : null
}

output "k8s_training_ref" {
  description = "Pinned nebius-solutions-library release used by this wrapper."
  value       = "main-v2026-05-25+local-cluster-patches"
}
