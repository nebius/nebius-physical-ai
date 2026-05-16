locals {
  ssh_private_key_path = trimsuffix(var.ssh_public_key_path, ".pub")
  _raw_public          = try(nebius_compute_v1_instance.workbench.status.network_interfaces[0].public_ip_address.address, "")
  _raw_private         = try(nebius_compute_v1_instance.workbench.status.network_interfaces[0].ip_address.address, "")
  instance_external_ip = local._raw_public != "" ? split("/", local._raw_public)[0] : ""
  instance_internal_ip = local._raw_private != "" ? split("/", local._raw_private)[0] : ""
}

output "vm_ip" {
  description = "External IP for SSH and HTTP"
  value       = local.instance_external_ip
}

output "ssh_user" {
  description = "SSH user for the VM"
  value       = var.ssh_user
}

output "ssh_key_path" {
  description = "Path to SSH private key"
  value       = local.ssh_private_key_path
}

output "storage_bucket" {
  description = "S3 bucket name"
  value       = var.s3_bucket
}

output "storage_endpoint" {
  description = "S3-compatible endpoint URL"
  value       = var.s3_endpoint
}

output "instance_id" {
  description = "Nebius instance ID"
  value       = nebius_compute_v1_instance.workbench.id
}

output "instance_name" {
  description = "Instance name"
  value       = var.instance_name
}

output "instance_state" {
  description = "Current instance state"
  value       = nebius_compute_v1_instance.workbench.status.state
}

output "nebius_region" {
  description = "Nebius region"
  value       = var.nebius_region
}

output "gpu_platform" {
  description = "GPU platform"
  value       = var.gpu_platform
}

output "gpu_preset" {
  description = "GPU preset"
  value       = var.gpu_preset
}

output "security_group_id" {
  description = "Security group ID"
  value       = nebius_vpc_v1_security_group.workbench.id
}
