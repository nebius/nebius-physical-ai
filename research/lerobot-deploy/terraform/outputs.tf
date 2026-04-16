locals {
  # Derive private key path from the public key path by stripping .pub
  ssh_private_key_path = trimsuffix(var.ssh_public_key_path, ".pub")
  instance_external_ip = split("/", nebius_compute_v1_instance.lerobot_gpu.status.network_interfaces[0].public_ip_address.address)[0]
  instance_internal_ip = split("/", nebius_compute_v1_instance.lerobot_gpu.status.network_interfaces[0].ip_address.address)[0]
}

output "instance_id" {
  description = "Instance ID"
  value       = nebius_compute_v1_instance.lerobot_gpu.id
}

output "instance_external_ip" {
  description = "External IP for SSH"
  value       = local.instance_external_ip
}

output "instance_internal_ip" {
  description = "Internal IP"
  value       = local.instance_internal_ip
}

output "ssh_command" {
  description = "SSH command"
  value       = "ssh -i ${local.ssh_private_key_path} ${var.ssh_user}@${local.instance_external_ip}"
}

output "instance_state" {
  description = "Current instance state"
  value       = nebius_compute_v1_instance.lerobot_gpu.status.state
}
