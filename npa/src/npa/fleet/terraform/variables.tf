# Static variable declarations for the npa fleet per-cluster wrapper.
# tenant_id / parent_id / region / iam_token / ssh_public_key are supplied via
# TF_VAR_* env at apply time; everything else comes from terraform.tfvars.

variable "tenant_id" {
  description = "Nebius tenant ID."
  type        = string
}

variable "parent_id" {
  description = "Nebius project ID that owns this cluster."
  type        = string
}

variable "region" {
  description = "Nebius region for the Managed Kubernetes cluster."
  type        = string
}

variable "iam_token" {
  description = "Nebius IAM token used by Terraform Kubernetes and Helm providers."
  type        = string
  sensitive   = true
}

variable "ssh_public_key" {
  description = "SSH public key material for Kubernetes node access."
  type        = string
  default     = ""
}

variable "subnet_id" {
  description = "Existing VPC subnet ID. Empty creates a dedicated network + subnet."
  type        = string
  default     = ""
}

variable "cluster_name" {
  description = "Managed Kubernetes cluster name."
  type        = string
}

variable "k8s_version" {
  description = "Kubernetes major.minor version. Null lets the backend choose."
  type        = string
  default     = null
}

variable "cpu_nodes_count" {
  description = "CPU-only node count (0 for GPU-only clusters)."
  type        = number
  default     = 0
}

variable "cpu_nodes_platform" {
  description = "CPU-only node platform."
  type        = string
  default     = "cpu-d3"
}

variable "cpu_nodes_preset" {
  description = "CPU-only node preset."
  type        = string
  default     = "16vcpu-64gb"
}

variable "gpu_nodes_count" {
  description = "GPU node count in the single GPU node group."
  type        = number
  default     = 0
}

variable "gpu_nodes_platform" {
  description = "GPU node platform."
  type        = string
  default     = "gpu-rtx6000"
}

variable "gpu_nodes_preset" {
  description = "GPU node preset."
  type        = string
  default     = "1gpu-24vcpu-218gb"
}

variable "gpu_disk_size" {
  description = "GPU node boot disk size in GiB."
  type        = string
  default     = "1023"
}

variable "enable_gpu_cluster" {
  description = "Enable Nebius GPU cluster + InfiniBand. Only valid for 8-GPU presets."
  type        = bool
  default     = false
}

variable "infiniband_fabric" {
  description = "InfiniBand fabric id when enable_gpu_cluster is true. Empty -> null."
  type        = string
  default     = ""
}

variable "enable_filestore" {
  description = "Create or attach a shared filesystem for cluster storage."
  type        = bool
  default     = true
}

variable "existing_filestore" {
  description = "Existing shared filesystem ID to attach instead of creating one."
  type        = string
  default     = ""
}

variable "filestore_disk_size_gibibytes" {
  description = "Shared filesystem size in GiB."
  type        = number
  default     = 1024
}
