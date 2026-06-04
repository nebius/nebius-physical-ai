variable "tenant_id" {
  description = "Nebius tenant ID."
  type        = string
}

variable "parent_id" {
  description = "Nebius project ID."
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

variable "subnet_id" {
  description = "Existing Nebius VPC subnet ID. Leave empty to create a dedicated network and subnet."
  type        = string
  default     = ""
}

variable "cluster_name" {
  description = "Managed Kubernetes cluster name."
  type        = string
  default     = "npa-cluster"
}

variable "ssh_user_name" {
  description = "SSH user configured on Kubernetes nodes."
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "SSH public key for Kubernetes node access."
  type = object({
    key  = optional(string)
    path = optional(string)
  })
  default = {
    path = "~/.ssh/id_rsa.pub"
  }
}

variable "cpu_nodes_count" {
  description = "CPU-only node count. Keep zero when the target cluster should contain only GPU worker nodes."
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
  default     = "4vcpu-16gb"
}

variable "gpu_nodes_count" {
  description = "GPU node count in the single GPU node group."
  type        = number
  default     = 2
}

variable "gpu_nodes_platform" {
  description = "GPU node platform."
  type        = string
  default     = "gpu-rtx6000"
}

variable "gpu_nodes_preset" {
  description = "GPU node preset. The default is the 8-GPU RTX PRO 6000 preset exposed by the Nebius platform catalog."
  type        = string
  default     = "8gpu-192vcpu-1744gb"
}

variable "gpu_disk_size" {
  description = "GPU node boot disk size in GiB."
  type        = string
  default     = "1023"
}

variable "enable_gpu_cluster" {
  description = "Enable Nebius GPU cluster and InfiniBand attachment for platforms that support it."
  type        = bool
  default     = false
}

variable "enable_k8s_node_group_sa" {
  description = "Create a dedicated node-group service account and add it to the tenant editors group."
  type        = bool
  default     = false
}

variable "infiniband_fabric" {
  description = "Optional InfiniBand fabric name when enable_gpu_cluster is true."
  type        = string
  default     = null
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

variable "filestore_block_size_kibibytes" {
  description = "Shared filesystem block size in KiB."
  type        = number
  default     = 4
}

variable "filestore_mount_path" {
  description = "Node mount path for the shared filesystem."
  type        = string
  default     = "/mnt/data"
}

variable "filesystem_csi_chart_version" {
  description = "Nebius Shared Filesystem CSI Helm chart version."
  type        = string
  default     = "0.1.5"
}

variable "previous_default_storage_class_name" {
  description = "StorageClass to demote before making the shared filesystem StorageClass the default. Empty disables demotion."
  type        = string
  default     = "compute-csi-default-sc"
}

variable "k8s_version" {
  description = "Kubernetes major.minor version. Null lets the Nebius backend choose its default."
  type        = string
  default     = null
}
