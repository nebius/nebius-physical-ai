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
  description = "SSH public key for Kubernetes node access. `npa cluster up` pins the first key that exists on the machine (NPA_SSH_PUBLIC_KEY, then ~/.ssh/id_ed25519.pub, id_rsa.pub, id_ecdsa.pub) unless this is set explicitly; the module rejects a path that does not exist."
  type = object({
    key  = optional(string)
    path = optional(string)
  })
  default = {
    path = "~/.ssh/id_ed25519.pub"
  }
}

variable "cpu_nodes_count" {
  description = "CPU-only node count. Default is one small CPU node for the FTUE / Physical AI Data Factory shape (CPU stages such as annotate/curate run here; GPU stages run on the GPU node). Set to 0 for a GPU-only cluster."
  type        = number
  default     = 1
}

variable "cpu_nodes_platform" {
  description = "CPU-only node platform."
  type        = string
  default     = "cpu-d3"
}

variable "cpu_nodes_preset" {
  description = "CPU-only node preset. The default matches the documented first-run / Physical AI Data Factory topology."
  type        = string
  default     = "8vcpu-32gb"
}

variable "gpu_nodes_count" {
  description = "GPU node count in the single GPU node group. Default is one node for the FTUE / Physical AI Data Factory shape. Raise it (with a multi-GPU preset and enable_gpu_cluster) for a training farm."
  type        = number
  default     = 1
}

variable "gpu_nodes_platform" {
  description = "GPU node platform."
  type        = string
  default     = "gpu-rtx6000"
}

variable "gpu_nodes_preset" {
  description = "GPU node preset. The default is the single-GPU RTX PRO 6000 preset for the small FTUE / Physical AI Data Factory shape. For a training farm, use a multi-GPU preset such as 8gpu-192vcpu-1744gb (and set enable_gpu_cluster = true for InfiniBand)."
  type        = string
  default     = "1gpu-24vcpu-218gb"
}

variable "gpu_nodes_preemptible" {
  description = "Run the GPU node group on preemptible VMs. Off by default; preemptible capacity does not bypass hard tenant instance, disk, or public-IP quotas."
  type        = bool
  default     = false
}

variable "capacity_block_group" {
  description = "Optional capacity block group ID used as a strict GPU node-group reservation selector. Leave empty for on-demand capacity."
  type        = string
  default     = ""
  sensitive   = true
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
  description = "Create a shared filesystem (SFS) for cross-node cluster storage. Off by default: npa.workflow stages (including the Physical AI Data Factory) hand off artifacts via S3 URIs, so a small 1-GPU + 1-CPU cluster does not need it, and creating it requires Shared Filesystem SSD quota. Set true when a workload needs a shared /mnt/data + the filesystem CSI default StorageClass; supplying existing_filestore enables the same wiring without creating a filesystem."
  type        = bool
  default     = false
}

variable "existing_filestore" {
  description = "Existing shared filesystem ID to attach instead of creating one. Setting it implies enable_filestore."
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
