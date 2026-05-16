# ── Nebius provider ─────────────────────────────────────────────────────────

variable "nebius_project_id" {
  description = "Nebius project ID"
  type        = string
  sensitive   = true
}

variable "iam_token" {
  description = "Nebius IAM token (from environment.sh)"
  type        = string
  sensitive   = true
  default     = ""
}

variable "nebius_region" {
  description = "Nebius region"
  type        = string
  default     = "eu-north1"
}

# ── Service account (created by environment.sh) ───────────────────────────

variable "service_account_id" {
  description = "Service account ID created by environment.sh"
  type        = string
  sensitive   = true
}

# ── Instance ───────────────────────────────────────────────────────────────

variable "instance_name" {
  description = "Name of the workbench instance"
  type        = string
  default     = "npa-workbench"
}

variable "gpu_platform" {
  description = "Compute platform (e.g. gpu-h100-sxm, gpu-h200-sxm, cpu-d3)"
  type        = string
  default     = "gpu-h200-sxm"
}

variable "gpu_preset" {
  description = "Compute preset (e.g. 1gpu-16vcpu-200gb)"
  type        = string
  default     = "1gpu-16vcpu-200gb"
}

variable "image_family" {
  description = "Boot disk image family"
  type        = string
  default     = "ubuntu24.04-cuda12"
}

variable "boot_disk_size_gb" {
  description = "Boot disk size in GiB"
  type        = number
  default     = 100
}

variable "cosmos_data_disk_size_gb" {
  description = "Cosmos data disk size in GiB. Only used when workbench_type is cosmos."
  type        = number
  default     = 200
}

variable "data_disk_size_gb" {
  description = "Generic data disk size in GiB for workbenches with an attached data volume. Currently used by GR00T."
  type        = number
  default     = 200
}

variable "enable_preemptible" {
  description = "Use preemptible instance (cheaper, can be interrupted)"
  type        = bool
  default     = true
}

variable "server_port" {
  description = "TCP port exposed for the workbench web app/server"
  type        = number
  default     = 8080
}

variable "workbench_type" {
  description = "Workbench bootstrap type rendered into cloud-init"
  type        = string
  default     = "lerobot"
}

# ── SSH ────────────────────────────────────────────────────────────────────

variable "ssh_user" {
  description = "SSH user"
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key (private key path is derived by stripping .pub)"
  type        = string
  default     = "~/.ssh/id_ed25519.pub"
}

variable "ssh_cidr_block" {
  description = "CIDR block allowed to SSH. Defaults to open access (0.0.0.0/0)."
  type        = string
  default     = "0.0.0.0/0"
}

# ── LeRobot ────────────────────────────────────────────────────────────────

variable "lerobot_version" {
  description = "LeRobot PyPI version to install on the instance"
  type        = string
  default     = "0.5.1"
}

# ── S3 credentials (from environment.sh) ──────────────────────────────────

variable "nebius_api_key" {
  description = "AWS-compatible access key for S3"
  type        = string
  sensitive   = true
  default     = ""
}

variable "nebius_secret_key" {
  description = "AWS-compatible secret key for S3"
  type        = string
  sensitive   = true
  default     = ""
}

variable "s3_bucket" {
  description = "S3 bucket for datasets and checkpoints"
  type        = string
  default     = "lerobot-data"
}

variable "s3_endpoint" {
  description = "Nebius S3-compatible endpoint"
  type        = string
  default     = "https://storage.eu-north1.nebius.cloud"
}

# ── FiftyOne ───────────────────────────────────────────────────────────────

variable "fiftyone_version" {
  description = "FiftyOne PyPI version to install when workbench_type is fiftyone"
  type        = string
  default     = "1.15.0"
}
