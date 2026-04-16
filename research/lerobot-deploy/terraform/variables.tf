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
  description = "Name of the training instance"
  type        = string
  default     = "lerobot-training"
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

variable "enable_preemptible" {
  description = "Use preemptible instance (cheaper, can be interrupted)"
  type        = bool
  default     = true
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
