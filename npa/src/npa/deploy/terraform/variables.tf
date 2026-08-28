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
  description = "Optional service account ID attached to the VM for metadata/API access"
  type        = string
  sensitive   = true
  default     = ""
}

# ── Instance ───────────────────────────────────────────────────────────────

variable "instance_name" {
  description = "Name of the workbench instance"
  type        = string
  default     = "npa-workbench"
}

variable "operation_id" {
  description = "Secret-free NPA provisioning operation ID used for exact recovery ownership"
  type        = string
  default     = ""
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

variable "extra_ingress_ports" {
  description = "Additional TCP ingress ports exposed alongside server_port, as a JSON array string (e.g. \"[443,9090]\")"
  # A JSON-string (decoded with jsondecode below) rather than list(number): a
  # `-var extra_ingress_ports=[443,9090]` value is parsed by Terraform as an HCL
  # expression for non-string types, which intermittently failed with
  # "Invalid expression". A string var is taken verbatim, so decoding is
  # deterministic.
  type    = string
  default = "[]"

  validation {
    condition     = can(jsondecode(var.extra_ingress_ports))
    error_message = "extra_ingress_ports must be a JSON array string, e.g. \"[443,9090]\" or \"[]\"."
  }
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
  description = "CIDR block allowed to SSH. Empty disables SSH ingress; set explicitly for remote bootstrap."
  type        = string
  default     = ""

  validation {
    condition = (
      trimspace(var.ssh_cidr_block) == "" ||
      can(cidrnetmask(var.ssh_cidr_block))
    )
    error_message = "ssh_cidr_block must be empty (no SSH ingress) or a valid IPv4 CIDR."
  }

  validation {
    condition     = !endswith(trimspace(var.ssh_cidr_block), "/0") || var.allow_world_open_ssh
    error_message = "World-open SSH (/0) requires allow_world_open_ssh=true."
  }
}

variable "application_cidr_block" {
  description = "CIDR block allowed to reach server_port and extra_ingress_ports. Empty disables public application ingress."
  type        = string
  default     = ""

  validation {
    condition = (
      trimspace(var.application_cidr_block) == "" ||
      can(cidrnetmask(var.application_cidr_block))
    )
    error_message = "application_cidr_block must be empty (no application ingress) or a valid IPv4 CIDR."
  }

  validation {
    condition = (
      !endswith(trimspace(var.application_cidr_block), "/0") ||
      var.allow_world_open_application
    )
    error_message = "World-open application ingress (/0) requires allow_world_open_application=true."
  }
}

variable "allow_world_open_ssh" {
  description = "Conspicuous operator acknowledgement required when SSH ingress is 0.0.0.0/0"
  type        = bool
  default     = false
}

variable "allow_world_open_application" {
  description = "Conspicuous operator acknowledgement required when application ingress is 0.0.0.0/0; especially hazardous for unauthenticated apps such as FiftyOne"
  type        = bool
  default     = false
}

# ── LeRobot ────────────────────────────────────────────────────────────────

variable "lerobot_version" {
  description = "LeRobot PyPI version to install on the instance (supported: 0.5.1 default, 0.6.0)"
  type        = string
  default     = "0.5.1"

  validation {
    condition     = contains(["0.5.1", "0.6.0"], var.lerobot_version)
    error_message = "lerobot_version must be one of: 0.5.1, 0.6.0."
  }
}

# ── S3 credentials (from environment.sh) ──────────────────────────────────

variable "nebius_api_key" {
  description = "Deprecated compatibility input for operator-side Terraform state access; never rendered into VM metadata or cloud-init"
  type        = string
  sensitive   = true
  default     = ""
}

variable "nebius_secret_key" {
  description = "Deprecated compatibility input for operator-side Terraform state access; never rendered into VM metadata or cloud-init"
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

variable "wait_for_ssh" {
  description = "Wait for the VM's SSH and cloud-init before finishing the apply. Set false when this machine cannot reach a fresh public IP on tcp/22 (corporate VPN / split tunnel): the VM still bootstraps, but the deploy cannot verify it and rolls nothing back."
  type        = bool
  default     = true
}
