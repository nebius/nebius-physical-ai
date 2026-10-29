variable "number_of_hosts" {
  type        = number
  description = "Number of GPU hosts to test"
  default     = 0
}

variable "image" {
  type        = string
  description = "NCCL test container image. Override this for a platform-specific CUDA/NCCL combination."
  default     = "ghcr.io/coreweave/nccl-tests@sha256:d575119394c310e9f501adf83cf109d0e89da7511e6d2b20c9702acfeed7fabb"

  validation {
    condition     = length(trimspace(var.image)) > 0
    error_message = "image must not be empty."
  }
}
