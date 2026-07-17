terraform {
  required_version = ">= 1.0"
  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = "~> 0.5.201"
    }
  }
}

provider "nebius" {
  parent_id = var.nebius_project_id
  token     = var.iam_token
}

# ── Network ────────────────────────────────────────────────────────────────

resource "nebius_vpc_v1_network" "workbench" {
  parent_id = var.nebius_project_id
  name      = "${var.instance_name}-network"
}

resource "nebius_vpc_v1_subnet" "workbench" {
  parent_id  = var.nebius_project_id
  network_id = nebius_vpc_v1_network.workbench.id
  name       = "${var.instance_name}-subnet"
}

# ── Security group + rules ─────────────────────────────────────────────────

resource "nebius_vpc_v1_security_group" "workbench" {
  parent_id  = var.nebius_project_id
  network_id = nebius_vpc_v1_network.workbench.id
  name       = "${var.instance_name}-sg"
}

resource "nebius_vpc_v1_security_rule" "allow_ssh" {
  parent_id = nebius_vpc_v1_security_group.workbench.id
  name      = "allow-ssh"
  protocol  = "TCP"
  access    = "ALLOW"

  ingress = {
    source_cidrs      = [var.ssh_cidr_block]
    destination_ports = [22]
  }
}

resource "nebius_vpc_v1_security_rule" "allow_server" {
  parent_id = nebius_vpc_v1_security_group.workbench.id
  name      = "allow-server"
  protocol  = "TCP"
  access    = "ALLOW"

  ingress = {
    source_cidrs      = [var.ssh_cidr_block]
    destination_ports = distinct(concat([var.server_port], var.extra_ingress_ports))
  }
}

resource "nebius_vpc_v1_security_rule" "allow_egress" {
  parent_id = nebius_vpc_v1_security_group.workbench.id
  name      = "allow-all-egress"
  protocol  = "ANY"
  access    = "ALLOW"

  egress = {
    destination_cidrs = ["0.0.0.0/0"]
  }
}

# ── Boot disk ──────────────────────────────────────────────────────────────

resource "nebius_compute_v1_disk" "boot" {
  parent_id      = var.nebius_project_id
  name           = "${var.instance_name}-boot"
  type           = "NETWORK_SSD"
  size_gibibytes = var.boot_disk_size_gb

  source_image_family = {
    image_family = var.image_family
  }
}

# ── GPU instance ───────────────────────────────────────────────────────────

resource "nebius_compute_v1_instance" "workbench" {
  parent_id = var.nebius_project_id
  name      = var.instance_name

  resources = {
    platform = var.gpu_platform
    preset   = var.gpu_preset
  }

  boot_disk = {
    attach_mode = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.boot.id
    }
  }

  secondary_disks = concat(
    var.workbench_type == "cosmos" ? [
      {
        attach_mode = "READ_WRITE"
        device_id   = "npa-cosmos-data"
        managed_disk = {
          name = "${var.instance_name}-cosmos-data"
          spec = {
            type           = "NETWORK_SSD"
            size_gibibytes = var.cosmos_data_disk_size_gb
          }
        }
      }
    ] : [],
    contains(["groot", "groot-container"], var.workbench_type) ? [
      {
        attach_mode = "READ_WRITE"
        device_id   = "npa-groot-data"
        managed_disk = {
          name = "${var.instance_name}-groot-data"
          spec = {
            type           = "NETWORK_SSD"
            size_gibibytes = var.data_disk_size_gb
          }
        }
      }
    ] : []
  )

  network_interfaces = [{
    name              = "eth0"
    subnet_id         = nebius_vpc_v1_subnet.workbench.id
    ip_address        = {}
    public_ip_address = {}
    security_groups = [{
      id = nebius_vpc_v1_security_group.workbench.id
    }]
  }]

  cloud_init_user_data = templatefile("${path.module}/cloud_init.yaml.tpl", {
    ssh_user         = var.ssh_user
    ssh_public_key   = trimspace(file(pathexpand(var.ssh_public_key_path)))
    workbench_type   = var.workbench_type
    server_port      = var.server_port
    lerobot_version  = var.lerobot_version
    fiftyone_version = var.fiftyone_version
    s3_bucket        = var.s3_bucket
    s3_endpoint      = var.s3_endpoint
    aws_access_key   = var.nebius_api_key
    aws_secret_key   = var.nebius_secret_key
    nebius_region    = var.nebius_region
  })

  preemptible = var.enable_preemptible ? {
    on_preemption = "STOP"
    priority      = 1
  } : null

  recovery_policy = var.enable_preemptible ? "FAIL" : "RECOVER"

  service_account_id = trimspace(var.service_account_id) != "" ? trimspace(var.service_account_id) : null

  labels = {
    environment = "ml-workbench"
    workload    = "physical-ai"
  }

  lifecycle {
    # Cloud-init is bootstrap-only here. Updating user_data on a running instance
    # fails in Nebius unless the VM is stopped first.
    ignore_changes = [cloud_init_user_data]
  }
}

# ── Wait for cloud-init to finish ─────────────────────────────────────────

resource "null_resource" "wait_for_cloud_init" {
  depends_on = [nebius_compute_v1_instance.workbench]

  triggers = {
    instance_id = nebius_compute_v1_instance.workbench.id
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-lc"]
    command     = <<-EOT
      set -euo pipefail

      key_path="${pathexpand(trimsuffix(var.ssh_public_key_path, ".pub"))}"
      host="${local.instance_external_ip}"
      user="${var.ssh_user}"
      ssh_cmd=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$key_path" "$user@$host")

      echo "Waiting for SSH on $host..."
      for _ in $(seq 1 60); do
        if "$${ssh_cmd[@]}" "true" >/dev/null 2>&1; then
          break
        fi
        sleep 5
      done
      "$${ssh_cmd[@]}" "true" >/dev/null 2>&1

      echo "Waiting for cloud-init boot-finished..."
      for _ in $(seq 1 120); do
        if "$${ssh_cmd[@]}" "test -f /var/lib/cloud/instance/boot-finished" >/dev/null 2>&1; then
          break
        fi
        sleep 10
      done

      # Do not use `cloud-init status --wait`: on some CUDA13 images it hangs
      # forever even when status is already "done" (boot-finished present).
      echo "Polling cloud-init status..."
      for _ in $(seq 1 60); do
        status="$("$${ssh_cmd[@]}" "cloud-init status 2>/dev/null | awk '{print \$2}'" || true)"
        echo "cloud-init status: $${status:-unknown}"
        case "$status" in
          done|error|disabled) break ;;
        esac
        sleep 5
      done
      "$${ssh_cmd[@]}" "cloud-init status --long || true" || true
    EOT
  }
}

# Runtime script sync is handled by the research repo layout
# (research/lerobot-deploy/terraform/main.tf) or by the npa deploy
# command's configurator.  The npa-bundled TF does not carry the
# research scripts, so this resource is intentionally absent here.
