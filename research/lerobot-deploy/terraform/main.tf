terraform {
  required_version = ">= 1.0"
  required_providers {
    nebius = {
      source  = "terraform-provider.storage.eu-north1.nebius.cloud/nebius/nebius"
      version = ">= 0.5.55"
    }
  }
}

provider "nebius" {
  parent_id = var.nebius_project_id
  token     = var.iam_token
}

# ── Network ────────────────────────────────────────────────────────────────

resource "nebius_vpc_v1_network" "lerobot" {
  parent_id = var.nebius_project_id
  name      = "lerobot-network"
}

resource "nebius_vpc_v1_subnet" "lerobot" {
  parent_id  = var.nebius_project_id
  network_id = nebius_vpc_v1_network.lerobot.id
  name       = "lerobot-subnet"
}

# ── Security group + rules ─────────────────────────────────────────────────

resource "nebius_vpc_v1_security_group" "lerobot" {
  parent_id  = var.nebius_project_id
  network_id = nebius_vpc_v1_network.lerobot.id
  name       = "lerobot-sg"
}

resource "nebius_vpc_v1_security_rule" "allow_ssh" {
  parent_id = nebius_vpc_v1_security_group.lerobot.id
  name      = "allow-ssh"
  protocol  = "TCP"
  access    = "ALLOW"

  ingress = {
    source_cidrs      = [var.ssh_cidr_block]
    destination_ports = [22]
  }
}

resource "nebius_vpc_v1_security_rule" "allow_egress" {
  parent_id = nebius_vpc_v1_security_group.lerobot.id
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

resource "nebius_compute_v1_instance" "lerobot_gpu" {
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

  network_interfaces = [{
    name              = "eth0"
    subnet_id         = nebius_vpc_v1_subnet.lerobot.id
    ip_address        = {}
    public_ip_address = {}
    security_groups = [{
      id = nebius_vpc_v1_security_group.lerobot.id
    }]
  }]

  cloud_init_user_data = templatefile("${path.module}/cloud_init.yaml.tpl", {
    ssh_user        = var.ssh_user
    ssh_public_key  = trimspace(file(pathexpand(var.ssh_public_key_path)))
    lerobot_version = var.lerobot_version
    s3_bucket       = var.s3_bucket
    s3_endpoint     = var.s3_endpoint
    aws_access_key  = var.nebius_api_key
    aws_secret_key  = var.nebius_secret_key
    nebius_region   = var.nebius_region
  })

  preemptible = var.enable_preemptible ? {
    on_preemption = "STOP"
    priority      = 1
  } : null

  recovery_policy = var.enable_preemptible ? "FAIL" : "RECOVER"

  service_account_id = var.service_account_id

  labels = {
    environment = "ml-training"
    framework   = "lerobot"
    version     = var.lerobot_version
  }

  lifecycle {
    # Cloud-init is bootstrap-only here. Updating user_data on a running instance
    # fails in Nebius unless the VM is stopped first.
    ignore_changes = [cloud_init_user_data]
  }
}

# ── Wait for cloud-init to finish ─────────────────────────────────────────

resource "null_resource" "wait_for_cloud_init" {
  depends_on = [nebius_compute_v1_instance.lerobot_gpu]

  triggers = {
    instance_id = nebius_compute_v1_instance.lerobot_gpu.id
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

      "$${ssh_cmd[@]}" "/usr/bin/cloud-init status --wait"
      "$${ssh_cmd[@]}" "/opt/lerobot/venv/bin/python -c 'import lerobot; print(\"LeRobot \" + lerobot.__version__ + \" ready\")'"
    EOT
  }
}

resource "null_resource" "sync_runtime_scripts" {
  depends_on = [null_resource.wait_for_cloud_init]

  triggers = {
    instance_id                 = nebius_compute_v1_instance.lerobot_gpu.id
    s3_sync_sha                 = filesha256("${path.module}/../s3_sync.py")
    train_sh_sha                = filesha256("${path.module}/../training/train.sh")
    eval_sh_sha                 = filesha256("${path.module}/../training/eval.sh")
    validate_policies_sh_sha    = filesha256("${path.module}/../training/validate_policies.sh")
    benchmark_metrics_py_sha    = filesha256("${path.module}/../training/benchmark_metrics.py")
    benchmark_policies_sh_sha   = filesha256("${path.module}/../training/benchmark_policies.sh")
    benchmark_sitecustomize_sha = filesha256("${path.module}/../training/benchmark_python/sitecustomize.py")
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-lc"]
    command     = <<-EOT
      set -euo pipefail

      key_path="${pathexpand(trimsuffix(var.ssh_public_key_path, ".pub"))}"
      host="${local.instance_external_ip}"
      user="${var.ssh_user}"
      repo_root="${path.module}/.."
      archive="$(mktemp /tmp/lerobot-runtime-sync.XXXXXX.tgz)"
      ssh_cmd=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$key_path" "$user@$host")
      scp_cmd=(scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$key_path")

      cleanup() {
        rm -f "$archive"
      }
      trap cleanup EXIT

      tar -C "$repo_root" -czf "$archive" \
        s3_sync.py \
        training/train.sh \
        training/eval.sh \
        training/validate_policies.sh \
        training/benchmark_metrics.py \
        training/benchmark_policies.sh \
        training/benchmark_python/sitecustomize.py

      "$${scp_cmd[@]}" "$archive" "$user@$host:/tmp/lerobot-runtime-sync.tgz"
      "$${ssh_cmd[@]}" 'set -euo pipefail
        sudo install -d -m 0755 /opt/lerobot /opt/lerobot/benchmark_python
        tmpdir=$$(mktemp -d)
        tar -xzf /tmp/lerobot-runtime-sync.tgz -C "$$tmpdir"
        sudo install -m 0755 "$$tmpdir/s3_sync.py" /opt/lerobot/s3_sync.py
        sudo install -m 0755 "$$tmpdir/training/train.sh" /opt/lerobot/train.sh
        sudo install -m 0755 "$$tmpdir/training/eval.sh" /opt/lerobot/eval.sh
        sudo install -m 0755 "$$tmpdir/training/validate_policies.sh" /opt/lerobot/validate_policies.sh
        sudo install -m 0644 "$$tmpdir/training/benchmark_metrics.py" /opt/lerobot/benchmark_metrics.py
        sudo install -m 0755 "$$tmpdir/training/benchmark_policies.sh" /opt/lerobot/benchmark_policies.sh
        sudo install -m 0644 "$$tmpdir/training/benchmark_python/sitecustomize.py" /opt/lerobot/benchmark_python/sitecustomize.py
        sudo chown ${var.ssh_user}:${var.ssh_user} \
          /opt/lerobot/s3_sync.py \
          /opt/lerobot/train.sh \
          /opt/lerobot/eval.sh \
          /opt/lerobot/validate_policies.sh \
          /opt/lerobot/benchmark_metrics.py \
          /opt/lerobot/benchmark_policies.sh \
          /opt/lerobot/benchmark_python/sitecustomize.py
        rm -rf "$$tmpdir" /tmp/lerobot-runtime-sync.tgz'
    EOT
  }
}
