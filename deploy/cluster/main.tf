locals {
  # The vendored module gates both filesystem creation and the existing-filesystem
  # lookup on enable_filestore, so `existing_filestore` alone would attach nothing.
  # Supplying an existing filesystem is itself an opt-in.
  enable_filestore     = var.enable_filestore || trimspace(var.existing_filestore) != ""
  create_subnet        = trimspace(var.subnet_id) == ""
  subnet_id            = local.create_subnet ? nebius_vpc_v1_subnet.cluster[0].id : var.subnet_id
  capacity_block_group = trimspace(var.capacity_block_group)
  gpu_nodes_driverfull_image = var.gpu_nodes_count > 0 && contains(
    ["auto", "managed-image"],
    var.gpu_driver_mode,
  )
  gpus_per_node = try(
    tonumber(regex("^([0-9]+)gpu-", lower(trimspace(var.gpu_nodes_preset)))[0]),
    0,
  )
  gpu_reservation_policy = local.capacity_block_group == "" ? null : {
    policy          = "STRICT"
    reservation_ids = [local.capacity_block_group]
  }
}

check "nvswitch_operator_acknowledgement" {
  assert {
    condition = !(
      var.gpu_nodes_count > 0 &&
      var.gpu_driver_mode == "operator" &&
      (var.enable_gpu_cluster || (local.gpus_per_node > 1 && can(regex("-(sxm|nvl)", lower(var.gpu_nodes_platform)))))
    ) || var.allow_unsafe_nvswitch_operator
    error_message = "gpu_driver_mode=operator is unsafe on NVSwitch systems because in-cluster driver/Fabric Manager startup can race host InfiniBand devices. Use auto/managed-image, or set allow_unsafe_nvswitch_operator=true only for a controlled diagnostic followed by GPU node-group recreation."
  }
}

resource "nebius_vpc_v1_network" "cluster" {
  count     = local.create_subnet ? 1 : 0
  parent_id = var.parent_id
  name      = "${var.cluster_name}-network"
}

resource "nebius_vpc_v1_subnet" "cluster" {
  count      = local.create_subnet ? 1 : 0
  parent_id  = var.parent_id
  network_id = nebius_vpc_v1_network.cluster[0].id
  name       = "${var.cluster_name}-subnet"
}

module "k8s_training" {
  source = "./vendor/nebius-solutions-library/k8s-training"

  tenant_id = var.tenant_id
  parent_id = var.parent_id
  region    = var.region
  subnet_id = local.subnet_id
  iam_token = var.iam_token

  cluster_name                    = var.cluster_name
  k8s_version                     = var.k8s_version
  ssh_user_name                   = var.ssh_user_name
  ssh_public_key                  = var.ssh_public_key
  mk8s_cluster_public_endpoint    = true
  enable_k8s_node_group_sa        = var.enable_k8s_node_group_sa
  enable_egress_gateway           = false
  cpu_nodes_public_ips            = false
  gpu_nodes_public_ips            = false
  cpu_nodes_preemptible           = false
  gpu_nodes_preemptible           = var.gpu_nodes_preemptible
  cpu_nodes_fixed_count           = var.cpu_nodes_count
  cpu_nodes_platform              = var.cpu_nodes_platform
  cpu_nodes_preset                = var.cpu_nodes_preset
  gpu_nodes_fixed_count_per_group = var.gpu_nodes_count
  gpu_node_groups                 = 1
  gpu_nodes_platform              = var.gpu_nodes_platform
  gpu_nodes_preset                = var.gpu_nodes_preset
  gpu_nodes_reservation_policy    = local.gpu_reservation_policy
  gpu_disk_size                   = var.gpu_disk_size
  gpu_nodes_driverfull_image      = local.gpu_nodes_driverfull_image
  gpu_nodes_driver_preset         = var.managed_driver_preset
  enable_gpu_cluster              = var.enable_gpu_cluster
  infiniband_fabric               = var.infiniband_fabric
  custom_driver                   = var.mig_enabled
  mig_strategy                    = var.mig_enabled ? var.mig_strategy : "none"
  mig_parted_config               = var.mig_enabled ? var.mig_parted_config : null
  gpu_operator_version            = var.gpu_operator_version
  gpu_driver_version              = var.gpu_driver_version
  gpu_device_plugin_version       = var.gpu_device_plugin_version
  gpu_gfd_version                 = var.gpu_gfd_version
  gpu_mig_manager_version         = var.gpu_mig_manager_version
  gpu_mig_with_reboot             = var.gpu_mig_with_reboot
  gpu_operator_rdma_enabled       = var.gpu_operator_rdma_enabled

  enable_filestore               = local.enable_filestore
  existing_filestore             = var.existing_filestore
  filestore_disk_size_gibibytes  = var.filestore_disk_size_gibibytes
  filestore_block_size_kibibytes = var.filestore_block_size_kibibytes
  filestore_mount_path           = var.filestore_mount_path
  filesystem_csi = {
    chart_version                       = var.filesystem_csi_chart_version
    namespace                           = "kube-system"
    make_default_storage_class          = true
    previous_default_storage_class_name = var.previous_default_storage_class_name
  }

  enable_nebius_o11y_agent = false
  enable_grafana           = false
  enable_prometheus        = false
  collectK8sClusterMetrics = false
  loki = {
    enabled            = false
    replication_factor = 1
  }

  enable_kuberay_cluster = false
  enable_kuberay_service = false
  enable_opa_gatekeeper  = false
}
