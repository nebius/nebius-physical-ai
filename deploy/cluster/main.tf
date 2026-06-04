locals {
  create_subnet        = trimspace(var.subnet_id) == ""
  subnet_id            = local.create_subnet ? nebius_vpc_v1_subnet.cluster[0].id : var.subnet_id
  capacity_block_group = trimspace(var.capacity_block_group)
  gpu_reservation_policy = local.capacity_block_group == "" ? null : {
    policy          = "STRICT"
    reservation_ids = [local.capacity_block_group]
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
  gpu_nodes_preemptible           = false
  cpu_nodes_fixed_count           = var.cpu_nodes_count
  cpu_nodes_platform              = var.cpu_nodes_platform
  cpu_nodes_preset                = var.cpu_nodes_preset
  gpu_nodes_fixed_count_per_group = var.gpu_nodes_count
  gpu_node_groups                 = 1
  gpu_nodes_platform              = var.gpu_nodes_platform
  gpu_nodes_preset                = var.gpu_nodes_preset
  gpu_nodes_reservation_policy    = local.gpu_reservation_policy
  gpu_disk_size                   = var.gpu_disk_size
  gpu_nodes_driverfull_image      = false
  enable_gpu_cluster              = var.enable_gpu_cluster
  infiniband_fabric               = var.infiniband_fabric
  custom_driver                   = false
  mig_strategy                    = "none"

  enable_filestore               = var.enable_filestore
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
