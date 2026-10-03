locals {
  # The marketplace defaults may omit zonal RTX platforms. Replace its driver
  # profile list only for an explicit homogeneous RTX rendering pool.
  rtx_driver_values = var.rtx_driver_profile == null ? null : yamlencode({
    # containerd config dump can migrate a v2 root to v4 in memory. The toolkit
    # must use the on-disk schema, otherwise its newer drop-in prevents startup.
    toolkit = {
      env = [{ name = "RUNTIME_CONFIG_SOURCE", value = "file" }]
    }
    driver = {
      repoConfig = {
        configMapName = length(var.rtx_driver_profile.package_repositories) > 0 ? "npa-rtx-driver-repositories" : ""
      }
    }
    extraObjects = length(var.rtx_driver_profile.package_repositories) == 0 ? [] : [{
      apiVersion = "v1"
      kind       = "ConfigMap"
      metadata   = { name = "npa-rtx-driver-repositories", namespace = "gpu-operator" }
      data       = var.rtx_driver_profile.package_repositories
    }]
    nebius = {
      nvidiaDriverCRDPatch = {
        profiles = [{
          name = var.rtx_driver_profile.platform
          nodeSelector = {
            "node.kubernetes.io/instance-type" = var.rtx_driver_profile.platform
            "nebius.com/resource-preset"       = var.rtx_driver_profile.preset
          }
          managerEnv = []
          rdma       = { enabled = false, useHostMofed = false }
        }]
      }
    }
  })
}

resource "nebius_applications_v1alpha1_k8s_release" "this" {
  cluster_id = var.cluster_id
  parent_id  = var.parent_id

  application_name = "gpu-operator"
  namespace        = "gpu-operator"
  product_slug     = "nebius/nvidia-gpu-operator"

  sensitive = {
    values = local.rtx_driver_values
    # Write-only values need an explicit revision to reconcile an existing
    # release. Hash only this public, non-secret driver configuration.
    version = local.rtx_driver_values == null ? null : sha256(local.rtx_driver_values)
    set = {
      "dcgmExporter.enabled"                                       = var.enable_dcgm_exporter
      "dcgmExporter.serviceMonitor.enabled"                        = var.enable_dcgm_service_monitor
      "dcgmExporter.serviceMonitor.honorLabels"                    = var.relabel_dcgm_exporter ? "false" : null
      "dcgmExporter.serviceMonitor.relabelings[0].action"          = var.relabel_dcgm_exporter ? "replace" : null
      "dcgmExporter.serviceMonitor.relabelings[0].regex"           = var.relabel_dcgm_exporter ? "nvidia-dcgm-exporter" : null
      "dcgmExporter.serviceMonitor.relabelings[0].replacement"     = var.relabel_dcgm_exporter ? "dcgm-exporter" : null
      "dcgmExporter.serviceMonitor.relabelings[0].sourceLabels[0]" = var.relabel_dcgm_exporter ? "__meta_kubernetes_pod_label_app" : null
      "dcgmExporter.serviceMonitor.relabelings[0].targetLabel"     = var.relabel_dcgm_exporter ? "app_kubernetes_io_name" : null
      "mig.strategy"                                               = var.mig_strategy != null ? var.mig_strategy : null
      "cdi.enabled"                                                = var.cdi_enabled
    }
  }
}
