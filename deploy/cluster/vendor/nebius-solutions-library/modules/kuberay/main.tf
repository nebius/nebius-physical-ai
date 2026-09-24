resource "nebius_applications_v1alpha1_k8s_release" "this" {
  parent_id        = var.parent_id
  cluster_id       = var.cluster_id
  application_name = var.name
  namespace        = var.namespace
  product_slug     = "nebius/ray-cluster"
  depends_on       = [kubernetes_network_policy_v1.cpu_cluster]
  values = var.cpu_cluster != null ? templatefile("${path.module}/files/ray-values-cpu.yaml.tftpl", {
    cpu_platform      = var.cpu_platform
    worker_replicas   = var.cpu_cluster.worker_replicas
    worker_cpus       = var.cpu_cluster.worker_cpus
    worker_memory_gib = var.cpu_cluster.worker_memory_gib
    }) : templatefile("${path.module}/files/ray-values.yaml.tftpl", {
    cpu_platform     = var.cpu_platform
    cpu_worker_image = var.cpu_worker_image
    min_cpu_replicas = var.min_cpu_replicas
    max_cpu_replicas = var.max_cpu_replicas
    gpu_platform     = var.gpu_platform
    cpu_resources    = var.cpu_resources
    gpu_worker_image = var.gpu_worker_image
    min_gpu_replicas = var.min_gpu_replicas
    max_gpu_replicas = var.max_gpu_replicas
    gpu_resources    = var.gpu_resources
  })
}

# NPA owns this namespace only for the opt-in CPU application. Install ingress
# isolation before the Jobs/GCS endpoints can become reachable.
resource "kubernetes_namespace_v1" "cpu_cluster" {
  count = var.cpu_cluster == null ? 0 : 1
  metadata {
    name   = var.namespace
    labels = { "app.kubernetes.io/managed-by" = "npa-kuberay" }
  }
}

resource "kubernetes_network_policy_v1" "cpu_cluster" {
  count = var.cpu_cluster == null ? 0 : 1
  metadata {
    name      = "ray-cluster-ingress"
    namespace = kubernetes_namespace_v1.cpu_cluster[0].metadata[0].name
  }
  spec {
    pod_selector {}
    policy_types = ["Ingress"]
    ingress {
      from {
        pod_selector {}
      }
    }
  }
}
