# Exercise the production VM request with real provider schemas and no cloud calls.
mock_provider "nebius" {}
mock_provider "null" {}

variables {
  nebius_project_id    = "project-test"
  ssh_public_key_path = "tests/spot-pricing.pub"
  wait_for_ssh        = false
}

run "preemptible_follows_spot_price" {
  command = plan

  variables {
    enable_preemptible = true
  }

  assert {
    condition = (
      nebius_compute_v1_instance.workbench.preemptible.on_preemption == "STOP" &&
      nebius_compute_v1_instance.workbench.preemptible.priority == null &&
      nebius_compute_v1_instance.workbench.follows_spot_price != null
    )
    error_message = "Preemptible VMs must explicitly follow the current spot price."
  }
}

run "on_demand_omits_spot_price" {
  command = plan

  variables {
    enable_preemptible = false
  }

  assert {
    condition = (
      nebius_compute_v1_instance.workbench.preemptible == null &&
      nebius_compute_v1_instance.workbench.follows_spot_price == null
    )
    error_message = "On-demand VMs must omit preemptible and spot-pricing settings."
  }
}
