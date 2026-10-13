# Validate the real provider schema and both VM pricing branches without cloud calls.
mock_provider "nebius" {}
mock_provider "null" {}

variables {
  nebius_project_id   = "synthetic-project"
  ssh_public_key_path = "tests/fixture.pub"
  wait_for_ssh        = false
}

run "preemptible_follows_spot_price" {
  command = plan

  variables {
    enable_preemptible = true
  }

  assert {
    condition     = nebius_compute_v1_instance.workbench.preemptible.on_preemption == "STOP" && nebius_compute_v1_instance.workbench.recovery_policy == "FAIL"
    error_message = "A reclaimed workbench must stop without automatic recovery."
  }

  assert {
    condition     = nebius_compute_v1_instance.workbench.follows_spot_price != null && nebius_compute_v1_instance.workbench.spot_pricing_policy == null && nebius_compute_v1_instance.workbench.on_demand == null
    error_message = "Preemptible workbenches must select exactly the current spot-price model."
  }
}

run "regular_vm_omits_spot_pricing" {
  command = plan

  variables {
    enable_preemptible = false
  }

  assert {
    condition     = nebius_compute_v1_instance.workbench.preemptible == null && nebius_compute_v1_instance.workbench.follows_spot_price == null && nebius_compute_v1_instance.workbench.spot_pricing_policy == null
    error_message = "Regular workbenches must not request preemption or spot pricing."
  }

  assert {
    condition     = nebius_compute_v1_instance.workbench.recovery_policy == "RECOVER"
    error_message = "Regular workbenches retain automatic recovery."
  }
}
