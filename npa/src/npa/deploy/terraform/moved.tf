moved {
  from = nebius_vpc_v1_network.lerobot
  to   = nebius_vpc_v1_network.workbench
}

moved {
  from = nebius_vpc_v1_subnet.lerobot
  to   = nebius_vpc_v1_subnet.workbench
}

moved {
  from = nebius_vpc_v1_security_group.lerobot
  to   = nebius_vpc_v1_security_group.workbench
}

moved {
  from = nebius_compute_v1_instance.lerobot_gpu
  to   = nebius_compute_v1_instance.workbench
}
