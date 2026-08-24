# Nebius VM network RTT benchmark

This benchmark measured ICMP round-trip time (RTT) between small CPU VMs in
Nebius AI Cloud on 2026-08-24. It covers same-VPC private networking,
same-region public networking across projects, and same-region public
networking across tenants.

## Results

All VMs were in `us-central1`. Each row contains 20 individual successful RTT
estimates; no packets were lost.

| Topology | Direction | Samples | Min | Median | Mean | p95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Same project and VPC, private IPv4 | A to B | 20 | 0.429 ms | 0.512 ms | 0.892 ms | 4.261 ms | 4.480 ms |
| Same project and VPC, private IPv4 | B to A | 20 | 0.373 ms | 0.521 ms | 0.794 ms | 2.337 ms | 4.740 ms |
| Same region, different projects, public IPv4 | A to B | 20 | 0.470 ms | 0.665 ms | 0.931 ms | 2.263 ms | 4.980 ms |
| Same region, different projects, public IPv4 | B to A | 20 | 0.502 ms | 0.585 ms | 1.142 ms | 5.317 ms | 6.960 ms |
| Same region, different tenants, public IPv4 | A to B | 20 | 0.520 ms | 0.583 ms | 0.912 ms | 1.859 ms | 4.880 ms |
| Same region, different tenants, public IPv4 | B to A | 20 | 0.519 ms | 0.609 ms | 0.834 ms | 1.380 ms | 4.600 ms |

These values describe one short measurement window, not an availability or
latency guarantee. The medians are more representative of the observed steady
path than the maxima in this small sample.

## Method

- The benchmark used four `cpu-d3/4vcpu-16gb` VMs with a driverless Ubuntu
  image. Two VMs shared one project, VPC, subnet, and security group; the other
  roles each used a separate disposable project and network.
- Private-path probes addressed the peer's private IPv4 address. Cross-project
  and cross-tenant probes addressed the peer's public IPv4 address.
- Each direction ran `ping` with numeric IPv4 output, 20 probes, a 0.2-second
  interval, and a two-second per-reply timeout. Statistics were calculated from
  the individual `time=` values. p95 uses linear interpolation at rank
  `(n - 1) * 0.95`.
- Firewall ingress admitted ICMP only from the same benchmark security group or
  an exact peer `/32`, as appropriate. SSH ingress for evidence collection was
  limited to the operator's exact `/32`. No customer workloads or data were
  accessed; traffic was synthetic ICMP between disposable benchmark VMs.
- The Compute VM creation surface for this CPU platform is regional and does
  not expose an availability-zone selector. The private-path pair was therefore
  colocated at the strongest available explicit scope: one region, project,
  VPC, and subnet.

Raw ping output, individual samples, provider responses, and a machine-readable
summary were retained in access-controlled operator evidence and were not
committed.

## Region limitation

An inter-region US comparison was not run. At measurement time, the official
[Nebius region catalog](https://docs.nebius.com/overview/regions) listed
`us-central1` as the only US region, so there was no second US region in which
to create a valid comparison VM. No latency value was inferred for this case.

## Teardown

After measurement, all benchmark VMs, disks, firewall rules, subnets, and
networks were deleted in dependency order. The three disposable projects were
then deleted. Provider-side reads confirmed every project was absent, final
tenant inventories contained no task-owned project, and all three local
Terraform states were empty. No chargeable benchmark resource remained.
