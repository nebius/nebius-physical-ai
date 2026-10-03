<!-- register: agent reference | reader: Workbench operator agent | consumed: retrieved for an authorized VM maintenance pause -->

# Keep VM maintenance jobs paused across reboot

Use this reference when an authorized pause of owned systemd jobs must survive
a VM reboot, or when reviewing whether paused jobs ran. Identify the exact
timers, associated services and maintenance owner. For NPA workflow cancellation
or infrastructure teardown, follow [teardown-and-cost](../../../atomic/teardown-and-cost/SKILL.md).
Routine status requests do not authorize disabling healthy timers.

Record both active and enabled states. Stopping a timer leaves its enablement
unchanged; an enabled calendar timer with `Persistent=true` can run missed work
when activated after reboot. An inactive service may already have completed
or failed after producing effects.

Before an authorized change, record existing enablement links, including custom
links. `disable` can remove links that `enable` will not recreate. Inspect
`[Install]`, `Also=` and user/global enablement for effects on other units and
remaining activation paths. Disable and stop only units in the authorized pause
scope. Check whether their services are still running and resolve other
activation paths through the owning maintenance procedure.

After reboot, inspect timer state, service execution timestamps and the current
boot journal before claiming the pause held. Check whether a service wrote data
or triggered an `OnFailure` service, without running its scripts or sending test
alerts. Restore the recorded original states and enablement links when the
maintenance owner authorizes resumption. Supplied receipts support conclusions
about the recorded boot; they do not establish the host's current state.

Related guidance: [PS Services compute reference](https://github.com/nebius/nebius-ps-services/blob/0ee453db63bbbe5ac30f91153b57fea3369a7c53/skills/nebius/references/compute.md#maintenance-pauses-across-vm-reboot)
([contribution 194](https://github.com/nebius/nebius-ps-services/pull/194)),
[systemctl](https://www.freedesktop.org/software/systemd/man/latest/systemctl.html),
[systemd timers](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html).
