# Safety

A machine that moves a plate near an animal's face deserves a few minutes of
thought. Most of the protection here is mechanical; the software just avoids
making things worse.

## Pinch risk

This is the one that matters. Mitigations, in order of importance:

1. **A light lid.** 3 mm acrylic at ~80 g cannot hurt a cat regardless of what
   the software does. This is worth more than every software measure combined.
2. **A lid that lifts and lowers, not one that slides.** A hinged plate pushes a
   head out of the way as it descends. A sliding plate can trap.
3. **Slew-rate limiting.** `actuator.move_speed_deg_s` (default 90°/s) makes the
   lid take about a second to travel instead of 0.15 s at full servo torque. A
   cat has time to withdraw and can physically push back against the motion.
4. **`detach_when_idle: true`.** Once the lid settles, PWM stops and the servo
   goes limp. A closed lid can then simply be lifted by a determined cat — which
   is the correct failure mode — and the servo does not cook itself holding.
5. **`close_delay_s`** (default 10 s). The bowl must look empty continuously for
   ten seconds before the lid comes down, so a cat that glances away does not get
   the lid on its ears.

The one case where the software deliberately closes on a present cat is
`close_on_intruder` — a wrong cat at an open bowl, after `intruder_grace_s`.
That is the whole point of the project, so it cannot be avoided, only made
gentle. If you are at all unsure, set `close_on_intruder: false` at first and
watch what actually happens with `run --collect` before enabling it.

## Failure modes and what happens

| Failure | Behaviour |
|---|---|
| Process crashes | systemd restarts it; `FeederApp.build()` drives every lid closed before anything else |
| Pi loses power | Servos go limp. A hinged lid falls shut under its own weight — design for that |
| Camera unplugged or frozen | After 3 s of stale frames the bowl is treated as unoccupied, so the lid closes after `close_delay_s`. Food is denied, not spilled |
| Model unsure | Below `min_confidence` every frame is `unknown`, and `unknown` can never win a vote. The lid stays shut |
| Cat never leaves | `max_open_s` (default 15 min) closes the lid regardless |
| Lid jammed | The servo stalls, then detaches after the move. Check the status page — a bowl stuck reporting `open` with no `closed` event is the signal |

The consistent bias is **fail closed**: when in doubt, no food. That is the safe
direction for a cat on a prescription diet, and the annoying-but-harmless
direction for everyone else.

## Before you leave it running unattended

- Run for a few days with `--dry-run` (lids simulated) and read the event log.
  Confirm the right cat is being recognised before anything moves.
- Then run with `close_on_intruder: false` and watch the lids move for real.
- Make sure there is always another source of food and water that this machine
  cannot deny, until you trust it. A misconfigured `min_confidence` that rejects
  everything means a cat that does not eat.
- Check `logs/events-*.jsonl` for a cat whose `opened` events stop. That is worth
  a vet conversation regardless of the machine.

## Electrical

- Fuse the servo supply, and keep the PSU and wiring away from the water bowl.
- Tie the servo PSU ground to the Pi ground; never power servos from the Pi.
- Shut down cleanly (`sudo systemctl stop catbowl`) before unplugging, so the
  lids park closed.
