# Moving Mass Concept

The single rigid-body vane model treats the drone body and battery as one rigid
mass. That is useful for vane sign checks, but it cannot represent a heavy
battery/Raspberry-Pi module moving near the top of the drone.

`MovingMassSingleFan2D` is therefore separate. It is experimental and simplified.

## Effects To Capture

- Moving the mass quickly creates reaction torque on the body.
- Holding the mass offset shifts the total CG.
- A shifted CG can move the thrust line away from the total CG and create a
  sustained pitch moment.

This is not just static CG shift. It combines internal angular momentum exchange
and thrust-offset torque.

## Current Simplification

The first implementation prioritizes a rotating module:

- `q_mass` is the relative moving-mass angle.
- `qdot_mass` is the relative angular velocity.
- q commands pass through lag, rate limiting, and acceleration limiting.
- Internal q acceleration applies opposite reaction to the body.
- Total CG is computed from body mass and moving mass position.

The formulas are intended for sign and architecture tests only. The mass,
inertia, hinge location, servo behavior, and actual force coefficients must be
measured experimentally before using this for design decisions.
