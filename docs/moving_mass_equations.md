# Moving Mass Equations

The experimental moving-mass model uses body angle `theta` and relative mass
angle `q`. The absolute moving-mass angle is:

```text
theta_m = theta + q
```

For short-duration motion with no external torque:

```text
I_body * theta_dot + I_moving * (theta_dot + q_dot) = constant
```

If the initial angular momentum is zero:

```text
theta_dot = -I_moving / (I_body + I_moving) * q_dot
```

For finite relative motion:

```text
Delta theta_body ~= -I_moving / (I_body + I_moving) * Delta q
```

This relation is implemented in helper functions:

- `moving_mass_reaction_body_delta`
- `moving_mass_reaction_rate`
- `moving_mass_reaction_accel`

## CG Offset Torque

When the moving mass remains offset, the total CG shifts relative to the thrust
line. With lateral offset `d`:

```text
M_cg = T * d
```

The moving-mass model reports reaction, CG-offset, and vane moments separately.

## Return Reaction

Moving the mass back toward center creates a reverse reaction. This means moving
mass can help with fast transients or trim, but it cannot provide unlimited
continuous reaction torque without oscillation. Vane control may be needed to
cancel or damp the return motion.
