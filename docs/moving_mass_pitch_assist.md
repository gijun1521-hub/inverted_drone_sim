# Moving Mass Pitch Assist

## Purpose

This note documents the disabled-by-default 2D moving-mass pitch assist added to
the analytical single-fan simulator.

The research motivation is the user's single-fan drone concept: one ducted fan
or EDF with thrust-vectoring or vane control, plus an upper moving mass that may
act as a secondary attitude actuator. The long-term question is whether moving
center-of-mass control can reduce pitch/roll excursions during horizontal
movement and later help aggressive maneuvers such as pitch or roll flips.

This PR implements only the deterministic 2D foundation. It is not a flip
controller and not reinforcement learning.

## Current Model

The current rigid-body simulator remains a 2D pitch-plane model. When moving
mass assist is disabled, the plant state is unchanged:

```text
[x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

When `RigidBodyConfig.moving_mass.enabled` is true, the plant appends:

```text
[moving_mass_offset_m, moving_mass_velocity_m_s, moving_mass_target_m]
```

The moving mass actuator is deterministic and limited by:

- `mass_kg`
- `max_offset_m`
- `max_rate_m_s`
- `max_accel_m_s2`
- `initial_offset_m`

Defaults are safe: `enabled=False`, `mass_kg=0.5`, `max_offset_m=0.05`,
`max_rate_m_s=0.20`, and `max_accel_m_s2=1.0`.

## Quasi-Static CG Torque

This first implementation uses only a quasi-static pitch assist moment:

```text
moving_mass_moment_Nm = moving_mass_kg * gravity * moving_mass_offset_m
```

Sign convention: a positive moving-mass offset produces a positive pitch moment
in this 2D pitch-assist channel. A negative offset produces the opposite moment.

This is an analytical assist channel for comparing attitude response. It is not
yet a full moving-body dynamics model.

## Reaction Kick Versus CG Torque

Two effects matter for future hardware studies:

- Quasi-static CG torque: a sustained moment from the mass being held away from
  the vehicle center line.
- Reaction kick or inertial effect: a transient body reaction caused by
  accelerating the mass.

This PR implements only the quasi-static CG torque in the 2D pitch model. The
reaction kick model is intentionally left for future work so it can be isolated,
tested, and calibrated separately.

## Headless Use

Headless LOITER scenarios can enable moving mass assist and either command a
fixed target offset or use a simple proportional assist path:

```text
moving_mass_target_m = moving_mass_assist_gain_m_per_Nm * desired_pitch_moment
```

The target is clamped by the moving mass offset limit, then rate and
acceleration limited by the actuator model.

Logged rows and metrics include moving-mass offset, velocity, target, moment,
and saturation activity so vane-only and vane-plus-moving-mass cases can be
compared without opening pygame.

## Intended Uses

- Compare vane-only and vane-plus-moving-mass headless scenarios.
- Measure whether moving mass reduces pitch excursions during horizontal
  movement or impulse recovery.
- Prepare a deterministic foundation for future flip feasibility studies.

The goal is to create a measurement framework, not to prove improvement in this
first PR.

## Limitations

- 2D only.
- Pitch only.
- No roll.
- No yaw.
- No full 3D dynamics.
- No reaction kick yet.
- No scripted flip controller yet.
- No trajectory search yet.
- No reinforcement learning yet.
- No exact ArduPilot firmware behavior.
- No bench-calibrated or real-flight prediction.

## Future Work

- Add an isolated reaction kick / inertial moving-mass model.
- Build a scripted pitch-flip baseline.
- Add trajectory search such as CEM or random shooting.
- Add a PPO/RL environment only after deterministic dynamics are stable.
- Calibrate moving mass hardware with bench measurements.
- Compare against real flight logs.
