# Moving Mass Pitch Assist

## Purpose

This note documents the disabled-by-default 2D moving-mass pitch assist added to
the analytical single-fan simulator.

The research motivation is the user's single-fan drone concept: one ducted fan
or EDF with thrust-vectoring or vane control, plus an upper moving mass that may
act as a secondary attitude actuator. The long-term question is whether moving
center-of-mass control can reduce pitch/roll excursions during horizontal
movement and later help aggressive maneuvers such as pitch or roll flips.

The implementation remains a deterministic 2D foundation. It is not a flip
controller and does not include reinforcement learning.

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

## Mass and geometry convention

`RigidBodyConfig.m` is the total vehicle mass, including the moving mass. The
moving mass is not added to it. In total-COM geometry mode:

```text
m_total = cfg.m
m_m = cfg.moving_mass.mass_kg
m_b = m_total - m_m
```

The fixed-body COM is the body-fixed geometry origin. Body coordinates are
written as `[body_right, body_up]`. The moving-mass rail rotates with the body,
and the moving-mass position is

```text
r_b = [0, 0]
r_m = [moving_mass_offset, moving_mass_body_up_offset_m]
r_C = (m_b * r_b + m_m * r_m) / m_total
```

The thrust application point is the geometry origin, `r_T = [0, 0]`. The vane
force application point is `r_V = [0, -cfg.l]`. Their arms about the
instantaneous total COM are therefore `r_T - r_C` and `r_V - r_C`.

An arm in body coordinates is converted to world coordinates with

```text
arm_world = arm_body_right * body_right + arm_body_up * body_up
```

The simulator then uses its established 2D moment convention for every
attitude:

```text
M = arm_z * force_x - arm_x * force_z
```

This geometry calculation is not replaced with an attitude-specific expression
such as `l * T * sin(theta)`.

When total-COM geometry is active, state `x` and `z` are the instantaneous
total-system COM world position. The body origin and force application points
are reconstructed from that position and `r_C`. Internal mass acceleration does
not directly move the total COM or produce a reaction term in this model.

Disabling the moving-mass actuator does not remove its physical mass. It fixes
the lateral offset at zero and preserves the 8-state representation. A nonzero
`moving_mass_body_up_offset_m` can therefore still move the total COM vertically
and change the vane arm.

## Legacy and total-COM modes

The backward-compatible defaults are:

```text
use_total_com_geometry = False
use_legacy_gravity_offset_moment = True
moving_mass_body_up_offset_m = 0.12
```

They preserve the original quasi-static pitch-assist term when the moving-mass
actuator is enabled:

```text
legacy_moving_mass_moment = m_m * g * moving_mass_offset
```

The optional geometry model must be selected explicitly:

```text
use_total_com_geometry = True
use_legacy_gravity_offset_moment = False
```

Enabling both models raises a configuration error. Adding them would count two
alternative approximations of the same moving-mass authority.

Uniform gravity has no net external torque about the instantaneous total COM.
In geometry mode the new pitch authority instead comes primarily from the
thrust and vane force lines being offset from that COM. Near hover,

```text
T * (m_m / m_total) * moving_mass_offset
approximately equals
m_m * g * moving_mass_offset
```

so the old and new results can look similar. Away from hover they diverge: the
geometry moment follows actual thrust. At zero thrust its thrust-line moment is
zero, whereas the legacy gravity-offset term would remain nonzero.

## Prototype example

The current design assumptions used for focused tests and examples are:

```text
cfg.m = 2.0 kg
moving_mass.mass_kg = 0.5 kg
moving_mass.max_offset_m = 0.05 m
moving_mass_body_up_offset_m = 0.12 m
```

At a lateral offset of `+0.05 m`, the total COM is
`[+0.0125, +0.03] m` in `[body_right, body_up]`. At zero lateral offset,
the right shift is zero but the upward shift remains `+0.03 m`; axial thrust
then has zero moment while the vane lever arm is longer by `0.03 m`. These are
current design assumptions, not calibrated flight values, and they do not
replace the existing global `RigidBodyConfig.m` default.

## Reaction Kick Versus CG Torque

Two effects matter for future hardware studies:

- Quasi-static CG torque: a sustained moment from the mass being held away from
  the vehicle center line.
- Reaction kick or inertial effect: a transient body reaction caused by
  accelerating the mass.

This implementation includes only quasi-static total-COM geometry or the legacy
offset term. Reaction kick, moving-mass acceleration reaction, rail/servo
reaction dynamics, and internal momentum coupling are intentionally deferred so
they can be isolated, tested, and calibrated separately.

Pitch inertia also remains the existing fixed `RigidBodyConfig.Iyy`. That value
already uses `cfg.m`, which includes the moving mass. Adding a moving point-mass
inertia before separating fixed-body inertia could double-count the moving mass;
position-dependent inertia is therefore deferred.

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
- No moving-mass acceleration reaction or internal momentum coupling.
- No position-dependent pitch inertia.
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
