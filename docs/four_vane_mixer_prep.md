# Four-Vane Mixer Preparation

## Purpose

This note describes the architecture-prep layer for a future four-vane
SingleCopter-style mixer. The current simulator behavior remains the validated
2D single-fan model with one equivalent pitch-axis vane/moment.

The new `mixers.py` module adds reusable data structures and a conceptual
adapter so future work can discuss front, right, rear, and left vane commands
without changing the existing physics loop.

## Current 2D Equivalent-Vane Model

The rigid-body simulator state is still:

```text
[x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

Only the pitch axis is modeled. The controller requests a pitch moment, the
existing `SingleCopterMixer` converts that request into one equivalent vane
angle command, and the plant applies one equivalent side force and pitch
moment. Motor lag, servo lag, rate limits, saturation flags, and authority
limits are unchanged.

## Real SingleCopter-Style Four-Vane Idea

A real single-fan ducted vehicle can use one motor and four downstream vanes or
flaps. Conceptually, independent front, right, rear, and left vane deflections
can contribute to roll, pitch, and yaw control.

This repository does not yet model those coupled 3D effects. The four-vane
types are preparation only.

## Coordinate Convention

The preparatory convention is:

- front and rear vanes form the pitch differential pair
- right and left vanes form the roll differential pair
- common same-sign vane deflection is reserved as a yaw placeholder
- thrust remains a scalar motor command

For the current 2D mode, only the pitch-equivalent channel is used. Roll and
yaw are held at zero by the adapter.

## Proposed Mixer Channels

- `roll_moment_cmd`: future differential right/left vane request
- `pitch_moment_cmd`: current equivalent pitch moment request
- `yaw_moment_cmd`: future common-mode swirl/yaw request
- `thrust_cmd`: scalar motor thrust request for diagnostics and future coupling

`ConceptualFourVaneMixer` maps those channels into a `FourVaneCommand`:

```text
front = +pitch + yaw
rear  = -pitch + yaw
right = +roll  + yaw
left  = -roll  + yaw
```

The mapping is intentionally simple and only establishes signs, saturation
reporting, and diagnostic fields.

## Current 2D Pitch Mapping

`equivalent_pitch_vane_to_four_vane()` maps the current equivalent pitch vane
into a front/rear differential command:

```text
front = equivalent_pitch_vane
rear  = -equivalent_pitch_vane
right = 0
left  = 0
```

The adapter carries the current pitch moment as `equivalent_2d_moment` and
marks `mode_2d_equivalent=True` in diagnostics.

## What Is Intentionally Not Modeled Yet

- full 3D position or attitude dynamics
- roll or yaw vehicle response
- four independent servo state variables
- aerodynamic interaction between vanes and duct flow
- yaw torque or swirl physics
- real ArduPilot SingleCopter firmware behavior
- calibrated hardware requirements or real-flight prediction

## Future Work

- full 3D rotational dynamics
- four independent servo models
- yaw torque and swirl modeling
- bench calibration of thrust and vane forces
- real flight log calibration
- comparison with ArduPilot SingleCopter motor/servo mapping
