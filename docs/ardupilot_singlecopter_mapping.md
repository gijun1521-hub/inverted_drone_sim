# ArduPilot SingleCopter Mapping

## Purpose

This note documents how this repository's preparatory four-vane convention
relates to ArduPilot SingleCopter output naming. It is a mapping reference only;
it does not change simulator physics.

The current simulator remains a validated 2D analytical model with one
equivalent pitch-axis vane/moment downstream of a single fan.

## ArduPilot SingleCopter Outputs

ArduPilot's SingleCopter frame convention assigns the vehicle outputs as:

- Motor 1: Forward Flap
- Motor 2: Right Flap
- Motor 3: Back Flap
- Motor 4: Left Flap
- Motor 5: Motor

In this repository, those names correspond to the future four-vane convention:

| ArduPilot output | ArduPilot label | Simulator convention |
| --- | --- | --- |
| Motor 1 | Forward Flap | `front` vane |
| Motor 2 | Right Flap | `right` vane |
| Motor 3 | Back Flap | `rear` vane |
| Motor 4 | Left Flap | `left` vane |
| Motor 5 | Motor | scalar fan thrust command |

The table is intended to keep documentation, diagnostics, and future code
discussions aligned. It is not a claim of exact ArduPilot firmware behavior.

## Simulator Four-Vane Convention

The preparatory four-vane convention uses vehicle-relative vane positions:

- `front`: forward/downstream flap position
- `right`: right-side flap position
- `rear`: rear/back flap position
- `left`: left-side flap position

The current architecture-prep layer in `mixers.py` uses these names in
`FourVaneCommand`. The helper `equivalent_pitch_vane_to_four_vane()` maps the
current equivalent pitch vane into a front/rear differential command for
diagnostics:

```text
front = equivalent_pitch_vane
rear  = -equivalent_pitch_vane
right = 0
left  = 0
```

This adapter is not wired into the plant. The physics loop still receives one
equivalent pitch-axis vane angle.

## Current 2D Equivalent Pitch Model

The current rigid-body simulator state is:

```text
[x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

Only pitch-plane motion is modeled. The controller requests a pitch moment, the
existing 2D mixer converts that request into one equivalent vane angle, and the
plant applies analytical thrust, side force, and pitch moment terms. Motor lag,
servo lag, rate limits, saturation reporting, authority limits, and safety
checks remain part of the 2D model.

There are no independent front/right/rear/left servo states in the simulator
loop yet.

## Future Roll, Pitch, And Yaw Interpretation

For future 3D work, the convention is:

- pitch: differential front/rear vane deflection
- roll: differential right/left vane deflection
- yaw: common or swirl-producing vane deflection
- thrust: scalar fan motor command

Yaw and swirl are conceptual placeholders in the current codebase. The
simulator does not yet model yaw torque, duct swirl, four-vane aerodynamic
coupling, or full 3D rotational dynamics.

## Boundaries

This mapping should be read with the same limits as the rest of the analytical
simulator documentation:

- no full 3D dynamics
- no four independent servo dynamics in the simulator loop
- no yaw or swirl physics
- no exact ArduPilot firmware implementation
- no bench-calibrated or real-flight prediction

The goal is to keep naming and sign discussions clear before later work
introduces a broader vehicle model.
