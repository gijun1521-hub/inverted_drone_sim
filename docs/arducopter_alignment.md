# ArduCopter Alignment

This simulator is ArduCopter-inspired, not a copy of ArduPilot firmware and not experimentally calibrated.

## What Is Similar

- Stabilize: pilot A/D input commands a lean angle and the attitude loop self-levels when the stick is released.
- AltHold: W/S commands climb or descent rate. Centered stick holds the captured altitude target.
- Loiter: A/D commands horizontal movement speed. Releasing the stick starts a delayed braking phase, then the vehicle holds the stopping position.
- The control stack is cascaded: pilot shaping, position, velocity, acceleration-to-lean, attitude, rate, mixer, and actuator dynamics.
- Saturation and authority limits are reported rather than hidden.

## 2D Simplifications

- One horizontal axis, `x`.
- One vertical axis, `z`.
- Pitch angle `theta` is the only attitude axis.
- No yaw, heading, roll, EKF, GPS, or full 3D dynamics are modeled yet.

## SingleCopter Simplifications

The current physics model represents one equivalent pitch vane/moment downstream of a single fan. It does not yet model the full four-flap SingleCopter dynamics.

Real ArduPilot SingleCopter vehicles use one motor and four independent flaps. Roll, pitch, and yaw are created by flap deflection. Yaw is controlled by all four fins pointing slightly clockwise or counter-clockwise. This simulator only models the pitch-axis equivalent for now.

The repository now includes a preparation layer in `mixers.py` and
`docs/four_vane_mixer_prep.md` with conceptual commands for:

- `front_flap`
- `right_flap`
- `rear_flap`
- `left_flap`

Those structures are not wired into the plant yet. They are a documentation,
type, and diagnostics layer for future four-vane work.

## Mapping To This Research Simulator

The goal is analytical clarity. LOITER should visibly drift, brake, and settle under finite thrust, finite vane authority, servo lag, motor lag, and rate limits. If authority is too low, position hold may fail; the expected behavior is honest saturation reporting, not perfect holding.
