# Pitch damping retune methodology

This workflow compares the PR #19 Vane-only controller with pitch-retuned Vane-only candidates in the same deterministic 2D analytical model. It is not real-flight, 3D, ArduPilot, Pixhawk, HIL, or hardware-safety validation.

## Fixed vehicle and controller values

- Total mass: 2.0 kg; physical moving mass: 0.5 kg; fixed-body mass: 1.5 kg.
- Moving mass enabled with total-COM geometry, centered actual/target position, and assist gain exactly 0.0 m/Nm.
- Rate I: 0.0; Position P: 0.55; Velocity P: 0.70.
- Brake delay/acceleration/jerk: 0.50 s / 1.00 m/s^2 / 3.00 m/s^3.
- Capture actual/desired velocity thresholds: 0.08 / 0.02 m/s.
- Persistent capture, shaped-velocity zero clamp, and capture without target jump remain enabled.
- Physics/controller timesteps: 0.005 s / 0.01 s; seed: 0.

## Scoring

Each metric is divided by the fresh Stage 0 value. Scenario score weights are 30% tail RMS pitch, 20% tail RMS pitch rate, 15% tail RMS horizontal velocity, 15% tail path, 10% tail RMS position error, 5% vane RMS, and 5% vane total variation. Each group uses mean + 0.5 times worst scenario; the final combination is 45% isolated pitch recovery and 55% integrated LOITER.

Stage 0 is a **FAILED / NON-ACCEPTABLE baseline used for normalization only**. Both +1 m and -1 m absolute-target runs fail the early-velocity-reversal hard gate. It is not a validated or acceptable controller, and its use does not relax any candidate gate.

The near-equivalent set was defined before final Stage 3C ranking inspection as every valid candidate whose raw aggregate score is less than or equal to the raw-score best plus exactly `0.010000`. Inside that inclusive set, the tie-break order is lower mean vane RMS, lower Rate D, lower vane total variation, better symmetry, then raw score.

## Chatter thresholds fixed before selection

- Command deadband: 0.5 deg.
- Meaningful command rate: 10.0 deg/s.
- Maximum meaningful sign changes: 80.
- Maximum total variation per second: 45.0 deg/s.
- Maximum tail high-frequency energy: 0.35 deg^2.
- High-frequency moving-average window: 0.1 s.

Logarithmic decrement and damping ratio are reported only when at least three significant absolute pitch peaks provide at least two monotonically decaying ratios; otherwise the reason is recorded.
