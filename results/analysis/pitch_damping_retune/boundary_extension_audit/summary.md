# Targeted Stage 3C boundary-extension audit

## Decision

A valid extended candidate improved the raw aggregate score and the extension continued until the best point was no longer on an upper boundary.

- Selected raw-score controller: Rate P `0.09375`, Rate I `0.0`, Rate D `0.02100`, Angle P `25.0`.
- Selected raw aggregate score: `0.391509724441`.
- Current-reference score: `0.396902568460`.
- Score change versus current: `-0.005392844019` (negative is better).
- Valid candidates: `25` of `25`.
- Full-duration scenario runs: `175` search runs plus 14 deterministic validation reruns.
- Upper-boundary extension rounds: `1`.
- Deterministic rerun digest: `9647cd26cf9f98fbe6a8c15a0f3f695b01697b45c5e0f7efd50a54f781b60601`.

No near-equivalent or lower-control-effort tie-break was used. Selection is the valid raw-score local rank-1 point.

## Fixed configuration

Rate I remained `0.0`, Angle P remained `25.0`, and every outer-loop, braking, capture, vehicle, and Vane-only setting remained unchanged. Moving-mass assist gain, actual displacement, and target displacement remained exactly zero in every run.

## Gate enforcement

Every candidate was evaluated in all seven full-duration scenarios. Any failure of either mirrored `+1 m` or `-1 m` early-velocity-reversal gate was a hard rejection. The existing premature-pause, second-lobe, capture-count, capture-discontinuity, shaped-vx-sign, finite-state, crash/ground-contact, chatter, saturation, effective-parameter, and symmetry gates were all retained.

## Normalization status

Stage 0 is **FAILED / NON-ACCEPTABLE** and is used for normalization and comparison only. Its preserved detector failures are: `forward_1m:early_velocity_reversal; backward_1m:early_velocity_reversal`. It is not a passing validation controller.

## Boundary rounds

| round | candidates | valid | best P | best D | raw score | P upper | D upper |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | 25 | 25 | 0.09375 | 0.02100 | 0.391509724441 | False | True |
| 1 | 25 | 25 | 0.09375 | 0.02100 | 0.391509724441 | False | False |
