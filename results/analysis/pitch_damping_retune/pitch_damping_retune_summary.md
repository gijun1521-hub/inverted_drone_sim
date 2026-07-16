# Vane-only pitch damping retune

The final controller is **the selected robust low-control-effort candidate under the predeclared near-equivalence rule**: Rate P/I/D `0.08500000 / 0.00000000 / 0.01850000` with Angle P `25.00000000`. It is raw-score rank `15`, not rank 1, and it is not the mathematical raw-score optimum. All outer-loop, braking, capture, physics, actuator, geometry, and scenario settings were fixed. Moving-mass assist remained exactly `0.0 m/Nm`, and the physical moving mass remained centered.

The predeclared inclusive near-equivalence set is `valid raw aggregate score <= raw-score best + 0.010000`. The raw-score best is `0.396902568`, the selected score is `0.406150436`, and the accepted penalty is `0.009247867` (`2.330%`). `19` of `120` valid Stage 3C candidates are inside the band. The selected candidate reduces mean vane RMS by `2.554%`, vane total variation by `4.278%`, and vane command-rate RMS by `5.372%` relative to raw-score rank 1.

## Baseline comparison

**Stage 0 status: FAILED / NON-ACCEPTABLE baseline used for normalization only.** It failed `forward_1m:early_velocity_reversal` and `backward_1m:early_velocity_reversal`. Stage 0 is not a validated or acceptable controller; its absolute metrics and detector failures are preserved in the baseline artifacts.

| metric | baseline mean | selected mean | improvement |
| --- | ---: | ---: | ---: |
| tail RMS pitch (deg) | 4.09193152 | 0.64635017 | 84.204% |
| tail RMS pitch rate (deg/s) | 12.05265143 | 2.60551640 | 78.382% |
| tail RMS horizontal velocity (m/s) | 0.26294265 | 0.03559767 | 86.462% |

## Search and validation

Both the raw-score rank-1 candidate and the selected rank-15 candidate pass every physical and hard gate in all seven full-duration scenarios. The selected candidate eliminates early velocity reversal in both +1 m and -1 m cases, records exactly one monotonic controller capture-count increment in stick release, has no capture discontinuity or shaped-vx reversal, and has zero vane/servo-rate/mixer saturation. Detailed side-by-side metrics, chatter, symmetry, and hard-gate results are in `selection_comparison.md`, `selection_comparison.csv`, and `selection_comparison.json`.

- Candidate counts: `{"stage1_rate_pd":180,"stage2_angle_p":57,"stage3a_local_rate_pd":81,"stage3b_local_angle_p":9,"stage3c_crosscheck":120}`.
- Total unique scenario rows: `3136`.
- Rejected candidates: `122`.
- Boundary flags: `{"angle_p_at_max":false,"angle_p_at_min":false,"rate_d_at_max":false,"rate_d_at_min":false,"rate_p_at_max":false,"rate_p_at_min":false}`.
- Deterministic selected-candidate reruns: `["f3ce803930fd40e4c583d7cbd5710a8aa224a5ec37908d65ec2d1faa0a85d59c","f3ce803930fd40e4c583d7cbd5710a8aa224a5ec37908d65ec2d1faa0a85d59c"]`.
- Stage 0 mismatch override record: `["forward_1m:early_velocity_reversal","backward_1m:early_velocity_reversal"]`.

These results apply only to the same deterministic 2D analytical Single Fan Drone-inspired model. They do not establish real-flight stability, 3D stability, validated ArduPilot/Pixhawk behavior, HIL validity, hardware safety, or commercial-aircraft equivalence.
