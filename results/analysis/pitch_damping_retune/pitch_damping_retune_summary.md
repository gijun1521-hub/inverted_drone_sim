# Vane-only pitch damping retune

The final selected controller is the **valid raw-score local rank 1** from the targeted boundary extension: Rate P/I/D `0.09375000 / 0.00000000 / 0.02100000`, Angle P `25.00000000`. Its raw aggregate score is `0.391509724441`, improving the previous Stage 3C rank-1 score `0.396902568460` by `0.005392844019` (1.359%). No near-equivalent lower-effort tie-break was used.

**Stage 0 status: FAILED / NON-ACCEPTABLE; normalization and comparison only.** Its absolute metrics and both early-reversal failures remain preserved. Stage 0 is not a validated controller.

| metric | Stage 0 mean | selected mean | improvement |
| --- | ---: | ---: | ---: |
| tail RMS pitch (deg) | 4.09193152 | 0.52554642 | 87.157% |
| tail RMS pitch rate (deg/s) | 12.05265143 | 2.37222359 | 80.318% |
| tail RMS horizontal velocity (m/s) | 0.26294265 | 0.03023909 | 88.500% |

All 25 targeted candidates passed all seven full-duration physical, behavioral, symmetry, chatter, saturation, and early-reversal gates. The selected controller eliminates early velocity reversal in both +1 m and -1 m cases, records exactly one stick-release capture, has no capture discontinuity or shaped-vx reversal, and keeps moving-mass gain, actual displacement, and target displacement exactly zero. Two fresh deterministic seven-scenario reruns were byte-identical at the metrics level (`9647cd26cf9f98fbe6a8c15a0f3f695b01697b45c5e0f7efd50a54f781b60601`).

The initial audit best was on the Rate D upper boundary at `0.02100`; one same-step extension to `0.02150` made the selected `D=0.02100` point interior. Rate P `0.09375` was already interior. This validates that the prior Stage 3C upper bounds did not conceal a better continuing-edge point.

These results apply only to the same deterministic 2D analytical model and do not establish real-flight, 3D, HIL, or hardware safety.
