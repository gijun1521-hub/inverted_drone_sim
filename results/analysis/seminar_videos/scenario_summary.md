# Seminar scenario comparison

These deterministic 2D simulations compare the same 2.0 kg vehicle with its 0.5 kg moving mass physically present in both variants. The Vane-only case commands and maintains a 0 mm offset; it does not remove the mass.

Both variants use the PR #19 LOITER controller from `params/loiter_transient_provisional.json`. The active moving-mass gain is `0.1325 m/Nm`, selected by the four-direction staged sweep documented in `moving_mass_gain_selection.md`.

The final 2.0 seconds form the tail window. Settling requires the remainder of the 8-second run to stay within 0.05 m position error and 0.05 m/s horizontal speed. Unsettled results are reported honestly rather than extending or cropping the video.

## Raw metrics from the exact rendered runs

| Scenario | Variant | Tail RMS x (m) | Tail RMS vx (m/s) | Tail p-p x (m) | Tail path (m) | Abs final x error (m) | Excursion/overshoot (m) | Peak pitch (deg) | Tail RMS pitch (deg) | Vane RMS (deg) | Vane max (deg) | Mass max (mm) | Tracking RMS (mm) | Settled | Pause | Second lobe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| LOITER disturbance recovery | Vane-only | 0.09058 | 0.25333 | 0.27364 | 0.44143 | 0.12296 | 0.67768 | 7.016 | 3.965 | 0.657 | 1.075 | 0.00 | 0.000 | no | no | no |
| LOITER disturbance recovery | Moving-mass assist | 0.07284 | 0.10891 | 0.20595 | 0.20595 | 0.04148 | 0.63627 | 5.300 | 0.346 | 0.168 | 0.679 | 5.77 | 0.396 | yes | no | no |
| +1 m position command and hold | Vane-only | 0.12582 | 0.24429 | 0.22727 | 0.43527 | 0.09147 | 0.22803 | 6.120 | 4.217 | 0.702 | 4.065 | 0.00 | 0.000 | no | no | no |
| +1 m position command and hold | Moving-mass assist | 0.06298 | 0.02383 | 0.03694 | 0.04133 | 0.03586 | 0.07280 | 3.691 | 0.200 | 0.210 | 4.065 | 4.17 | 1.323 | yes | no | no |

## Vane-only to assist percentage changes

Negative values mean the assist result is lower than Vane-only.

| Scenario | Tail RMS x | Tail RMS vx | Tail p-p x | Tail path | Final error | Excursion/overshoot | Peak pitch | Tail RMS pitch | Vane RMS | Mass travel delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LOITER disturbance recovery | -19.59% | -57.01% | -24.73% | -53.34% | -66.27% | -6.11% | -24.46% | -91.28% | -74.36% | +5.77 mm |
| +1 m position command and hold | -49.94% | -90.24% | -83.74% | -90.50% | -60.80% | -68.07% | -39.68% | -95.26% | -70.08% | +4.17 mm |

## PPT Page 3 replacement values

### LOITER

- Tail RMS position error: 0.09058 m -> 0.07284 m (19.6% reduction)
- Tail path length: 0.44143 m -> 0.20595 m (53.3% reduction)
- Vane command RMS: 0.657 deg -> 0.168 deg (74.4% reduction)
- Final position error: 0.12296 m -> 0.04148 m (66.3% reduction)

### +1 m

- Tail RMS position error: 0.12582 m -> 0.06298 m (49.9% reduction)
- Peak pitch: 6.120 deg -> 3.691 deg (39.7% reduction)
- Final position error: 0.09147 m -> 0.03586 m (60.8% reduction); assist overshoot 0.07280 m
- Vane command RMS: 0.702 deg -> 0.210 deg (70.1% reduction)

## Interpretation limit

These results compare implementations inside the same deterministic 2D model. They are not evidence of real-flight equivalence, hardware safety, 3D stability, or calibrated aerodynamic performance.
