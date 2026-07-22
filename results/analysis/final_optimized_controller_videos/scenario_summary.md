# Final independently optimized controller videos

These deterministic 2D simulation-only renders use the two full-precision profiles selected in PR #25. They do not claim hardware, HIL, or real-flight validation.

Both variants use identical scenario definitions, vehicle/moving-mass physics, actuator limits, integration timesteps, camera bounds, frame timing, and resolution. Only the profile-selected controller and assist behavior differ.

| Scenario | Variant | Profile | Gain (m/Nm) | Final |x error| (m) | Overshoot/excursion (m) | Settling (s) | Peak pitch (deg) | Mass max (mm) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LOITER: 8 N world-frame disturbance | Independently optimized Vane-only | `results/analysis/variant_controller_optimization/profiles/vane_only.json` | 0 | 0.019562 | 0.496378 | 5.385 | 7.638 | 0.000 |
| LOITER: 8 N world-frame disturbance | Independently optimized moving-mass assist | `results/analysis/variant_controller_optimization/profiles/moving_mass_assist.json` | 0.018680405097860724 | 0.017559 | 0.495894 | 5.060 | 7.434 | 6.896 |
| LOITER: absolute +1 m command | Independently optimized Vane-only | `results/analysis/variant_controller_optimization/profiles/vane_only.json` | 0 | 0.003345 | 0.069968 | 6.250 | 5.912 | 0.000 |
| LOITER: absolute +1 m command | Independently optimized moving-mass assist | `results/analysis/variant_controller_optimization/profiles/moving_mass_assist.json` | 0.018680405097860724 | 0.000005 | 0.070767 | 6.315 | 5.920 | 6.728 |

## Validation

- PR #25 selected-scenario comparisons: **PASS** (48 explicit checks).
- Vane-only assist gain, target, offset, velocity, and acceleration: **exactly zero**.
- Moving-mass assist gain: **loaded from and matched to the PR #25 selected candidate**.
- Physical moving mass remains installed in both variants; the Vane-only rail state is locked at center.
