# Moving Mass Comparison Analysis

## Purpose

This analysis measures whether the existing disabled-by-default 2D moving-mass
pitch assist model changes headless LOITER behavior compared with the vane-only
baseline.

It is a reporting workflow only. It does not change physics, controller logic,
or default simulator behavior.

## Variants

The comparison runs the same scenario under three variants:

- `vane_only`: moving mass disabled.
- `moving_mass_fixed_target`: moving mass enabled with a fixed target offset.
- `moving_mass_proportional_assist`: moving mass enabled with target offset
  proportional to desired pitch moment.

The defaults are intentionally conservative. They establish measurement
infrastructure rather than tuned controller gains.

## Scenarios

The default run uses `pitch_assist_probe`. The CLI can also run a single named
headless scenario or `--scenario all`, which includes:

- `pitch_assist_probe`
- `stick_move_release`
- `horizontal_impulse_recovery`
- `initial_x_offset_recovery`
- `authority_stress`

## Metrics

The CSV and Markdown reports include position, attitude, saturation, and moving
mass metrics:

- `final_abs_x_error`
- `rms_x_error`
- `final_abs_z_error`
- `rms_z_error`
- `max_theta_deg`
- `rms_theta_deg`
- `final_theta_deg`
- `max_omega_deg_s`
- vane, mixer, servo, authority, and motor saturation percentages
- `moving_mass_max_offset_m`
- `moving_mass_saturation_percent`
- effective moving mass settings

Derived deltas are computed against `vane_only` for the same scenario and
parameter file. Negative delta values mean the moving-mass variant reduced that
metric. Positive `attitude_improvement_score` means max and RMS pitch angle
decreased on average.

## Running

```bash
python compare_moving_mass_assist.py --scenario pitch_assist_probe --no-plots
python compare_moving_mass_assist.py --scenario all --no-plots
```

Outputs are written to:

- `results/analysis/moving_mass_comparison/moving_mass_comparison.csv`
- `results/analysis/moving_mass_comparison/moving_mass_comparison.md`

Plots are optional and skipped with `--no-plots`.

## Safe Conclusions

Use cautious, scenario-specific language when interpreting the report:

- "Improved max_theta in this scenario."
- "Reduced RMS theta but increased final x error."
- "No improvement under this configuration."
- "Needs tuning."

Do not claim that moving mass is globally better unless broader data and
calibration support that conclusion.

## Limitations

- Still 2D only.
- Uses the existing quasi-static CG pitch moment only.
- Does not yet include an explicit total-CG geometry shift.
- Does not yet include a moving-mass-induced thrust-line pitch moment.
- Does not yet include position-dependent inertia (`Iyy`) changes.
- Does not yet include inertial reaction kick from moving-mass acceleration.
- No flip controller.
- No reinforcement learning.
- No full 3D dynamics.
- No four-vane physics wiring.
- No real-flight calibration.

Results from the current simplified model are provisional. They must not be
used as final validation for a large moving mass, such as approximately 0.5 kg,
until the missing physics terms above are modeled and calibrated.
