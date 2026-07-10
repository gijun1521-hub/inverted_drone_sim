# Moving Mass Comparison Analysis

## Purpose

This headless analysis compares the legacy gravity-offset moving-mass model and
the optional total-COM geometry model in identical scenarios. It changes only
experiment configuration and reporting; it does not change rigid-body physics,
controller logic, gains, or scenario definitions.

The comparison separates three effects:

1. changing from the historical nominal-CG model to total-COM geometry while
   the moving mass remains laterally centered;
2. actively moving the mass within total-COM geometry;
3. using the legacy `m_m * g * offset` formulation versus thrust- and
   vane-line moments about the instantaneous total COM.

## Variants

All six variants run by default.

### Historical baseline

- `vane_only`: moving-mass actuation and total-COM geometry are disabled. The
  compatible legacy flag remains configured, but no legacy moment is active.

### Legacy gravity-offset variants

- `moving_mass_fixed_target`: the existing fixed lateral target using the
  legacy gravity-offset moment.
- `moving_mass_proportional_assist`: the existing target proportional to
  desired pitch moment using the legacy gravity-offset moment.

### Total-COM geometry baseline

- `total_com_geometry_centered`: total-COM geometry is active while the
  moving-mass actuator is disabled. The state remains 8-dimensional, lateral
  offset is fixed at zero, and the physical moving mass can still shift the COM
  vertically through `moving_mass_body_up_offset_m`.

### Total-COM geometry active variants

- `total_com_geometry_fixed_target`: uses exactly the same fixed-target command
  definition as `moving_mass_fixed_target`.
- `total_com_geometry_proportional_assist`: uses exactly the same proportional
  command definition as `moving_mass_proportional_assist`.

Only the plant formulation differs between paired legacy and geometry active
variants. The comparison does not override total mass, moving-mass mass, travel
limit, vertical location, controller gains, or scenario definitions.

## Baselines And Delta Conventions

Every row has two explicit baseline fields. Baselines are matched by parameter
source, scenario name, and variant name; row position is never used. A missing
or duplicate required baseline is an analysis error.

- `baseline_variant` is `vane_only`. Existing fields such as
  `delta_rms_theta_deg` remain differences versus this historical baseline.
- `mode_baseline_variant` is `total_com_geometry_centered` for the two active
  geometry variants and `vane_only` for the other variants. New
  `delta_vs_mode_baseline_*` fields answer whether active mass motion changed a
  metric relative to the same geometry with centered mass.

For raw error and angle deltas, negative means the metric decreased and positive
means it increased. `attitude_improvement_score` is a separate score: positive
means the average of max and RMS pitch angle decreased versus `vane_only`.

The geometry-centered row itself is compared with `vane_only`, showing the
overall formulation and vertical-COM effect. Geometry active variants should
not be judged only against `vane_only`, because that would mix the geometry
change with active lateral mass motion.

## Schema And Effective Configuration

Stable model metadata includes:

- `moving_mass_model_mode`: `disabled`, `legacy_gravity_offset`, or
  `total_com_geometry`
- `baseline_variant`
- `mode_baseline_variant`
- `total_com_geometry_active`
- `legacy_gravity_offset_active`
- `state_dimension`

Effective configuration is reported rather than silently replaced:

- `total_mass_kg`
- `moving_mass_mass_kg`
- `effective_moving_mass_max_offset_m` (the configured travel limit)
- `moving_mass_body_up_offset_m`
- `effective_moving_mass_max_rate_m_s`
- `effective_moving_mass_max_accel_m_s2`

The existing `moving_mass_max_offset_m` column remains the maximum actual offset
observed during a run. Existing CSV columns and vane-only delta meanings are
preserved. Requested variant flags are checked against the effective headless
configuration before a row is published; a model-mode mismatch is an analysis
error rather than a mislabeled result.

## Geometry Diagnostics

The following metrics are explicit maxima or RMS values over the logged
time-series:

- `max_abs_total_com_body_right_m`
- `max_abs_total_com_body_up_m`
- `max_abs_thrust_moment_from_com_offset`
- `rms_thrust_moment_from_com_offset`
- `max_abs_vane_moment_about_total_com`
- `max_abs_legacy_moving_mass_moment`

The two COM metrics use metres. All thrust, vane, and legacy moment metrics use
N·m. Maxima and RMS values use every logged post-step time-series sample in the
run.

Geometry-specific COM and thrust-moment metrics are zero in inactive modes.
Legacy moment is zero in total-COM geometry mode. The centered geometry variant
can have nonzero vertical COM shift while its lateral COM and axial-thrust
moment remain zero. `vane_moment_about_total_com` is the complete vane moment
about the selected COM, not a moment attributed solely to moving-mass motion.

## Scenarios And Running

The default scenario is `pitch_assist_probe`. `--scenario all` runs:

- `pitch_assist_probe`
- `stick_move_release`
- `horizontal_impulse_recovery`
- `initial_x_offset_recovery`
- `authority_stress`

```bash
python compare_moving_mass_assist.py --scenario pitch_assist_probe --no-plots
python compare_moving_mass_assist.py --scenario all --duration 0.1 --no-plots
```

Outputs are written to:

- `results/analysis/moving_mass_comparison/moving_mass_comparison.csv`
- `results/analysis/moving_mass_comparison/moving_mass_comparison.md`

Plots remain optional and are skipped with `--no-plots`.

## Interpretation And Limitations

Safe conclusions are scenario-specific, for example: "Under this analytical
scenario, the geometry formulation changed RMS theta by ..." or "Relative to
the geometry-centered baseline, active mass motion reduced this metric by ...".

The analysis does not establish that total-COM geometry is automatically more
accurate in real flight or that moving mass is globally beneficial. It does not
include reaction kick, moving-mass acceleration reaction, internal momentum
coupling, position-dependent inertia (`Iyy`), 3D dynamics, yaw/swirl physics,
four-vane physics wiring, flip control, trajectory search, reinforcement
learning, or real-flight calibration. Loaded parameter values are design
assumptions, not calibrated flight values.
