# Controller gain grid search

`sweep_controller_gains.py` runs a deterministic, staged search through the same headless plant, motor, servo, mixer, and cascaded controller used by the interactive simulator. It changes parameter overrides only. It does not change rigid-body physics, actuator dynamics, active controller equations, or global defaults.

The workflow addresses persistent ribbon/comet-like LOITER motion with repeatable scenario metrics instead of manual gain guessing. Results are analytical comparisons within this simulator, not real-flight optimality claims.

## Stages

The stages always follow this dependency order when `--stage all` is used:

1. `rate_pd`: moving mass disabled, `atc_rat_pit_i=0`, joint coarse P/D grid, then a half-step local search around the five best accepted coarse candidates.
2. `rate_i`: fixes the top three P/D candidates and sweeps I, including `I=0` and positive/negative persistent moment-bias cases.
3. `attitude_p`: fixes the selected rate PID and sweeps Angle P with ±5, ±10, and ±15 degree initial attitudes.
4. `loiter_xy`: fixes the rate and attitude loops and sweeps only the active horizontal position P and velocity P fields. Moving mass remains disabled.
5. `moving_mass_gain`: fixes the selected vane-only controller, enables total-COM geometry, and sweeps the proportional assist gain against a mode-matched centered baseline.

The active LOITER implementation does not use `psc_ne_vel_i` or `psc_ne_vel_d`. They are intentionally absent from every candidate key and are not reported as optimized.

## Full grids

| Stage | Grid | Candidate count before dependent replication |
| --- | --- | ---: |
| RATE P-D coarse | P `0.005..0.080` by `0.005`; D `0.0000..0.0080` by `0.0005` | 272 |
| RATE I | I `0.000..0.030` by `0.002` | 16 per selected P-D pair |
| Angle P | `2.0..10.0` by `0.5` | 17 |
| LOITER P/P | position P `0.1..1.5` by `0.1`; velocity P `0.2..2.5` by `0.1` | 360 |
| Moving-mass gain | `0.000..0.080 m/Nm` by `0.0025` | 33 |

Decimal grids are constructed from decimal arithmetic so endpoints and counts are exact. Candidate keys are stable hashes of sorted parameter/value pairs. Run keys also contain the scenario configuration, tail window, and parameter-file content hash.

## Scenarios and scoring scope

RATE P-D uses 1.5-second positive and negative 60 deg/s and 120 deg/s initial-rate recovery cases plus mirrored low-authority cases. RATE I keeps the recovery cases and adds mirrored persistent ±0.005 Nm disturbances in LOITER so the non-rate axes remain bounded while steady-state error, integrator output, inhibition, and anti-windup are measured. Angle P uses the six required signed initial attitudes. RATE and STABILIZE tests start with 6 m of analytical altitude margin because those modes intentionally do not run the altitude controller; this prevents a pitch-loop experiment from being rejected merely because hover thrust at a temporary tilt has less vertical component.

LOITER uses upright hold, mirrored initial-offset recovery, mirrored horizontal impulses, mirrored stick move/release, and mirrored authority-stress cases. Authority stress participates in crash and hard-rejection checks, but its metrics do not enter the primary aggregate score.

Moving-mass gain uses the established five scenarios:

- `pitch_assist_probe`
- `stick_move_release`
- `horizontal_impulse_recovery`
- `initial_x_offset_recovery`
- `authority_stress`

Each moving-mass candidate is compared with a total-COM centered baseline using the same selected controller and scenario.

## Metrics

RATE rows include full and tail RMS rate error, steady-state absolute error, overshoot, settling, zero crossings, vane RMS/max, vane and servo saturation, control effort, integrator output, inhibition, and anti-windup correction.

Attitude rows include RMS/tail RMS theta, overshoot, settling, tail peak-to-peak theta, maximum omega, vane activity, saturation, and signed-case symmetry.

LOITER rows use a final two-second tail window by default. The ribbon/comet assessment is driven primarily by:

- tail RMS x error
- tail peak-to-peak x
- tail x-z path length
- tail RMS horizontal velocity

The report also includes x-error zero crossings, oscillation peaks, settling, final/RMS x error, RMS theta, vane RMS, and saturation.

Moving-mass rows include pitch improvement and horizontal change versus the centered baseline, final x error, tail path length, vane activity, moving-mass travel/saturation, target tracking error, and target/actual direction-reversal counts.

## Normalized score and rejection

Raw values with different units are never added. Each metric is divided by a stage-specific reference scale, capped at five reference units, multiplied by its documented weight, and combined into a dimensionless weighted average. Exact scales and weights are stored in workbook metadata.

Candidates are rejected for:

- crash, ground contact, NaN/Inf, or the scenario attitude safety limit
- accelerating growth across three consecutive run windows, above the stage reference threshold
- more than 85% sustained tail actuator/mixer saturation
- missing scenario rows or duplicate run keys
- requested/effective parameter mismatches
- in a full LOITER search, failure of a primary scenario's documented analytical recovery threshold
- for moving mass, materially worse RMS x, final x, or tail path than the centered baseline

Ranking first uses the normalized score in `1e-6` equivalence buckets. Tie-breakers then prefer lower tail oscillation, saturation, effort, and smaller total gain magnitude. The report separately marks best aggregate, most stable, fastest settling, and lowest-saturation candidates, and exports the top 50 where requested.

## Commands and resume

Full workflow:

```powershell
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --output-dir results/analysis/controller_grid_search
```

Quick end-to-end smoke:

```powershell
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --quick --output-dir results/analysis/controller_grid_search_quick
```

One stage:

```powershell
.venv\Scripts\python.exe sweep_controller_gains.py --stage loiter_xy --output-dir results/analysis/controller_grid_search
```

The default behavior reloads `scenario_results.csv` and skips exact completed run keys. Use `--no-resume` to rebuild that output directory's scenario CSV. When a later stage is run alone, it uses preceding aggregate CSVs from the same output directory when available and otherwise falls back to the source profile values.

## Outputs

The output directory contains per-stage raw candidate CSVs, `scenario_results.csv`, `controller_gain_search_summary.md`, and `controller_gain_search.xlsx`. The workbook has these exact sheets:

1. `01_rate_pd_all`
2. `02_rate_pd_top50`
3. `03_rate_i_all`
4. `04_attitude_p_all`
5. `05_loiter_xy_all`
6. `06_loiter_xy_top50`
7. `07_moving_mass_gain_all`
8. `08_scenario_summary`
9. `09_best_parameters`
10. `10_metadata`

Generated profiles are separate from the source profiles:

- `params/loiter_tuned_vane_only.json`
- `params/moving_mass_prototype_2kg_tuned.json`

The first copies `loiter_example.json` and overrides only selected active controller gains. The second copies `moving_mass_prototype_2kg.json`, applies the same selected controller gains, and records the analysis-only proportional assist gain in an `analysis` metadata section. Loading either profile leaves the active controller formulas unchanged.
