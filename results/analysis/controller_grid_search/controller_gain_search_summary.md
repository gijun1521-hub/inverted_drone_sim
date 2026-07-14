# Controller Gain Grid Search

This report ranks deterministic analytical simulations. It does not claim real-flight optimality.
Raw metrics are normalized by the documented reference scales before weighting; raw values with different units are never summed directly.

## Reproduction

```powershell
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --no-resume --output-dir results/analysis/controller_grid_search
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --quick --no-resume --output-dir results/analysis/controller_grid_search_quick_review
```

- Git SHA: `6abc6aa3c8e3f3f44669c0d2b7ae48477345431e`
- Workflow fingerprint: `b983de865755429aeb46b1f4760f3fe549b4b17b72b0a047dde697fab5fafcbe`
- Cache schema: `3`; grid definition: `2026-07-13.3`
- Parameter source: `params\loiter_example.json`
- Profile output: `params` (canonical params directory).
- Quick mode defaults to `<output-dir>/profiles`; it does not overwrite canonical full-search profiles unless `--profile-output-dir` is explicit.
- Tail windows: RATE P/D and recovery 0.75 s; RATE I bias and attitude 1.0 s; LOITER and moving mass 2.0 s by default.
- Runtime: `1162.801818` s
- Ribbon/comet assessment: **reduced but not eliminated under the documented tail thresholds**.
- `psc_ne_vel_i` and `psc_ne_vel_d` remain inactive and were not swept or optimized.

## Best Aggregate Candidates

| stage | candidate | score | selected parameters | rejected |
| --- | --- | ---: | --- | --- |
| rate_pd | rate_pd-84ed1879d5 | 0.448139 | atc_rat_pit_p=0.0725, atc_rat_pit_i=0, atc_rat_pit_d=0.008 | False |
| rate_i | rate_i-9c54834c68 | 2.134130 | atc_rat_pit_p=0.07, atc_rat_pit_i=0, atc_rat_pit_d=0.008 | False |
| attitude_p | attitude_p-0be053ab42 | 1.137475 | atc_rat_pit_p=0.07, atc_rat_pit_i=0, atc_rat_pit_d=0.008, atc_ang_pit_p=10 | False |
| loiter_xy | loiter_xy-ce3cdef9c3 | 1.004894 | atc_rat_pit_p=0.07, atc_rat_pit_i=0, atc_rat_pit_d=0.008, atc_ang_pit_p=10, psc_ne_pos_p=0.5, psc_ne_vel_p=0.9 | False |
| moving_mass_gain | moving_mass_gain-f9d7de3672 | 0.644174 | atc_rat_pit_p=0.07, atc_rat_pit_i=0, atc_rat_pit_d=0.008, atc_ang_pit_p=10, psc_ne_pos_p=0.5, psc_ne_vel_p=0.9, moving_mass_assist_gain_m_per_Nm=0.055 | False |

## Selected Metrics

| stage | selected evidence |
| --- | --- |
| rate_pd | RMS/tail rate error 15.115/3.240 deg/s; overshoot 4.123 deg/s; settling 1.800 s; vane RMS 0.329 deg |
| rate_i | I=0.000; tail mean absolute rate error 12.768 deg/s; integrator RMS 0.00000 Nm; inhibition 0.00% |
| attitude_p | RMS/tail theta 3.148/1.884 deg; tail peak-to-peak 3.567 deg; max omega 28.378 deg/s |
| loiter_xy | tail RMS x 0.0595 m; tail RMS vx 0.1547 m/s; tail peak-to-peak x 0.1565 m; tail path 0.2780 m |
| moving_mass_gain | gain 0.0550 m/Nm; theta/x/path ratios 0.7005/0.9887/0.6893; mass max 0.00340 m |

## Search Boundary Diagnostics

A selected boundary value is only the best value within the tested grid; it is not evidence of an interior or global optimum.
The parameter grids were not expanded in this review.

| parameter | selected | search min | search max | lower boundary | upper boundary |
| --- | ---: | ---: | ---: | --- | --- |
| atc_rat_pit_p | 0.07 | 0.005 | 0.08 | False | False |
| atc_rat_pit_i | 0 | 0 | 0.03 | True | False |
| atc_rat_pit_d | 0.008 | 0 | 0.008 | False | True |
| atc_ang_pit_p | 10 | 2 | 10 | False | True |
| psc_ne_pos_p | 0.5 | 0.1 | 1.5 | False | False |
| psc_ne_vel_p | 0.9 | 0.2 | 2.5 | False | False |
| moving_mass_assist_gain_m_per_Nm | 0.055 | 0 | 0.08 | False | False |

Warnings:
- atc_rat_pit_i selected at the lower search boundary; this is the best value within the tested grid, not evidence of an interior or global optimum.
- atc_rat_pit_d selected at the upper search boundary; this is the best value within the tested grid, not evidence of an interior or global optimum.
- atc_ang_pit_p selected at the upper search boundary; this is the best value within the tested grid, not evidence of an interior or global optimum.

## Ribbon Tail Comparison

Selected LOITER P/P versus the mode-matched `psc_ne_pos_p=0.8`, `psc_ne_vel_p=1.1` candidate:

| metric | 0.8/1.1 | selected | change |
| --- | ---: | ---: | ---: |
| tail RMS x | 0.075540 m | 0.059454 m | -21.29% |
| tail RMS vx | 0.187147 m/s | 0.154705 m/s | -17.34% |
| tail peak-to-peak x | 0.175438 m | 0.156523 m | -10.78% |
| tail x-z path length | 0.333777 m | 0.278014 m | -16.71% |

## Stage Commands

- `rate_pd`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage rate_pd --output-dir results/analysis/controller_grid_search`
- `rate_i`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage rate_i --output-dir results/analysis/controller_grid_search`
- `attitude_p`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage attitude_p --output-dir results/analysis/controller_grid_search`
- `loiter_xy`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage loiter_xy --output-dir results/analysis/controller_grid_search`
- `moving_mass_gain`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage moving_mass_gain --output-dir results/analysis/controller_grid_search`

## Grid And Run Counts

| stage | candidates | scenario rows |
| --- | ---: | ---: |
| rate_pd | 289 | 2312 |
| rate_i | 48 | 192 |
| attitude_p | 17 | 102 |
| loiter_xy | 360 | 3240 |
| moving_mass_gain | 33 | 165 |

## Rejection And Tie-Break Rules

Candidates are rejected for crashes, ground contact, non-finite data, attitude-limit violations, unbounded growth, excessive sustained saturation, missing scenarios, duplicate run keys, effective-parameter mismatches, or failure of a required stage validity gate. Moving-mass candidates are also rejected when horizontal hold is materially worse than the total-COM centered baseline.
Signed 10 deg/s RATE recoveries must settle. Signed 60 deg/s cases remain scored without a settling gate; 120 deg/s and low-authority RATE cases are robustness-only. RATE I bias rows and attitude rows use the documented terminal/tail thresholds stored in metadata.
Resume rows and prerequisite aggregate CSVs are accepted only when their workflow fingerprint matches the current implementation, parameter sources, quick/full mode, tail policy, and grid version.
Ties favor lower tail oscillation, then lower saturation, lower control effort, and finally smaller gain magnitude.
Authority-stress LOITER rows participate in hard rejection and robustness reporting but not the primary aggregate score.
