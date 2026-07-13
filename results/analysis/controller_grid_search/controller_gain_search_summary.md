# Controller Gain Grid Search

This report ranks deterministic analytical simulations. It does not claim real-flight optimality.
Raw metrics are normalized by the documented reference scales before weighting; raw values with different units are never summed directly.

## Reproduction

```powershell
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --output-dir results/analysis/controller_grid_search
.venv\Scripts\python.exe sweep_controller_gains.py --stage all --quick --output-dir results/analysis/controller_grid_search_quick
```

- Git SHA: `c498f873df1e2e93cef0aee6aa8b78c05569b5bd`
- Parameter source: `params\loiter_example.json`
- Tail window: `2.0` s
- Runtime: `1026.282` s
- Ribbon/comet assessment: **reduced but not eliminated under the documented tail thresholds**.
- `psc_ne_vel_i` and `psc_ne_vel_d` remain inactive and were not swept or optimized.

## Best Aggregate Candidates

| stage | candidate | score | selected parameters | rejected |
| --- | --- | ---: | --- | --- |
| rate_pd | rate_pd-ceff8eb68b | 1.580740 | atc_rat_pit_p=0.0425, atc_rat_pit_i=0, atc_rat_pit_d=0.00725 | False |
| rate_i | rate_i-8913059c5e | 2.130664 | atc_rat_pit_p=0.0425, atc_rat_pit_i=0, atc_rat_pit_d=0.00775 | False |
| attitude_p | attitude_p-7c6b5f7621 | 1.659295 | atc_rat_pit_p=0.0425, atc_rat_pit_i=0, atc_rat_pit_d=0.00775, atc_ang_pit_p=9.5 | False |
| loiter_xy | loiter_xy-2bf8648b07 | 1.897191 | atc_rat_pit_p=0.0425, atc_rat_pit_i=0, atc_rat_pit_d=0.00775, atc_ang_pit_p=9.5, psc_ne_pos_p=0.4, psc_ne_vel_p=0.6 | False |
| moving_mass_gain | moving_mass_gain-a950f78c88 | 0.800000 | atc_rat_pit_p=0.0425, atc_rat_pit_i=0, atc_rat_pit_d=0.00775, atc_ang_pit_p=9.5, psc_ne_pos_p=0.4, psc_ne_vel_p=0.6, moving_mass_assist_gain_m_per_Nm=0 | False |

## Selected Metrics

| stage | selected evidence |
| --- | --- |
| rate_pd | RMS/tail rate error 39.877/39.877 deg/s; overshoot 0.006 deg/s; settling 1.478 s; vane RMS 0.477 deg |
| rate_i | I=0.000; tail mean absolute rate error 31.848 deg/s; integrator RMS 0.00000 Nm; inhibition 0.00% |
| attitude_p | RMS/tail theta 4.207/3.129 deg; tail peak-to-peak 7.348 deg; max omega 21.992 deg/s |
| loiter_xy | tail RMS x 0.1622 m; tail RMS vx 0.3132 m/s; tail peak-to-peak x 0.3691 m; tail path 0.5641 m |
| moving_mass_gain | gain 0.0000 m/Nm; theta/x/path ratios 1.0000/1.0000/1.0000; mass max 0.00000 m |

## Ribbon Tail Comparison

Selected LOITER P/P versus the mode-matched `psc_ne_pos_p=0.8`, `psc_ne_vel_p=1.1` candidate:

| metric | 0.8/1.1 | selected | change |
| --- | ---: | ---: | ---: |
| tail RMS x | 0.201214 m | 0.162207 m | -19.39% |
| tail RMS vx | 0.416832 m/s | 0.313233 m/s | -24.85% |
| tail peak-to-peak x | 0.541461 m | 0.369072 m | -31.84% |
| tail x-z path length | 0.732531 m | 0.564113 m | -22.99% |

## Stage Commands

- `rate_pd`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage rate_pd --output-dir results/analysis/controller_grid_search`
- `rate_i`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage rate_i --output-dir results/analysis/controller_grid_search`
- `attitude_p`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage attitude_p --output-dir results/analysis/controller_grid_search`
- `loiter_xy`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage loiter_xy --output-dir results/analysis/controller_grid_search`
- `moving_mass_gain`: `.venv\Scripts\python.exe sweep_controller_gains.py --stage moving_mass_gain --output-dir results/analysis/controller_grid_search`

## Grid And Run Counts

| stage | candidates | scenario rows |
| --- | ---: | ---: |
| rate_pd | 299 | 1794 |
| rate_i | 48 | 192 |
| attitude_p | 17 | 102 |
| loiter_xy | 360 | 3240 |
| moving_mass_gain | 33 | 165 |

## Rejection And Tie-Break Rules

Candidates are rejected for crashes, ground contact, non-finite data, attitude-limit violations, unbounded growth, excessive sustained saturation, missing scenarios, duplicate run keys, or effective-parameter mismatches. Moving-mass candidates are also rejected when horizontal hold is materially worse than the total-COM centered baseline.
Ties favor lower tail oscillation, then lower saturation, lower control effort, and finally smaller gain magnitude.
Authority-stress LOITER rows participate in hard rejection and robustness reporting but not the primary aggregate score.
