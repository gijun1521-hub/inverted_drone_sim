# Parameter Sweep Guide

Run:

```bash
python analyze_authority.py
python analyze_moving_mass.py
```

Generated files are placed under `results/analysis/` and ignored by git.

Important sweep axes:

- `moving_mass_ratio`: fraction of total mass that can move
- `inertia_ratio`: moving inertia relative to body inertia
- `q_limit`: moving mass travel
- `q_rate_limit` and `q_accel_limit`
- `thrust_to_weight`
- `vane_angle_max`
- `vane_area / duct_area`

Read the results as trends and feasibility checks, not exact predictions.
