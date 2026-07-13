# 2 kg Moving-Mass Prototype Profile

`params/moving_mass_prototype_2kg.json` records the current design assumptions for the 2 kg moving-mass prototype. These values are analytical design inputs, not measured calibration or real-flight validation.

## Parameter assumptions

- Total vehicle mass, `cfg.m`: `2.0 kg`
- Moving mass: `0.5 kg`
- Maximum lateral moving-mass offset: `0.05 m`
- Moving-mass body-up position: `+0.12 m`
- Moving-mass rail: body-fixed
- Fixed-body geometry origin: `[0, 0]` in `[body_right, body_up]`
- Thrust application point: `[0, 0]`
- Vehicle height, `cfg.H`: `0.50 m`
- Vane application point: `[0, -cfg.l] = [0, -0.25] m`

The mass convention is important: `cfg.m` already includes the moving mass. The implied fixed-body mass is therefore `2.0 - 0.5 = 1.5 kg`; the moving mass must not be added to `cfg.m` again.

The profile preserves the LOITER example's unrelated rigid-body, interactive, and controller values. Its safe default mode flags are:

```text
moving_mass.enabled = False
moving_mass.use_total_com_geometry = False
moving_mass.use_legacy_gravity_offset_moment = True
```

The comparison runner overrides only these mode flags for each of its six variants.

## Geometry expectations

The existing model derives `cfg.l` as `cfg.H / 2`, so this profile gives `cfg.l = 0.25 m`. The fixed body and thrust application point are at the geometry origin; the vane application point is derived from `cfg.l` without a separate vane-position parameter.

For total mass `2.0 kg`, moving mass `0.5 kg`, and body-up position `0.12 m`:

- At zero lateral offset, total COM is `[0.0, 0.03] m`.
- At `+0.05 m` lateral offset, total COM is `[+0.0125, +0.03] m`.

Consequently, `total_com_geometry_centered` remains 8-state, has zero lateral COM shift and zero axial-thrust pitch moment from lateral offset, retains the `+0.03 m` vertical COM shift, and uses a vane arm extended by that vertical COM shift.

## Reproducing the comparison

Run all five native-duration scenarios and all six variants without plots:

```powershell
.venv\Scripts\python.exe compare_moving_mass_assist.py --params params/moving_mass_prototype_2kg.json --scenario all --no-plots --output-dir results/analysis/moving_mass_prototype_2kg
```

The generated files are:

- `results/analysis/moving_mass_prototype_2kg/moving_mass_comparison.csv`
- `results/analysis/moving_mass_prototype_2kg/moving_mass_comparison.md`

Generated analysis results are transient and ignored by Git. The current model retains its existing fixed `Iyy` calculation. It does not include position-dependent inertia, reaction kick, moving-mass acceleration reaction, internal momentum coupling, or rail/servo reaction forces.
