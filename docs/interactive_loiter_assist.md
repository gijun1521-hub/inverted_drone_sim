# Interactive LOITER moving-mass assist

The seminar comparison's active-assist panels used a headless-only moving-mass command law. The interactive simulator now exposes the same command path behind an explicit, disabled-by-default configuration.

Run the validated 2 kg interactive profile directly with Python:

```powershell
.venv\Scripts\python.exe interactive_sim.py --params params/interactive_loiter_assist_2kg.json
```

The profile starts in LOITER, enables total-COM geometry, and commands the moving-mass target as:

```text
moving_mass_target_m = 0.055 m/Nm * desired_pitch_moment_Nm
```

The physical rail, velocity, and acceleration limits still apply. Leaving LOITER commands the mass back to center. The existing defaults and the canonical vane-only and prototype profiles remain unchanged.

This profile also uses the provisional stick-release capture settings from the LOITER transient diagnosis. It is an analytical 2D simulation profile, not flight-hardware validation.
