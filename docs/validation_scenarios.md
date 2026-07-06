# Validation Scenarios

Run:

```bash
python validate_sim.py
```

The runner is headless and does not import or open pygame windows. It writes:

- `results/validation_summary.csv`
- `results/validation_report.md`

Scenarios:

- DIRECT throttle step: motor lag and vertical acceleration sign.
- DIRECT vane step: side-force and pitch-moment signs.
- RATE pitch-rate command: requested moment follows omega target.
- STABILIZE initial tilt recovery: restoring moment for positive/negative tilt.
- External horizontal impulse: velocity changes and force disappears afterward.
- External pitch impulse: angular velocity changes and moment disappears.
- Low thrust authority: mixer reports insufficient authority.
- Emergency motor cut: zero command eventually produces crash.
- Nonlinear vane model: axial thrust loss appears with vane deflection.
- Long finite run: state remains finite unless a crash is expected.

Validation failures are reported directly rather than hidden through gain tuning.
