# Independent variant-controller optimization

This is deterministic 2D simulation research only. It makes no real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety claim, and it does not modify rigid-body physics.

## Result categories

1. **PR #24 shared-controller actuator isolation:** the merged controller and gain 0 / 0.0415 actuator variants are preserved as read-only references.
2. **Independently optimized Vane-only:** its PID/position gains were selected by this executable workflow; the physical 0.5 kg mass stayed present and its gain, target, offset, rate, and acceleration were exactly zero.
3. **Independently optimized Moving-mass-assist:** its PID/position gains and assist gain were searched independently and are not forced to match Vane-only.

Candidates evaluated: **259**. Selected validation cases per rerun: **46**. Two-rerun deterministic validation: **True**. All finalist gates passed: **True**.

## Selected parameters

| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) | settle (s) | rise (s) | overshoot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 0.11356035 | 0.02711948 | 28.377440 | 0.78243440 | 0.96229151 | 0.00000000 | 7.1800 | 1.9850 | 0.1120 |
| moving_mass_assist | 0.11964316 | 0.02469923 | 18.139040 | 0.71700583 | 0.95948159 | 0.02670551 | 6.1600 | 2.0650 | 0.1194 |

Selection was lexicographic: all hard gates, settled status, settling time, rise time, preferred-band overshoot distance, steady-state error, actuator effort/limiter use, then mirrored/scenario robustness. Zero overshoot received a distance penalty whenever it fell below the preferred band; it was not treated as automatically optimal.

The Pareto CSV retains valid tradeoffs rather than collapsing them into one weighted score. The shared-objective comparison CSV evaluates the PR #24 shared references and both independently optimized controllers with the same screening objectives.
