# Seminar scenario comparison

These deterministic 2D simulations compare the same 2.0 kg vehicle with its 0.5 kg moving mass physically present in both variants. The locked case commands and maintains a 0 mm offset; it does not remove the mass.

The final 2.0 seconds form the tail window. `tail_rms_x_m` is RMS target error. Settling requires the remaining run to stay within 0.05 m position error and 0.05 m/s horizontal speed; an unsettled run is reported at the available observation-window limit. For the disturbance scenario, `position_overshoot_m` is the peak absolute recovery excursion after the force ends. For the +1 m step it is excursion beyond +1.0 m.

| Scenario | Variant | Tail RMS x (m) | Final |x error| (m) | Peak |pitch| (deg) | Vane RMS (deg) | Moving mass max (mm) | Settled |
|---|---|---:|---:|---:|---:|---:|:---:|
| LOITER disturbance recovery | Mass locked at center | 0.12060 | 0.02198 | 4.682 | 0.666 | 0.00 | no |
| LOITER disturbance recovery | Active moving-mass assist | 0.14375 | 0.03487 | 4.697 | 0.560 | 8.49 | no |
| +1 m position command and hold | Mass locked at center | 0.09656 | 0.09180 | 3.062 | 0.865 | 0.00 | no |
| +1 m position command and hold | Active moving-mass assist | 0.13374 | 0.13164 | 2.837 | 0.757 | 6.10 | no |

## Pairwise comparison

For **LOITER disturbance recovery**, active assist changes tail RMS x error by +0.02315 m relative to the locked-mass simulation.

For **+1 m position command and hold**, active assist changes tail RMS x error by +0.03718 m relative to the locked-mass simulation.

## Interpretation limit

These results compare implementations inside the same deterministic 2D model. They are not evidence of real-flight equivalence, hardware safety, 3D stability, or calibrated aerodynamic performance.
