# LOITER transient diagnosis and provisional candidate

## Diagnosed cause

The absolute-target pause is an outer-loop/inner-loop interaction. With the current 0.50/0.90 position/velocity P pair, the velocity loop changes acceleration demand faster than the delayed attitude/rate/servo plant settles. Actual pitch remains on the braking side while total desired velocity is still forward, so vx reaches the pause threshold; the lagging attitude then crosses and produces the next forward lobe. Vane deadband and actuator lag contribute phase lag, but the logged command remains available and neither saturation nor target shaping initiates the absolute-target pause.

The stick-release defect is different: jerk-limited shaped velocity overshoots through zero and reverses. Integrating that reversed velocity walks target_x backward, while the one-control-tick capture condition can be missed or repeated. The provisional behavior clamps the shaper at its zero target and records one capture without replacing the already-shaped final hold target, eliminating capture jumps.

## Selected provisional parameters

| Parameter | Current | Selected |
|---|---:|---:|
| PSC_NE_POS_P | 0.50 | 0.55 |
| PSC_NE_VEL_P | 0.90 | 0.70 |
| LOIT_BRK_DELAY | 0.40 s | 0.50 s |
| LOIT_BRK_ACC | 0.80 m/s^2 | 1.00 m/s^2 |
| LOIT_BRK_JERK | 2.00 m/s^3 | 3.00 m/s^3 |
| capture vx threshold | 0.08 m/s | 0.08 m/s |
| persistent capture handshake | off | on |
| clamp shaped velocity at target | off | on |
| capture without target_x jump | off | on |

ATC rate PID, Angle P, physics, actuator/vane model, mass, and COM geometry are unchanged. Moving-mass assist remains disabled in the selected vane-only profile.

## Absolute +1 m before/after

| Metric | Current | Selected |
|---|---:|---:|
| premature pause | True | False |
| pause start | 2.825 s | none |
| pause position error | 0.329 m | none |
| prior peak vx | 0.600 m/s | 0.595 m/s |
| vx reversals before x=0.98 m | 0 | 0 |
| final absolute error | 0.0784 m | 0.0004 m |
| tail RMS error | 0.0597 m | 0.0703 m |
| tail path length | 0.3356 m | 0.2931 m |
| overshoot | 0.1367 m | 0.1689 m |
| peak pitch | 4.578 deg | 4.243 deg |
| vane RMS | 0.450 deg | 0.378 deg |

First pause context (radians are retained for attitude/rate fields):

```json
{
  "pause": {
    "duration_s": 0.15,
    "end_index": 593,
    "end_time_s": 2.969999999999959,
    "position_error_m": 0.3290011774531114,
    "prior_peak_vx_ms": 0.6000033773243525,
    "start_index": 564,
    "start_time_s": 2.824999999999962,
    "vx_ms": 0.02998366885631668,
    "x_m": 0.6709988225468886
  },
  "pause_start": {
    "actual_vane_angle": -0.006462015385440704,
    "desired_ax": 0.11997625937051457,
    "desired_moment": -0.009397536091595628,
    "omega": 0.22213251256699235,
    "omega_target": 0.34392504798594015,
    "position_velocity_correction": 0.1645755478986965,
    "shaped_desired_vx": 0.0,
    "target_x": 1.0,
    "theta": -0.021052456080918828,
    "theta_target": 0.012229386154840214,
    "time": 2.824999999999962,
    "total_desired_vx": 0.1645755478986965,
    "vane_angle_cmd": 0.003400898530756551,
    "vx": 0.02998366885631668,
    "x": 0.6709988225468886,
    "x_error": 0.3290011774531114
  },
  "prior_peak": {
    "actual_vane_angle": 0.003555254444334609,
    "desired_ax": -0.24146211058166256,
    "desired_moment": 0.01580291561079414,
    "omega": -0.24069892654689595,
    "omega_target": -0.2637296699472236,
    "position_velocity_correction": 0.3316313070959399,
    "shaped_desired_vx": 0.0,
    "target_x": 1.0,
    "theta": -0.0006420185451048038,
    "theta_target": -0.024608905769994704,
    "time": 1.8399999999999828,
    "total_desired_vx": 0.3316313070959399,
    "vane_angle_cmd": -0.005725880649550759,
    "vx": 0.6000033773243525,
    "x": 0.3427374021615312,
    "x_error": 0.6572625978384687
  }
}
```

## Stick pulse/release before/after

| Metric | Current | Selected |
|---|---:|---:|
| shaped-vx sign changes after release | 2 | 0 |
| capture events | 2 | 1 |
| target discontinuities > 0.02 m | 2 | 0 |
| maximum target step | 0.1683 m | 0.0078 m |
| tail RMS position error | 0.1124 m | 0.0940 m |
| tail path length | 0.3449 m | 0.3507 m |
| final vx | -0.0843 m/s | -0.0053 m/s |

## Moving-mass follow-up

| Gain (m/Nm) | Pause | Final error (m) | Tail RMS (m) | Peak pitch (deg) |
|---:|:---:|---:|---:|---:|
| 0.000 | False | 0.0004 | 0.0703 | 4.243 |
| 0.025 | False | 0.0468 | 0.0687 | 3.929 |
| 0.040 | False | 0.1077 | 0.0607 | 3.772 |
| 0.055 | False | 0.1387 | 0.1660 | 3.641 |
| 0.070 | False | 0.0071 | 0.0157 | 3.528 |

Gain 0.055 m/Nm is pause-free but is no longer a reasonable preferred gain under the explicit stepped target: final error=0.1387 m, tail RMS=0.1660 m, and peak pitch=3.641 deg. Gain 0.070 performs best in this one scenario. The selected moving-mass gain is not changed by this LOITER task; broader moving-mass regressions are required before adopting a replacement.

## Remaining limitations

The fixed attitude/rate controller remains lightly damped, so small post-target velocity lobes remain after the vehicle has first reached the target region. The selected outer pair removes the defined premature pause and pre-0.98 m reversal, but it does not retune the validated inner loop. This is a provisional profile and does not replace the canonical tuned profile.
