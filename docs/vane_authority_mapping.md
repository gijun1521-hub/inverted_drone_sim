# Vane Authority Mapping

## Purpose

The vane authority workflow sweeps vane angle limit, servo rate limit, and thrust margin in the analytical 2D single-fan simulator. It helps compare relative recovery, saturation, and authority margin trends before changing physical design assumptions.

The results are not calibrated flight requirements.

## Why Authority Mapping Matters

A single-fan vehicle must recover horizontal disturbances by leaning and by generating enough pitch control moment through the vane model. Too little vane angle, slow servo motion, or low thrust margin can show up as poor recovery, high pitch demand, servo rate saturation, mixer saturation, or authority-limited moment requests.

## Swept Parameters

- `vane_angle_max_deg`: maximum equivalent pitch-axis vane deflection.
- `vane_rate_limit_deg_s`: servo slew limit for the equivalent vane.
- `T_max_factor`: maximum thrust divided by weight.

## Scenario Definitions

- `authority_stress`: low altitude margin, initial x offset, and horizontal impulse.
- `impulse_light`: moderate horizontal impulse; many reasonable configurations should recover.
- `impulse_heavy`: strong horizontal impulse; low-authority configurations should fail or saturate.
- `offset_recovery_2m`: starts 2 m away from target_x with no stick input.
- `stick_step_aggressive`: near-full horizontal stick step, release, then settle.
- `low_thrust_margin`: low vertical margin and climb demand to expose thrust coupling.

## Metrics

- `final_abs_x_error`, `rms_x_error`: horizontal recovery quality.
- `final_abs_z_error`, `rms_z_error`: altitude coupling and recovery quality.
- `max_theta_deg`, `max_omega_deg_s`: attitude demand and aggressiveness.
- `max_vane_cmd_deg`, `max_vane_actual_deg`: requested and achieved vane motion.
- `mixer_saturation_percent`, `authority_limited_percent`: pitch moment authority indicators.
- `servo_rate_saturation_percent`, `motor_saturation_percent`: actuator limit indicators.
- `combined_design_score`: simple analytical ranking score. It rewards lower final/RMS error, lower saturation, lower pitch peak, and passing cases.

## Failure Reasons

- `crash`: safety checks stopped the run.
- `large_final_error`: thresholds failed without explicit saturation or crash.
- `saturation_or_authority_limit`: actuator or authority limits were active.
- `inconclusive`: sweep parameters did not affect important metrics and did not trigger limits.

## Interpreting Heatmaps

Heatmaps are written under `results/analysis/authority_maps/`. The x axis is `vane_angle_max_deg`, the y axis is `vane_rate_limit_deg_s`, and each image is scoped to one scenario and one `T_max_factor`.

Use them comparatively. Look for transitions from crash or large error into stable recovery, and for regions where increasing thrust margin helps only after enough vane angle and servo rate are available.

## Commands

```bash
python sweep_loiter_authority.py
python sweep_loiter_authority.py --quick
python sweep_loiter_authority.py --scenario all --no-plots
python sweep_loiter_authority.py --scenario impulse_heavy
python sweep_loiter_authority.py --vane-angle-max-deg 1,2,3,5,8 --vane-rate-limit-deg-s 10,20,40,80 --T-max-factor 1.1,1.4,1.8
```

## Physical Design Use

Use the maps to identify relative under-actuated regions and promising regions before changing vane geometry, servo selection, or thrust margin assumptions. Treat the output as a guide for what to test next, not as proof that a real vehicle will fly.

## Limitations

- The model is analytical, 2D, and uses one equivalent pitch-axis vane/moment.
- It is ArduCopter-inspired, not exact ArduPilot firmware.
- It is not calibrated with bench thrust data or real flight logs.
- Real SingleCopter vehicles use four flaps; full four-vane physics are future work.

## Future Work

- Use the four-vane mixer prep structures in `mixers.py` when a future PR is
  ready to introduce a broader vehicle model.
- Add 3D modeling later.
- Calibrate thrust and vane parameters with bench data.
- Calibrate controller and disturbance assumptions with real flight logs.
