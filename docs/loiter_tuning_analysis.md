# Headless LOITER Tuning Analysis

## Purpose

The headless LOITER tools turn the interactive 2D single-fan simulator into a repeatable analysis workflow. They run the same rigid-body plant, motor lag, servo lag, cascaded controller, LOITER input shaper, mixer saturation reporting, and safety checks without opening pygame.

These results are analytical indicators only. The model is not experimentally calibrated and should not be treated as real-flight prediction.

## Scenario Definitions

- `stick_move_release`: applies horizontal stick input for a short period, releases it, then observes braking and hold behavior.
- `initial_x_offset_recovery`: starts away from the horizontal target with no pilot input and measures recovery.
- `horizontal_impulse_recovery`: applies a short world-frame horizontal force impulse and measures the recovery transient.
- `vertical_offset_althold`: starts above the altitude target and exercises the altitude-hold path.
- `low_authority_probe`: reduces vane authority and rate limit to expose saturation and authority flags.

Scenario defaults are deterministic and configurable in `LoiterScenarioConfig`.

## Metrics

- `final_abs_x_error`: absolute horizontal hold error at the end of the run.
- `rms_x_error`: root-mean-square horizontal error over the run.
- `max_theta_deg`: peak absolute pitch angle.
- `max_vane_cmd_deg`: peak requested vane angle.
- `mixer_saturation_percent`: percent of samples where the mixer saturated.
- `authority_limited_percent`: percent of samples where requested pitch moment exceeded modeled authority.
- `servo_rate_saturation_percent`: percent of samples where the servo rate limit clipped motion.

## Interpreting Saturation

Saturation is not always a failure. It is a design signal. Short saturation during a disturbance may be acceptable; sustained saturation or high authority-limited percentage means the controller is asking for more moment than the current thrust/vane geometry can provide.

Authority-limited means the desired pitch moment exceeded the moment available from current thrust and vane limits. Servo rate saturation means the vane could eventually reach the command, but not as quickly as requested.

## Comparing Parameter Sets

Use `compare_loiter_params.py` to compare sluggish, nominal, and aggressive LOITER parameter files. Sluggish settings should generally move less abruptly but leave larger residual errors. Aggressive settings may reduce error faster but can increase pitch, vane demand, and saturation.

Compare parameter sets relatively. The most useful signals are changes in error, attitude peaks, and saturation percentages across the same deterministic scenario set.

## Future Vane Authority Design

Use `sweep_loiter_authority.py` as a vane authority map. It sweeps vane angle limit, vane rate limit, and thrust margin to identify regions that are obviously under-actuated or saturation-heavy.

The default authority sweep uses an `authority_stress` scenario with a low-altitude margin, horizontal offset, and horizontal impulse. Its default grid intentionally includes very low vane angles, slow vane rates, and low thrust margins so the report can show transitions between crash, large error, actuator/authority limits, and acceptable analytical recovery. If the selected grid does not change important metrics, the Markdown report marks the sweep as `INCONCLUSIVE`.

The expanded workflow supports multiple stress scenarios, a `--quick` grid, `--scenario all`, per-scenario heatmaps, and simple design scores. See [vane_authority_mapping.md](vane_authority_mapping.md).

## Limitations

- The simulator is 2D and analytical.
- Vane and thrust parameters are not calibrated against bench or flight data.
- The controller is ArduCopter-inspired, not exact ArduPilot firmware.
- The current model uses one equivalent pitch vane/moment, not full four-vane physics.
- Results should guide design questions and tuning sensitivity, not certify real-world flight behavior.
