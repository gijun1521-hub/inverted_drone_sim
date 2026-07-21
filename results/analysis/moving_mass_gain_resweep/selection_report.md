# Moving-mass assist-gain resweep

The merged Pitch controller is fixed at P/I/D `0.09375 / 0.0 / 0.02100`, Angle P `25.0`. Only the simulation-only `moving_mass_assist_gain_m_per_Nm` changed.

Stage 0 gain `0.0` is the valid Vane-only comparison baseline: it passes all seven full-duration scenarios and maintains exactly zero moving-mass actual and target displacement.

Raw-score rank 1 is gain `0.04150` with score `0.585840087765`. Gain-zero score is `1.500000000000`; improvement is `60.944%`. Decision: **adopt_valid_raw_score_rank1**. The selected nonzero gain satisfies the required meaningful-improvement threshold.

The old PR #20 gain `0.1325` is included only as a reference. PR #20 uses the obsolete pre-PR-22 Pitch controller and was not modified.

All selected hard gates, symmetry, chatter, saturation, and moving-mass physical-limit results are in `selection_comparison.json` and the scenario results. No hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.

# Moving-mass gain selection

The merged PR #22 Pitch controller is fixed. This is a simulation-only virtual-actuator sweep; no hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.

Stage 0 gain `0.0` is a **VALID Vane-only comparison baseline**: it passes all seven full-duration scenarios with exactly zero moving-mass target and actual displacement.

Raw-score rank 1 is gain `0.04150` with score `0.585840087765`. Decision: **adopt_valid_raw_score_rank1**.

Normal Moving-mass limiter engagement is recorded and penalized, not automatically rejected. Actual offset/rate/acceleration violations use a +1e-9 numerical tolerance.

| metric | gain 0 Vane-only | old PR #20 reference 0.1325 | selected | selected vs gain 0 |
| --- | ---: | ---: | ---: | ---: |
| tail_rms_pitch_deg | 0.525546418 | 1.085698848 | 0.053698180 | -89.782% |
| tail_rms_pitch_rate_deg_s | 2.372223592 | 10.490676146 | 0.029630482 | -98.751% |
| tail_rms_horizontal_velocity_m_s | 0.030239088 | 0.039889226 | 0.015021717 | -50.324% |
| tail_path_length_m | 0.063139595 | 0.084689013 | 0.033957655 | -46.218% |
| final_abs_position_error_m | 0.017361590 | 0.011098084 | 0.022014444 | 26.800% |
| position_overshoot_m | 0.330149368 | 0.331038314 | 0.337922623 | 2.354% |
| recovery_excursion_m | 0.585183881 | 0.589338292 | 0.583386258 | -0.307% |
| strict_settling_time_s | 5.134285714 | 9.692142857 | 5.130000000 | -0.083% |
| vane_command_rms_deg | 0.732413779 | 1.824421029 | 0.574999284 | -21.493% |
| vane_command_total_variation_deg | 34.517019467 | 170.130519689 | 20.499985137 | -40.609% |
| vane_command_rate_rms_deg_s | 30.716419702 | 40.177003561 | 30.757650712 | 0.134% |
| moving_mass_rms_displacement_m | 0.000000000 | 0.008011024 | 0.000919780 | new; delta 0.000919780 |
| moving_mass_total_travel_m | 0.000000000 | 0.699613193 | 0.028547739 | new; delta 0.028547739 |
| moving_mass_rate_rms_m_s | 0.000000000 | 0.078769685 | 0.011025899 | new; delta 0.011025899 |
| moving_mass_acceleration_rms_m_s2 | 0.000000000 | 0.889595794 | 0.298102669 | new; delta 0.298102669 |
| moving_mass_max_abs_offset_m | 0.000000000 | 0.014159949 | 0.006535913 | new; delta 0.006535913 |
| moving_mass_max_abs_velocity_m_s | 0.000000000 | 0.158990995 | 0.093413879 | new; delta 0.093413879 |
| moving_mass_max_abs_acceleration_m_s2 | 0.000000000 | 1.000000000 | 1.000000000 | new; delta 1.000000000 |
| moving_mass_command_clipping_duty_percent | 0.000000000 | 0.214285714 | 0.000000000 | new; delta 0.000000000 |
| moving_mass_rail_contact_duty_percent | 0.000000000 | 0.000000000 | 0.000000000 | new; delta 0.000000000 |
| moving_mass_rate_limiter_duty_percent | 0.000000000 | 51.065476190 | 0.178571429 | new; delta 0.178571429 |
| moving_mass_acceleration_limiter_duty_percent | 0.000000000 | 82.232142857 | 4.892857143 | new; delta 4.892857143 |

Metrics worse than gain zero: final_abs_position_error_m, position_overshoot_m, vane_command_rate_rms_deg_s, moving_mass_rms_displacement_m, moving_mass_total_travel_m, moving_mass_rate_rms_m_s, moving_mass_acceleration_rms_m_s2, moving_mass_max_abs_offset_m, moving_mass_max_abs_velocity_m_s, moving_mass_max_abs_acceleration_m_s2, moving_mass_rate_limiter_duty_percent, moving_mass_acceleration_limiter_duty_percent.

| selected limiter diagnostic | maximum duty | longest continuous duration | hard-gate duty / duration |
| --- | ---: | ---: | ---: |
| command_clipping | 0.000% | 0.000 s | 5.0% / 0.5 s |
| rail_contact | 0.000% | 0.000 s | 5.0% / 0.5 s |
| rate_limiter | 0.375% | 0.030 s | 20.0% / 1.0 s |
| acceleration_limiter | 5.750% | 0.430 s | 30.0% / 1.0 s |

Actual selected maxima remain within the unchanged physical limits (offset 0.050 m, rate 0.200 m/s, acceleration 1.000 m/s²), using the required +1e-9 tolerance for violation checks.
The comparison CSV/JSON preserves every scenario-level hard-gate, chatter, saturation, symmetry, and transient field.
