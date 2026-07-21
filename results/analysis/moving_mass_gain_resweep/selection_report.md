# Moving-mass assist-gain resweep

The merged Pitch controller is fixed at P/I/D `0.09375 / 0.0 / 0.02100`, Angle P `25.0`. Only the simulation-only `moving_mass_assist_gain_m_per_Nm` changed.

Stage 0 gain `0.0` is the valid Vane-only comparison baseline: it passes all seven full-duration scenarios and maintains exactly zero moving-mass actual and target displacement.

Raw-score rank 1 is gain `0.00000` with score `1.500000000000`. Gain-zero score is `1.500000000000`; improvement is `0.000%`. Decision: **retain_gain_zero_valid_raw_score_rank1**. Gain zero is the valid raw-score rank-1 result, so no nonzero moving-mass gain is adopted.

The old PR #20 gain `0.1325` is included only as a reference. PR #20 uses the obsolete pre-PR-22 Pitch controller and was not modified.

All selected hard gates, symmetry, chatter, saturation, and moving-mass physical-limit results are in `selection_comparison.json` and the scenario results. No hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.

# Moving-mass gain selection

The merged PR #22 Pitch controller is fixed. This is a simulation-only virtual-actuator sweep; no hardware, HIL, Pixhawk, Raspberry Pi, or real-flight claim is made.

Stage 0 gain `0.0` is a **VALID Vane-only comparison baseline**: it passes all seven full-duration scenarios with exactly zero moving-mass target and actual displacement.

Raw-score rank 1 is gain `0.00000` with score `1.500000000000`. Decision: **retain_gain_zero_valid_raw_score_rank1**.

All nonzero candidates are hard-rejected for Moving-mass saturation and/or acceleration-limit saturation. The old PR #20 reference gain `0.1325` is hard-rejected; PR #20 uses the obsolete pre-PR-22 Pitch controller and is comparison-only.

| metric | gain 0 Vane-only | old PR #20 reference 0.1325 | selected | selected vs gain 0 |
| --- | ---: | ---: | ---: | ---: |
| tail_rms_pitch_deg | 0.525546418 | 1.085698848 | 0.525546418 | 0.000% |
| tail_rms_pitch_rate_deg_s | 2.372223592 | 10.490676146 | 2.372223592 | 0.000% |
| tail_rms_horizontal_velocity_m_s | 0.030239088 | 0.039889226 | 0.030239088 | 0.000% |
| tail_path_length_m | 0.063139595 | 0.084689013 | 0.063139595 | 0.000% |
| final_abs_position_error_m | 0.017361590 | 0.011098084 | 0.017361590 | 0.000% |
| position_overshoot_m | 0.330149368 | 0.331038314 | 0.330149368 | 0.000% |
| recovery_excursion_m | 0.585183881 | 0.589338292 | 0.585183881 | 0.000% |
| strict_settling_time_s | 5.134285714 | 9.692142857 | 5.134285714 | 0.000% |
| vane_command_rms_deg | 0.732413779 | 1.824421029 | 0.732413779 | 0.000% |
| vane_command_total_variation_deg | 34.517019467 | 170.130519689 | 34.517019467 | 0.000% |
| vane_command_rate_rms_deg_s | 30.716419702 | 40.177003561 | 30.716419702 | 0.000% |
| moving_mass_rms_displacement_m | 0.000000000 | 0.008011024 | 0.000000000 | 0.000% |
| moving_mass_total_travel_m | 0.000000000 | 0.699613193 | 0.000000000 | 0.000% |
| moving_mass_rate_rms_m_s | 0.000000000 | 0.078769685 | 0.000000000 | 0.000% |
| moving_mass_acceleration_rms_m_s2 | 0.000000000 | 0.889595794 | 0.000000000 | 0.000% |

The comparison CSV/JSON preserves every scenario-level hard-gate, chatter, saturation, symmetry, and transient field. Since gain zero is selected, no performance or effort metric is hidden as worse than the adopted baseline.
