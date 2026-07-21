from __future__ import annotations

import unittest

import numpy as np

from analysis.moving_mass_gain_resweep import (
    FIXED_CONTROLLER,
    MOVING_MASS_LIMITER_HARD_GATES,
    OLD_PR20_REFERENCE_GAIN,
    STAGE1_GAINS,
    STAGE2_STEP,
    STAGE3_STEP,
    _duty_and_longest_duration,
    _grid,
    _hard_gate_reasons,
    _scenario_score,
)
from analysis.pitch_damping_retune import SCORE_WEIGHTS, required_scenarios


class MovingMassGainResweepTests(unittest.TestCase):
    def test_coarse_grid_includes_zero_upper_value_and_old_reference(self):
        self.assertIn(0.0, STAGE1_GAINS)
        self.assertIn(0.2, STAGE1_GAINS)
        self.assertIn(OLD_PR20_REFERENCE_GAIN, STAGE1_GAINS)

    def test_refinement_grids_use_declared_steps(self):
        fine = _grid(0.10, 0.025, STAGE2_STEP)
        local = _grid(0.10, 0.005, STAGE3_STEP)
        self.assertAlmostEqual(fine[1] - fine[0], STAGE2_STEP)
        self.assertAlmostEqual(local[1] - local[0], STAGE3_STEP)

    def test_limiter_diagnostic_reports_duty_and_continuous_duration(self):
        duty, longest = _duty_and_longest_duration(
            np.asarray([0, 1, 1, 0, 1], dtype=bool),
            np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        self.assertAlmostEqual(duty, 60.0)
        self.assertAlmostEqual(longest, 0.2)

    def test_normal_limiter_activity_is_not_an_automatic_rejection(self):
        definition = required_scenarios(False)[0]
        metrics = self._valid_metrics(definition)
        metrics.update({
            "moving_mass_rate_limiter_duty_percent": MOVING_MASS_LIMITER_HARD_GATES["rate_limiter"]["max_duty_percent"],
            "moving_mass_rate_limiter_longest_continuous_duration_s": MOVING_MASS_LIMITER_HARD_GATES["rate_limiter"]["max_continuous_duration_s"],
        })
        reasons = _hard_gate_reasons(definition, metrics)
        self.assertFalse(any("moving_mass_rate_limiter" in reason for reason in reasons))

    def test_actual_physical_limit_uses_declared_tolerance(self):
        definition = required_scenarios(False)[0]
        metrics = self._valid_metrics(definition)
        metrics["moving_mass_max_abs_offset_m"] = 0.05 + 0.5e-9
        self.assertNotIn("moving_mass_actual_offset_physical_limit_violation", _hard_gate_reasons(definition, metrics))
        metrics["moving_mass_max_abs_offset_m"] = 0.05 + 1.1e-9
        self.assertIn("moving_mass_actual_offset_physical_limit_violation", _hard_gate_reasons(definition, metrics))

    def test_normal_limiter_duty_is_penalized_in_score(self):
        baseline = {metric: 1.0 for metric in SCORE_WEIGHTS}
        row = dict(baseline)
        for name in MOVING_MASS_LIMITER_HARD_GATES:
            row[f"moving_mass_{name}_duty_percent"] = 0.0
        score_without = _scenario_score(row, baseline)
        row["moving_mass_rate_limiter_duty_percent"] = 20.0
        self.assertGreater(_scenario_score(row, baseline), score_without)

    @staticmethod
    def _valid_metrics(definition):
        metrics = {
            "finite": True, "crash": False, "ground_contact": False, "peak_abs_pitch_deg": 0.0,
            "premature_pause": False, "early_velocity_reversal": False,
            "second_acceleration_lobe_after_full_pause": False, "capture_discontinuity": False,
            "shaped_velocity_sign_reversal_after_release": False,
            "target_capture_count": 1, "vane_saturation_percent": 0.0,
            "servo_rate_saturation_percent": 0.0, "mixer_saturation_percent": 0.0,
            "meaningful_vane_sign_change_count": 0, "vane_total_variation_per_second_deg_s": 0.0,
            "tail_high_frequency_vane_energy_deg2": 0.0,
            "moving_mass_max_abs_offset_m": 0.0, "moving_mass_max_abs_target_m": 0.0,
            "moving_mass_max_abs_velocity_m_s": 0.0, "moving_mass_max_abs_acceleration_m_s2": 0.0,
            "effective_moving_mass_max_offset_m": 0.05, "effective_moving_mass_max_rate_m_s": 0.2,
            "effective_moving_mass_max_accel_m_s2": 1.0, "meaningful_moving_mass_direction_change_count": 0,
            "moving_mass_total_travel_per_second_m_s": 0.0, "tail_high_frequency_moving_mass_energy_m2": 0.0,
            "total_mass_kg": 2.0, "physical_moving_mass_kg": 0.5, "moving_mass_enabled": True,
            "total_com_geometry_active": True, "legacy_gravity_offset_active": False,
            "moving_mass_assist_gain_m_per_Nm": 0.0,
        }
        for name in MOVING_MASS_LIMITER_HARD_GATES:
            metrics[f"moving_mass_{name}_duty_percent"] = 0.0
            metrics[f"moving_mass_{name}_longest_continuous_duration_s"] = 0.0
        metrics.update({f"effective_{key}": value for key, value in FIXED_CONTROLLER.items()})
        return metrics


if __name__ == "__main__":
    unittest.main()
