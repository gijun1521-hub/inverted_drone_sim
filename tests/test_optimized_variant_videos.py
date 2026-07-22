from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from analysis.optimized_variant_video_scenarios import (
    compare_to_pr25,
    expected_profile_values,
    optimized_scenarios,
    optimized_variants,
    run_all_scenarios,
    validate_profile_sources,
)
from analysis.seminar_video_renderer import RenderConfig, render_optimized_comparison


class OptimizedVariantScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_all_scenarios(duration_s=0.05)
        cls.by_key = {result.key: result for result in cls.results}

    def test_profiles_are_the_only_full_precision_controller_sources(self):
        report = validate_profile_sources()["profiles"]
        self.assertEqual(set(report), {"vane_only", "moving_mass_assist"})
        self.assertNotEqual(report["vane_only"]["controller"], report["moving_mass_assist"]["controller"])
        self.assertEqual(report["vane_only"]["assist_gain_m_per_Nm"], 0.0)
        self.assertEqual(
            report["moving_mass_assist"]["assist_gain_m_per_Nm"],
            expected_profile_values()["moving_mass_assist"]["assist_gain_m_per_Nm"],
        )

    def test_video_scenarios_use_pr24_timing_and_ten_second_default(self):
        loiter, forward = optimized_scenarios()
        self.assertEqual(loiter.config.duration_s, 10.0)
        self.assertEqual(
            (
                loiter.config.initial_x,
                loiter.config.initial_z,
                loiter.config.initial_theta_deg,
                loiter.config.disturbance_start_s,
                loiter.config.disturbance_duration_s,
                loiter.config.disturbance_force_x,
            ),
            (0.0, 1.0, 0.0, 1.5, 0.2, 8.0),
        )
        self.assertEqual((forward.config.target_step_time_s, forward.config.target_step_x), (1.0, 1.0))

    def test_paired_runs_share_scenario_vehicle_and_timing(self):
        for scenario in ("loiter", "forward_1m"):
            vane = self.by_key[(scenario, "vane_only")]
            assist = self.by_key[(scenario, "moving_mass_assist")]
            self.assertEqual(vane.scenario.config, assist.scenario.config)
            self.assertEqual(vane.rb_config, assist.rb_config)
            self.assertEqual(vane.metrics["physics_dt_s"], assist.metrics["physics_dt_s"])
            self.assertEqual(vane.metrics["controller_dt_s"], assist.metrics["controller_dt_s"])

    def test_runtime_controllers_are_distinct_and_match_profiles(self):
        vane = self.by_key[("loiter", "vane_only")]
        assist = self.by_key[("loiter", "moving_mass_assist")]
        self.assertNotEqual(vane.controller_config, assist.controller_config)
        expected_values = expected_profile_values()
        for result in (vane, assist):
            expected = expected_values[result.variant.key]
            for key in (
                "atc_rat_pit_p",
                "atc_rat_pit_i",
                "atc_rat_pit_d",
                "atc_ang_pit_p",
                "psc_ne_pos_p",
                "psc_ne_vel_p",
            ):
                self.assertTrue(math.isclose(getattr(result.controller_config, key), expected[key], abs_tol=1e-15))

    def test_vane_only_mass_command_and_state_are_exact_zero(self):
        for scenario in ("loiter", "forward_1m"):
            result = self.by_key[(scenario, "vane_only")]
            self.assertEqual(result.variant.assist_gain_m_per_Nm, 0.0)
            for row in result.run.rows:
                self.assertEqual(float(row["moving_mass_target_m"]), 0.0)
                self.assertEqual(float(row["moving_mass_offset_m"]), 0.0)
                self.assertEqual(float(row["moving_mass_velocity_m_s"]), 0.0)
            self.assertEqual(result.metrics["moving_mass_max_acceleration_m_s2"], 0.0)

    def test_assist_uses_pr25_selected_gain(self):
        expected = expected_profile_values()["moving_mass_assist"]["assist_gain_m_per_Nm"]
        for scenario in ("loiter", "forward_1m"):
            self.assertEqual(
                self.by_key[(scenario, "moving_mass_assist")].variant.assist_gain_m_per_Nm,
                expected,
            )

    def test_short_horizon_pr25_check_still_verifies_all_effective_parameters(self):
        report = compare_to_pr25(self.results)
        self.assertTrue(report["passed"])
        self.assertTrue(report["dynamic_metrics_skipped_for_short_horizon"])
        self.assertEqual(report["comparison_count"], 28)

    def test_reduced_resolution_renderer_produces_composite_assets(self):
        config = RenderConfig(
            fps=20,
            panel_width=240,
            panel_height=136,
            gif_fps=10,
            gif_width=240,
            gif_height=136,
            optimized_hud=True,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            report = render_optimized_comparison(self.results, output, config=config, write_mp4=False)
            self.assertEqual(report["panel_order"][0], "loiter/vane_only")
            self.assertEqual(report["individual_size_px"], [240, 136])
            self.assertEqual(report["composite_size_px"], [480, 272])
            self.assertGreater((output / "final_optimized_comparison.gif").stat().st_size, 0)
            self.assertGreater((output / "final_optimized_thumbnail.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
