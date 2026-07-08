import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.headless_loiter import LoiterScenarioConfig, run_headless_loiter
from analysis.vane_authority import (
    default_grid,
    resolve_authority_scenarios,
    write_markdown as write_authority_markdown,
    write_plots as write_authority_plots,
)
from compare_loiter_params import run_comparison, write_csv as write_comparison_csv, write_markdown as write_comparison_md
from params import load_interactive_config
from sweep_loiter_authority import (
    run_sweep,
    sweep_sensitivity,
    write_csv as write_sweep_csv,
    write_markdown as write_sweep_md,
)


def synthetic_authority_row(
    *,
    scenario_name="synthetic",
    angle=5.0,
    rate=80.0,
    thrust=1.4,
    passed=True,
    failure_reason="",
    final_abs_x_error=1.0,
    combined_design_score=0.6,
    authority_limited_percent=0.0,
    servo_rate_saturation_percent=0.0,
    mixer_saturation_percent=0.0,
    motor_saturation_percent=0.0,
):
    return {
        "scenario_name": scenario_name,
        "param_file": "params/loiter_example.json",
        "vane_angle_max_deg": angle,
        "vane_rate_limit_deg_s": rate,
        "T_max_factor": thrust,
        "effective_vane_angle_max_deg": angle,
        "effective_vane_rate_limit_deg_s": rate,
        "effective_T_max_factor": thrust,
        "effective_T_max_N": thrust,
        "pass": passed,
        "failure_reason": failure_reason,
        "crash_reason": "",
        "final_abs_x_error": final_abs_x_error,
        "final_abs_z_error": 0.1,
        "rms_x_error": final_abs_x_error,
        "rms_z_error": 0.1,
        "max_theta_deg": 10.0,
        "max_omega_deg_s": 5.0,
        "max_vane_cmd_deg": angle,
        "max_vane_actual_deg": angle,
        "max_thrust_cmd_N": 1.0,
        "max_thrust_actual_N": 1.0,
        "mixer_saturation_percent": mixer_saturation_percent,
        "mixer_angle_saturation_percent": 0.0,
        "authority_limited_percent": authority_limited_percent,
        "servo_angle_saturation_percent": 0.0,
        "servo_rate_saturation_percent": servo_rate_saturation_percent,
        "motor_saturation_percent": motor_saturation_percent,
        "final_x": 0.0,
        "final_z": 1.0,
        "final_vx": 0.0,
        "final_vz": 0.0,
        "authority_margin_score": 1.0,
        "recovery_score": 0.8,
        "saturation_score": 1.0,
        "combined_design_score": combined_design_score,
    }


class HeadlessLoiterTests(unittest.TestCase):
    def test_headless_runner_returns_finite_metrics(self):
        scenario = LoiterScenarioConfig(name="short", duration_s=0.4, capture_current_target=True)
        result = run_headless_loiter("params/loiter_example.json", scenario)

        self.assertGreater(len(result.rows), 0)
        for key in ("final_abs_x_error", "rms_x_error", "max_theta_deg", "max_vane_cmd_deg"):
            self.assertTrue(np.isfinite(float(result.metrics[key])), key)

    def test_stick_move_release_runs_without_crash_for_default_params(self):
        scenario = LoiterScenarioConfig(
            name="short_stick",
            duration_s=1.0,
            stick_start_s=0.1,
            stick_end_s=0.3,
            stick_x=0.4,
            capture_current_target=True,
        )
        result = run_headless_loiter("params/loiter_example.json", scenario)

        self.assertFalse(result.crashed, result.crash_reason)
        self.assertEqual(result.metrics["crash_reason"], "")

    def test_comparison_script_writes_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                params=["params/loiter_example.json"],
                scenario="horizontal_impulse_recovery",
                duration=0.5,
                output_dir=tmp,
                save_timeseries=False,
            )
            results = run_comparison(args)
            csv_path = write_comparison_csv(results, Path(tmp) / "comparison.csv")
            md_path = write_comparison_md(results, Path(tmp) / "comparison.md")

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertIn("analytical 2D", md_path.read_text(encoding="utf-8"))

    def test_authority_sweep_writes_csv_and_markdown_with_small_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                params="params/loiter_example.json",
                scenario="authority_stress",
                duration=0.4,
                output_dir=tmp,
                vane_angle_max_deg="5",
                vane_rate_limit_deg_s="60",
                T_max_factor="1.4",
            )
            rows = run_sweep(args)
            csv_path = write_sweep_csv(rows, Path(tmp) / "sweep.csv")
            md_path = write_sweep_md(rows, Path(tmp) / "sweep.md", args.scenario)

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(len(rows), 1)
            self.assertIn("vane authority mapping", md_path.read_text(encoding="utf-8").lower())

    def test_authority_scenario_names_resolve(self):
        names = {scenario.name for scenario in resolve_authority_scenarios("all")}

        self.assertIn("authority_stress", names)
        self.assertIn("impulse_light", names)
        self.assertIn("impulse_heavy", names)
        self.assertIn("offset_recovery_2m", names)
        self.assertIn("stick_step_aggressive", names)
        self.assertIn("low_thrust_margin", names)

    def test_quick_grid_and_scenario_all_expand(self):
        grid = default_grid(quick=True)
        scenarios = resolve_authority_scenarios("all", duration_s=0.05)

        self.assertGreater(len(grid.vane_angle_max_deg), 1)
        self.assertGreater(len(scenarios), 1)

    def test_sweep_overrides_change_effective_config_values(self):
        low = LoiterScenarioConfig(name="short_low", duration_s=0.1, capture_current_target=True)
        high = LoiterScenarioConfig(name="short_high", duration_s=0.1, capture_current_target=True)

        low_result = run_headless_loiter(
            "params/loiter_example.json",
            low,
            rb_overrides={"vane_angle_max_deg": 0.5, "vane_rate_limit_deg_s": 5.0, "T_max_factor": 1.05},
        )
        high_result = run_headless_loiter(
            "params/loiter_example.json",
            high,
            rb_overrides={"vane_angle_max_deg": 20.0, "vane_rate_limit_deg_s": 160.0, "T_max_factor": 2.5},
        )

        self.assertNotEqual(low_result.metrics["effective_vane_angle_max_rad"], high_result.metrics["effective_vane_angle_max_rad"])
        self.assertNotEqual(low_result.metrics["effective_vane_rate_limit_rad_s"], high_result.metrics["effective_vane_rate_limit_rad_s"])
        self.assertNotEqual(low_result.metrics["effective_T_max_N"], high_result.metrics["effective_T_max_N"])

    def test_authority_sweep_extremes_are_not_invariant(self):
        args = SimpleNamespace(
            params="params/loiter_example.json",
            scenario="authority_stress",
            duration=None,
            output_dir="",
            vane_angle_max_deg="0.5,2,20",
            vane_rate_limit_deg_s="5,160",
            T_max_factor="1.05,2.5",
        )

        rows = run_sweep(args)
        stats = sweep_sensitivity(rows)

        self.assertGreater(int(stats["unique_final_abs_x_error"]), 1)
        self.assertGreater(int(stats["unique_max_vane_actual_deg"]), 1)
        self.assertFalse(stats["inconclusive"])

    def test_authority_design_scores_are_finite_and_ordered(self):
        args = SimpleNamespace(
            params="params/loiter_example.json",
            scenario="authority_stress",
            duration=None,
            output_dir="",
            vane_angle_max_deg="0.5,20",
            vane_rate_limit_deg_s="5,160",
            T_max_factor="1.05,2.5",
        )

        rows = run_sweep(args)
        for row in rows:
            self.assertTrue(np.isfinite(float(row["combined_design_score"])))
        low = [row for row in rows if float(row["vane_angle_max_deg"]) == 0.5]
        high = [row for row in rows if float(row["vane_angle_max_deg"]) == 20.0]
        self.assertLess(
            max(float(row["combined_design_score"]) for row in low),
            max(float(row["combined_design_score"]) for row in high),
        )

    def test_authority_sweep_markdown_reports_sensitivity_or_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                params="params/loiter_example.json",
                scenario="authority_stress",
                duration=None,
                output_dir=tmp,
                vane_angle_max_deg="0.5,2,20",
                vane_rate_limit_deg_s="5,160",
                T_max_factor="1.05,2.5",
            )
            rows = run_sweep(args)
            md_path = write_sweep_md(rows, Path(tmp) / "sweep.md", args.scenario)
            text = md_path.read_text(encoding="utf-8")

            self.assertIn("## Sensitivity Check", text)
            self.assertIn("## Recommended Design Regions", text)
            self.assertIn("## Limitations", text)
            self.assertRegex(text, r"unique final_abs_x_error values: [2-9]")
            self.assertNotIn("**INCONCLUSIVE:**", text)

    def test_authority_markdown_separates_failed_best_error_from_recommendations(self):
        rows = [
            synthetic_authority_row(
                angle=2.0,
                rate=5.0,
                thrust=1.05,
                passed=False,
                failure_reason="saturation_or_authority_limit",
                final_abs_x_error=0.2,
                combined_design_score=0.3,
                authority_limited_percent=25.0,
            ),
            synthetic_authority_row(
                angle=8.0,
                rate=80.0,
                thrust=1.4,
                passed=True,
                final_abs_x_error=0.8,
                combined_design_score=0.85,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            md_path = write_authority_markdown(rows, Path(tmp) / "authority.md", "synthetic")
            text = md_path.read_text(encoding="utf-8")

        self.assertIn("Best overall final horizontal error, but failed due to saturation_or_authority_limit", text)
        self.assertIn("Best passing final horizontal error: 0.800 m", text)
        self.assertIn("Best passing combined design score: 0.850", text)
        self.assertNotIn("Best passing combined design score: 0.300", text)

    def test_authority_markdown_no_passing_rows_avoids_lowest_passing_claims(self):
        rows = [
            synthetic_authority_row(
                angle=2.0,
                rate=5.0,
                thrust=1.05,
                passed=False,
                failure_reason="large_final_error",
                final_abs_x_error=2.2,
                combined_design_score=0.25,
            ),
            synthetic_authority_row(
                angle=5.0,
                rate=80.0,
                thrust=1.4,
                passed=False,
                failure_reason="large_final_error",
                final_abs_x_error=1.8,
                combined_design_score=0.4,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            md_path = write_authority_markdown(rows, Path(tmp) / "authority.md", "synthetic")
            text = md_path.read_text(encoding="utf-8")

        self.assertIn("No passing cases were found in this scenario/grid.", text)
        self.assertNotIn("Lowest passing vane angle", text)
        self.assertNotIn("Lowest passing servo rate", text)
        self.assertNotIn("Lowest passing T_max_factor", text)

    def test_authority_plot_smoke_or_clean_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                params="params/loiter_example.json",
                scenario="impulse_light",
                duration=0.2,
                output_dir=tmp,
                vane_angle_max_deg="0.5,20",
                vane_rate_limit_deg_s="5,80",
                T_max_factor="1.05",
            )
            rows = run_sweep(args)
            paths = write_authority_plots(rows, Path(tmp), required=False)

            for path in paths:
                self.assertTrue(path.exists())

    def test_authority_all_fail_plot_edge_case(self):
        rows = [
            synthetic_authority_row(angle=2.0, rate=5.0, thrust=1.05, passed=False, failure_reason="large_final_error"),
            synthetic_authority_row(angle=5.0, rate=5.0, thrust=1.05, passed=False, failure_reason="large_final_error"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_authority_plots(rows, Path(tmp), required=False)
            expected = Path(tmp) / "authority_maps" / "synthetic_pass_Tmax_1p05.png"
            summary = Path(tmp) / "authority_maps" / "synthetic_minimum_passing_Tmax_summary.png"

            if paths:
                self.assertTrue(expected.exists())
                self.assertTrue(summary.exists())

    def test_authority_all_zero_percent_plot_edge_case(self):
        rows = [
            synthetic_authority_row(angle=2.0, rate=5.0, thrust=1.05),
            synthetic_authority_row(angle=5.0, rate=5.0, thrust=1.05),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_authority_plots(rows, Path(tmp), required=False)
            expected = Path(tmp) / "authority_maps" / "synthetic_authority_limited_percent_Tmax_1p05.png"

            if paths:
                self.assertTrue(expected.exists())

    def test_authority_quick_grid_annotation_smoke(self):
        rows = [
            synthetic_authority_row(angle=2.0, rate=5.0, thrust=1.05, passed=False, failure_reason="large_final_error"),
            synthetic_authority_row(angle=5.0, rate=5.0, thrust=1.05, passed=True, final_abs_x_error=0.5),
            synthetic_authority_row(angle=2.0, rate=80.0, thrust=1.05, passed=True, final_abs_x_error=0.7),
            synthetic_authority_row(angle=5.0, rate=80.0, thrust=1.05, passed=True, final_abs_x_error=0.4),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_authority_plots(rows, Path(tmp), required=False)

            for path in paths:
                self.assertTrue(path.exists())

    def test_legacy_flat_and_structured_params_load(self):
        flat_rb, _flat_ui, _flat_controller = load_interactive_config("params/low_authority_example.json")
        structured_rb, structured_ui, structured_controller = load_interactive_config("params/loiter_example.json")

        self.assertEqual(flat_rb.vane_angle_max_deg, 15.0)
        self.assertGreater(structured_ui.vane_visual_scale, 0.0)
        self.assertGreater(structured_controller.loit_speed_ms, 0.0)
        self.assertEqual(structured_rb.vane_model, "nonlinear_with_axial_loss")

    def test_headless_tools_do_not_import_pygame(self):
        sys.modules.pop("pygame", None)

        scenario = LoiterScenarioConfig(name="short_no_pygame", duration_s=0.05, capture_current_target=True)
        run_headless_loiter("params/loiter_example.json", scenario)

        self.assertNotIn("pygame", sys.modules)


if __name__ == "__main__":
    unittest.main()
