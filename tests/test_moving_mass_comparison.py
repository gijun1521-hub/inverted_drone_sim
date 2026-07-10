import copy
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.moving_mass_comparison import (
    CSV_FIELDS,
    add_baseline_deltas,
    default_variants,
    resolve_comparison_scenarios,
    run_moving_mass_comparison,
    write_csv,
    write_markdown,
)


EXPECTED_VARIANTS = {
    "vane_only",
    "moving_mass_fixed_target",
    "moving_mass_proportional_assist",
    "total_com_geometry_centered",
    "total_com_geometry_fixed_target",
    "total_com_geometry_proportional_assist",
}


class MovingMassComparisonTests(unittest.TestCase):
    def test_variant_definitions_preserve_legacy_behavior_and_command_parity(self):
        variants = {variant.name: variant for variant in default_variants(
            fixed_target_m=0.031,
            proportional_gain_m_per_Nm=0.047,
        )}
        self.assertEqual(set(variants), EXPECTED_VARIANTS)

        vane_only = variants["vane_only"]
        self.assertFalse(vane_only.moving_mass_enabled)
        self.assertFalse(vane_only.use_total_com_geometry)
        self.assertTrue(vane_only.use_legacy_gravity_offset_moment)

        for name in ("moving_mass_fixed_target", "moving_mass_proportional_assist"):
            variant = variants[name]
            self.assertTrue(variant.moving_mass_enabled)
            self.assertFalse(variant.use_total_com_geometry)
            self.assertTrue(variant.use_legacy_gravity_offset_moment)
            self.assertEqual(variant.mode_baseline_variant, "vane_only")

        centered = variants["total_com_geometry_centered"]
        self.assertFalse(centered.moving_mass_enabled)
        self.assertTrue(centered.use_total_com_geometry)
        self.assertFalse(centered.use_legacy_gravity_offset_moment)

        legacy_fixed = variants["moving_mass_fixed_target"]
        geometry_fixed = variants["total_com_geometry_fixed_target"]
        self.assertEqual(legacy_fixed.moving_mass_target_m, geometry_fixed.moving_mass_target_m)
        self.assertEqual(legacy_fixed.moving_mass_target_m, 0.031)

        legacy_proportional = variants["moving_mass_proportional_assist"]
        geometry_proportional = variants["total_com_geometry_proportional_assist"]
        self.assertEqual(
            legacy_proportional.moving_mass_assist_gain_m_per_Nm,
            geometry_proportional.moving_mass_assist_gain_m_per_Nm,
        )
        self.assertEqual(legacy_proportional.moving_mass_assist_gain_m_per_Nm, 0.047)
        for variant in (geometry_fixed, geometry_proportional):
            self.assertTrue(variant.use_total_com_geometry)
            self.assertFalse(variant.use_legacy_gravity_offset_moment)
            self.assertEqual(variant.mode_baseline_variant, "total_com_geometry_centered")

    def test_comparison_runs_six_variants_with_expected_modes_and_diagnostics(self):
        sys.modules.pop("pygame", None)
        scenarios = resolve_comparison_scenarios("pitch_assist_probe", duration_s=0.15)
        results = run_moving_mass_comparison(
            param_path="params/loiter_example.json",
            scenarios=scenarios,
        )

        self.assertEqual(len(results), 6)
        self.assertEqual({result.variant.name for result in results}, EXPECTED_VARIANTS)
        by_variant = {result.variant.name: result.row for result in results}

        self.assertEqual(by_variant["vane_only"]["moving_mass_model_mode"], "disabled")
        self.assertFalse(by_variant["vane_only"]["moving_mass_enabled"])
        self.assertFalse(by_variant["vane_only"]["total_com_geometry_active"])
        self.assertFalse(by_variant["vane_only"]["legacy_gravity_offset_active"])
        self.assertEqual(by_variant["vane_only"]["state_dimension"], 8)
        self.assertEqual(float(by_variant["vane_only"]["moving_mass_max_offset_m"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["moving_mass_saturation_percent"]), 0.0)
        for field in (
            "delta_max_theta_deg",
            "delta_rms_theta_deg",
            "delta_final_abs_x_error",
            "delta_rms_x_error",
        ):
            self.assertEqual(float(by_variant["vane_only"][field]), 0.0)

        for name in ("moving_mass_fixed_target", "moving_mass_proportional_assist"):
            row = by_variant[name]
            self.assertEqual(row["moving_mass_model_mode"], "legacy_gravity_offset")
            self.assertTrue(row["moving_mass_enabled"])
            self.assertFalse(row["total_com_geometry_active"])
            self.assertTrue(row["legacy_gravity_offset_active"])
            self.assertEqual(row["state_dimension"], 11)
            self.assertEqual(float(row["max_abs_thrust_moment_from_com_offset"]), 0.0)
            self.assertGreater(float(row["max_abs_legacy_moving_mass_moment"]), 0.0)

        centered = by_variant["total_com_geometry_centered"]
        self.assertEqual(centered["moving_mass_model_mode"], "total_com_geometry")
        self.assertFalse(centered["moving_mass_enabled"])
        self.assertTrue(centered["total_com_geometry_active"])
        self.assertFalse(centered["legacy_gravity_offset_active"])
        self.assertEqual(centered["state_dimension"], 8)
        self.assertEqual(float(centered["max_abs_total_com_body_right_m"]), 0.0)
        self.assertGreater(float(centered["max_abs_total_com_body_up_m"]), 0.0)
        self.assertAlmostEqual(float(centered["max_abs_thrust_moment_from_com_offset"]), 0.0)

        for name in (
            "total_com_geometry_fixed_target",
            "total_com_geometry_proportional_assist",
        ):
            row = by_variant[name]
            self.assertEqual(row["moving_mass_model_mode"], "total_com_geometry")
            self.assertTrue(row["moving_mass_enabled"])
            self.assertTrue(row["total_com_geometry_active"])
            self.assertFalse(row["legacy_gravity_offset_active"])
            self.assertEqual(row["state_dimension"], 11)
            self.assertGreater(float(row["max_abs_total_com_body_right_m"]), 0.0)
            self.assertGreater(float(row["max_abs_thrust_moment_from_com_offset"]), 0.0)
            self.assertEqual(float(row["max_abs_legacy_moving_mass_moment"]), 0.0)

        for row in by_variant.values():
            self.assertEqual(float(row["total_mass_kg"]), 1.5)
            self.assertEqual(float(row["moving_mass_mass_kg"]), 0.5)
            self.assertEqual(float(row["effective_moving_mass_max_offset_m"]), 0.05)
            self.assertEqual(float(row["moving_mass_body_up_offset_m"]), 0.12)
            self.assertFalse(
                bool(row["total_com_geometry_active"])
                and bool(row["legacy_gravity_offset_active"])
            )
        self.assertNotIn("pygame", sys.modules)

    def test_baselines_are_explicit_order_independent_and_missing_rows_fail(self):
        results = run_moving_mass_comparison(
            param_path="params/loiter_example.json",
            scenarios=resolve_comparison_scenarios("pitch_assist_probe", duration_s=0.10),
        )
        expected = {
            result.variant.name: (
                result.row["delta_rms_theta_deg"],
                result.row["delta_vs_mode_baseline_rms_theta_deg"],
            )
            for result in results
        }
        reordered = [copy.deepcopy(result.row) for result in reversed(results)]
        add_baseline_deltas(reordered)
        self.assertEqual(
            {
                str(row["variant"]): (
                    row["delta_rms_theta_deg"],
                    row["delta_vs_mode_baseline_rms_theta_deg"],
                )
                for row in reordered
            },
            expected,
        )

        by_variant = {result.variant.name: result.row for result in results}
        active = by_variant["total_com_geometry_fixed_target"]
        centered = by_variant["total_com_geometry_centered"]
        self.assertAlmostEqual(
            float(active["delta_vs_mode_baseline_rms_theta_deg"]),
            float(active["rms_theta_deg"]) - float(centered["rms_theta_deg"]),
        )
        self.assertAlmostEqual(
            float(active["delta_rms_theta_deg"]),
            float(active["rms_theta_deg"]) - float(by_variant["vane_only"]["rms_theta_deg"]),
        )

        without_vane = [copy.deepcopy(row) for row in reordered if row["variant"] != "vane_only"]
        with self.assertRaisesRegex(ValueError, "missing baseline variant 'vane_only'"):
            add_baseline_deltas(without_vane)

        without_centered = [
            copy.deepcopy(row)
            for row in reordered
            if row["variant"] != "total_com_geometry_centered"
        ]
        with self.assertRaisesRegex(
            ValueError, "missing mode baseline variant 'total_com_geometry_centered'"
        ):
            add_baseline_deltas(without_centered)

    def test_comparison_writes_stable_csv_and_grouped_markdown(self):
        results = run_moving_mass_comparison(
            param_path="params/loiter_example.json",
            scenarios=resolve_comparison_scenarios("pitch_assist_probe", duration_s=0.10),
        )

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(results, Path(tmp) / "moving_mass_comparison.csv")
            md_path = write_markdown(results, Path(tmp) / "moving_mass_comparison.md")

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 6)
            self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
            self.assertEqual({row["variant"] for row in rows}, EXPECTED_VARIANTS)
            for old_field in (
                "delta_max_theta_deg",
                "delta_rms_theta_deg",
                "delta_final_abs_x_error",
                "delta_rms_x_error",
                "attitude_improvement_score",
            ):
                self.assertIn(old_field, CSV_FIELDS)
            for new_field in (
                "moving_mass_model_mode",
                "baseline_variant",
                "mode_baseline_variant",
                "total_com_geometry_active",
                "legacy_gravity_offset_active",
                "delta_vs_mode_baseline_rms_theta_deg",
                "max_abs_thrust_moment_from_com_offset",
            ):
                self.assertIn(new_field, CSV_FIELDS)

            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Historical baseline", md_text)
            self.assertIn("Legacy gravity-offset variants", md_text)
            self.assertIn("Total-COM centered baseline", md_text)
            self.assertIn("Total-COM active variants", md_text)
            self.assertIn("Deltas Versus Mode-Matched Baseline", md_text)
            self.assertIn("vane_moment_about_total_com", md_text)
            self.assertIn("does not include reaction kick", md_text)
            self.assertIn("not calibrated flight values", md_text)

    def test_cli_all_short_duration_no_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    "compare_moving_mass_assist.py",
                    "--scenario",
                    "all",
                    "--duration",
                    "0.02",
                    "--no-plots",
                    "--output-dir",
                    tmp,
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with (Path(tmp) / "moving_mass_comparison.csv").open(
                newline="", encoding="utf-8"
            ) as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5 * 6)
            scenarios = {row["scenario_name"] for row in rows}
            self.assertEqual(len(scenarios), 5)
            for scenario_name in scenarios:
                scenario_rows = [row for row in rows if row["scenario_name"] == scenario_name]
                self.assertEqual(len(scenario_rows), 6)
                self.assertEqual({row["variant"] for row in scenario_rows}, EXPECTED_VARIANTS)
            self.assertTrue((Path(tmp) / "moving_mass_comparison.md").is_file())


if __name__ == "__main__":
    unittest.main()
