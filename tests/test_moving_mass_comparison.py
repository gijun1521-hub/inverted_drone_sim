import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.moving_mass_comparison import (
    CSV_FIELDS,
    default_variants,
    resolve_comparison_scenarios,
    run_moving_mass_comparison,
    write_csv,
    write_markdown,
)


class MovingMassComparisonTests(unittest.TestCase):
    def test_comparison_runs_all_variants_without_pygame(self):
        sys.modules.pop("pygame", None)
        scenarios = resolve_comparison_scenarios("pitch_assist_probe", duration_s=0.15)

        results = run_moving_mass_comparison(
            param_path="params/loiter_example.json",
            scenarios=scenarios,
            variants=default_variants(),
        )

        self.assertEqual({result.variant.name for result in results}, {
            "vane_only",
            "moving_mass_fixed_target",
            "moving_mass_proportional_assist",
        })
        by_variant = {result.variant.name: result.row for result in results}
        self.assertFalse(by_variant["vane_only"]["moving_mass_enabled"])
        self.assertTrue(by_variant["moving_mass_fixed_target"]["moving_mass_enabled"])
        self.assertTrue(by_variant["moving_mass_proportional_assist"]["moving_mass_enabled"])
        self.assertEqual(float(by_variant["vane_only"]["moving_mass_max_offset_m"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["moving_mass_saturation_percent"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["delta_max_theta_deg"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["delta_rms_theta_deg"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["delta_final_abs_x_error"]), 0.0)
        self.assertEqual(float(by_variant["vane_only"]["delta_rms_x_error"]), 0.0)
        self.assertGreater(float(by_variant["moving_mass_fixed_target"]["moving_mass_max_offset_m"]), 0.0)
        self.assertGreater(float(by_variant["moving_mass_proportional_assist"]["moving_mass_max_offset_m"]), 0.0)
        for row in by_variant.values():
            for key in (
                "final_abs_x_error",
                "max_theta_deg",
                "rms_theta_deg",
                "moving_mass_max_offset_m",
                "moving_mass_saturation_percent",
            ):
                self.assertIn(key, row)
        self.assertNotIn("pygame", sys.modules)

    def test_comparison_writes_csv_and_markdown(self):
        scenarios = resolve_comparison_scenarios("pitch_assist_probe", duration_s=0.10)
        results = run_moving_mass_comparison(param_path="params/loiter_example.json", scenarios=scenarios)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = write_csv(results, Path(tmp) / "moving_mass_comparison.csv")
            md_path = write_markdown(results, Path(tmp) / "moving_mass_comparison.md")

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 3)
            self.assertEqual(list(rows[0].keys()), CSV_FIELDS)
            self.assertEqual({row["variant"] for row in rows}, {
                "vane_only",
                "moving_mass_fixed_target",
                "moving_mass_proportional_assist",
            })
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("Moving Mass Comparison Analysis", md_text)
            self.assertIn("Cases Where Moving Mass Worsens Performance", md_text)
            self.assertIn("explicit total-CG geometry shift", md_text)
            self.assertIn("approximately 0.5 kg", md_text)


if __name__ == "__main__":
    unittest.main()
