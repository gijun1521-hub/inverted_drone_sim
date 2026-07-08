import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.headless_loiter import LoiterScenarioConfig, run_headless_loiter
from compare_loiter_params import run_comparison, write_csv as write_comparison_csv, write_markdown as write_comparison_md
from params import load_interactive_config
from sweep_loiter_authority import run_sweep, write_csv as write_sweep_csv, write_markdown as write_sweep_md


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
                scenario="horizontal_impulse_recovery",
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
            self.assertIn("preliminary authority map", md_path.read_text(encoding="utf-8").lower())

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
