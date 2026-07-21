from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

from analysis.seminar_video_renderer import (
    RenderConfig,
    detect_ffmpeg,
    render_seminar_comparison,
    render_single_result,
    synchronized_frame_timestamps,
)
from analysis.seminar_video_scenarios import (
    ASSIST_GAIN_M_PER_NM,
    METRIC_COLUMNS,
    run_all_scenarios,
    run_seminar_variant,
    seminar_scenarios,
    seminar_variants,
    write_metrics_csv,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "seminar_scenario_videos.ipynb"


class SeminarScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.short_results = run_all_scenarios(duration_s=0.05)
        cls.by_key = {result.key: result for result in cls.short_results}

    def test_paired_variants_share_scenario_and_timing_definitions(self):
        for scenario in seminar_scenarios():
            locked = scenario.config
            assist = scenario.config
            self.assertEqual(locked, assist)
            self.assertEqual(locked.duration_s, 8.0)
            self.assertEqual((locked.initial_x, locked.initial_z, locked.initial_theta_deg), (0.0, 1.0, 0.0))
            self.assertEqual((locked.target_x, locked.target_z), (0.0, 1.0))
        loiter, forward = seminar_scenarios()
        self.assertEqual(
            (loiter.config.disturbance_start_s, loiter.config.disturbance_duration_s, loiter.config.disturbance_force_x),
            (1.5, 0.2, 8.0),
        )
        self.assertEqual(
            (forward.config.target_step_time_s, forward.config.target_step_x),
            (1.0, 1.0),
        )

    def test_locked_variant_retains_physical_mass_and_stays_centered(self):
        for scenario_key in ("loiter", "forward_1m"):
            result = self.by_key[(scenario_key, "locked")]
            self.assertTrue(result.rb_config.moving_mass.enabled)
            self.assertEqual(result.rb_config.m, 2.0)
            self.assertEqual(result.rb_config.moving_mass.mass_kg, 0.5)
            self.assertEqual(result.rb_config.moving_mass.max_offset_m, 0.05)
            self.assertEqual(result.variant.assist_gain_m_per_Nm, 0.0)
            self.assertTrue(all(float(row["moving_mass_offset_m"]) == 0.0 for row in result.run.rows))
            self.assertTrue(all(float(row["moving_mass_target_m"]) == 0.0 for row in result.run.rows))

    def test_active_variant_uses_selected_gain(self):
        for scenario_key in ("loiter", "forward_1m"):
            result = self.by_key[(scenario_key, "assist")]
            self.assertEqual(result.variant.assist_gain_m_per_Nm, ASSIST_GAIN_M_PER_NM)
            self.assertEqual(result.metrics["assist_gain_m_per_Nm"], 0.0415)

    def test_total_com_geometry_and_legacy_moment_policy_match(self):
        for result in self.short_results:
            mm = result.rb_config.moving_mass
            self.assertTrue(mm.use_total_com_geometry)
            self.assertFalse(mm.use_legacy_gravity_offset_moment)
            self.assertEqual(mm.moving_mass_body_up_offset_m, 0.12)
            self.assertTrue(all(int(row["total_com_geometry_active"]) == 1 for row in result.run.rows))
            self.assertTrue(all(float(row["legacy_moving_mass_moment"]) == 0.0 for row in result.run.rows))

    def test_target_step_is_applied_at_exact_configured_time(self):
        scenario = seminar_scenarios(duration_s=1.05)[1]
        result = run_seminar_variant(scenario, seminar_variants()[0])
        self.assertEqual(scenario.config.target_step_time_s, 1.0)
        before = [row for row in result.run.rows if float(row["sim_time"]) <= 1.0 + 1e-9]
        after = [row for row in result.run.rows if float(row["sim_time"]) > 1.0 + 1e-9]
        self.assertTrue(before and after)
        self.assertTrue(all(float(row["target_x"]) == 0.0 for row in before))
        self.assertTrue(all(float(row["target_x"]) == 1.0 for row in after))
        first_commanded_row = after[0]
        integration_start = float(first_commanded_row["sim_time"]) - float(first_commanded_row["physics_dt"])
        self.assertAlmostEqual(integration_start, scenario.config.target_step_time_s, places=12)

    def test_repeated_runs_produce_identical_metric_rows(self):
        repeated = run_all_scenarios(duration_s=0.05)
        self.assertEqual(
            [result.metrics for result in self.short_results],
            [result.metrics for result in repeated],
        )

    def test_frame_timestamps_are_synchronized_for_all_four_panels(self):
        timestamps = synchronized_frame_timestamps(self.short_results, fps=20)
        self.assertEqual(len(self.short_results), 4)
        self.assertEqual(timestamps.tolist(), [0.0])
        self.assertEqual({len(result.run.rows) for result in self.short_results}, {10})

    def test_metric_csv_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_metrics_csv(self.short_results, Path(temp_dir) / "metrics.csv")
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, METRIC_COLUMNS)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({(row["scenario"], row["variant"]) for row in rows}), 4)
            for row in rows:
                for key in METRIC_COLUMNS:
                    self.assertIn(key, row)
                    self.assertIsNotNone(row[key])

    def test_renderer_generates_small_png_gif_and_four_panel_report(self):
        config = RenderConfig(
            fps=20,
            panel_width=240,
            panel_height=136,
            gif_fps=10,
            gif_width=240,
            gif_height=136,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            png_path = temp / "single.png"
            gif_path = temp / "single.gif"
            render_single_result(
                self.short_results[0], config=config, png_path=png_path, gif_path=gif_path
            )
            self.assertGreater(png_path.stat().st_size, 0)
            self.assertGreater(gif_path.stat().st_size, 0)
            report = render_seminar_comparison(
                self.short_results, temp / "composite", config=config, write_mp4=False
            )
            self.assertEqual(report["panel_count"], 4)
            self.assertEqual(report["frame_count"], 1)
            self.assertGreater((temp / "composite" / "seminar_video_thumbnail.png").stat().st_size, 0)
            self.assertGreater((temp / "composite" / "seminar_scenario_comparison.gif").stat().st_size, 0)

    def test_short_mp4_only_when_encoder_is_available(self):
        encoder = detect_ffmpeg()
        if not encoder.available:
            self.skipTest("FFmpeg encoder is unavailable")
        config = RenderConfig(
            fps=20,
            panel_width=240,
            panel_height=136,
            gif_fps=10,
            gif_width=240,
            gif_height=136,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            mp4_path = Path(temp_dir) / "short.mp4"
            render_single_result(
                self.short_results[0], config=config, mp4_path=mp4_path, encoder=encoder
            )
            self.assertGreater(mp4_path.stat().st_size, 0)

    def test_metrics_are_finite(self):
        for result in self.short_results:
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value), f"{result.key} {key}")


class SeminarNotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_has_no_stored_execution_errors(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                self.assertNotEqual(output.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
