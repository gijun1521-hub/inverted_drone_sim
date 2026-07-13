import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from analysis.controller_grid_search import (
    SCORE_SPECS,
    SHEET_NAMES,
    Candidate,
    ScenarioResultStore,
    WorkflowOptions,
    attitude_p_candidates,
    best_parameter_rows,
    compute_scenario_metrics,
    direction_reversal_count,
    loiter_xy_candidates,
    moving_mass_gain_candidates,
    normalized_score,
    path_length,
    rate_i_candidates,
    rate_pd_coarse_candidates,
    run_workflow,
    top_candidates,
    zero_crossing_count,
)
from analysis.headless_loiter import LoiterRunResult, LoiterScenarioConfig
from params import load_interactive_config


class ControllerGridDefinitionTests(unittest.TestCase):
    def test_full_grid_endpoints_and_counts(self):
        rate_pd = rate_pd_coarse_candidates()
        self.assertEqual(len(rate_pd), 16 * 17)
        self.assertEqual(rate_pd[0].parameters, {"atc_rat_pit_p": 0.005, "atc_rat_pit_i": 0.0, "atc_rat_pit_d": 0.0})
        self.assertEqual(rate_pd[-1].parameters, {"atc_rat_pit_p": 0.08, "atc_rat_pit_i": 0.0, "atc_rat_pit_d": 0.008})

        rate_i = rate_i_candidates([rate_pd[0].parameters])
        self.assertEqual(len(rate_i), 16)
        self.assertEqual(rate_i[0].parameters["atc_rat_pit_i"], 0.0)
        self.assertEqual(rate_i[-1].parameters["atc_rat_pit_i"], 0.03)

        attitude = attitude_p_candidates(rate_i[0].parameters)
        self.assertEqual(len(attitude), 17)
        self.assertEqual(attitude[0].parameters["atc_ang_pit_p"], 2.0)
        self.assertEqual(attitude[-1].parameters["atc_ang_pit_p"], 10.0)

        loiter = loiter_xy_candidates(attitude[0].parameters)
        self.assertEqual(len(loiter), 15 * 24)
        self.assertEqual(loiter[0].parameters["psc_ne_pos_p"], 0.1)
        self.assertEqual(loiter[0].parameters["psc_ne_vel_p"], 0.2)
        self.assertEqual(loiter[-1].parameters["psc_ne_pos_p"], 1.5)
        self.assertEqual(loiter[-1].parameters["psc_ne_vel_p"], 2.5)

        gains = moving_mass_gain_candidates(loiter[0].parameters)
        self.assertEqual(len(gains), 33)
        self.assertEqual(gains[0].parameters["moving_mass_assist_gain_m_per_Nm"], 0.0)
        self.assertEqual(gains[-1].parameters["moving_mass_assist_gain_m_per_Nm"], 0.08)

    def test_candidate_order_and_keys_are_deterministic_and_unique(self):
        first = rate_pd_coarse_candidates()
        second = rate_pd_coarse_candidates()
        self.assertEqual([candidate.key for candidate in first], [candidate.key for candidate in second])
        self.assertEqual(len({candidate.key for candidate in first}), len(first))
        self.assertEqual(len({candidate.candidate_id for candidate in first}), len(first))

    def test_loiter_grid_excludes_inactive_velocity_i_and_d(self):
        candidate = loiter_xy_candidates(
            {
                "atc_rat_pit_p": 0.035,
                "atc_rat_pit_i": 0.01,
                "atc_rat_pit_d": 0.002,
                "atc_ang_pit_p": 7.0,
            },
            quick=True,
        )[0]
        self.assertNotIn("psc_ne_vel_i", candidate.parameters)
        self.assertNotIn("psc_ne_vel_d", candidate.parameters)

    def test_tail_metric_helpers(self):
        self.assertEqual(zero_crossing_count([1.0, 0.0, -1.0, -0.5, 1.0], 1e-9), 2)
        self.assertEqual(direction_reversal_count([0.0, 1.0, 2.0, 1.0, 0.0, 1.0], 1e-9), 2)
        self.assertAlmostEqual(path_length([0.0, 3.0, 3.0], [0.0, 0.0, 4.0]), 7.0)

    def test_normalized_score_uses_documented_reference_scales(self):
        row = {spec.metric: spec.reference_scale for spec in SCORE_SPECS["rate_pd"]}
        score, components = normalized_score(row, "rate_pd")
        self.assertAlmostEqual(score, 1.0)
        self.assertTrue(all(value == 1.0 for value in json.loads(components).values()))

    def test_top_n_excludes_hard_rejections(self):
        rows = [
            {"candidate_key": "a", "normalized_score": 0.1, "rejected": True},
            {"candidate_key": "b", "normalized_score": 0.2, "rejected": False},
            {"candidate_key": "c", "normalized_score": 0.3, "rejected": False},
        ]
        self.assertEqual([row["candidate_key"] for row in top_candidates(rows, 1)], ["b"])

    def test_resume_store_rejects_duplicate_run_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScenarioResultStore(Path(tmp) / "rows.csv")
            row = {"run_key": "same"}
            store.add(row)
            with self.assertRaisesRegex(ValueError, "duplicate run key"):
                store.add(row)

    def test_hard_rejection_flags_ground_contact(self):
        rows = [
            {
                "time": index * 0.1,
                "min_body_z": -0.01,
                "theta": 0.0,
                "omega": 0.0,
                "rate_error": 0.0,
                "x_error": 0.0,
                "x": 0.0,
                "z": 0.0,
                "vx": 0.0,
            }
            for index in range(3)
        ]
        scenario = LoiterScenarioConfig(name="ground_contact", duration_s=0.3)
        result = LoiterRunResult("<synthetic>", scenario, rows, {}, False, "")
        metrics = compute_scenario_metrics("loiter_xy", result, tail_window_s=0.2)
        self.assertTrue(metrics["rejected"])
        self.assertIn("ground contact", metrics["rejection_reasons"])


class ControllerGridQuickEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.output_dir = root / "output"
        cls.profile_dir = root / "profiles"
        cls.result = run_workflow(
            WorkflowOptions(
                stage="all",
                output_dir=cls.output_dir,
                quick=True,
                resume=False,
                profile_output_dir=cls.profile_dir,
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_quick_end_to_end_outputs_and_unique_keys(self):
        self.assertTrue(self.result["workbook_path"].exists())
        self.assertTrue(self.result["markdown_path"].exists())
        self.assertEqual(
            len({row["run_key"] for row in self.result["scenario_rows"]}),
            len(self.result["scenario_rows"]),
        )
        for stage in ("rate_pd", "rate_i", "attitude_p", "loiter_xy", "moving_mass_gain"):
            self.assertGreater(len(self.result["stage_aggregates"][stage]), 0)

    def test_workbook_sheet_names_and_row_counts(self):
        workbook = load_workbook(self.result["workbook_path"], read_only=False, data_only=False)
        self.assertEqual(tuple(workbook.sheetnames), SHEET_NAMES)
        aggregates = self.result["stage_aggregates"]
        expected = {
            "01_rate_pd_all": len(aggregates["rate_pd"]),
            "02_rate_pd_top50": len(top_candidates(aggregates["rate_pd"], 50)),
            "03_rate_i_all": len(aggregates["rate_i"]),
            "04_attitude_p_all": len(aggregates["attitude_p"]),
            "05_loiter_xy_all": len(aggregates["loiter_xy"]),
            "06_loiter_xy_top50": len(top_candidates(aggregates["loiter_xy"], 50)),
            "07_moving_mass_gain_all": len(aggregates["moving_mass_gain"]),
            "08_scenario_summary": len(self.result["scenario_rows"]),
            "09_best_parameters": len(best_parameter_rows(aggregates)),
            "10_metadata": len(self.result["metadata"]),
        }
        for sheet_name, row_count in expected.items():
            self.assertEqual(workbook[sheet_name].max_row, row_count + 1, sheet_name)
            self.assertEqual(workbook[sheet_name].freeze_panes, "A2")
            self.assertIsNotNone(workbook[sheet_name].auto_filter.ref)

    def test_profiles_preserve_structure_and_are_loadable(self):
        vane_path = self.result["vane_profile_path"]
        moving_path = self.result["moving_mass_profile_path"]
        vane = json.loads(vane_path.read_text(encoding="utf-8"))
        moving = json.loads(moving_path.read_text(encoding="utf-8"))
        self.assertEqual(set(vane), {"rigid_body", "interactive", "controller"})
        self.assertEqual(set(moving), {"rigid_body", "interactive", "controller", "analysis"})
        self.assertIn("moving_mass_assist_gain_m_per_Nm", moving["analysis"])
        self.assertNotIn("psc_ne_vel_i", vane["controller"])
        self.assertNotIn("psc_ne_vel_d", vane["controller"])
        load_interactive_config(vane_path)
        load_interactive_config(moving_path)


if __name__ == "__main__":
    unittest.main()
