import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.headless_loiter import run_headless_loiter
from analysis.loiter_transient_diagnosis import (
    CURRENT_PROFILE,
    PROVISIONAL_PROFILE,
    _jsonable,
    absolute_target_scenario,
    compute_transient_metrics,
    detect_premature_pause,
    second_acceleration_lobe_after_full_pause,
    stick_release_scenario,
    target_capture_events,
    velocity_sign_changes_before_target,
)


class LoiterTransientDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.absolute_current = run_headless_loiter(CURRENT_PROFILE, absolute_target_scenario())
        cls.absolute_selected = run_headless_loiter(PROVISIONAL_PROFILE, absolute_target_scenario())
        cls.stick_selected = run_headless_loiter(PROVISIONAL_PROFILE, stick_release_scenario())

    def test_deterministic_diagnostic_runs(self):
        repeat = run_headless_loiter(PROVISIONAL_PROFILE, absolute_target_scenario())

        self.assertEqual(self.absolute_selected.rows, repeat.rows)

    def test_pause_detector_finds_current_first_pause(self):
        pause = detect_premature_pause(self.absolute_current.rows)

        self.assertIsNotNone(pause)
        self.assertAlmostEqual(float(pause["start_time_s"]), 2.825, places=3)
        self.assertGreater(float(pause["position_error_m"]), 0.10)
        self.assertGreaterEqual(float(pause["duration_s"]), 0.15)

    def test_absolute_diagnostic_logs_zero_to_one_target_step(self):
        before = [row for row in self.absolute_current.rows if float(row["time"]) < 0.5]
        after = [row for row in self.absolute_current.rows if float(row["time"]) >= 0.5]

        self.assertTrue(before)
        self.assertTrue(after)
        self.assertTrue(all(float(row["target_x"]) == 0.0 for row in before))
        self.assertTrue(any(float(row["target_x"]) == 1.0 for row in after))
        self.assertTrue(any(int(row["target_step_event"]) for row in after))

    def test_velocity_sign_change_detector(self):
        rows = [
            {"time": 0.0, "x": 0.0, "vx": 0.00},
            {"time": 0.1, "x": 0.1, "vx": 0.11},
            {"time": 0.2, "x": 0.2, "vx": 0.05},
            {"time": 0.3, "x": 0.2, "vx": -0.02},
            {"time": 0.4, "x": 0.3, "vx": 0.04},
        ]

        changes = velocity_sign_changes_before_target(rows)

        self.assertEqual(len(changes), 2)
        self.assertAlmostEqual(changes[0]["time_s"], 0.2)

    def test_target_capture_event_logging_is_single_and_monotonic(self):
        events = target_capture_events(self.stick_selected.rows)
        counts = [int(row["target_capture_count"]) for row in self.stick_selected.rows]

        self.assertEqual(len(events), 1)
        self.assertEqual(max(counts), 1)
        self.assertTrue(any(int(row["target_capture_event"]) for row in self.stick_selected.rows))
        self.assertEqual(counts, sorted(counts))

    def test_identical_repeated_metric_output(self):
        repeat = run_headless_loiter(PROVISIONAL_PROFILE, stick_release_scenario())
        first = json.dumps(_jsonable(compute_transient_metrics(self.stick_selected)), sort_keys=True)
        second = json.dumps(_jsonable(compute_transient_metrics(repeat)), sort_keys=True)

        self.assertEqual(first, second)

    def test_selected_absolute_candidate_has_no_premature_pause_or_early_reversal(self):
        metrics = compute_transient_metrics(self.absolute_selected)

        self.assertFalse(metrics["premature_pause"])
        self.assertEqual(metrics["vx_sign_change_count_before_0p98m"], 0)
        self.assertFalse(self.absolute_selected.crashed)

    def test_stick_release_has_no_second_lobe_after_full_pause(self):
        metrics = compute_transient_metrics(self.stick_selected)

        self.assertFalse(second_acceleration_lobe_after_full_pause(self.stick_selected.rows))
        self.assertFalse(metrics["second_acceleration_lobe_after_full_pause"])
        self.assertEqual(metrics["target_capture_count"], 1)
        self.assertEqual(metrics["target_discontinuity_count"], 0)

    def test_second_lobe_detector_synthetic_positive_case(self):
        rows = []
        for index in range(50):
            time = index * 0.05
            if time < 0.5:
                vx = 0.2
            elif time < 0.7:
                vx = 0.0
            else:
                vx = 0.15
            rows.append({"time": time, "vx": vx})

        self.assertTrue(
            second_acceleration_lobe_after_full_pause(rows, release_time_s=0.4)
        )


if __name__ == "__main__":
    unittest.main()
