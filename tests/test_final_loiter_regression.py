from __future__ import annotations
import unittest
from dataclasses import replace
from analysis.final_loiter_regression import (
    VARIANTS,
    _case_row,
    _scenarios,
    _second_lobe_in_release_window,
    build_summary_markdown,
)

class FinalLoiterRegressionTests(unittest.TestCase):
    def test_variants_have_distinct_executed_gains(self):
        scenario=next(_scenarios())[2]
        executed=[replace(scenario,moving_mass_enabled=True,moving_mass_target_m=0.0,moving_mass_assist_gain_m_per_Nm=g).moving_mass_assist_gain_m_per_Nm for _,g in VARIANTS]
        self.assertEqual(executed,[0.0,0.0415])
        self.assertNotEqual(*executed)

    def test_reversal_and_interrupted_release_are_declared_patterns(self):
        keys={item[0] for item in _scenarios()}
        self.assertIn("commanded_reversal",keys)
        self.assertIn("repeated_pulse_release",keys)

    @staticmethod
    def _trace(values):
        return [{"time": index * 0.05, "vx": value} for index, value in enumerate(values)]

    @staticmethod
    def _summary_row(pattern="ordinary", direction="positive", count=1):
        return {
            "pattern": pattern,
            "direction": direction,
            "variant": "vane_only",
            "gain": 0.0,
            "passed": True,
            "failures": "",
            "final_cumulative_capture_count": count,
            "vane_command_rms_deg": 0.1,
            "moving_mass_max_abs_offset_m": 0.0,
            "moving_mass_max_abs_velocity_m_s": 0.0,
            "moving_mass_max_abs_acceleration_m_s2": 0.0,
        }

    @staticmethod
    def _event_rows(pattern="ordinary", direction="positive", count=1, increments=(1,)):
        return [
            {
                "pattern": pattern,
                "direction": direction,
                "variant": "vane_only",
                "final_cumulative_capture_count": count,
                "capture_increments": increment,
            }
            for increment in increments
        ]

    def test_mirrored_same_direction_traces_have_identical_second_lobe_decisions(self):
        positive = [0.15, 0.20, 0.12, 0.01, 0.01, 0.01, 0.01, 0.12]
        negative = [-value for value in positive]
        self.assertTrue(_second_lobe_in_release_window(self._trace(positive), 0.0, 1.0, 0.6))
        self.assertEqual(
            _second_lobe_in_release_window(self._trace(positive), 0.0, 1.0, 0.6),
            _second_lobe_in_release_window(self._trace(negative), 0.0, 1.0, -0.6),
        )

    def test_opposite_direction_correction_after_negative_travel_is_not_second_lobe(self):
        negative_travel_then_positive_correction = [-0.15, -0.20, -0.12, -0.01, -0.01, -0.01, -0.01, 0.12]
        self.assertFalse(
            _second_lobe_in_release_window(
                self._trace(negative_travel_then_positive_correction), 0.0, 1.0, -0.6
            )
        )

    def test_new_intentional_command_is_excluded_from_previous_release_window(self):
        trace = self._trace([0.15, 0.20, 0.12, 0.01, 0.01, 0.01, 0.01, 0.12])
        self.assertFalse(_second_lobe_in_release_window(trace, 0.0, 0.30, 0.6))

    def test_case_direction_label_cannot_be_overwritten_by_metrics(self):
        positive = _case_row("case", 1, "vane_only", 0.0, {"direction": -1}, [])
        negative = _case_row("case", -1, "vane_only", 0.0, {"direction": 1}, [])
        self.assertEqual((positive["direction"], negative["direction"]), ("positive", "negative"))
        self.assertEqual((positive["metric_direction"], negative["metric_direction"]), (-1, 1))

    def test_summary_uses_final_cumulative_capture_count(self):
        row = self._summary_row(count=1)
        row["target_capture_count"] = 0
        summary = build_summary_markdown(
            [row], self._event_rows(count=1)
        )
        self.assertIn("| 1 | pass |", summary)
        self.assertNotIn("| 0 | pass |", summary)

    def test_summary_values_agree_with_event_audit(self):
        cases = [
            ("ordinary", 1, (1,)),
            ("move_stop_move", 2, (1, 1)),
            ("repeated_pulse_release", 1, (0, 0, 1)),
        ]
        for pattern, count, increments in cases:
            with self.subTest(pattern=pattern):
                summary = build_summary_markdown(
                    [self._summary_row(pattern=pattern, count=count)],
                    self._event_rows(pattern=pattern, count=count, increments=increments),
                )
                self.assertIn(f"| {count} | pass |", summary)
        with self.assertRaises(ValueError):
            build_summary_markdown([self._summary_row(count=2)], self._event_rows(count=1))

    def test_missing_required_capture_metric_fails(self):
        row = self._summary_row()
        del row["final_cumulative_capture_count"]
        with self.assertRaisesRegex(KeyError, "final_cumulative_capture_count"):
            build_summary_markdown([row], self._event_rows())

if __name__=="__main__": unittest.main()
