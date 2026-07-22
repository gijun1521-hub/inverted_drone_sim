from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.variant_controller_optimizer import (
    DEFAULT_SPEC,
    CandidateCache,
    VariantCandidate,
    _boundary_axes,
    _write_manifest,
    boundary_extension_candidates,
    candidate_rank_key,
    coarse_candidates,
    detect_settling_time,
    evaluate_candidates,
    evaluate_scenario,
    load_search_spec,
    overshoot_band_distance,
    pareto_front,
    rank_candidates,
    screening_scenarios,
    target_crossing_count,
    verify_manifest,
)


class VariantControllerOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_search_spec(DEFAULT_SPEC)

    @staticmethod
    def candidate(variant="vane_only", **overrides):
        values = {
            "variant": variant,
            "stage": "test",
            "atc_rat_pit_p": 0.08,
            "atc_rat_pit_d": 0.015,
            "atc_ang_pit_p": 20.0,
            "psc_ne_pos_p": 0.6,
            "psc_ne_vel_p": 0.75,
            "moving_mass_assist_gain_m_per_Nm": 0.0 if variant == "vane_only" else 0.05,
        }
        values.update(overrides)
        return VariantCandidate(**values)

    @staticmethod
    def row(key, *, overshoot_distance=0.0, settle=2.0, rise=1.0, effort=1.0, robustness=1.0):
        return {
            "candidate_key": key,
            "all_hard_gates_pass": True,
            "all_step_scenarios_settled": True,
            "worst_settling_time_s": settle,
            "worst_rise_time_s": rise,
            "mean_overshoot_band_distance": overshoot_distance,
            "worst_final_abs_position_error_m": 0.01,
            "actuator_effort_index": effort,
            "robustness_index": robustness,
        }

    def test_overshoot_band_scoring(self):
        preferred = (0.02, 0.05)
        self.assertAlmostEqual(overshoot_band_distance(0.00, preferred), 0.02)
        self.assertEqual(overshoot_band_distance(0.03, preferred), 0.0)
        self.assertAlmostEqual(overshoot_band_distance(0.08, preferred), 0.03)

    def test_zero_overshoot_does_not_automatically_win(self):
        zero = self.row("zero", overshoot_distance=0.02)
        preferred = self.row("preferred", overshoot_distance=0.0)
        self.assertEqual(rank_candidates([zero, preferred])[0]["candidate_key"], "preferred")

    def test_settling_time_detection(self):
        t = np.arange(0.0, 2.01, 0.05)
        error = np.where(t < 0.8, 0.1, 0.01)
        velocity = np.where(t < 0.8, 0.1, 0.01)
        settling, duration, settled = detect_settling_time(
            t,
            error,
            velocity,
            event_time_s=0.0,
            position_band_m=0.025,
            velocity_band_m_s=0.03,
            required_duration_s=0.75,
        )
        self.assertTrue(settled)
        self.assertAlmostEqual(settling, 0.8, places=8)
        self.assertGreaterEqual(duration, 0.75)

    def test_continuous_settling_rejects_late_escape(self):
        t = np.arange(0.0, 2.01, 0.05)
        error = np.where((t >= 0.5) & (t < 1.4), 0.01, 0.1)
        velocity = np.where((t >= 0.5) & (t < 1.4), 0.01, 0.1)
        settling, _, settled = detect_settling_time(
            t,
            error,
            velocity,
            event_time_s=0.0,
            position_band_m=0.025,
            velocity_band_m_s=0.03,
            required_duration_s=0.75,
        )
        self.assertFalse(settled)
        self.assertIsNone(settling)

    def test_target_crossing_count_ignores_deadband_noise(self):
        error = [-1.0, -0.1, -0.001, 0.001, 0.08, 0.001, -0.001, -0.03]
        self.assertEqual(target_crossing_count(error, deadband=0.002), 2)

    def test_lexicographic_ranking_prioritizes_settling_then_rise(self):
        slower_rise = self.row("slower-rise", settle=1.0, rise=1.2)
        faster_settle = self.row("faster-settle", settle=0.9, rise=2.0)
        self.assertLess(candidate_rank_key(faster_settle), candidate_rank_key(slower_rise))

    def test_pareto_front_keeps_tradeoffs_and_drops_dominated(self):
        fast = self.row("fast", settle=0.8, effort=2.0)
        efficient = self.row("efficient", settle=1.0, effort=1.0)
        dominated = self.row("dominated", settle=1.2, effort=2.5, robustness=1.5)
        keys = {row["candidate_key"] for row in pareto_front([fast, efficient, dominated])}
        self.assertEqual(keys, {"fast", "efficient"})

    def test_resume_cache_round_trip_and_fingerprint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            cache = CandidateCache(path, "fingerprint-a", resume=True)
            cache.record({"candidate_key": "one", "value": 1})
            resumed = CandidateCache(path, "fingerprint-a", resume=True)
            self.assertEqual(resumed.get("one")["value"], 1)
            invalidated = CandidateCache(path, "fingerprint-b", resume=True)
            self.assertIsNone(invalidated.get("one"))
            json.loads(path.read_text(encoding="utf-8"))

    def test_coarse_search_includes_old_and_current_controllers(self):
        rows = coarse_candidates("vane_only", self.spec, quick=True)
        parameters = [row.parameters for row in rows]
        self.assertTrue(any(row["atc_rat_pit_p"] == 0.070 and row["atc_rat_pit_d"] == 0.008 for row in parameters))
        self.assertTrue(any(row["atc_rat_pit_p"] == 0.09375 and row["atc_rat_pit_d"] == 0.021 for row in parameters))

    def test_two_variants_can_have_separate_pid_values(self):
        vane = self.candidate(atc_rat_pit_p=0.08)
        assist = self.candidate("moving_mass_assist", atc_rat_pit_p=0.11)
        self.assertNotEqual(vane.atc_rat_pit_p, assist.atc_rat_pit_p)
        self.assertNotEqual(vane.key, assist.key)

    def test_boundary_extension_expands_selected_upper_axis(self):
        best = self.candidate(atc_rat_pit_p=0.14).parameters
        best.update({"variant": "vane_only", "candidate_key": "boundary"})
        ranges = {name: list(bounds) for name, bounds in self.spec["ranges"].items()}
        axes = _boundary_axes(best, "vane_only", ranges)
        candidates, diagnostics = boundary_extension_candidates(
            "vane_only", best, ranges, axes, self.spec, 1, quick=True
        )
        self.assertGreater(ranges["atc_rat_pit_p"][1], 0.14)
        self.assertTrue(candidates)
        self.assertTrue(any(row["axis"] == "atc_rat_pit_p" and row["action"] == "extended" for row in diagnostics))

    def test_manifest_verification_excludes_its_own_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.txt").write_text("payload", encoding="utf-8")
            (root / "validation").mkdir()
            (root / "validation" / "artifact_hash_verification.json").write_text("{}", encoding="utf-8")
            _write_manifest(root)
            verification = verify_manifest(root)
            self.assertTrue(verification["passed"])
            manifest = json.loads((root / "sha256_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["artifacts"]), {"payload.txt"})

    def test_vane_only_exact_moving_mass_lock(self):
        candidate = self.candidate()
        definition = screening_scenarios(self.spec)[0]
        definition = replace(definition, config=replace(definition.config, duration_s=1.20))
        _, result = evaluate_scenario(candidate, definition, self.spec, keep_result=True)
        self.assertIsNotNone(result)
        for key in ("moving_mass_target_m", "moving_mass_offset_m", "moving_mass_velocity_m_s"):
            self.assertEqual(max(abs(float(row[key])) for row in result.rows), 0.0)
        velocity = np.asarray([float(row["moving_mass_velocity_m_s"]) for row in result.rows])
        self.assertEqual(float(np.max(np.abs(np.diff(velocity, prepend=0.0)))), 0.0)

    def test_worker_count_deterministic_equivalence(self):
        spec = json.loads(json.dumps(self.spec))
        spec["search"]["fast_screen_duration_s"] = 1.20
        candidate = self.candidate()
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            serial_cache = CandidateCache(Path(left) / "cache.json", "test", resume=False)
            parallel_cache = CandidateCache(Path(right) / "cache.json", "test", resume=False)
            serial = evaluate_candidates([candidate], spec, "test", serial_cache, workers=1)
            parallel = evaluate_candidates([candidate], spec, "test", parallel_cache, workers=2)
        self.assertEqual(serial, parallel)


if __name__ == "__main__":
    unittest.main()
