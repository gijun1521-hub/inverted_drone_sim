from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis.pitch_damping_retune import (
    BASELINE_GAINS,
    CANONICAL_PROFILES,
    CHATTER_THRESHOLDS,
    FIXED_CONTROLLER_VALUES,
    NEAR_EQUIVALENCE_ABSOLUTE_MARGIN,
    PHYSICAL_CONFIGURATION,
    PROVISIONAL_PROFILE,
    SOURCE_PROFILE,
    BaselineMismatchError,
    Candidate,
    ScenarioResultStore,
    WorkflowOptions,
    _profile_and_preservation_hashes,
    _profile_payload,
    asymmetry_fraction,
    baseline_mismatch_reasons,
    build_fingerprint,
    capture_gate_metrics,
    hard_gate_reasons,
    required_scenarios,
    run_workflow,
    selected_boundary_flags,
    select_near_equivalent,
    valid_near_equivalent_candidates,
    sha256_file,
    stage_candidates_from_rows,
    stage1_candidates,
    stage2_candidates,
    stage3a_candidates,
    stage3b_candidates,
    stage3c_candidates,
    transient_gate_metrics,
    validate_parameter_sources,
)
from analysis.headless_loiter import run_headless_loiter
from params import load_interactive_config


class PitchDampingStaticTests(unittest.TestCase):
    def test_authoritative_baseline_profile_loads_exact_controller(self):
        _rb, _ui, controller = load_interactive_config(SOURCE_PROFILE)
        validate_parameter_sources()
        for key, expected in {**FIXED_CONTROLLER_VALUES, **BASELINE_GAINS}.items():
            actual = getattr(controller, key)
            if isinstance(expected, bool):
                self.assertIs(actual, expected)
            else:
                self.assertAlmostEqual(float(actual), float(expected), places=12)

    def test_only_requested_pitch_fields_vary(self):
        candidates = stage1_candidates()
        self.assertEqual(len(candidates), 72)
        self.assertEqual({candidate.angle_p for candidate in candidates}, {10.0})
        for candidate in candidates:
            overrides = candidate.controller_overrides()
            self.assertEqual(overrides["atc_rat_pit_i"], 0.0)
            self.assertEqual(overrides["psc_ne_pos_p"], 0.55)
            self.assertEqual(overrides["psc_ne_vel_p"], 0.70)
            self.assertEqual(overrides["loit_brk_delay_s"], 0.50)
            self.assertEqual(overrides["loit_brk_acc_mss"], 1.00)
            self.assertEqual(overrides["loit_brk_jerk_msss"], 3.00)

    def test_capture_and_shaper_values_remain_fixed(self):
        overrides = Candidate("test", 0.06, 0.01, 11.0).controller_overrides()
        self.assertEqual(overrides["loit_capture_vx_threshold_ms"], 0.08)
        self.assertEqual(overrides["loit_capture_desired_vx_threshold_ms"], 0.02)
        self.assertIs(overrides["loit_capture_persistent"], True)
        self.assertIs(overrides["loit_shaper_clamp_target"], True)
        self.assertIs(overrides["loit_capture_without_jump"], True)

    def test_physical_moving_mass_configuration_is_present_and_centered(self):
        moving_mass = PHYSICAL_CONFIGURATION["moving_mass"]
        self.assertEqual(PHYSICAL_CONFIGURATION["m"], 2.0)
        self.assertEqual(moving_mass["mass_kg"], 0.5)
        self.assertEqual(PHYSICAL_CONFIGURATION["m"] - moving_mass["mass_kg"], 1.5)
        self.assertTrue(moving_mass["enabled"])
        self.assertEqual(moving_mass["initial_offset_m"], 0.0)
        self.assertEqual(moving_mass["moving_mass_body_up_offset_m"], 0.12)
        self.assertTrue(moving_mass["use_total_com_geometry"])
        self.assertFalse(moving_mass["use_legacy_gravity_offset_moment"])
        for scenario in required_scenarios():
            self.assertEqual(scenario.config.moving_mass_target_m, 0.0)
            self.assertEqual(scenario.config.moving_mass_assist_gain_m_per_Nm, 0.0)

    def test_mirrored_scenario_signs(self):
        scenarios = {scenario.key: scenario for scenario in required_scenarios()}
        self.assertEqual(scenarios["loiter_positive_disturbance"].config.disturbance_force_x, 8.0)
        self.assertEqual(scenarios["loiter_negative_disturbance"].config.disturbance_force_x, -8.0)
        self.assertEqual(scenarios["forward_1m"].config.target_step_x, 1.0)
        self.assertEqual(scenarios["backward_1m"].config.target_step_x, -1.0)
        self.assertEqual(scenarios["pitch_positive_recovery"].config.initial_theta_deg, 3.0)
        self.assertEqual(scenarios["pitch_negative_recovery"].config.initial_theta_deg, -3.0)
        self.assertEqual(scenarios["forward_1m"].direction, 1)
        self.assertEqual(scenarios["backward_1m"].direction, -1)

    def test_required_grid_counts(self):
        stage1 = stage1_candidates()
        self.assertEqual(len(stage1), 72)
        self.assertEqual(len(stage2_candidates(stage1[:3])), 21)
        center = Candidate("center", 0.070, 0.008, 10.0)
        self.assertEqual(len(stage3a_candidates(center)), 81)
        self.assertEqual(len(stage3b_candidates(center)), 9)
        self.assertEqual(len(stage3c_candidates(center)), 75)

    def test_quick_mode_has_required_smoke_scenarios_and_two_pd_candidates(self):
        self.assertEqual(
            {scenario.key for scenario in required_scenarios(True)},
            {"forward_1m", "pitch_positive_recovery", "stick_release"},
        )
        self.assertGreaterEqual(len(stage1_candidates(True)), 2)

    def test_symmetry_metric(self):
        self.assertEqual(asymmetry_fraction(2.0, 2.0), 0.0)
        self.assertAlmostEqual(asymmetry_fraction(2.0, 1.0), 2.0 / 3.0)

    def test_boundary_flags(self):
        candidates = [
            Candidate("test", 0.06, 0.008, 9.0),
            Candidate("test", 0.07, 0.009, 10.0),
            Candidate("test", 0.08, 0.010, 11.0),
        ]
        flags = selected_boundary_flags(
            {"rate_p": 0.08, "rate_d": 0.009, "angle_p": 9.0}, candidates
        )
        self.assertTrue(flags["rate_p_at_max"])
        self.assertTrue(flags["angle_p_at_min"])
        self.assertFalse(flags["rate_d_at_min"])
        self.assertFalse(flags["rate_d_at_max"])

    def test_extension_selection_keeps_prior_stage_winner_in_the_ranking_pool(self):
        prior = {
            "candidate_key": "prior",
            "rejected": False,
            "final_score": 0.40,
            "mean_vane_command_rms_deg": 0.7,
            "rate_d": 0.02,
            "mean_vane_total_variation_deg": 20.0,
            "worst_asymmetry_fraction": 0.0,
        }
        extension = {
            "candidate_key": "extension",
            "rejected": False,
            "final_score": 0.60,
            "mean_vane_command_rms_deg": 0.6,
            "rate_d": 0.01,
            "mean_vane_total_variation_deg": 15.0,
            "worst_asymmetry_fraction": 0.0,
        }
        self.assertEqual(
            select_near_equivalent([prior, extension])["candidate_key"], "prior"
        )

    def test_predeclared_absolute_near_equivalence_rule_selects_lower_effort(self):
        self.assertEqual(NEAR_EQUIVALENCE_ABSOLUTE_MARGIN, 0.010000)
        raw_best = {
            "candidate_key": "raw_best",
            "rejected": False,
            "final_score": 0.39690256845961375,
            "mean_vane_command_rms_deg": 0.7123899409223641,
            "rate_d": 0.0195,
            "mean_vane_total_variation_deg": 33.536981401396254,
            "worst_asymmetry_fraction": 0.0,
        }
        lower_effort = {
            "candidate_key": "lower_effort",
            "rejected": False,
            "final_score": 0.406150435755185,
            "mean_vane_command_rms_deg": 0.6941952163735429,
            "rate_d": 0.0185,
            "mean_vane_total_variation_deg": 32.10237017235434,
            "worst_asymmetry_fraction": 0.0,
        }
        outside = {
            **lower_effort,
            "candidate_key": "outside",
            "final_score": raw_best["final_score"] + 0.010001,
            "mean_vane_command_rms_deg": 0.1,
        }
        near = valid_near_equivalent_candidates([raw_best, lower_effort, outside])
        self.assertEqual({row["candidate_key"] for row in near}, {"raw_best", "lower_effort"})
        self.assertEqual(
            select_near_equivalent([raw_best, lower_effort, outside])["candidate_key"],
            "lower_effort",
        )

    def test_resume_boundary_ranges_include_cached_extension_candidates(self):
        cached = stage_candidates_from_rows(
            [
                {
                    "stage": "stage1_rate_pd",
                    "rate_p": "0.100",
                    "rate_d": "0.020",
                    "angle_p": "10.0",
                }
            ],
            "stage1_rate_pd",
        )
        flags = selected_boundary_flags(
            {"rate_p": 0.100, "rate_d": 0.020, "angle_p": 10.0}, cached
        )
        self.assertTrue(flags["rate_p_at_max"])
        self.assertTrue(flags["rate_d_at_max"])

    def test_fingerprint_is_deterministic_and_includes_required_inputs(self):
        first_payload, first_digest = build_fingerprint(True)
        second_payload, second_digest = build_fingerprint(True)
        self.assertEqual(first_payload, second_payload)
        self.assertEqual(first_digest, second_digest)
        self.assertEqual(first_payload["moving_mass_assist_gain_m_per_Nm"], 0.0)
        self.assertIn("source_hashes", first_payload)
        self.assertIn("scenario_fingerprint", first_payload)
        self.assertIn("hard_gate_thresholds", first_payload)
        self.assertIn("chatter_thresholds", first_payload)
        self.assertEqual(
            first_payload["search_ranges"]["boundary_extension_policy"]["maximum_rounds_per_stage"],
            12,
        )

    def test_resume_and_stale_cache_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scenario_results.csv"
            store = ScenarioResultStore(path, "fingerprint-a", resume=False)
            store.add(
                {
                    "run_key": "one",
                    "workflow_fingerprint": "fingerprint-a",
                    "stage": "test",
                }
            )
            store.save()
            resumed = ScenarioResultStore(path, "fingerprint-a", resume=True)
            self.assertIsNotNone(resumed.get("one"))
            with self.assertRaisesRegex(ValueError, "stale scenario cache"):
                ScenarioResultStore(path, "fingerprint-b", resume=True)

    def test_provisional_profile_generation_does_not_change_canonical_profiles(self):
        before = {path: sha256_file(path) for path in CANONICAL_PROFILES}
        payload, digest = build_fingerprint(True)
        profile = _profile_payload(
            Candidate("selected", 0.07125, 0.009, 10.25),
            payload,
            digest,
            {"rate_p_at_min": False, "rate_p_at_max": False},
            ["same", "same"],
            ["2D analytical model only"],
        )
        self.assertEqual(profile["analysis"]["profile_status"], "provisional")
        self.assertEqual(profile["controller"]["atc_rat_pit_i"], 0.0)
        self.assertEqual(profile["analysis"]["moving_mass_assist_gain_m_per_Nm"], 0.0)
        self.assertEqual(before, {path: sha256_file(path) for path in CANONICAL_PROFILES})
        self.assertNotIn(str(Path.home()), json.dumps(profile))


class PitchDampingDetectorTests(unittest.TestCase):
    @staticmethod
    def _forward_definition():
        return next(scenario for scenario in required_scenarios() if scenario.key == "forward_1m")

    def test_premature_pause_detector(self):
        rows = []
        for index in range(20):
            time_s = 0.5 + index * 0.05
            vx = 0.2 if index < 4 else (0.0 if index < 9 else 0.15)
            rows.append({"time": time_s, "vx": vx, "x_error": 0.5})
        metrics = transient_gate_metrics(self._forward_definition(), rows)
        self.assertTrue(metrics["premature_pause"])
        self.assertTrue(metrics["second_acceleration_lobe_after_full_pause"])

    def test_early_reversal_detector(self):
        rows = [
            {"time": 0.50, "vx": 0.12, "x_error": 0.8},
            {"time": 0.55, "vx": 0.05, "x_error": 0.7},
            {"time": 0.60, "vx": -0.01, "x_error": 0.6},
            {"time": 0.65, "vx": 0.1, "x_error": 0.5},
        ]
        metrics = transient_gate_metrics(self._forward_definition(), rows)
        self.assertTrue(metrics["early_velocity_reversal"])

    def test_capture_discontinuity_detector(self):
        definition = next(
            scenario for scenario in required_scenarios() if scenario.key == "stick_release"
        )
        rows = [
            {"time": 2.20, "target_capture_event": 0, "target_capture_count": 0, "target_x": 1.0, "shaped_desired_vx": 0.0},
            {"time": 2.25, "target_capture_event": 1, "target_capture_count": 1, "target_x": 1.03, "shaped_desired_vx": 0.0},
        ]
        metrics = capture_gate_metrics(definition, rows)
        self.assertEqual(metrics["target_capture_count"], 1)
        self.assertTrue(metrics["capture_discontinuity"])

    def test_hard_gate_chatter_rejection(self):
        definition = next(
            scenario for scenario in required_scenarios() if scenario.key == "pitch_positive_recovery"
        )
        metrics = {
            "finite": True,
            "crash": False,
            "ground_contact": False,
            "peak_abs_pitch_deg": 3.0,
            "premature_pause": False,
            "early_velocity_reversal": False,
            "second_acceleration_lobe_after_full_pause": False,
            "capture_discontinuity": False,
            "shaped_velocity_sign_reversal_after_release": False,
            "vane_saturation_percent": 0.0,
            "servo_rate_saturation_percent": 0.0,
            "mixer_saturation_percent": 0.0,
            "meaningful_vane_sign_change_count": CHATTER_THRESHOLDS["max_meaningful_sign_changes"] + 1,
            "vane_total_variation_per_second_deg_s": 1.0,
            "tail_high_frequency_vane_energy_deg2": 0.0,
            "moving_mass_assist_gain_m_per_Nm": 0.0,
            "moving_mass_max_abs_offset_m": 0.0,
            "moving_mass_max_abs_target_m": 0.0,
            "effective_atc_rat_pit_i": 0.0,
            "effective_psc_ne_pos_p": 0.55,
            "effective_psc_ne_vel_p": 0.70,
            "total_mass_kg": 2.0,
            "physical_moving_mass_kg": 0.5,
            "moving_mass_enabled": True,
            "total_com_geometry_active": True,
            "legacy_gravity_offset_active": False,
        }
        self.assertIn("vane_chatter", hard_gate_reasons(definition, metrics))


class PitchDampingRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = required_scenarios(True)
        cls.results = {}
        cls.metric_rows = []
        candidate = Candidate("stage0_baseline", 0.070, 0.008, 10.0)
        from analysis.pitch_damping_retune import compute_metrics

        for definition in cls.scenarios:
            result = run_headless_loiter(
                SOURCE_PROFILE,
                definition.config,
                rb_overrides=PHYSICAL_CONFIGURATION,
                controller_overrides=candidate.controller_overrides(),
            )
            cls.results[definition.key] = result
            cls.metric_rows.append(compute_metrics(definition, result, quick=True))

    def test_moving_mass_actual_and_target_remain_exactly_zero(self):
        for result in self.results.values():
            self.assertTrue(all(float(row["moving_mass_offset_m"]) == 0.0 for row in result.rows))
            self.assertTrue(all(float(row["moving_mass_target_m"]) == 0.0 for row in result.rows))

    def test_capture_count_resets_between_simulations(self):
        definition = next(s for s in self.scenarios if s.key == "stick_release")
        first = self.results["stick_release"]
        second = run_headless_loiter(
            SOURCE_PROFILE,
            definition.config,
            rb_overrides=PHYSICAL_CONFIGURATION,
            controller_overrides=Candidate("test", 0.070, 0.008, 10.0).controller_overrides(),
        )
        self.assertEqual(int(first.rows[-1]["target_capture_count"]), 1)
        self.assertEqual(int(second.rows[-1]["target_capture_count"]), 1)
        self.assertEqual(int(second.rows[0]["target_capture_count"]), 0)

    def test_required_2kg_baseline_mismatch_is_visible(self):
        reasons = baseline_mismatch_reasons(self.metric_rows, self.scenarios)
        self.assertIn("forward_1m:early_velocity_reversal", reasons)

    def test_default_quick_workflow_stops_before_candidate_search(self):
        before = _profile_and_preservation_hashes()
        profile_before = PROVISIONAL_PROFILE.read_bytes() if PROVISIONAL_PROFILE.exists() else None
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pitch_damping"
            with self.assertRaises(BaselineMismatchError):
                run_workflow(
                    WorkflowOptions(
                        output_dir=output,
                        quick=True,
                        resume=False,
                        allow_baseline_mismatch=False,
                        stage="all",
                    )
                )
            stop = json.loads((output / "baseline_mismatch_stop.json").read_text())
            self.assertFalse(stop["search_started"])
            self.assertFalse((output / "candidate_results.csv").exists())
        self.assertEqual(before, _profile_and_preservation_hashes())
        profile_after = PROVISIONAL_PROFILE.read_bytes() if PROVISIONAL_PROFILE.exists() else None
        self.assertEqual(profile_before, profile_after)

    def test_authorized_quick_smoke_reports_rejections_without_selecting(self):
        profile_before = PROVISIONAL_PROFILE.read_bytes() if PROVISIONAL_PROFILE.exists() else None
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pitch_damping_quick"
            first = run_workflow(
                WorkflowOptions(
                    output_dir=output,
                    quick=True,
                    resume=False,
                    allow_baseline_mismatch=True,
                    stage="all",
                )
            )
            self.assertEqual(first["mode"], "quick")
            self.assertEqual(first["metadata"]["candidate_count"], 2)
            self.assertEqual(first["metadata"]["valid_candidate_count"], 0)
            self.assertTrue((output / "candidate_results.csv").is_file())
            second = run_workflow(
                WorkflowOptions(
                    output_dir=output,
                    quick=True,
                    resume=True,
                    allow_baseline_mismatch=True,
                    stage="all",
                )
            )
            self.assertEqual(
                first["metadata"]["scenario_run_count"],
                second["metadata"]["scenario_run_count"],
            )
        profile_after = PROVISIONAL_PROFILE.read_bytes() if PROVISIONAL_PROFILE.exists() else None
        self.assertEqual(profile_before, profile_after)


if __name__ == "__main__":
    unittest.main()
