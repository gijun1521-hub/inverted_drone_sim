from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from interactive_logging import INTERACTIVE_FIELDS
from interactive_sim import ControlMode, InteractiveApp, configure_actuator_lab
from params import load_interactive_config
from replay_interactive import load_csv


PROFILE_PATH = "params/moving_mass_prototype_2kg.json"


class ActuatorLabTests(unittest.TestCase):
    def make_app(self, *, ui_cfg=None, rb_cfg=None) -> InteractiveApp:
        loaded_rb, loaded_ui, controller_cfg = load_interactive_config(PROFILE_PATH)
        effective_rb = configure_actuator_lab(rb_cfg or loaded_rb)
        return InteractiveApp(
            effective_rb,
            ui_cfg or loaded_ui,
            controller_cfg,
            actuator_lab_enabled=True,
        )

    def test_lab_configuration_is_a_validated_copy_and_preserves_prototype_values(self):
        rb_cfg, _ui_cfg, _controller_cfg = load_interactive_config(PROFILE_PATH)
        effective = configure_actuator_lab(rb_cfg)

        self.assertIsNot(effective, rb_cfg)
        self.assertIsNot(effective.moving_mass, rb_cfg.moving_mass)
        self.assertFalse(rb_cfg.moving_mass.enabled)
        self.assertFalse(rb_cfg.moving_mass.use_total_com_geometry)
        self.assertTrue(rb_cfg.moving_mass.use_legacy_gravity_offset_moment)
        self.assertTrue(effective.moving_mass.enabled)
        self.assertTrue(effective.moving_mass.use_total_com_geometry)
        self.assertFalse(effective.moving_mass.use_legacy_gravity_offset_moment)
        self.assertEqual(effective.m, 2.0)
        self.assertEqual(effective.moving_mass.mass_kg, 0.5)
        self.assertEqual(effective.moving_mass.max_offset_m, 0.05)
        self.assertEqual(effective.moving_mass.moving_mass_body_up_offset_m, 0.12)
        self.assertEqual(effective.moving_mass.max_rate_m_s, 0.20)
        self.assertEqual(effective.moving_mass.max_accel_m_s2, 1.0)
        self.assertEqual(effective.H, 0.50)
        self.assertEqual(effective.l, 0.25)
        effective.validate()

    def test_lab_starts_centered_in_eleven_state_form_near_hover(self):
        app = self.make_app()

        self.assertEqual(app.mode, ControlMode.ACTUATOR_LAB)
        self.assertEqual(app.state.shape, (11,))
        self.assertEqual(float(app.state[8]), 0.0)
        self.assertEqual(float(app.state[9]), 0.0)
        self.assertEqual(float(app.state[10]), 0.0)
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)
        self.assertEqual(app.commands.direct_vane, 0.0)
        self.assertEqual(float(app.state[6]), app.rb_cfg.hover_thrust)

    def test_fine_and_coarse_commands_have_exact_steps(self):
        app = self.make_app()

        self.assertAlmostEqual(app.adjust_actuator_lab_mass_target(1.0), 0.001)
        self.assertAlmostEqual(app.adjust_actuator_lab_mass_target(1.0, coarse=True), 0.006)
        self.assertAlmostEqual(np.rad2deg(app.adjust_actuator_lab_vane_command(1.0)), 0.5)
        self.assertAlmostEqual(
            np.rad2deg(app.adjust_actuator_lab_vane_command(1.0, coarse=True)),
            2.5,
        )

    def test_keydown_mapping_works_exactly_while_paused_and_centers_commands(self):
        class FakeEventQueue:
            def __init__(self):
                self.items = []

            def get(self):
                items, self.items = self.items, []
                return items

        class FakePygame:
            QUIT = -1
            KEYDOWN = 1
            KMOD_SHIFT = 1
            K_ESCAPE = 10
            K_SPACE = 11
            K_n = 12
            K_r = 13
            K_l = 14
            K_BACKSPACE = 15
            K_a = 16
            K_d = 17
            K_v = 18
            K_f = 19
            K_h = 20
            K_g = 21

            def __init__(self):
                self.event = FakeEventQueue()

        pygame = FakePygame()

        def post(key, mod=0):
            pygame.event.items.append(
                SimpleNamespace(type=pygame.KEYDOWN, key=key, mod=mod)
            )

        app = self.make_app()
        app.paused = True
        for key, mod in (
            (pygame.K_d, 0),
            (pygame.K_d, pygame.KMOD_SHIFT),
            (pygame.K_h, 0),
            (pygame.K_h, pygame.KMOD_SHIFT),
        ):
            post(key, mod)
        self.assertTrue(app.handle_events(pygame))
        self.assertAlmostEqual(np.rad2deg(app.commands.direct_vane), 2.5)
        self.assertAlmostEqual(app.commands.moving_mass_target_m, 0.006)

        post(pygame.K_n)
        post(pygame.K_v)
        post(pygame.K_g)
        self.assertTrue(app.handle_events(pygame))
        self.assertTrue(app.step_once)
        self.assertEqual(app.commands.direct_vane, 0.0)
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)

        app.adjust_actuator_lab_vane_command(1.0)
        app.adjust_actuator_lab_mass_target(1.0)
        post(pygame.K_BACKSPACE)
        self.assertTrue(app.handle_events(pygame))
        self.assertEqual(app.commands.direct_vane, 0.0)
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)

    def test_manual_limits_clamp_to_ui_and_physical_limits(self):
        app = self.make_app()
        for _ in range(20):
            app.adjust_actuator_lab_mass_target(1.0, coarse=True)
            app.adjust_actuator_lab_vane_command(1.0, coarse=True)
        self.assertAlmostEqual(app.commands.moving_mass_target_m, 0.010)
        self.assertAlmostEqual(np.rad2deg(app.commands.direct_vane), 5.0)
        for _ in range(20):
            app.adjust_actuator_lab_mass_target(-1.0, coarse=True)
            app.adjust_actuator_lab_vane_command(-1.0, coarse=True)
        self.assertAlmostEqual(app.commands.moving_mass_target_m, -0.010)
        self.assertAlmostEqual(np.rad2deg(app.commands.direct_vane), -5.0)

        rb_cfg, ui_cfg, _controller_cfg = load_interactive_config(PROFILE_PATH)
        physical_rb = replace(
            rb_cfg,
            moving_mass=replace(rb_cfg.moving_mass, max_offset_m=0.006),
        )
        physical_app = self.make_app(ui_cfg=ui_cfg, rb_cfg=physical_rb)
        for _ in range(20):
            physical_app.adjust_actuator_lab_mass_target(1.0, coarse=True)
        self.assertAlmostEqual(physical_app.actuator_lab_mass_limit_m, 0.006)
        self.assertAlmostEqual(physical_app.commands.moving_mass_target_m, 0.006)

    def test_center_reset_and_leaving_lab_clear_stale_targets(self):
        app = self.make_app()
        app.adjust_actuator_lab_mass_target(1.0, coarse=True)
        app.adjust_actuator_lab_vane_command(1.0, coarse=True)
        app.commands.moving_mass_target_m = 0.0  # G
        app.commands.direct_vane = 0.0  # V
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)
        self.assertEqual(app.commands.direct_vane, 0.0)

        app.adjust_actuator_lab_mass_target(1.0, coarse=True)
        app.adjust_actuator_lab_vane_command(1.0, coarse=True)
        app.commands.zero(app.thrust_curve.throttle_for_hover())  # Backspace
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)
        self.assertEqual(app.commands.direct_vane, 0.0)

        app.adjust_actuator_lab_mass_target(1.0, coarse=True)
        app.adjust_actuator_lab_vane_command(1.0, coarse=True)
        app.reset()
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)
        self.assertEqual(app.commands.direct_vane, 0.0)
        self.assertEqual(float(app.state[10]), 0.0)

        app.adjust_actuator_lab_mass_target(1.0, coarse=True)
        self.assertTrue(app.set_mode(ControlMode.DIRECT))
        self.assertEqual(app.commands.moving_mass_target_m, 0.0)
        self.assertEqual(app.state.shape, (11,))

    def test_manual_modes_to_rate_seed_current_angular_velocity(self):
        app = self.make_app()

        for source_mode in (
            ControlMode.ACTUATOR_LAB,
            ControlMode.DIRECT,
            ControlMode.STABILIZE,
        ):
            with self.subTest(source_mode=source_mode):
                app.reset()
                self.assertTrue(app.set_mode(source_mode))
                app.state[5] = 1.25
                app.commands.omega_target = -3.0

                self.assertTrue(app.set_mode(ControlMode.RATE))

                self.assertEqual(app.commands.omega_target, app.state[5])
                self.assertEqual(app.control.rate._last_error, 0.0)

    def test_positive_manual_target_moves_actual_mass_with_limits_without_teleporting(self):
        app = self.make_app()
        target = 0.005
        app.commands.moving_mass_target_m = target
        app.commands.direct_vane = np.deg2rad(1.0)
        before_velocity = float(app.state[9])

        app.physics_step(0.0, 1.0)

        offset = float(app.state[8])
        velocity = float(app.state[9])
        self.assertGreater(offset, 0.0)
        self.assertLess(offset, target)
        self.assertLessEqual(abs(velocity), app.rb_cfg.moving_mass.max_rate_m_s + 1e-12)
        self.assertLessEqual(
            abs(velocity - before_velocity),
            app.rb_cfg.moving_mass.max_accel_m_s2 * app.rb_cfg.dt + 1e-12,
        )
        self.assertAlmostEqual(float(app.state[10]), target)
        self.assertAlmostEqual(app.last_control.vane_angle_cmd, np.deg2rad(1.0))
        self.assertEqual(float(app.state[7]), 0.0)
        for _ in range(5):
            app.physics_step(0.0, 1.0)
        self.assertGreater(float(app.state[7]), 0.0)
        self.assertLess(float(app.state[7]), np.deg2rad(1.0))

    def test_static_signs_show_opposing_and_reinforcing_components(self):
        app = self.make_app()
        offset = 0.005
        positive_vane_state = np.array(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, app.rb_cfg.hover_thrust, np.deg2rad(2.0), offset, 0.0, offset]
        )
        negative_vane_state = positive_vane_state.copy()
        negative_vane_state[7] *= -1.0

        opposing = app.plant.force_moment_breakdown(positive_vane_state)
        reinforcing = app.plant.force_moment_breakdown(negative_vane_state)

        self.assertGreater(opposing.thrust_moment_from_com_offset, 0.0)
        self.assertLess(opposing.vane_moment, 0.0)
        self.assertGreater(reinforcing.thrust_moment_from_com_offset, 0.0)
        self.assertGreater(reinforcing.vane_moment, 0.0)
        self.assertLess(
            opposing.thrust_moment_from_com_offset * opposing.vane_moment,
            0.0,
        )
        self.assertGreater(
            reinforcing.thrust_moment_from_com_offset * reinforcing.vane_moment,
            0.0,
        )
        app.last_forces = opposing
        _body_up, body_right = app.plant.body_axes(0.0)
        self.assertEqual(app.vane_side_force_direction(body_right), "BODY-RIGHT (+)")

    def test_expected_pitch_direction_uses_configured_deadband(self):
        app = self.make_app()

        self.assertEqual(app.expected_pitch_direction(0.0), "NEAR ZERO")
        self.assertEqual(app.expected_pitch_direction(0.001), "NEAR ZERO")
        self.assertEqual(app.expected_pitch_direction(0.003), "RIGHT / POSITIVE")
        self.assertEqual(app.expected_pitch_direction(-0.003), "LEFT / NEGATIVE")

    def test_actuator_pair_diagnostic_excludes_damping_and_disturbance(self):
        app = self.make_app()
        app.last_forces = replace(
            app.last_forces,
            thrust_moment_from_com_offset=0.006,
            vane_moment_about_total_com=-0.003,
            damping_moment=-0.5,
            disturbance_moment=0.4,
            total_moment=-0.097,
        )

        self.assertAlmostEqual(app.actuator_pair_moment(), 0.003)
        self.assertEqual(app.actuator_pair_direction(), "PAIR RIGHT / POSITIVE")
        panel = "\n".join(app.actuator_lab_panel_lines())
        self.assertIn("actuator-pair moment: +0.0030 N m", panel)
        self.assertIn("actuator pair: PAIR RIGHT / POSITIVE", panel)
        self.assertIn("total pitch moment: -0.0970 N m", panel)
        self.assertIn("expected pitch: LEFT / NEGATIVE", panel)

        self.assertEqual(app.actuator_pair_direction(-0.003), "PAIR LEFT / NEGATIVE")
        self.assertEqual(app.actuator_pair_direction(0.001), "PAIR NEAR ZERO")

    def test_total_com_visual_geometry_uses_fixed_body_origin(self):
        app = self.make_app()
        app.commands.moving_mass_target_m = 0.010
        app.state[8] = 0.005
        app.state[10] = 0.010
        app.plant.state = app.state.copy()
        app.last_forces = app.plant.force_moment_breakdown(app.state)
        body_up, body_right = app.plant.body_axes(0.0)

        geometry = app.actuator_lab_visual_geometry(body_up, body_right)

        np.testing.assert_allclose(geometry["total_com"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(
            geometry["fixed_body_origin"],
            np.array([-0.00125, 0.97]),
        )
        np.testing.assert_allclose(
            geometry["moving_mass_actual"],
            np.array([0.00375, 1.09]),
        )
        np.testing.assert_allclose(
            geometry["moving_mass_target"],
            np.array([0.00875, 1.09]),
        )

    def test_normal_startup_and_mode_six_refusal_remain_eight_state(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(PROFILE_PATH)
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        self.assertEqual(app.state.shape, (8,))
        self.assertFalse(app.set_mode(ControlMode.ACTUATOR_LAB))
        self.assertEqual(app.mode, ControlMode.DIRECT)
        self.assertIn("restart with --actuator-lab", app.mode_status)
        self.assertEqual(
            {
                ControlMode.DIRECT,
                ControlMode.RATE,
                ControlMode.STABILIZE,
                ControlMode.ALT_HOLD,
                ControlMode.LOITER,
            },
            set(ControlMode) - {ControlMode.ACTUATOR_LAB},
        )

    def test_logging_adds_only_lab_activity_and_old_csv_still_loads(self):
        self.assertIn("actuator_lab_active", INTERACTIVE_FIELDS)
        self.assertEqual(INTERACTIVE_FIELDS.count("actuator_lab_active"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            rb_cfg, ui_cfg, controller_cfg = load_interactive_config(PROFILE_PATH)
            ui_cfg.log_directory = tmp
            app = InteractiveApp(
                configure_actuator_lab(rb_cfg),
                ui_cfg,
                controller_cfg,
                actuator_lab_enabled=True,
            )
            log_path = app.logger.start()
            app.physics_step(0.0, 1.0)
            app.logger.close()
            with log_path.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["mode"], "ACTUATOR_LAB")
            self.assertEqual(row["actuator_lab_active"], "1")
            self.assertIn("moving_mass_target_m", row)
            self.assertIn("total_com_body_right_m", row)

            old_path = Path(tmp) / "old_log.csv"
            old_path.write_text("sim_time,x_cg,z_cg\n0.0,0.0,1.0\n", encoding="utf-8")
            old_data = load_csv(old_path)
            self.assertIn("sim_time", old_data)
            self.assertNotIn("actuator_lab_active", old_data)

    def test_short_hover_and_panel_smoke(self):
        app = self.make_app()
        for _ in range(10):
            app.physics_step(0.0, 1.0)
        self.assertTrue(np.all(np.isfinite(app.state)))
        self.assertEqual(app.crash_reason, "")
        panel = "\n".join(app.actuator_lab_panel_lines())
        self.assertIn("moving-mass target", panel)
        self.assertIn("total COM body-right", panel)
        self.assertIn("expected pitch", panel)


if __name__ == "__main__":
    unittest.main()
