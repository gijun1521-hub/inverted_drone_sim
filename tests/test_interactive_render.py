import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_sim import ControlMode, InteractiveApp, configure_actuator_lab
from params import load_interactive_config


class InteractiveRenderSmokeTests(unittest.TestCase):
    def test_interactive_loiter_assist_profile_starts_active(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
            "params/interactive_loiter_assist_2kg.json"
        )
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        self.assertEqual(app.mode, ControlMode.LOITER)
        self.assertTrue(app.moving_mass_assist_active)
        self.assertEqual(app.state.shape, (11,))
        self.assertTrue(rb_cfg.moving_mass.use_total_com_geometry)
        self.assertFalse(rb_cfg.moving_mass.use_legacy_gravity_offset_moment)

    def test_interactive_loiter_assist_commands_mass_from_pitch_moment(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
            "params/interactive_loiter_assist_2kg.json"
        )
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)
        app.targets.target_x = 1.0

        app.physics_step(0.0, 1.0)

        expected = ui_cfg.moving_mass_assist_gain_m_per_Nm * app.last_control.desired_moment
        self.assertAlmostEqual(float(app.state[10]), expected)
        self.assertNotEqual(float(app.state[10]), 0.0)

    def test_interactive_loiter_assist_recenters_outside_loiter(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
            "params/interactive_loiter_assist_2kg.json"
        )
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)
        app.state[10] = 0.02
        app.set_mode(ControlMode.STABILIZE)

        app.physics_step(0.0, 1.0)

        self.assertEqual(float(app.state[10]), 0.0)

    def test_interactive_loiter_assist_rejects_disabled_mass(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        ui_cfg.moving_mass_assist_enabled = True
        ui_cfg.moving_mass_assist_gain_m_per_Nm = 0.055

        with self.assertRaisesRegex(ValueError, "total-COM geometry"):
            InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

    def test_full_stick_release_captures_once_and_returns(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
            "params/interactive_loiter_assist_2kg.json"
        )
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        for index in range(int(10.0 / rb_cfg.dt)):
            time_s = index * rb_cfg.dt
            app.commands.stick_x = 1.0 if 0.5 <= time_s < 2.0 else 0.0
            app.physics_step(time_s, 1.0)

        self.assertEqual(app.targets.target_capture_count, 1)
        self.assertFalse(app.targets.target_capture_pending)
        self.assertAlmostEqual(app.targets.desired_vx, 0.0)
        self.assertLess(abs(app.targets.target_x - float(app.state[0])), 0.10)
        self.assertEqual(app.crash_reason, "")

    def test_assist_profile_reproduces_seminar_recovery_cases(self):
        for scenario in ("disturbance", "step"):
            rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
                "params/interactive_loiter_assist_2kg.json"
            )
            app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

            for index in range(int(8.0 / rb_cfg.dt)):
                time_s = index * rb_cfg.dt
                if scenario == "disturbance":
                    app.disturbance.force = (
                        np.array([8.0, 0.0])
                        if 1.5 <= time_s < 1.7
                        else np.zeros(2)
                    )
                elif time_s >= 1.0:
                    app.targets.target_x = 1.0
                app.physics_step(time_s, 1.0)

            final_error = abs(app.targets.target_x - float(app.state[0]))
            limit = 0.15 if scenario == "disturbance" else 0.05
            self.assertLess(final_error, limit, scenario)
            self.assertGreater(float(np.max(np.abs(app.state[8:11]))), 0.0)
            self.assertEqual(app.crash_reason, "")

    def test_actuator_lab_render_smoke_with_dummy_surface(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:
            raise unittest.SkipTest("pygame is not installed") from exc

        rb_cfg, ui_cfg, controller_cfg = load_interactive_config(
            "params/moving_mass_prototype_2kg.json"
        )
        app = InteractiveApp(
            configure_actuator_lab(rb_cfg),
            ui_cfg,
            controller_cfg,
            actuator_lab_enabled=True,
        )

        pygame.init()
        try:
            screen = pygame.display.set_mode((1200, 820))
            font = pygame.font.SysFont("consolas", 18)
            small_font = pygame.font.SysFont("consolas", 16)
            app.render(pygame, screen, font, small_font)
        finally:
            pygame.quit()

    def test_render_errors_follow_mode(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)
        app.targets.target_x = 2.0
        app.targets.target_z = 3.0

        self.assertEqual(app.render_errors(1.0, 1.5), (0.0, 0.0))

        app.set_mode(ControlMode.ALT_HOLD)
        app.targets.target_x = 2.0
        app.targets.target_z = 3.0
        self.assertEqual(app.render_errors(1.0, 1.5), (0.0, 1.5))

        app.set_mode(ControlMode.LOITER)
        app.targets.target_x = 2.0
        app.targets.target_z = 3.0
        self.assertEqual(app.render_errors(1.0, 1.5), (1.0, 1.5))

    def test_vane_visual_geometry_uses_overlay_config(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        ui_cfg.vane_visual_scale = 3.0
        ui_cfg.vane_visual_length_m = 0.55
        ui_cfg.vane_visual_offset_m = 0.11
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        bottom = np.array([0.0, 1.0])
        body_up = np.array([0.0, 1.0])
        body_right = np.array([1.0, 0.0])
        geom = app.vane_visual_geometry(bottom, body_up, body_right, actual_vane=0.1, command_vane=-0.1)

        np.testing.assert_allclose(geom["hinge"], np.array([0.0, 0.89]))
        self.assertEqual(geom["length"], 0.55)
        np.testing.assert_allclose(geom["neutral_dir"], np.array([0.0, -1.0]))
        np.testing.assert_allclose(geom["actual_dir"], np.array([np.sin(0.3), -np.cos(0.3)]))
        np.testing.assert_allclose(geom["command_dir"], np.array([np.sin(-0.3), -np.cos(-0.3)]))

    def test_zero_vane_visual_aligns_with_downstream_body_axis(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        theta = np.deg2rad(18.0)
        body_up, body_right = app.plant.body_axes(theta)
        bottom = np.array([0.2, 0.7])
        geom = app.vane_visual_geometry(bottom, body_up, body_right, actual_vane=0.0, command_vane=0.0)

        np.testing.assert_allclose(geom["neutral_dir"], -body_up)
        np.testing.assert_allclose(geom["actual_dir"], -body_up)
        np.testing.assert_allclose(geom["command_dir"], -body_up)

    def test_positive_and_negative_vane_visual_deflect_to_opposite_sides(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        ui_cfg.vane_visual_scale = 1.0
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        bottom = np.array([0.0, 0.0])
        body_up = np.array([0.0, 1.0])
        body_right = np.array([1.0, 0.0])
        pos = app.vane_visual_geometry(bottom, body_up, body_right, actual_vane=0.2, command_vane=0.0)
        neg = app.vane_visual_geometry(bottom, body_up, body_right, actual_vane=-0.2, command_vane=0.0)

        self.assertGreater(np.dot(pos["actual_dir"], body_right), 0.0)
        self.assertLess(np.dot(neg["actual_dir"], body_right), 0.0)
        self.assertAlmostEqual(float(np.dot(pos["actual_dir"], body_up)), float(np.dot(neg["actual_dir"], body_up)))

    def test_vane_visual_scale_only_changes_display_angle_not_physics(self):
        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)

        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, rb_cfg.hover_thrust, 0.1])
        before = app.plant.force_moment_breakdown(state)
        ui_cfg.vane_visual_scale = 4.0
        after = app.plant.force_moment_breakdown(state)

        np.testing.assert_allclose(after.vane_force, before.vane_force)
        self.assertAlmostEqual(after.vane_moment, before.vane_moment)

        bottom = np.array([0.0, 0.0])
        body_up = np.array([0.0, 1.0])
        body_right = np.array([1.0, 0.0])
        geom = app.vane_visual_geometry(bottom, body_up, body_right, actual_vane=0.1, command_vane=0.0)
        np.testing.assert_allclose(geom["actual_dir"], np.array([np.sin(0.4), -np.cos(0.4)]))

    def test_loiter_render_smoke_with_dummy_surface(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:
            raise unittest.SkipTest("pygame is not installed") from exc

        rb_cfg, ui_cfg, controller_cfg = load_interactive_config("params/loiter_example.json")
        app = InteractiveApp(rb_cfg, ui_cfg, controller_cfg)
        app.set_mode(ControlMode.LOITER)

        pygame.init()
        try:
            screen = pygame.display.set_mode((640, 480))
            font = pygame.font.SysFont("consolas", 14)
            small_font = pygame.font.SysFont("consolas", 12)
            app.render(pygame, screen, font, small_font)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
