import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interactive_sim import ControlMode, InteractiveApp
from params import load_interactive_config


class InteractiveRenderSmokeTests(unittest.TestCase):
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
