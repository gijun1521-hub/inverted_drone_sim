import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverted_drone_sim.interactive_sim import ControlMode, InteractiveApp
from inverted_drone_sim.params import load_interactive_config


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
        np.testing.assert_allclose(geom["actual_dir"], np.array([np.cos(0.3), np.sin(0.3)]))
        np.testing.assert_allclose(geom["command_dir"], np.array([np.cos(-0.3), np.sin(-0.3)]))

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