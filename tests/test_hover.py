import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DroneConfig
from drone_model import InvertedDrone2D


class HoverTests(unittest.TestCase):
    def test_upright_hover_has_zero_acceleration(self):
        cfg = DroneConfig()
        drone = InvertedDrone2D(cfg)
        state = np.array([0.0, cfg.target_z, 0.0, 0.0, 0.0, 0.0], dtype=float)
        action = np.array([cfg.hover_throttle, 0.0], dtype=float)

        accelerations = drone.accelerations(state, action)

        np.testing.assert_allclose(accelerations, np.zeros(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
