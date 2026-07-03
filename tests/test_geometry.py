import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverted_drone_sim.config import DroneConfig
from inverted_drone_sim.diagnostics import assert_geometry_sign, cg_position


class GeometryTests(unittest.TestCase):
    def test_theta_sign_matches_cg_offset(self):
        cfg = DroneConfig()

        left_lean = np.array([0.0, 1.0, np.deg2rad(-5.0), 0.0, 0.0, 0.0])
        right_lean = np.array([0.0, 1.0, np.deg2rad(5.0), 0.0, 0.0, 0.0])

        self.assertLess(cg_position(left_lean, cfg)[0], left_lean[0])
        self.assertGreater(cg_position(right_lean, cfg)[0], right_lean[0])
        assert_geometry_sign(left_lean, cfg)
        assert_geometry_sign(right_lean, cfg)


if __name__ == "__main__":
    unittest.main()
