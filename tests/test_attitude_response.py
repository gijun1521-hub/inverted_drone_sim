import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DroneConfig
from pid_controller import PIDController
from simulate_pid import run_simulation


class AttitudeResponseTests(unittest.TestCase):
    def test_attitude_only_commands_base_under_cg(self):
        cfg = DroneConfig(max_time=1.5)
        controller = PIDController(cfg, Kx=0.0, Kvx=0.0)

        left_lean = np.array([0.0, cfg.target_z, np.deg2rad(-5.0), 0.0, 0.0, 0.0])
        right_lean = np.array([0.0, cfg.target_z, np.deg2rad(5.0), 0.0, 0.0, 0.0])

        self.assertLess(controller.compute_action(left_lean)[1], 0.0)
        controller.reset()
        self.assertGreater(controller.compute_action(right_lean)[1], 0.0)

    def test_attitude_only_reduces_small_initial_tilt(self):
        cfg = DroneConfig(max_time=1.5)
        controller = PIDController(cfg, Kx=0.0, Kvx=0.0)

        _times, states, _actions, _rows, _cfg = run_simulation(cfg, controller)

        self.assertLess(abs(states[-1, 2]), abs(states[0, 2]))


if __name__ == "__main__":
    unittest.main()
