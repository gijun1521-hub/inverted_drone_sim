from __future__ import annotations

import unittest

from analysis.moving_mass_gain_resweep import (
    OLD_PR20_REFERENCE_GAIN,
    STAGE1_GAINS,
    STAGE2_STEP,
    STAGE3_STEP,
    _grid,
)


class MovingMassGainResweepTests(unittest.TestCase):
    def test_coarse_grid_includes_zero_upper_value_and_old_reference(self):
        self.assertIn(0.0, STAGE1_GAINS)
        self.assertIn(0.2, STAGE1_GAINS)
        self.assertIn(OLD_PR20_REFERENCE_GAIN, STAGE1_GAINS)

    def test_refinement_grids_use_declared_steps(self):
        fine = _grid(0.10, 0.025, STAGE2_STEP)
        local = _grid(0.10, 0.005, STAGE3_STEP)
        self.assertAlmostEqual(fine[1] - fine[0], STAGE2_STEP)
        self.assertAlmostEqual(local[1] - local[0], STAGE3_STEP)


if __name__ == "__main__":
    unittest.main()
