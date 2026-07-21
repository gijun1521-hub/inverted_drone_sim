from __future__ import annotations

import unittest

from analysis.pitch_damping_boundary_audit import (
    ANGLE_P,
    INITIAL_RATE_D,
    INITIAL_RATE_P,
    RATE_D_STEP,
    RATE_I,
    RATE_P_STEP,
    extension_candidates,
    initial_candidates,
)


class PitchDampingBoundaryAuditTests(unittest.TestCase):
    def test_exact_initial_grid_and_fixed_parameters(self):
        candidates = initial_candidates()
        self.assertEqual(len(candidates), 20)
        self.assertEqual({candidate.rate_p for candidate in candidates}, set(INITIAL_RATE_P))
        self.assertEqual({candidate.rate_d for candidate in candidates}, set(INITIAL_RATE_D))
        self.assertEqual({candidate.angle_p for candidate in candidates}, {ANGLE_P})
        self.assertEqual(RATE_I, 0.0)

    def test_upper_p_extension_uses_only_declared_step(self):
        candidates = initial_candidates()
        additions = extension_candidates(
            candidates, {"rate_p": max(INITIAL_RATE_P), "rate_d": min(INITIAL_RATE_D)}
        )
        self.assertEqual(len(additions), len(INITIAL_RATE_D))
        self.assertEqual(
            {candidate.rate_p for candidate in additions},
            {max(INITIAL_RATE_P) + RATE_P_STEP},
        )

    def test_both_upper_axes_add_one_rectangular_layer(self):
        candidates = initial_candidates()
        additions = extension_candidates(
            candidates, {"rate_p": max(INITIAL_RATE_P), "rate_d": max(INITIAL_RATE_D)}
        )
        expected = len(INITIAL_RATE_D) + len(INITIAL_RATE_P) + 1
        self.assertEqual(len(additions), expected)
        self.assertEqual(max(candidate.rate_p for candidate in additions), max(INITIAL_RATE_P) + RATE_P_STEP)
        self.assertEqual(max(candidate.rate_d for candidate in additions), max(INITIAL_RATE_D) + RATE_D_STEP)


if __name__ == "__main__":
    unittest.main()
