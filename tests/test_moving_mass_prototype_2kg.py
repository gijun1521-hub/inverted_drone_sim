from __future__ import annotations

import unittest

from analysis.moving_mass_comparison import (
    default_variants,
    resolve_comparison_scenarios,
    run_moving_mass_comparison,
)
from params import load_interactive_config


PROFILE_PATH = "params/moving_mass_prototype_2kg.json"
EXPECTED_VARIANTS = {
    "vane_only",
    "moving_mass_fixed_target",
    "moving_mass_proportional_assist",
    "total_com_geometry_centered",
    "total_com_geometry_fixed_target",
    "total_com_geometry_proportional_assist",
}


class MovingMassPrototype2kgTests(unittest.TestCase):
    def test_profile_loads_with_expected_mass_and_geometry_values(self):
        cfg, _ui_cfg, _controller_cfg = load_interactive_config(PROFILE_PATH)

        self.assertEqual(cfg.m, 2.0)
        self.assertEqual(cfg.moving_mass.mass_kg, 0.5)
        self.assertEqual(cfg.moving_mass.max_offset_m, 0.05)
        self.assertEqual(cfg.moving_mass.moving_mass_body_up_offset_m, 0.12)
        self.assertEqual(cfg.l, 0.25)
        self.assertEqual(cfg.m - cfg.moving_mass.mass_kg, 1.5)
        self.assertFalse(cfg.moving_mass.enabled)
        self.assertFalse(cfg.moving_mass.use_total_com_geometry)
        self.assertTrue(cfg.moving_mass.use_legacy_gravity_offset_moment)
        self.assertFalse(
            cfg.moving_mass.use_total_com_geometry
            and cfg.moving_mass.use_legacy_gravity_offset_moment
        )

    def test_short_comparison_has_six_variants_and_profile_metadata(self):
        results = run_moving_mass_comparison(
            param_path=PROFILE_PATH,
            scenarios=resolve_comparison_scenarios(
                "pitch_assist_probe", duration_s=0.02
            ),
        )

        self.assertEqual(len(default_variants()), 6)
        self.assertEqual(len(results), 6)
        self.assertEqual({result.variant.name for result in results}, EXPECTED_VARIANTS)
        for result in results:
            row = result.row
            self.assertEqual(float(row["total_mass_kg"]), 2.0)
            self.assertEqual(float(row["moving_mass_mass_kg"]), 0.5)
            self.assertEqual(float(row["effective_moving_mass_max_offset_m"]), 0.05)
            self.assertEqual(float(row["moving_mass_body_up_offset_m"]), 0.12)


if __name__ == "__main__":
    unittest.main()
