import unittest

import numpy as np

from config import MovingMassPitchAssistConfig, RigidBodyConfig
from rigid_body_model import RigidBodySingleFan2D


def geometry_config(*, enabled: bool = True) -> RigidBodyConfig:
    return RigidBodyConfig(
        m=2.0,
        translational_drag=0.0,
        angular_damping=0.0,
        moving_mass=MovingMassPitchAssistConfig(
            enabled=enabled,
            mass_kg=0.5,
            max_offset_m=0.05,
            moving_mass_body_up_offset_m=0.12,
            use_total_com_geometry=True,
            use_legacy_gravity_offset_moment=False,
        ),
    )


def geometry_state(
    cfg: RigidBodyConfig,
    offset: float,
    *,
    thrust: float | None = None,
    theta: float = 0.0,
    vane_angle: float = 0.0,
) -> np.ndarray:
    actual_thrust = cfg.hover_thrust if thrust is None else thrust
    return np.array(
        [0.0, 1.0, theta, 0.0, 0.0, 0.0, actual_thrust, vane_angle, offset, 0.0, offset],
        dtype=float,
    )


class TotalComGeometryTests(unittest.TestCase):
    def test_defaults_preserve_nominal_cg_geometry_and_legacy_mode(self):
        cfg = RigidBodyConfig(translational_drag=0.0, angular_damping=0.0)
        self.assertFalse(cfg.moving_mass.use_total_com_geometry)
        self.assertTrue(cfg.moving_mass.use_legacy_gravity_offset_moment)

        state = np.array([0.2, 1.1, 0.3, 0.0, 0.0, 0.0, 8.0, 0.1])
        terms = RigidBodySingleFan2D(cfg).force_moment_breakdown(state)
        body_up, _body_right = RigidBodySingleFan2D(cfg).body_axes(state[2])

        np.testing.assert_array_equal(terms.thrust_application_point, state[:2])
        np.testing.assert_allclose(terms.vane_application_point, state[:2] - cfg.l * body_up)
        np.testing.assert_array_equal(terms.thrust_moment_arm, np.zeros(2))
        self.assertEqual(terms.thrust_moment, 0.0)
        self.assertFalse(terms.total_com_geometry_active)

    def test_state_shapes_remain_compatible(self):
        disabled_cfg = geometry_config(enabled=False)
        disabled_plant = RigidBodySingleFan2D(disabled_cfg)
        disabled_state = disabled_plant.reset()
        disabled_terms = disabled_plant.force_moment_breakdown(disabled_state)

        enabled_cfg = geometry_config(enabled=True)
        enabled_state = RigidBodySingleFan2D(enabled_cfg).reset()

        self.assertEqual(disabled_state.shape, (8,))
        self.assertEqual(enabled_state.shape, (11,))
        self.assertAlmostEqual(disabled_terms.moving_mass_offset_m, 0.0)
        self.assertAlmostEqual(disabled_terms.total_com_body_right_m, 0.0)
        self.assertAlmostEqual(disabled_terms.total_com_body_up_m, 0.03)
        self.assertAlmostEqual(disabled_terms.thrust_moment, 0.0)

    def test_geometry_and_legacy_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot both be enabled"):
            RigidBodyConfig(
                moving_mass=MovingMassPitchAssistConfig(
                    use_total_com_geometry=True,
                    use_legacy_gravity_offset_moment=True,
                )
            )

    def test_nonpositive_moving_mass_is_rejected_when_geometry_is_active(self):
        for mass_kg in (0.0, -0.1):
            with self.subTest(mass_kg=mass_kg), self.assertRaisesRegex(
                ValueError, "greater than zero"
            ):
                RigidBodyConfig(
                    moving_mass=MovingMassPitchAssistConfig(
                        mass_kg=mass_kg,
                        use_total_com_geometry=True,
                        use_legacy_gravity_offset_moment=False,
                    )
                )

    def test_moving_mass_must_be_less_than_total_mass_in_geometry_mode(self):
        for mass_kg in (2.0, 2.1):
            with self.subTest(mass_kg=mass_kg), self.assertRaisesRegex(
                ValueError, "less than RigidBodyConfig.m"
            ):
                RigidBodyConfig(
                    m=2.0,
                    moving_mass=MovingMassPitchAssistConfig(
                        mass_kg=mass_kg,
                        use_total_com_geometry=True,
                        use_legacy_gravity_offset_moment=False,
                    ),
                )

    def test_prototype_total_com_and_application_arms(self):
        cfg = geometry_config()
        terms = RigidBodySingleFan2D(cfg).force_moment_breakdown(geometry_state(cfg, 0.05))

        self.assertAlmostEqual(terms.total_com_body_right_m, 0.0125)
        self.assertAlmostEqual(terms.total_com_body_up_m, 0.03)
        self.assertAlmostEqual(terms.thrust_application_arm_body_right_m, -0.0125)
        self.assertAlmostEqual(terms.thrust_application_arm_body_up_m, -0.03)
        self.assertAlmostEqual(terms.vane_application_arm_body_right_m, -0.0125)
        self.assertAlmostEqual(terms.vane_application_arm_body_up_m, -cfg.l - 0.03)
        self.assertTrue(terms.total_com_geometry_active)

    def test_zero_lateral_offset_retains_vertical_com_shift(self):
        cfg = geometry_config()
        terms = RigidBodySingleFan2D(cfg).force_moment_breakdown(geometry_state(cfg, 0.0))

        self.assertAlmostEqual(terms.total_com_body_right_m, 0.0)
        self.assertAlmostEqual(terms.total_com_body_up_m, 0.03)
        self.assertAlmostEqual(terms.thrust_moment, 0.0)
        self.assertAlmostEqual(terms.vane_application_arm_body_up_m, -cfg.l - 0.03)

    def test_positive_and_negative_offsets_have_symmetric_thrust_moments(self):
        cfg = geometry_config()
        plant = RigidBodySingleFan2D(cfg)
        positive = plant.force_moment_breakdown(geometry_state(cfg, 0.05))
        negative = plant.force_moment_breakdown(geometry_state(cfg, -0.05))

        self.assertGreater(positive.thrust_moment, 0.0)
        self.assertAlmostEqual(positive.thrust_moment, -negative.thrust_moment)

    def test_thrust_offset_moment_scales_with_actual_thrust(self):
        cfg = geometry_config()
        plant = RigidBodySingleFan2D(cfg)
        moments = []
        for scale in (0.5, 1.0, 1.5):
            terms = plant.force_moment_breakdown(
                geometry_state(cfg, 0.05, thrust=scale * cfg.hover_thrust)
            )
            moments.append(terms.thrust_moment_from_com_offset)
            self.assertAlmostEqual(terms.thrust_moment, 0.0125 * scale * cfg.hover_thrust)

        self.assertAlmostEqual(moments[0], 0.5 * moments[1])
        self.assertAlmostEqual(moments[2], 1.5 * moments[1])

    def test_zero_thrust_has_no_thrust_or_legacy_offset_moment(self):
        cfg = geometry_config()
        terms = RigidBodySingleFan2D(cfg).force_moment_breakdown(
            geometry_state(cfg, 0.05, thrust=0.0)
        )

        self.assertEqual(terms.thrust_moment, 0.0)
        self.assertEqual(terms.thrust_moment_from_com_offset, 0.0)
        self.assertEqual(terms.legacy_moving_mass_moment, 0.0)
        self.assertEqual(terms.moving_mass_moment, 0.0)

    def test_axial_thrust_moment_is_attitude_invariant(self):
        cfg = geometry_config()
        plant = RigidBodySingleFan2D(cfg)
        expected = 0.0125 * cfg.hover_thrust

        for theta in (-1.1, -0.35, 0.0, 0.6, 1.4):
            with self.subTest(theta=theta):
                terms = plant.force_moment_breakdown(geometry_state(cfg, 0.05, theta=theta))
                self.assertAlmostEqual(terms.thrust_moment, expected)

    def test_vane_moment_uses_application_point_relative_to_total_com(self):
        cfg = geometry_config()
        vane_angle = 0.1
        terms = RigidBodySingleFan2D(cfg).force_moment_breakdown(
            geometry_state(cfg, 0.0, vane_angle=vane_angle)
        )
        side_force = cfg.k_vane_force * cfg.hover_thrust * vane_angle
        expected = (-cfg.l - 0.03) * side_force
        nominal_cg_moment = -cfg.l * side_force

        self.assertAlmostEqual(terms.vane_moment, expected)
        self.assertAlmostEqual(terms.vane_moment_about_total_com, expected)
        self.assertGreater(abs(terms.vane_moment), abs(nominal_cg_moment))


if __name__ == "__main__":
    unittest.main()
