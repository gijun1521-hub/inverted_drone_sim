from __future__ import annotations

import unittest

import numpy as np

from config import MovingMassPitchAssistConfig, RigidBodyConfig
from rigid_body_model import RigidBodySingleFan2D


class MovingMassTrajectoryTests(unittest.TestCase):
    def make_plant(
        self,
        *,
        dt: float = 0.005,
        max_offset: float = 0.05,
        max_rate: float = 0.20,
        max_accel: float = 1.0,
    ) -> RigidBodySingleFan2D:
        cfg = RigidBodyConfig(
            dt=dt,
            moving_mass=MovingMassPitchAssistConfig(
                enabled=True,
                max_offset_m=max_offset,
                max_rate_m_s=max_rate,
                max_accel_m_s2=max_accel,
            ),
        )
        plant = RigidBodySingleFan2D(cfg)
        plant.reset()
        return plant

    def run_until_settled(
        self,
        plant: RigidBodySingleFan2D,
        target: float,
        *,
        max_steps: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
        dt = plant.cfg.dt
        offsets = [float(plant.state[8])]
        velocities = [float(plant.state[9])]
        saturated = []
        for _ in range(max_steps):
            state = plant.step(0.0, 0.0, moving_mass_target_m=target)
            offsets.append(float(state[8]))
            velocities.append(float(state[9]))
            saturated.append(bool(plant.last_moving_mass_saturated))
            bounded_target = float(
                np.clip(
                    target,
                    -plant.cfg.moving_mass.max_offset_m,
                    plant.cfg.moving_mass.max_offset_m,
                )
            )
            if state[8] == bounded_target and state[9] == 0.0:
                break
        velocity_array = np.asarray(velocities)
        accelerations = np.diff(velocity_array) / dt
        return np.asarray(offsets), velocity_array, accelerations, saturated

    def assert_motion_is_bounded(
        self,
        plant: RigidBodySingleFan2D,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ) -> None:
        self.assertLessEqual(
            float(np.max(np.abs(velocities))),
            plant.cfg.moving_mass.max_rate_m_s + 1e-12,
        )
        self.assertLessEqual(
            float(np.max(np.abs(accelerations))),
            plant.cfg.moving_mass.max_accel_m_s2 + 1e-10,
        )

    def assert_same_position_motion_brakes_and_returns(
        self,
        *,
        offset: float,
        velocity: float,
    ) -> None:
        plant = self.make_plant()
        target = offset
        plant.reset(
            np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    plant.cfg.hover_thrust,
                    0.0,
                    offset,
                    velocity,
                    target,
                ]
            )
        )

        state = plant.step(0.0, 0.0, moving_mass_target_m=target)
        expected_velocity = velocity - np.sign(velocity) * plant.cfg.moving_mass.max_accel_m_s2 * plant.cfg.dt
        expected_offset = offset + expected_velocity * plant.cfg.dt
        self.assertAlmostEqual(state[9], expected_velocity)
        self.assertAlmostEqual(state[8], expected_offset)
        self.assertNotEqual(state[9], 0.0)

        offsets, velocities, accelerations, _saturated = self.run_until_settled(
            plant, target
        )
        all_velocities = np.r_[velocity, state[9], velocities[1:]]
        all_accelerations = np.diff(all_velocities) / plant.cfg.dt

        self.assertLessEqual(float(np.max(np.abs(offsets))), 0.05 + 1e-12)
        self.assertEqual(offsets[-1], target)
        self.assertEqual(velocities[-1], 0.0)
        self.assert_motion_is_bounded(plant, all_velocities, all_accelerations)

    def test_same_position_positive_velocity_brakes_and_returns(self):
        self.assert_same_position_motion_brakes_and_returns(offset=0.010, velocity=0.100)

    def test_same_position_negative_velocity_brakes_and_returns(self):
        self.assert_same_position_motion_brakes_and_returns(offset=-0.010, velocity=-0.100)

    def test_same_position_settled_state_remains_stationary(self):
        plant = self.make_plant()
        offset, velocity, target, _saturated = plant._moving_mass_update(
            0.010, 0.0, 0.010, plant.cfg.dt
        )
        self.assertEqual((offset, velocity, target), (0.010, 0.0, 0.010))

    def test_fixed_targets_are_monotonic_bounded_and_settle(self):
        for target in (0.001, 0.005, 0.010, -0.001, -0.005, -0.010):
            with self.subTest(target=target):
                plant = self.make_plant()
                offsets, velocities, accelerations, _saturated = self.run_until_settled(
                    plant, target
                )
                direction = float(np.sign(target))

                self.assertGreater(direction * offsets[1], 0.0)
                self.assertTrue(np.all(direction * np.diff(offsets) >= -1e-12))
                self.assertTrue(np.all(direction * (offsets - target) <= 1e-12))
                self.assertEqual(offsets[-1], target)
                self.assertEqual(velocities[-1], 0.0)
                self.assert_motion_is_bounded(plant, velocities, accelerations)
                held_state = plant.step(0.0, 0.0, moving_mass_target_m=target)
                self.assertEqual((held_state[8], held_state[9]), (target, 0.0))

                if target == 0.005:
                    settling_time = (len(offsets) - 1) * plant.cfg.dt
                    self.assertAlmostEqual(settling_time, 0.165, delta=0.020)
                    self.assertEqual(float(np.max(offsets)), target)

                remaining = np.abs(target - offsets[:-1])
                moving_speed = np.abs(velocities[:-1])
                self.assertTrue(
                    np.any(
                        (remaining > 1e-12)
                        & (moving_speed > 0.0)
                        & (np.diff(np.r_[0.0, moving_speed]) < 0.0)
                    )
                )

    def test_target_reversal_is_bounded_and_arrives_stably(self):
        plant = self.make_plant()
        velocities = [float(plant.state[9])]
        for _ in range(20):
            state = plant.step(0.0, 0.0, moving_mass_target_m=0.010)
            velocities.append(float(state[9]))
        self.assertGreater(state[8], 0.0)
        self.assertGreater(state[9], 0.0)

        offsets, reversed_velocities, _accelerations, _saturated = self.run_until_settled(
            plant, -0.010
        )
        velocities.extend(reversed_velocities[1:])
        velocity_array = np.asarray(velocities)
        accelerations = np.diff(velocity_array) / plant.cfg.dt

        self.assertGreaterEqual(float(np.min(offsets)), -0.010 - 1e-12)
        self.assertLessEqual(float(np.max(np.abs(offsets))), 0.05 + 1e-12)
        self.assertEqual(offsets[-1], -0.010)
        self.assertEqual(reversed_velocities[-1], 0.0)
        self.assert_motion_is_bounded(plant, velocity_array, accelerations)
        held_state = plant.step(0.0, 0.0, moving_mass_target_m=-0.010)
        self.assertEqual((held_state[8], held_state[9]), (-0.010, 0.0))

    def test_close_target_change_while_moving_preserves_every_step_bounds(self):
        plant = self.make_plant()
        velocities = [float(plant.state[9])]
        for _ in range(20):
            state = plant.step(0.0, 0.0, moving_mass_target_m=0.010)
            velocities.append(float(state[9]))
        close_target = float(state[8] + 0.001)
        self.assertGreater(state[9], 0.0)

        offsets, changed_velocities, _accelerations, _saturated = self.run_until_settled(
            plant, close_target
        )
        velocities.extend(changed_velocities[1:])
        velocity_array = np.asarray(velocities)
        accelerations = np.diff(velocity_array) / plant.cfg.dt

        self.assertGreater(float(np.max(offsets)), close_target)
        self.assertLessEqual(float(np.max(np.abs(offsets))), 0.05 + 1e-12)
        self.assertEqual(offsets[-1], close_target)
        self.assertEqual(changed_velocities[-1], 0.0)
        self.assert_motion_is_bounded(plant, velocity_array, accelerations)

    def test_manual_style_centering_while_moving_preserves_bounds(self):
        plant = self.make_plant()
        velocities = [float(plant.state[9])]
        for _ in range(20):
            state = plant.step(0.0, 0.0, moving_mass_target_m=0.010)
            velocities.append(float(state[9]))
        offset_at_center_command = float(state[8])
        self.assertGreater(state[9], 0.0)

        offsets, centered_velocities, _accelerations, _saturated = self.run_until_settled(
            plant, 0.0
        )
        velocities.extend(centered_velocities[1:])
        velocity_array = np.asarray(velocities)
        accelerations = np.diff(velocity_array) / plant.cfg.dt

        self.assertGreater(float(np.max(offsets)), offset_at_center_command)
        self.assertLessEqual(float(np.max(np.abs(offsets))), 0.05 + 1e-12)
        self.assertEqual(offsets[-1], 0.0)
        self.assertEqual(centered_velocities[-1], 0.0)
        self.assert_motion_is_bounded(plant, velocity_array, accelerations)

    def test_physical_rail_clips_target_and_offset(self):
        for command in (-1.0, 1.0):
            with self.subTest(command=command):
                plant = self.make_plant(max_offset=0.05)
                offsets, velocities, accelerations, saturated = self.run_until_settled(
                    plant, command
                )
                expected = float(np.copysign(0.05, command))

                self.assertLessEqual(float(np.max(np.abs(offsets))), 0.05 + 1e-12)
                self.assertEqual(offsets[-1], expected)
                self.assertEqual(velocities[-1], 0.0)
                self.assertEqual(plant.state[10], expected)
                self.assertTrue(all(saturated))
                self.assert_motion_is_bounded(plant, velocities, accelerations)

    def test_degenerate_inputs_stop_safely_without_travel(self):
        plant = self.make_plant()
        self.assertEqual(
            plant._moving_mass_update(0.02, 0.1, 0.03, 0.0)[:2],
            (0.02, 0.0),
        )
        self.assertEqual(
            plant._moving_mass_update(0.01, 0.0, 0.01, plant.cfg.dt)[:2],
            (0.01, 0.0),
        )

        zero_rate = self.make_plant(max_rate=0.0)
        self.assertEqual(
            zero_rate._moving_mass_update(0.0, 0.1, 0.01, zero_rate.cfg.dt)[:2],
            (0.0, 0.0),
        )
        zero_accel = self.make_plant(max_accel=0.0)
        self.assertEqual(
            zero_accel._moving_mass_update(0.0, 0.1, 0.01, zero_accel.cfg.dt)[:2],
            (0.0, 0.0),
        )

        offset, velocity, target, saturated = plant._moving_mass_update(
            0.08, 0.0, 0.08, plant.cfg.dt
        )
        self.assertEqual((offset, velocity, target), (0.05, 0.0, 0.05))
        self.assertTrue(saturated)


if __name__ == "__main__":
    unittest.main()
