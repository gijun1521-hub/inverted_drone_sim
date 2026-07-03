import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverted_drone_sim.actuators import FirstOrderMotor, VaneServo
from inverted_drone_sim.cascaded_controller import AttitudeController, RatePIDController
from inverted_drone_sim.config import RigidBodyConfig
from inverted_drone_sim.rigid_body_model import RigidBodySingleFan2D
from inverted_drone_sim.singlecopter_mixer import SingleCopterMixer


class RigidBodyPhysicsTests(unittest.TestCase):
    def test_zero_force_free_fall(self):
        cfg = RigidBodyConfig(translational_drag=0.0, angular_damping=0.0)
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        terms = plant.force_moment_breakdown(state)

        self.assertAlmostEqual(terms.x_ddot, 0.0)
        self.assertAlmostEqual(terms.z_ddot, -cfg.g)
        self.assertAlmostEqual(terms.theta_ddot, 0.0)

    def test_upright_hover_with_mg_thrust(self):
        cfg = RigidBodyConfig(translational_drag=0.0, angular_damping=0.0)
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        terms = plant.force_moment_breakdown(state)

        self.assertAlmostEqual(terms.x_ddot, 0.0)
        self.assertAlmostEqual(terms.z_ddot, 0.0)
        self.assertAlmostEqual(terms.theta_ddot, 0.0)

    def test_tilted_main_thrust_has_horizontal_component(self):
        cfg = RigidBodyConfig()
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, np.deg2rad(10.0), 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        terms = plant.force_moment_breakdown(state)

        self.assertGreater(terms.thrust_force[0], 0.0)

    def test_vane_force_direction_upright(self):
        cfg = RigidBodyConfig()
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.1])

        terms = plant.force_moment_breakdown(state)

        self.assertGreater(terms.vane_force[0], 0.0)
        self.assertAlmostEqual(terms.vane_force[1], 0.0)

    def test_vane_moment_direction(self):
        cfg = RigidBodyConfig()
        plant = RigidBodySingleFan2D(cfg)
        positive_vane = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.1])
        negative_vane = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, -0.1])

        self.assertLess(plant.force_moment_breakdown(positive_vane).vane_moment, 0.0)
        self.assertGreater(plant.force_moment_breakdown(negative_vane).vane_moment, 0.0)

    def test_pitch_inertia_changes_angular_acceleration(self):
        small = RigidBodyConfig(H=0.30, W=0.08)
        large = RigidBodyConfig(H=0.80, W=0.08)
        small_plant = RigidBodySingleFan2D(small)
        large_plant = RigidBodySingleFan2D(large)
        small_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, small.hover_thrust, 0.1])
        large_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, large.hover_thrust, 0.1])

        small_alpha = abs(small_plant.force_moment_breakdown(small_state).theta_ddot)
        large_alpha = abs(large_plant.force_moment_breakdown(large_state).theta_ddot)

        self.assertLess(large_alpha, small_alpha)


class ActuatorAndMixerTests(unittest.TestCase):
    def test_motor_lag(self):
        motor = FirstOrderMotor(thrust_max=100.0, time_constant=0.5)

        output = motor.update(thrust=10.0, thrust_cmd=20.0)

        self.assertAlmostEqual(output.thrust_dot, 20.0)
        self.assertFalse(output.saturated)

    def test_servo_lag_and_rate_limit(self):
        servo = VaneServo(dt=0.01, angle_limit=1.0, rate_limit=0.5, time_constant=0.1)

        output = servo.update(vane_angle=0.0, vane_angle_cmd=1.0)

        self.assertAlmostEqual(output.vane_angle_dot, 0.5)
        self.assertTrue(output.rate_saturated)

    def test_mixer_command_increases_when_thrust_decreases(self):
        cfg = RigidBodyConfig()
        mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)

        high_thrust = mixer.mix(desired_moment=-0.1, thrust=cfg.hover_thrust)
        low_thrust = mixer.mix(desired_moment=-0.1, thrust=0.5 * cfg.hover_thrust)

        self.assertGreater(abs(low_thrust.vane_angle_cmd), abs(high_thrust.vane_angle_cmd))

    def test_mixer_saturates_at_low_thrust(self):
        cfg = RigidBodyConfig()
        mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)

        output = mixer.mix(desired_moment=10.0, thrust=0.0)

        self.assertTrue(output.saturated)
        self.assertNotEqual(output.unattainable_moment, 0.0)


class ControllerArchitectureTests(unittest.TestCase):
    def test_rate_pid_stabilizing_moment_direction(self):
        rate = RatePIDController(moment_limit=1.0, kp=0.1, ki=0.0, kd=0.0)

        desired_moment, *_ = rate.compute(omega_target=0.0, omega=2.0, dt=0.01)

        self.assertLess(desired_moment, 0.0)

    def test_attitude_controller_outputs_rate_target(self):
        cfg = RigidBodyConfig()
        attitude = AttitudeController(cfg)

        omega_target = attitude.compute(theta=np.deg2rad(5.0), theta_target=0.0)

        self.assertIsInstance(omega_target, float)
        self.assertLess(omega_target, 0.0)


if __name__ == "__main__":
    unittest.main()
