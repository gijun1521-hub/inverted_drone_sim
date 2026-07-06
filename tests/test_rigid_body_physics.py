import sys
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from inverted_drone_sim.actuators import FirstOrderMotor, VaneServo
from inverted_drone_sim.cascaded_controller import AttitudeController, RatePIDController
from inverted_drone_sim.config import RigidBodyConfig
from inverted_drone_sim.interactive_sim import ControlMode, InteractiveApp, ManualCommands, ManualControlSystem
from inverted_drone_sim.math_utils import shortest_angle_error, wrap_pi
from inverted_drone_sim.moment_allocator import MomentAllocator
from inverted_drone_sim.moving_mass_model import MovingMassSingleFan2D
from inverted_drone_sim.rigid_body_model import RigidBodySingleFan2D
from inverted_drone_sim.safety import check_safety
from inverted_drone_sim.singlecopter_mixer import SingleCopterMixer
from inverted_drone_sim.thrust_curve import ThrottleToThrustModel
from inverted_drone_sim.wind import SimpleWindModel


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

    def test_nonlinear_vane_has_axial_loss(self):
        cfg = RigidBodyConfig(vane_model="nonlinear_with_axial_loss")
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.4])

        terms = plant.force_moment_breakdown(state)

        self.assertLess(terms.axial_efficiency, 1.0)
        self.assertLess(terms.thrust_force[1], cfg.hover_thrust)

    def test_left_right_vane_symmetry(self):
        cfg = RigidBodyConfig(vane_model="nonlinear_with_axial_loss")
        plant = RigidBodySingleFan2D(cfg)
        positive = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.2])
        negative = positive.copy()
        negative[7] = -0.2

        pos = plant.force_moment_breakdown(positive)
        neg = plant.force_moment_breakdown(negative)

        self.assertAlmostEqual(pos.vane_force[0], -neg.vane_force[0])
        self.assertAlmostEqual(pos.vane_moment, -neg.vane_moment)

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

    def test_external_force_changes_acceleration_not_position_directly(self):
        cfg = RigidBodyConfig(translational_drag=0.0, angular_damping=0.0)
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        terms = plant.force_moment_breakdown(state, disturbance_force=np.array([3.0, 0.0]))

        self.assertAlmostEqual(state[0], 0.0)
        self.assertGreater(terms.x_ddot, 0.0)

    def test_external_moment_changes_angular_acceleration(self):
        cfg = RigidBodyConfig(translational_drag=0.0, angular_damping=0.0)
        plant = RigidBodySingleFan2D(cfg)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        terms = plant.force_moment_breakdown(state, disturbance_moment=0.2)

        self.assertGreater(terms.theta_ddot, 0.0)

    def test_open_loop_run_remains_finite(self):
        cfg = RigidBodyConfig(dt=0.005, translational_drag=0.0, angular_damping=0.0)
        plant = RigidBodySingleFan2D(cfg)
        plant.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))

        for _ in range(1000):
            state = plant.step(0.0, 0.0)

        self.assertTrue(np.all(np.isfinite(state)))

    def test_dt_convergence_for_hover(self):
        def run(dt):
            cfg = RigidBodyConfig(dt=dt, translational_drag=0.0, angular_damping=0.0)
            plant = RigidBodySingleFan2D(cfg)
            plant.reset(np.array([0.0, 1.0, 0.05, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))
            for _ in range(int(0.2 / dt)):
                plant.step(0.0, 0.0)
            return plant.state

        coarse = run(0.005)
        fine = run(0.0025)

        np.testing.assert_allclose(coarse[:6], fine[:6], atol=3e-3)

    def test_crash_detection_uses_lowest_body_corner(self):
        cfg = RigidBodyConfig()
        state = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        status = check_safety(state, cfg)

        self.assertTrue(status.crashed)
        self.assertEqual(status.reason, "ground contact")

    def test_wind_changes_drag_direction(self):
        tailwind_cfg = RigidBodyConfig(wind_velocity_world=(2.0, 0.0))
        headwind_cfg = RigidBodyConfig(wind_velocity_world=(-2.0, 0.0))
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, tailwind_cfg.hover_thrust, 0.0])

        tailwind = RigidBodySingleFan2D(tailwind_cfg).force_moment_breakdown(state)
        headwind = RigidBodySingleFan2D(headwind_cfg).force_moment_breakdown(state)

        self.assertGreater(tailwind.drag_force[0], 0.0)
        self.assertLess(headwind.drag_force[0], 0.0)

    def test_wind_gust_impulse_has_duration(self):
        cfg = RigidBodyConfig(gust_force_world=(3.0, 0.0), gust_moment=0.2, gust_duration_s=0.1)
        wind = SimpleWindModel(cfg)

        early = wind.disturbance_at(0.05)
        late = wind.disturbance_at(0.2)

        self.assertGreater(early.force_world[0], 0.0)
        self.assertGreater(early.moment, 0.0)
        self.assertAlmostEqual(late.force_world[0], 0.0)
        self.assertAlmostEqual(late.moment, 0.0)


class ActuatorAndMixerTests(unittest.TestCase):
    def test_polynomial_throttle_zero_is_not_max_thrust(self):
        base = RigidBodyConfig()
        cfg = RigidBodyConfig(thrust_curve_model="polynomial", thrust_curve_coefficients=(base.T_max, 0.0))
        model = ThrottleToThrustModel(cfg)

        self.assertAlmostEqual(model.thrust(0.0), 0.0)

    def test_missing_polynomial_coefficients_raise(self):
        cfg = RigidBodyConfig(thrust_curve_model="polynomial", thrust_curve_coefficients=())
        model = ThrottleToThrustModel(cfg)

        with self.assertRaisesRegex(ValueError, "requires coefficients"):
            model.thrust(0.5)

    def test_lookup_csv_requires_monotonic_throttle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_lookup.csv"
            path.write_text("throttle,thrust\n0.0,0\n0.5,2\n0.4,3\n", encoding="utf-8")
            cfg = RigidBodyConfig(thrust_curve_model="lookup_csv", thrust_curve_lookup_csv=str(path))
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                ThrottleToThrustModel(cfg)

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
        self.assertTrue(output.authority_limited)
        self.assertNotEqual(output.unattainable_moment, 0.0)

    def test_zero_thrust_physical_moment_is_zero(self):
        cfg = RigidBodyConfig()
        mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)

        output = mixer.mix(desired_moment=0.1, thrust=0.0)

        self.assertAlmostEqual(output.physically_achievable_moment, 0.0)


class ControllerArchitectureTests(unittest.TestCase):
    def test_wrap_pi_shortest_error_across_boundary(self):
        self.assertAlmostEqual(wrap_pi(np.pi), -np.pi)
        err = shortest_angle_error(np.deg2rad(-179.0), np.deg2rad(179.0))
        self.assertAlmostEqual(np.rad2deg(err), 2.0)

    def test_attitude_shaping_uses_controller_dt_not_physics_dt(self):
        fast_physics = AttitudeController(RigidBodyConfig(dt=0.0025))
        slow_physics = AttitudeController(RigidBodyConfig(dt=0.005))

        a = fast_physics.compute(theta=1.0, theta_target=0.0, dt=0.01)
        b = slow_physics.compute(theta=1.0, theta_target=0.0, dt=0.01)

        self.assertAlmostEqual(a, b)

    def test_rate_pid_stabilizing_moment_direction(self):
        rate = RatePIDController(moment_limit=1.0, kp=0.1, ki=0.0, kd=0.0)

        desired_moment, *_ = rate.compute(omega_target=0.0, omega=2.0, dt=0.01)

        self.assertLess(desired_moment, 0.0)

    def test_attitude_controller_outputs_rate_target(self):
        cfg = RigidBodyConfig()
        attitude = AttitudeController(cfg)

        omega_target = attitude.compute(theta=np.deg2rad(5.0), theta_target=0.0, dt=cfg.dt)

        self.assertIsInstance(omega_target, float)
        self.assertLess(omega_target, 0.0)

    def test_bumpless_mode_switch_seeds_targets_without_throttle_reset(self):
        app = InteractiveApp()
        app.state[5] = 1.25
        app.commands.throttle = 0.77

        app.set_mode(ControlMode.RATE)

        self.assertAlmostEqual(app.commands.omega_target, 1.25)
        self.assertAlmostEqual(app.commands.throttle, 0.77)
        app.state[2] = 3.5
        app.set_mode(ControlMode.STABILIZE)
        self.assertAlmostEqual(app.commands.theta_target, wrap_pi(3.5))

    def test_no_large_mode_switch_moment_spike(self):
        app = InteractiveApp()
        app.set_mode(ControlMode.RATE)
        app.state[5] = 0.6
        app.commands.omega_target = 0.6
        before = app.control.compute(app.mode, app.state, app.commands, app.ui_cfg.controller_dt).desired_moment
        app.set_mode(ControlMode.STABILIZE)
        after = app.control.compute(app.mode, app.state, app.commands, app.ui_cfg.controller_dt).desired_moment

        self.assertLess(abs(after - before), 0.2)

    def test_rate_pid_anti_windup_under_low_thrust_saturation(self):
        cfg = RigidBodyConfig()
        rate = RatePIDController(moment_limit=1.0, kp=0.4, ki=0.2, kd=0.0)
        mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)

        desired, *_ = rate.compute(omega_target=5.0, omega=0.0, dt=0.01)
        output = mixer.mix(desired, thrust=0.0)
        rate.apply_mixer_feedback(desired, output, dt=0.01)

        self.assertTrue(rate.last_integrator_inhibited)
        self.assertLess(rate.last_anti_windup_correction, 0.0)

    def test_direct_mode_outputs_actuator_commands(self):
        cfg = RigidBodyConfig()
        control = ManualControlSystem(cfg)
        commands = ManualCommands(throttle=0.4, direct_vane=0.1)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        output = control.compute(ControlMode.DIRECT, state, commands)

        self.assertAlmostEqual(output.thrust_cmd, 0.4 * cfg.T_max)
        self.assertAlmostEqual(output.vane_angle_cmd, 0.1)
        self.assertAlmostEqual(output.desired_moment, 0.0)

    def test_rate_mode_uses_mixer_not_direct_vane(self):
        cfg = RigidBodyConfig()
        control = ManualControlSystem(cfg)
        commands = ManualCommands(throttle=0.4, direct_vane=0.2, omega_target=-1.0)
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        output = control.compute(ControlMode.RATE, state, commands)

        self.assertNotAlmostEqual(output.vane_angle_cmd, commands.direct_vane)
        self.assertLess(output.desired_moment, 0.0)

    def test_stabilize_mode_generates_rate_target(self):
        cfg = RigidBodyConfig()
        control = ManualControlSystem(cfg)
        commands = ManualCommands(throttle=0.4, theta_target=-0.1)
        state = np.array([0.0, 1.0, 0.1, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])

        output = control.compute(ControlMode.STABILIZE, state, commands)

        self.assertLess(output.omega_target, 0.0)

    def test_crash_logging_writes_post_step_ground_contact(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = InteractiveApp()
            app.logger.directory = Path(tmp)
            app.state = app.plant.reset(np.array([0.0, 0.26, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0]))
            app.logger.start()
            for _ in range(10):
                app.physics_step(wall_time=0.0, real_time_factor=1.0)
                if app.crash_reason:
                    break
            path = app.logger.path
            app.logger.stop()
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertTrue(rows)
        self.assertEqual(rows[-1]["crash_reason"], "ground contact")


class MovingMassAndAllocationTests(unittest.TestCase):
    def test_moving_mass_internal_motion_conserves_angular_momentum(self):
        from inverted_drone_sim.config import MovingMassConfig

        plant = MovingMassSingleFan2D(MovingMassConfig(thrust=0.0, g=0.0))
        plant.reset()
        for _ in range(20):
            plant.step(0.3, thrust=0.0)
        self.assertAlmostEqual(plant.last_breakdown.angular_momentum, 0.0, places=6)

    def test_positive_q_accel_creates_opposite_body_reaction(self):
        from inverted_drone_sim.config import MovingMassConfig

        plant = MovingMassSingleFan2D(MovingMassConfig(thrust=0.0, g=0.0))
        plant.reset()
        plant.step(0.3, thrust=0.0)
        self.assertLess(plant.last_breakdown.reaction_moment_body, 0.0)

    def test_moving_mass_offset_shifts_cg(self):
        from inverted_drone_sim.config import MovingMassConfig

        plant = MovingMassSingleFan2D(MovingMassConfig())
        state = plant.reset()
        cg0, _ = plant.total_cg_world(state)
        state[8] = 0.3
        cg1, _ = plant.total_cg_world(state)
        self.assertNotAlmostEqual(cg0[0], cg1[0])

    def test_moving_mass_limits(self):
        from inverted_drone_sim.config import MovingMassConfig

        cfg = MovingMassConfig(q_limit=0.05, q_rate_limit=0.1)
        plant = MovingMassSingleFan2D(cfg)
        plant.reset()
        for _ in range(50):
            plant.step(1.0)
        self.assertLessEqual(abs(plant.state[8]), cfg.q_limit + 1e-12)
        self.assertLessEqual(abs(plant.state[9]), cfg.q_rate_limit + 1e-12)

    def test_moving_mass_vane_force_creates_moment_at_application_point(self):
        from inverted_drone_sim.config import MovingMassConfig

        cfg = MovingMassConfig()
        plant = MovingMassSingleFan2D(cfg)
        plant.reset()
        plant.state[7] = 0.2
        plant.step(0.0)
        self.assertNotAlmostEqual(plant.last_breakdown.vane_moment, 0.0)

    def test_vane_only_allocator_uses_mixer(self):
        cfg = RigidBodyConfig()
        allocator = MomentAllocator("vane_only")
        state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])
        out = allocator.allocate(-0.1, state, cfg)
        self.assertNotEqual(out.vane_angle_cmd, 0.0)
        self.assertEqual(out.moment_to_moving_mass, 0.0)


if __name__ == "__main__":
    unittest.main()
