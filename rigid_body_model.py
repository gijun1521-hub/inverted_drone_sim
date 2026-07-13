from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import RigidBodyConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import RigidBodyConfig


@dataclass(frozen=True)
class ForceMomentBreakdown:
    body_up: np.ndarray
    body_right: np.ndarray
    thrust_force: np.ndarray
    vane_force: np.ndarray
    axial_efficiency: float
    axial_force_magnitude: float
    side_force_magnitude: float
    thrust_application_point: np.ndarray
    vane_application_point: np.ndarray
    thrust_moment_arm: np.ndarray
    vane_moment_arm: np.ndarray
    gravity_force: np.ndarray
    drag_force: np.ndarray
    disturbance_force: np.ndarray
    total_force: np.ndarray
    thrust_moment: float
    vane_moment: float
    damping_moment: float
    disturbance_moment: float
    moving_mass_offset_m: float
    moving_mass_velocity_m_s: float
    moving_mass_target_m: float
    moving_mass_moment: float
    moving_mass_saturated: bool
    total_com_body_right_m: float
    total_com_body_up_m: float
    thrust_application_arm_body_right_m: float
    thrust_application_arm_body_up_m: float
    vane_application_arm_body_right_m: float
    vane_application_arm_body_up_m: float
    thrust_moment_from_com_offset: float
    vane_moment_about_total_com: float
    legacy_moving_mass_moment: float
    total_com_geometry_active: bool
    total_moment: float
    x_ddot: float
    z_ddot: float
    theta_ddot: float


class RigidBodySingleFan2D:
    """2D rigid body with fan thrust and a lower vane force."""

    def __init__(self, config: RigidBodyConfig):
        config.validate()
        self.cfg = config
        self.state = np.zeros(8, dtype=float)
        self.last_breakdown: ForceMomentBreakdown | None = None
        self.last_moving_mass_saturated = False

    def _with_moving_mass_state(self, state: np.ndarray) -> np.ndarray:
        mm = self.cfg.moving_mass
        state = np.asarray(state, dtype=float)
        if state.shape == (11,):
            return state.copy()
        if state.shape != (8,):
            raise ValueError("state must have shape (8,) or (11,)")
        if not mm.enabled:
            return state.copy()
        offset = float(np.clip(mm.initial_offset_m, -mm.max_offset_m, mm.max_offset_m))
        return np.concatenate([state, np.array([offset, 0.0, offset], dtype=float)])

    def _moving_mass_update(
        self,
        offset: float,
        velocity: float,
        command: float,
        dt: float,
    ) -> tuple[float, float, float, bool]:
        mm = self.cfg.moving_mass
        limit = abs(float(mm.max_offset_m))
        target = float(np.clip(command, -limit, limit))
        command_clipped = abs(float(command) - target) > 1e-12
        bounded_offset = float(np.clip(offset, -limit, limit))
        rail_clipped = abs(float(offset) - bounded_offset) > 1e-12
        if dt <= 0.0:
            return bounded_offset, 0.0, target, command_clipped or rail_clipped

        max_rate = max(0.0, float(mm.max_rate_m_s))
        max_accel = max(0.0, float(mm.max_accel_m_s2))
        if max_rate <= 0.0 or max_accel <= 0.0:
            return bounded_offset, 0.0, target, command_clipped or rail_clipped

        error = target - bounded_offset
        if error == 0.0:
            saturated = command_clipped or rail_clipped or abs(bounded_offset) >= limit - 1e-12
            return bounded_offset, 0.0, target, bool(saturated)

        accel_step = max_accel * dt
        # Include the next semi-implicit position step in the stopping bound so
        # the terminal target clamp does not require an over-limit velocity jump.
        braking_speed = float(
            np.sqrt(accel_step**2 + 2.0 * max_accel * abs(error)) - accel_step
        )
        desired_velocity = float(np.copysign(min(max_rate, braking_speed), error))
        raw_delta_v = desired_velocity - velocity
        delta_v = float(np.clip(raw_delta_v, -accel_step, accel_step))
        new_velocity = float(np.clip(velocity + delta_v, -max_rate, max_rate))
        new_offset = float(bounded_offset + new_velocity * dt)

        if error * (target - new_offset) <= 0.0:
            new_offset = target
            new_velocity = 0.0
        if abs(new_offset) > limit:
            new_offset = float(np.clip(new_offset, -limit, limit))
            new_velocity = 0.0

        saturated = (
            command_clipped
            or rail_clipped
            or abs(raw_delta_v - delta_v) > 1e-12
            or abs(desired_velocity) >= max_rate - 1e-12
            or abs(new_offset) >= limit - 1e-12
        )
        return new_offset, new_velocity, target, bool(saturated)

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.array(
                [
                    self.cfg.target_x,
                    self.cfg.target_z,
                    np.deg2rad(8.0),
                    0.0,
                    0.0,
                    0.0,
                    self.cfg.hover_thrust,
                    0.0,
                ],
                dtype=float,
            )

        self.state = self._with_moving_mass_state(np.asarray(state, dtype=float))
        self.last_breakdown = None
        self.last_moving_mass_saturated = False
        return self.state.copy()

    def body_axes(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        body_up = np.array([np.sin(theta), np.cos(theta)], dtype=float)
        body_right = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
        return body_up, body_right

    def _total_com_body(self, moving_mass_offset: float) -> np.ndarray:
        """Return the instantaneous total COM in [body_right, body_up]."""
        mm = self.cfg.moving_mass
        fixed_body_mass = self.cfg.m - mm.mass_kg
        fixed_body_com = np.zeros(2, dtype=float)
        moving_mass_position = np.array(
            [moving_mass_offset, mm.moving_mass_body_up_offset_m], dtype=float
        )
        return (fixed_body_mass * fixed_body_com + mm.mass_kg * moving_mass_position) / self.cfg.m

    def force_moment_breakdown(
        self,
        state: np.ndarray,
        disturbance_force: np.ndarray | None = None,
        disturbance_moment: float = 0.0,
    ) -> ForceMomentBreakdown:
        # RigidBodyConfig is intentionally mutable. Revalidate here so a
        # post-construction mode change cannot combine geometry and legacy
        # moving-mass moments or invalidate the total-mass convention.
        self.cfg.validate()
        full_state = np.asarray(state, dtype=float)
        if full_state.shape not in ((8,), (11,)):
            raise ValueError("state must have shape (8,) or (11,)")
        x, z, theta, vx, vz, omega, thrust, vane_angle = full_state[:8]
        if disturbance_force is None:
            disturbance_force = np.zeros(2, dtype=float)
        else:
            disturbance_force = np.asarray(disturbance_force, dtype=float)
            if disturbance_force.shape != (2,):
                raise ValueError("disturbance_force must have shape (2,)")

        body_up, body_right = self.body_axes(theta)

        if self.cfg.vane_model == "linear_legacy":
            axial_efficiency = 1.0
            axial_force_mag = thrust
            side_force_mag = self.cfg.k_vane_force * thrust * vane_angle
        elif self.cfg.vane_model == "nonlinear_with_axial_loss":
            axial_efficiency = float(np.clip(1.0 - self.cfg.k_vane_axial_loss * vane_angle**2, 0.0, 1.0))
            axial_force_mag = thrust * axial_efficiency
            side_force_mag = self.cfg.k_vane_side * thrust * np.sin(vane_angle)
        elif self.cfg.vane_model == "analytical_plate":
            disk_area = np.pi * (0.5 * self.cfg.duct_diameter) ** 2
            area_ratio = self.cfg.vane_count_effective * self.cfg.vane_area / max(disk_area, 1e-9)
            k_side_analytic = self.cfg.vane_lift_slope * self.cfg.vane_efficiency * area_ratio
            axial_efficiency = float(np.clip(1.0 - self.cfg.vane_axial_loss_coefficient * vane_angle**2, 0.0, 1.0))
            axial_force_mag = thrust * axial_efficiency
            side_force_mag = k_side_analytic * thrust * np.sin(vane_angle)
        else:
            raise ValueError(f"unknown vane_model: {self.cfg.vane_model}")

        thrust_force = axial_force_mag * body_up
        vane_force = side_force_mag * body_right

        moving_mass_offset = 0.0
        moving_mass_velocity = 0.0
        moving_mass_target = 0.0
        moving_mass_saturated = False
        if self.cfg.moving_mass.enabled and full_state.shape == (11,):
            moving_mass_offset = float(full_state[8])
            moving_mass_velocity = float(full_state[9])
            moving_mass_target = float(full_state[10])
            limit = abs(float(self.cfg.moving_mass.max_offset_m))
            moving_mass_saturated = bool(
                self.last_moving_mass_saturated
                or abs(moving_mass_offset) >= limit - 1e-12
                or abs(moving_mass_target) >= limit - 1e-12
            )

        cg = np.array([x, z], dtype=float)
        total_com_geometry_active = bool(self.cfg.moving_mass.use_total_com_geometry)
        if total_com_geometry_active:
            total_com_body = self._total_com_body(moving_mass_offset)
            thrust_arm_body = -total_com_body
            vane_arm_body = np.array([0.0, -self.cfg.l], dtype=float) - total_com_body
            thrust_moment_arm = thrust_arm_body[0] * body_right + thrust_arm_body[1] * body_up
            vane_moment_arm = vane_arm_body[0] * body_right + vane_arm_body[1] * body_up
            thrust_application_point = cg + thrust_moment_arm
            vane_application_point = cg + vane_moment_arm
        else:
            total_com_body = np.zeros(2, dtype=float)
            thrust_arm_body = np.zeros(2, dtype=float)
            vane_arm_body = np.array([0.0, -self.cfg.l], dtype=float)
            thrust_application_point = cg.copy()
            vane_application_point = cg - self.cfg.l * body_up
            thrust_moment_arm = thrust_application_point - cg
            vane_moment_arm = vane_application_point - cg

        gravity_force = np.array([0.0, -self.cfg.m * self.cfg.g], dtype=float)
        relative_velocity = np.array([vx, vz], dtype=float) - np.asarray(self.cfg.wind_velocity_world, dtype=float)
        drag_force = -self.cfg.translational_drag * relative_velocity
        total_force = thrust_force + vane_force + gravity_force + drag_force + disturbance_force

        thrust_moment = float(thrust_moment_arm[1] * thrust_force[0] - thrust_moment_arm[0] * thrust_force[1])
        vane_moment = float(vane_moment_arm[1] * vane_force[0] - vane_moment_arm[0] * vane_force[1])
        damping_moment = -self.cfg.angular_damping * omega
        legacy_moving_mass_moment = 0.0
        if (
            self.cfg.moving_mass.use_legacy_gravity_offset_moment
            and self.cfg.moving_mass.enabled
            and full_state.shape == (11,)
        ):
            # Sign convention: positive moving-mass offset produces positive
            # pitch moment in this 2D pitch-assist channel. This is a
            # quasi-static CG torque placeholder, not reaction-kick dynamics.
            legacy_moving_mass_moment = float(
                self.cfg.moving_mass.mass_kg * self.cfg.g * moving_mass_offset
            )
        moving_mass_moment = legacy_moving_mass_moment
        thrust_moment_from_com_offset = thrust_moment if total_com_geometry_active else 0.0
        vane_moment_about_total_com = vane_moment if total_com_geometry_active else 0.0
        total_moment = thrust_moment + vane_moment + damping_moment + disturbance_moment + moving_mass_moment

        x_ddot = float(total_force[0] / self.cfg.m)
        z_ddot = float(total_force[1] / self.cfg.m)
        theta_ddot = float(total_moment / self.cfg.Iyy)

        return ForceMomentBreakdown(
            body_up=body_up,
            body_right=body_right,
            thrust_force=thrust_force,
            vane_force=vane_force,
            axial_efficiency=float(axial_efficiency),
            axial_force_magnitude=float(axial_force_mag),
            side_force_magnitude=float(side_force_mag),
            thrust_application_point=thrust_application_point,
            vane_application_point=vane_application_point,
            thrust_moment_arm=thrust_moment_arm,
            vane_moment_arm=vane_moment_arm,
            gravity_force=gravity_force,
            drag_force=drag_force,
            disturbance_force=disturbance_force.copy(),
            total_force=total_force,
            thrust_moment=thrust_moment,
            vane_moment=vane_moment,
            damping_moment=float(damping_moment),
            disturbance_moment=float(disturbance_moment),
            moving_mass_offset_m=moving_mass_offset,
            moving_mass_velocity_m_s=moving_mass_velocity,
            moving_mass_target_m=moving_mass_target,
            moving_mass_moment=moving_mass_moment,
            moving_mass_saturated=moving_mass_saturated,
            total_com_body_right_m=float(total_com_body[0]),
            total_com_body_up_m=float(total_com_body[1]),
            thrust_application_arm_body_right_m=float(thrust_arm_body[0]),
            thrust_application_arm_body_up_m=float(thrust_arm_body[1]),
            vane_application_arm_body_right_m=float(vane_arm_body[0]),
            vane_application_arm_body_up_m=float(vane_arm_body[1]),
            thrust_moment_from_com_offset=float(thrust_moment_from_com_offset),
            vane_moment_about_total_com=float(vane_moment_about_total_com),
            legacy_moving_mass_moment=float(legacy_moving_mass_moment),
            total_com_geometry_active=total_com_geometry_active,
            total_moment=float(total_moment),
            x_ddot=x_ddot,
            z_ddot=z_ddot,
            theta_ddot=theta_ddot,
        )

    def derivatives(
        self,
        state: np.ndarray,
        thrust_dot: float,
        vane_angle_dot: float,
        disturbance_force: np.ndarray | None = None,
        disturbance_moment: float = 0.0,
    ) -> np.ndarray:
        _x, _z, _theta, vx, vz, omega, _thrust, _vane_angle = state[:8]
        terms = self.force_moment_breakdown(state, disturbance_force, disturbance_moment)
        base = np.array(
            [
                vx,
                vz,
                omega,
                terms.x_ddot,
                terms.z_ddot,
                terms.theta_ddot,
                thrust_dot,
                vane_angle_dot,
            ],
            dtype=float,
        )
        if self.cfg.moving_mass.enabled and np.asarray(state).shape == (11,):
            offset, velocity, target = [float(v) for v in state[8:11]]
            new_offset, new_velocity, _new_target, _saturated = self._moving_mass_update(offset, velocity, target, self.cfg.dt)
            return np.concatenate([base, np.array([velocity, (new_velocity - velocity) / self.cfg.dt, 0.0])])
        return base

    def step(
        self,
        thrust_dot: float,
        vane_angle_dot: float,
        disturbance_force: np.ndarray | None = None,
        disturbance_moment: float = 0.0,
        moving_mass_target_m: float | None = None,
    ) -> np.ndarray:
        dt = self.cfg.dt
        x, z, theta, vx, vz, omega, thrust, vane_angle = self.state[:8]
        terms = self.force_moment_breakdown(self.state, disturbance_force, disturbance_moment)

        vx += terms.x_ddot * dt
        vz += terms.z_ddot * dt
        omega += terms.theta_ddot * dt
        thrust += thrust_dot * dt
        vane_angle += vane_angle_dot * dt

        thrust = float(np.clip(thrust, 0.0, self.cfg.T_max))
        vane_angle = float(np.clip(vane_angle, -self.cfg.vane_angle_max, self.cfg.vane_angle_max))

        x += vx * dt
        z += vz * dt
        theta += omega * dt

        state_values = [x, z, theta, vx, vz, omega, thrust, vane_angle]
        if self.cfg.moving_mass.enabled:
            if self.state.shape == (8,):
                self.state = self._with_moving_mass_state(self.state)
            offset, velocity, current_target = [float(v) for v in self.state[8:11]]
            command = current_target if moving_mass_target_m is None else float(moving_mass_target_m)
            new_offset, new_velocity, target, saturated = self._moving_mass_update(offset, velocity, command, dt)
            self.last_moving_mass_saturated = saturated
            state_values.extend([new_offset, new_velocity, target])
        else:
            self.last_moving_mass_saturated = False

        self.state = np.array(state_values, dtype=float)
        self.last_breakdown = self.force_moment_breakdown(self.state, disturbance_force, disturbance_moment)
        return self.state.copy()
