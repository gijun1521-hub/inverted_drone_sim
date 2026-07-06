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
    total_moment: float
    x_ddot: float
    z_ddot: float
    theta_ddot: float


class RigidBodySingleFan2D:
    """CG-referenced 2D rigid body with fan thrust and a lower vane force."""

    def __init__(self, config: RigidBodyConfig):
        self.cfg = config
        self.state = np.zeros(8, dtype=float)
        self.last_breakdown: ForceMomentBreakdown | None = None

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

        self.state = np.asarray(state, dtype=float).copy()
        if self.state.shape != (8,):
            raise ValueError("state must have shape (8,)")
        self.last_breakdown = None
        return self.state.copy()

    def body_axes(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        body_up = np.array([np.sin(theta), np.cos(theta)], dtype=float)
        body_right = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
        return body_up, body_right

    def force_moment_breakdown(
        self,
        state: np.ndarray,
        disturbance_force: np.ndarray | None = None,
        disturbance_moment: float = 0.0,
    ) -> ForceMomentBreakdown:
        x, z, theta, vx, vz, omega, thrust, vane_angle = np.asarray(state, dtype=float)
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
        cg = np.array([x, z], dtype=float)
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
        total_moment = thrust_moment + vane_moment + damping_moment + disturbance_moment

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
        _x, _z, _theta, vx, vz, omega, _thrust, _vane_angle = state
        terms = self.force_moment_breakdown(state, disturbance_force, disturbance_moment)
        return np.array(
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

    def step(
        self,
        thrust_dot: float,
        vane_angle_dot: float,
        disturbance_force: np.ndarray | None = None,
        disturbance_moment: float = 0.0,
    ) -> np.ndarray:
        dt = self.cfg.dt
        x, z, theta, vx, vz, omega, thrust, vane_angle = self.state
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

        self.state = np.array([x, z, theta, vx, vz, omega, thrust, vane_angle], dtype=float)
        self.last_breakdown = self.force_moment_breakdown(self.state, disturbance_force, disturbance_moment)
        return self.state.copy()
