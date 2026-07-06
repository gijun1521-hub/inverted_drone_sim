from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import MovingMassConfig
except ImportError:  # pragma: no cover
    from config import MovingMassConfig


@dataclass(frozen=True)
class MovingMassBreakdown:
    total_cg_world: np.ndarray
    moving_mass_world: np.ndarray
    q_cmd: float
    qddot_mass: float
    reaction_moment_body: float
    thrust_moment: float
    total_external_moment: float
    angular_momentum: float
    q_limited: bool
    q_rate_limited: bool


class MovingMassSingleFan2D:
    """Experimental 2D single-fan plant with a rotating moving mass.

    The internal q acceleration creates equal/opposite angular momentum exchange
    between body and moving mass. This model is intentionally separate from the
    single-rigid-body vane model.
    """

    def __init__(self, cfg: MovingMassConfig):
        self.cfg = cfg
        self.state = np.zeros(10, dtype=float)
        self.last_breakdown: MovingMassBreakdown | None = None

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is None:
            state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, self.cfg.thrust, 0.0, 0.0, 0.0])
        self.state = np.asarray(state, dtype=float).copy()
        if self.state.shape != (10,):
            raise ValueError("state must have shape (10,)")
        return self.state.copy()

    def body_axes(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        body_up = np.array([np.sin(theta), np.cos(theta)], dtype=float)
        body_right = np.array([np.cos(theta), -np.sin(theta)], dtype=float)
        return body_up, body_right

    def moving_mass_position_body(self, q: float) -> np.ndarray:
        hinge = np.asarray(self.cfg.hinge_position_body, dtype=float)
        if self.cfg.moving_mass_geometry == "rotating":
            offset = np.asarray(self.cfg.mass_center_offset_body, dtype=float)
            c, s = np.cos(q), np.sin(q)
            rot = np.array([[c, -s], [s, c]], dtype=float)
            return hinge + rot @ offset
        if self.cfg.moving_mass_geometry == "sliding":
            axis = np.asarray(self.cfg.rail_axis_body, dtype=float)
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            return hinge + axis * np.clip(q, -self.cfg.rail_limit, self.cfg.rail_limit)
        raise ValueError(f"unknown moving_mass_geometry: {self.cfg.moving_mass_geometry}")

    def _world_from_body(self, theta: float, p_body: np.ndarray) -> np.ndarray:
        body_up, body_right = self.body_axes(theta)
        return p_body[0] * body_right + p_body[1] * body_up

    def total_cg_world(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, z, theta, *_rest, q, _qdot = state
        body_cg = np.array([x, z], dtype=float)
        p_mass_body = self.moving_mass_position_body(q)
        p_mass_world = body_cg + self._world_from_body(theta, p_mass_body)
        total_cg = (
            self.cfg.m_body_without_battery * body_cg + self.cfg.m_moving * p_mass_world
        ) / self.cfg.m_total
        return total_cg, p_mass_world

    def step(self, q_cmd: float, thrust: float | None = None) -> np.ndarray:
        x, z, theta, vx, vz, omega, state_thrust, vane_angle, q, qdot = self.state
        thrust = state_thrust if thrust is None else thrust
        dt = self.cfg.dt

        q_cmd = float(np.clip(q_cmd, -self.cfg.q_limit, self.cfg.q_limit))
        qdot_des = (q_cmd - q) / max(self.cfg.q_servo_time_constant, 1e-6)
        qdot_des = float(np.clip(qdot_des, -self.cfg.q_rate_limit, self.cfg.q_rate_limit))
        qddot = float(np.clip((qdot_des - qdot) / dt, -self.cfg.q_accel_limit, self.cfg.q_accel_limit))

        I_body_total = self.cfg.I_body_without_battery + self.cfg.I_moving_about_hinge
        alpha_reaction = -(self.cfg.I_moving_about_hinge * qddot) / max(I_body_total, 1e-9)
        reaction_moment = self.cfg.I_body_without_battery * alpha_reaction

        body_up, _body_right = self.body_axes(theta)
        force = np.array([thrust * body_up[0], thrust * body_up[1] - self.cfg.m_total * self.cfg.g])
        ax, az = force / self.cfg.m_total

        total_cg, _mass_world = self.total_cg_world(self.state)
        body_cg = np.array([x, z], dtype=float)
        thrust_point = body_cg + self._world_from_body(theta, np.asarray(self.cfg.thrust_offset_body, dtype=float))
        arm = thrust_point - total_cg
        thrust_force = thrust * body_up
        thrust_moment = float(arm[0] * thrust_force[1] - arm[1] * thrust_force[0])
        alpha_external = thrust_moment / max(I_body_total, 1e-9)

        vx += ax * dt
        vz += az * dt
        omega += (alpha_reaction + alpha_external) * dt
        qdot += qddot * dt
        qdot = float(np.clip(qdot, -self.cfg.q_rate_limit, self.cfg.q_rate_limit))
        q += qdot * dt
        q_limited = False
        if abs(q) > self.cfg.q_limit:
            q = float(np.clip(q, -self.cfg.q_limit, self.cfg.q_limit))
            qdot = 0.0
            q_limited = True

        x += vx * dt
        z += vz * dt
        theta += omega * dt
        self.state = np.array([x, z, theta, vx, vz, omega, thrust, vane_angle, q, qdot], dtype=float)
        total_cg, moving_world = self.total_cg_world(self.state)
        angular_momentum = I_body_total * omega + self.cfg.I_moving_about_hinge * qdot
        self.last_breakdown = MovingMassBreakdown(
            total_cg_world=total_cg,
            moving_mass_world=moving_world,
            q_cmd=q_cmd,
            qddot_mass=qddot,
            reaction_moment_body=reaction_moment,
            thrust_moment=thrust_moment,
            total_external_moment=thrust_moment,
            angular_momentum=float(angular_momentum),
            q_limited=q_limited,
            q_rate_limited=abs(qdot_des) >= self.cfg.q_rate_limit - 1e-9,
        )
        return self.state.copy()
