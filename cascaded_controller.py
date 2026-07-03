from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .config import RigidBodyConfig
    from .singlecopter_mixer import MixerOutput, SingleCopterMixer
except ImportError:  # pragma: no cover - supports direct script execution
    from config import RigidBodyConfig
    from singlecopter_mixer import MixerOutput, SingleCopterMixer


@dataclass(frozen=True)
class RigidBodyControlOutput:
    thrust_cmd: float
    vane_angle_cmd: float
    ax_target: float
    theta_target: float
    omega_target: float
    desired_moment: float
    rate_error: float
    rate_p: float
    rate_i: float
    rate_d: float
    rate_ff: float
    mixer: MixerOutput


class PositionController:
    def __init__(self, cfg: RigidBodyConfig, kp_x: float = 0.8, kd_vx: float = 1.1):
        self.cfg = cfg
        self.kp_x = kp_x
        self.kd_vx = kd_vx

    def compute(self, x: float, vx: float) -> tuple[float, float]:
        ax_target = self.kp_x * (self.cfg.target_x - x) - self.kd_vx * vx
        theta_target = np.arctan2(ax_target, self.cfg.g)
        theta_target = float(np.clip(theta_target, -self.cfg.theta_max, self.cfg.theta_max))
        return float(ax_target), theta_target


class AltitudeController:
    def __init__(self, cfg: RigidBodyConfig, kp_z: float = 6.0, kd_vz: float = 4.0):
        self.cfg = cfg
        self.kp_z = kp_z
        self.kd_vz = kd_vz

    def compute(self, z: float, vz: float, theta: float) -> float:
        vertical_accel = self.kp_z * (self.cfg.target_z - z) - self.kd_vz * vz
        body_up_z = max(0.2, float(np.cos(theta)))
        thrust_cmd = self.cfg.m * (self.cfg.g + vertical_accel) / body_up_z
        return float(np.clip(thrust_cmd, 0.0, self.cfg.T_max))


class AttitudeController:
    def __init__(self, cfg: RigidBodyConfig, kp_theta: float = 7.0):
        self.cfg = cfg
        self.kp_theta = kp_theta
        self._last_omega_target = 0.0

    def reset(self) -> None:
        self._last_omega_target = 0.0

    def compute(self, theta: float, theta_target: float) -> float:
        omega_cmd = self.kp_theta * (theta_target - theta)
        omega_cmd = float(np.clip(omega_cmd, -self.cfg.omega_target_max, self.cfg.omega_target_max))
        max_delta = self.cfg.alpha_target_max * self.cfg.dt
        omega_target = float(np.clip(omega_cmd, self._last_omega_target - max_delta, self._last_omega_target + max_delta))
        self._last_omega_target = omega_target
        return omega_target


class RatePIDController:
    def __init__(
        self,
        moment_limit: float,
        kp: float = 0.035,
        ki: float = 0.010,
        kd: float = 0.002,
        kff: float = 0.0,
        integrator_limit: float = 0.15,
    ):
        self.moment_limit = moment_limit
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.kff = kff
        self.integrator_limit = integrator_limit
        self.integrator = 0.0
        self._last_error = 0.0

    def reset(self) -> None:
        self.integrator = 0.0
        self._last_error = 0.0

    def compute(self, omega_target: float, omega: float, dt: float) -> tuple[float, float, float, float, float, float]:
        error = omega_target - omega
        derivative = (error - self._last_error) / max(dt, 1e-6)
        self._last_error = error

        p = self.kp * error
        d = self.kd * derivative
        ff = self.kff * omega_target
        candidate_i = self.integrator + self.ki * error * dt

        unclipped = p + candidate_i + d + ff
        desired_moment = float(np.clip(unclipped, -self.moment_limit, self.moment_limit))
        saturated = not np.isclose(desired_moment, unclipped)
        if not saturated or np.sign(error) != np.sign(unclipped):
            self.integrator = float(np.clip(candidate_i, -self.integrator_limit, self.integrator_limit))

        desired_moment = float(np.clip(p + self.integrator + d + ff, -self.moment_limit, self.moment_limit))
        return desired_moment, error, p, self.integrator, d, ff


class ArduPilotLikeController:
    def __init__(self, cfg: RigidBodyConfig):
        self.cfg = cfg
        moment_limit = abs(cfg.k_moment) * cfg.hover_thrust * cfg.vane_angle_max
        self.position = PositionController(cfg)
        self.altitude = AltitudeController(cfg)
        self.attitude = AttitudeController(cfg)
        self.rate = RatePIDController(moment_limit=moment_limit)
        self.mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)

    def reset(self) -> None:
        self.attitude.reset()
        self.rate.reset()

    def compute(self, state: np.ndarray) -> RigidBodyControlOutput:
        x, z, theta, vx, vz, omega, thrust, _vane_angle = state
        ax_target, theta_target = self.position.compute(x, vx)
        thrust_cmd = self.altitude.compute(z, vz, theta)
        omega_target = self.attitude.compute(theta, theta_target)
        desired_moment, rate_error, p, i, d, ff = self.rate.compute(omega_target, omega, self.cfg.dt)
        mixer_output = self.mixer.mix(desired_moment, thrust)

        return RigidBodyControlOutput(
            thrust_cmd=thrust_cmd,
            vane_angle_cmd=mixer_output.vane_angle_cmd,
            ax_target=ax_target,
            theta_target=theta_target,
            omega_target=omega_target,
            desired_moment=desired_moment,
            rate_error=rate_error,
            rate_p=p,
            rate_i=i,
            rate_d=d,
            rate_ff=ff,
            mixer=mixer_output,
        )
