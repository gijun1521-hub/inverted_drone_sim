from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotorOutput:
    thrust_cmd: float
    thrust_dot: float
    saturated: bool


class FirstOrderMotor:
    def __init__(self, thrust_max: float, time_constant: float):
        self.thrust_max = thrust_max
        self.time_constant = time_constant

    def update(self, thrust: float, thrust_cmd: float) -> MotorOutput:
        clipped_cmd = float(np.clip(thrust_cmd, 0.0, self.thrust_max))
        tau = max(self.time_constant, 1e-6)
        thrust_dot = (clipped_cmd - thrust) / tau
        return MotorOutput(
            thrust_cmd=clipped_cmd,
            thrust_dot=float(thrust_dot),
            saturated=not np.isclose(clipped_cmd, thrust_cmd),
        )


@dataclass(frozen=True)
class ServoOutput:
    vane_angle_cmd: float
    delayed_cmd: float
    vane_angle_dot: float
    angle_saturated: bool
    rate_saturated: bool


class VaneServo:
    def __init__(
        self,
        dt: float,
        angle_limit: float,
        rate_limit: float,
        time_constant: float,
        deadband: float = 0.0,
        command_delay: float = 0.0,
    ):
        self.dt = dt
        self.angle_limit = angle_limit
        self.rate_limit = rate_limit
        self.time_constant = time_constant
        self.deadband = deadband
        delay_steps = max(0, int(round(command_delay / dt)))
        self._commands = deque([0.0] * (delay_steps + 1), maxlen=delay_steps + 1)

    def reset(self) -> None:
        for i in range(len(self._commands)):
            self._commands[i] = 0.0

    def update(self, vane_angle: float, vane_angle_cmd: float) -> ServoOutput:
        clipped_cmd = float(np.clip(vane_angle_cmd, -self.angle_limit, self.angle_limit))
        self._commands.append(clipped_cmd)
        delayed_cmd = float(self._commands[0])

        error = delayed_cmd - vane_angle
        if abs(error) <= self.deadband:
            desired_rate = 0.0
        else:
            desired_rate = error / max(self.time_constant, 1e-6)

        vane_angle_dot = float(np.clip(desired_rate, -self.rate_limit, self.rate_limit))
        return ServoOutput(
            vane_angle_cmd=clipped_cmd,
            delayed_cmd=delayed_cmd,
            vane_angle_dot=vane_angle_dot,
            angle_saturated=not np.isclose(clipped_cmd, vane_angle_cmd),
            rate_saturated=not np.isclose(vane_angle_dot, desired_rate),
        )
