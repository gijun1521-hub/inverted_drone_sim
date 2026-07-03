from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MixerOutput:
    vane_angle_cmd: float
    requested_moment: float
    achievable_moment: float
    unattainable_moment: float
    saturated: bool
    thrust_for_control: float
    min_thrust_for_authority: float


class SingleCopterMixer:
    def __init__(self, k_moment: float, vane_angle_limit: float, thrust_control_floor: float):
        self.k_moment = k_moment
        self.vane_angle_limit = vane_angle_limit
        self.thrust_control_floor = thrust_control_floor

    def mix(self, desired_moment: float, thrust: float) -> MixerOutput:
        thrust_for_control = max(thrust, self.thrust_control_floor)
        denom = self.k_moment * thrust_for_control

        if abs(denom) < 1e-9:
            raw_cmd = 0.0
        else:
            raw_cmd = desired_moment / denom

        vane_angle_cmd = float(np.clip(raw_cmd, -self.vane_angle_limit, self.vane_angle_limit))
        achievable_moment = float(self.k_moment * thrust_for_control * vane_angle_cmd)
        unattainable_moment = float(desired_moment - achievable_moment)
        max_moment = abs(self.k_moment) * self.vane_angle_limit
        min_thrust = abs(desired_moment) / max_moment if max_moment > 1e-9 else np.inf

        return MixerOutput(
            vane_angle_cmd=vane_angle_cmd,
            requested_moment=float(desired_moment),
            achievable_moment=achievable_moment,
            unattainable_moment=unattainable_moment,
            saturated=not np.isclose(vane_angle_cmd, raw_cmd),
            thrust_for_control=float(thrust_for_control),
            min_thrust_for_authority=float(min_thrust),
        )
