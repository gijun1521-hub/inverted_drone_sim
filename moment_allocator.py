from __future__ import annotations

from dataclasses import dataclass

try:
    from .config import RigidBodyConfig
    from .singlecopter_mixer import SingleCopterMixer
except ImportError:  # pragma: no cover
    from config import RigidBodyConfig
    from singlecopter_mixer import SingleCopterMixer


@dataclass(frozen=True)
class ActuatorMomentCommands:
    vane_angle_cmd: float
    moving_mass_cmd: float
    moment_to_vane: float
    moment_to_moving_mass: float
    unavailable_moment: float
    saturated: bool


class MomentAllocator:
    def __init__(self, mode: str = "vane_only"):
        self.mode = mode

    def allocate(self, desired_moment: float, state, cfg: RigidBodyConfig) -> ActuatorMomentCommands:
        if self.mode == "vane_only":
            mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)
            mixed = mixer.mix(desired_moment, float(state[6]))
            return ActuatorMomentCommands(
                vane_angle_cmd=mixed.vane_angle_cmd,
                moving_mass_cmd=0.0,
                moment_to_vane=mixed.physically_achievable_moment,
                moment_to_moving_mass=0.0,
                unavailable_moment=mixed.unattainable_moment,
                saturated=mixed.saturated,
            )
        if self.mode in {"moving_mass_only", "hybrid"}:
            return ActuatorMomentCommands(0.0, 0.0, 0.0, 0.0, float(desired_moment), True)
        raise ValueError(f"unknown allocator mode: {self.mode}")
