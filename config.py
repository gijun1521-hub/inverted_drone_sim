from dataclasses import dataclass


@dataclass
class DroneConfig:
    """Tunable parameters for the 2D moving-base inverted drone model."""

    H: float = 0.50
    W: float = 0.10
    m: float = 1.50
    g: float = 9.81
    dt: float = 0.01

    T_max_factor: float = 2.5
    damping: float = 0.35
    ax_cmd_max: float = 15.0

    # Legacy notebook compatibility only. The moving-base model no longer
    # uses vane angle or vane torque as control inputs.
    vane_angle_max_deg: float = 0.0
    vane_torque_coeff: float = 0.0

    target_x: float = 0.0
    target_z: float = 1.0
    target_theta: float = 0.0

    max_time: float = 10.0

    @property
    def l(self) -> float:
        return self.H / 2

    @property
    def I_cg(self) -> float:
        return (1 / 12) * self.m * (self.H**2 + self.W**2)

    @property
    def I(self) -> float:
        return self.I_cg + self.m * self.l**2

    @property
    def T_max(self) -> float:
        return self.T_max_factor * self.m * self.g

    @property
    def hover_throttle(self) -> float:
        return self.m * self.g / self.T_max
