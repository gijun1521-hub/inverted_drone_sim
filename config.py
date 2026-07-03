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


@dataclass
class RigidBodyConfig:
    """Parameters for the CG-referenced single-fan rigid-body model."""

    H: float = 0.50
    W: float = 0.10
    m: float = 1.50
    g: float = 9.81
    dt: float = 0.005

    T_max_factor: float = 2.5
    motor_time_constant: float = 0.08

    vane_angle_max_deg: float = 25.0
    vane_rate_limit_deg_s: float = 300.0
    servo_time_constant: float = 0.04
    servo_deadband_deg: float = 0.0
    servo_delay: float = 0.0

    k_vane_force: float = 0.75
    translational_drag: float = 0.10
    angular_damping: float = 0.04

    target_x: float = 0.0
    target_z: float = 1.0
    target_theta: float = 0.0
    max_time: float = 8.0

    theta_max_deg: float = 20.0
    omega_target_max: float = 5.0
    alpha_target_max: float = 30.0
    thrust_control_floor_factor: float = 0.20
    attitude_priority_thrust_mixing: bool = False

    @property
    def l(self) -> float:
        return self.H / 2

    @property
    def Iyy(self) -> float:
        return (1 / 12) * self.m * (self.H**2 + self.W**2)

    @property
    def T_max(self) -> float:
        return self.T_max_factor * self.m * self.g

    @property
    def hover_thrust(self) -> float:
        return self.m * self.g

    @property
    def vane_angle_max(self) -> float:
        import math

        return math.radians(self.vane_angle_max_deg)

    @property
    def vane_rate_limit(self) -> float:
        import math

        return math.radians(self.vane_rate_limit_deg_s)

    @property
    def servo_deadband(self) -> float:
        import math

        return math.radians(self.servo_deadband_deg)

    @property
    def theta_max(self) -> float:
        import math

        return math.radians(self.theta_max_deg)

    @property
    def k_moment(self) -> float:
        return -self.l * self.k_vane_force

    @property
    def thrust_control_floor(self) -> float:
        return self.thrust_control_floor_factor * self.hover_thrust
