from dataclasses import dataclass
from dataclasses import field


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
    k_vane_side: float = 0.75
    k_vane_axial_loss: float = 1.6
    vane_model: str = "linear_legacy"
    thrust_curve_model: str = "linear"
    thrust_curve_coefficients: tuple[float, ...] = ()
    thrust_curve_lookup_csv: str = ""
    translational_drag: float = 0.10
    angular_damping: float = 0.04
    wind_velocity_world: tuple[float, float] = (0.0, 0.0)
    gust_force_world: tuple[float, float] = (0.0, 0.0)
    gust_moment: float = 0.0
    gust_duration_s: float = 0.0

    target_x: float = 0.0
    target_z: float = 1.0
    target_theta: float = 0.0
    max_time: float = 8.0

    theta_max_deg: float = 20.0
    omega_target_max: float = 5.0
    alpha_target_max: float = 30.0
    thrust_control_floor_factor: float = 0.20
    attitude_priority_thrust_mixing: bool = False

    x_limit_abs: float = 8.0
    z_limit_min: float = -0.5
    z_limit_max: float = 8.0
    theta_limit_abs: float = 4.0 * 3.141592653589793
    velocity_limit_abs: float = 30.0
    omega_limit_abs: float = 60.0

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


@dataclass
class MovingMassConfig:
    H: float = 0.50
    W: float = 0.10
    g: float = 9.81
    dt: float = 0.005
    m_body_without_battery: float = 1.0
    m_moving: float = 0.5
    I_body_without_battery: float = 0.025
    I_moving_about_hinge: float = 0.004
    moving_mass_geometry: str = "rotating"
    q_limit: float = 0.45
    q_rate_limit: float = 4.0
    q_accel_limit: float = 30.0
    q_servo_time_constant: float = 0.08
    q_command_delay: float = 0.0
    hinge_position_body: tuple[float, float] = (0.0, 0.15)
    mass_center_offset_body: tuple[float, float] = (0.0, 0.12)
    rail_axis_body: tuple[float, float] = (1.0, 0.0)
    rail_limit: float = 0.12
    thrust_offset_body: tuple[float, float] = (0.0, -0.25)
    vane_offset_body: tuple[float, float] = (0.0, -0.25)
    thrust: float = 14.715

    @property
    def m_total(self) -> float:
        return self.m_body_without_battery + self.m_moving


@dataclass
class ResetPreset:
    name: str
    state: list[float]


@dataclass
class InteractiveSimConfig:
    """Runtime configuration for the pygame interactive simulator."""

    physics_dt: float = 0.005
    controller_dt: float = 0.01
    render_rate: float = 60.0
    initial_speed: float = 1.0
    slow_motion_speed: float = 0.25
    speed_step: float = 0.25
    min_speed: float = 0.05
    max_speed: float = 4.0

    throttle_slew_per_s: float = 0.6
    vane_slew_deg_s: float = 80.0
    theta_target_slew_deg_s: float = 60.0
    omega_target_slew_deg_s: float = 180.0
    command_return_rate: float = 3.5

    direct_vane_max_deg: float = 20.0
    manual_theta_max_deg: float = 18.0
    manual_omega_max_deg_s: float = 120.0

    disturbance_force_x_N: float = 4.0
    disturbance_force_z_N: float = 4.0
    disturbance_moment_Nm: float = 0.20
    impulse_duration_s: float = 0.15

    pixels_per_meter: float = 180.0
    zoom_step: float = 1.15
    min_pixels_per_meter: float = 60.0
    max_pixels_per_meter: float = 420.0
    trace_length: int = 900
    log_directory: str = "results/interactive_logs"

    presets: dict[str, ResetPreset] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.presets:
            return
        rb = RigidBodyConfig(dt=self.physics_dt)
        self.presets = {
            "F1": ResetPreset(
                "upright hover",
                [0.0, rb.target_z, 0.0, 0.0, 0.0, 0.0, rb.hover_thrust, 0.0],
            ),
            "F2": ResetPreset(
                "positive initial tilt",
                [0.0, rb.target_z, 0.15, 0.0, 0.0, 0.0, rb.hover_thrust, 0.0],
            ),
            "F3": ResetPreset(
                "negative initial tilt",
                [0.0, rb.target_z, -0.15, 0.0, 0.0, 0.0, rb.hover_thrust, 0.0],
            ),
            "F4": ResetPreset(
                "horizontal velocity",
                [0.0, rb.target_z, 0.0, 1.0, 0.0, 0.0, rb.hover_thrust, 0.0],
            ),
            "F5": ResetPreset(
                "angular velocity",
                [0.0, rb.target_z, 0.0, 0.0, 0.0, 1.0, rb.hover_thrust, 0.0],
            ),
            "F6": ResetPreset(
                "low thrust authority",
                [0.0, rb.target_z, 0.0, 0.0, 0.0, 0.0, 0.45 * rb.hover_thrust, 0.0],
            ),
        }
