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
class ControllerConfig:
    """ArduCopter-inspired controller parameters for the 2D simulator."""

    loit_speed_ms: float = 1.5
    loit_acc_max_mss: float = 1.0
    loit_brk_acc_mss: float = 1.0
    loit_brk_delay_s: float = 0.4
    loit_brk_jerk_msss: float = 2.5
    loit_capture_vx_threshold_ms: float = 0.08
    loit_capture_desired_vx_threshold_ms: float = 0.02
    loit_capture_persistent: bool = False
    loit_shaper_clamp_target: bool = False
    loit_capture_without_jump: bool = False
    loit_angle_max_deg: float = 15.0
    psc_ne_pos_p: float = 0.8
    psc_ne_vel_p: float = 1.1
    psc_ne_vel_i: float = 0.0
    psc_ne_vel_d: float = 0.0
    psc_accel_xy_max_mss: float = 2.0
    psc_jerk_xy_max_msss: float = 4.0

    pilot_speed_up_ms: float = 0.8
    pilot_speed_dn_ms: float = 0.6
    pilot_accel_z_mss: float = 1.0
    thr_dz: float = 0.1
    psc_posz_p: float = 1.6
    psc_velz_p: float = 4.0
    psc_velz_i: float = 0.0
    psc_velz_d: float = 0.0
    psc_accz_p: float = 1.0
    psc_accz_i: float = 0.0
    psc_accz_d: float = 0.0

    atc_angle_max_deg: float = 20.0
    atc_input_tc: float = 0.15
    atc_ang_pit_p: float = 7.0
    atc_rat_pit_p: float = 0.035
    atc_rat_pit_i: float = 0.010
    atc_rat_pit_d: float = 0.002
    atc_rat_pit_ff: float = 0.0
    atc_rat_pit_imax: float = 0.15

    enable_noise: bool = False
    random_seed: int = 0
    sensor_theta_noise_std_deg: float = 0.0
    sensor_omega_noise_std_deg_s: float = 0.0
    sensor_x_noise_std: float = 0.0
    sensor_z_noise_std: float = 0.0
    servo_bias_deg: float = 0.0
    vane_effectiveness_scale: float = 1.0
    motor_thrust_bias: float = 0.0
    disturbance_wind_noise_std: float = 0.0
    controller_update_jitter: float = 0.0

    @property
    def loit_angle_max(self) -> float:
        import math

        return math.radians(self.loit_angle_max_deg)

    @property
    def atc_angle_max(self) -> float:
        import math

        return math.radians(self.atc_angle_max_deg)


@dataclass
class MovingMassPitchAssistConfig:
    """Disabled-by-default 2D pitch-axis moving mass assist."""

    enabled: bool = False
    mass_kg: float = 0.5
    max_offset_m: float = 0.05
    max_rate_m_s: float = 0.20
    max_accel_m_s2: float = 1.0
    initial_offset_m: float = 0.0
    use_total_com_geometry: bool = False
    use_legacy_gravity_offset_moment: bool = True
    moving_mass_body_up_offset_m: float = 0.12


@dataclass
class RigidBodyConfig:
    """Parameters for the single-fan rigid-body model and optional COM geometry."""

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
    duct_diameter: float = 0.18
    vane_area: float = 0.004
    vane_count_effective: float = 2.0
    vane_lift_slope: float = 2.0
    vane_efficiency: float = 0.55
    vane_axial_loss_coefficient: float = 1.6
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

    moving_mass: MovingMassPitchAssistConfig = field(default_factory=MovingMassPitchAssistConfig)

    x_limit_abs: float = 8.0
    z_limit_min: float = -0.5
    z_limit_max: float = 8.0
    theta_limit_abs: float = 4.0 * 3.141592653589793
    velocity_limit_abs: float = 30.0
    omega_limit_abs: float = 60.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate moving-mass mode and mass-accounting invariants."""
        mm = self.moving_mass
        if mm.use_total_com_geometry and mm.use_legacy_gravity_offset_moment:
            raise ValueError(
                "moving_mass.use_total_com_geometry and "
                "moving_mass.use_legacy_gravity_offset_moment cannot both be enabled"
            )
        if (mm.enabled or mm.use_total_com_geometry) and mm.mass_kg <= 0.0:
            raise ValueError("moving_mass.mass_kg must be greater than zero")
        if mm.use_total_com_geometry and mm.mass_kg >= self.m:
            raise ValueError(
                "moving_mass.mass_kg must be less than RigidBodyConfig.m when total-COM geometry is enabled"
            )

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

    actuator_lab_mass_limit_m: float = 0.010
    actuator_lab_mass_step_m: float = 0.001
    actuator_lab_mass_coarse_step_m: float = 0.005
    actuator_lab_vane_limit_deg: float = 5.0
    actuator_lab_vane_step_deg: float = 0.5
    actuator_lab_vane_coarse_step_deg: float = 2.0
    actuator_lab_moment_deadband_Nm: float = 0.002

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

    vane_visual_scale: float = 2.5
    vane_visual_length_m: float = 0.45
    vane_visual_offset_m: float = 0.08
    show_vane_command_ghost: bool = True
    show_vane_overlay: bool = True
    show_target_marker: bool = True
    show_loiter_error_vector: bool = True
    show_desired_accel_arrow: bool = True
    show_theta_target_vector: bool = True

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
