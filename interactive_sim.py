from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
import argparse

import numpy as np

try:
    from .actuators import FirstOrderMotor, MotorOutput, ServoOutput, VaneServo
    from .cascaded_controller import AttitudeController, RatePIDController, RigidBodyControlOutput
    from .config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from .interactive_logging import InteractiveCSVLogger, interactive_row
    from .math_utils import wrap_pi
    from .params import load_interactive_config
    from .rigid_body_model import ForceMomentBreakdown, RigidBodySingleFan2D
    from .safety import check_safety
    from .singlecopter_mixer import MixerOutput, SingleCopterMixer
    from .thrust_curve import ThrottleToThrustModel
except ImportError:  # pragma: no cover - supports direct script execution
    from actuators import FirstOrderMotor, MotorOutput, ServoOutput, VaneServo
    from cascaded_controller import AttitudeController, RatePIDController, RigidBodyControlOutput
    from config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from interactive_logging import InteractiveCSVLogger, interactive_row
    from math_utils import wrap_pi
    from params import load_interactive_config
    from rigid_body_model import ForceMomentBreakdown, RigidBodySingleFan2D
    from safety import check_safety
    from singlecopter_mixer import MixerOutput, SingleCopterMixer
    from thrust_curve import ThrottleToThrustModel


class ControlMode(str, Enum):
    DIRECT = "DIRECT"
    RATE = "RATE"
    STABILIZE = "STABILIZE"
    ALT_HOLD = "ALT_HOLD"
    LOITER = "LOITER"
    ACTUATOR_LAB = "ACTUATOR_LAB"


@dataclass
class ManualCommands:
    throttle: float
    direct_vane: float = 0.0
    theta_target: float = 0.0
    omega_target: float = 0.0
    stick_x: float = 0.0
    stick_z: float = 0.0
    moving_mass_target_m: float = 0.0

    def zero(self, hover_throttle: float) -> None:
        self.throttle = hover_throttle
        self.direct_vane = 0.0
        self.theta_target = 0.0
        self.omega_target = 0.0
        self.stick_x = 0.0
        self.stick_z = 0.0
        self.moving_mass_target_m = 0.0


@dataclass
class Disturbance:
    force: np.ndarray
    moment: float
    impulse_time_remaining: float = 0.0
    impulse_force: np.ndarray | None = None
    impulse_moment: float = 0.0

    def combined_force(self) -> np.ndarray:
        if self.impulse_time_remaining > 0.0 and self.impulse_force is not None:
            return self.force + self.impulse_force
        return self.force

    def combined_moment(self) -> float:
        if self.impulse_time_remaining > 0.0:
            return self.moment + self.impulse_moment
        return self.moment

    def tick(self, dt: float) -> None:
        self.impulse_time_remaining = max(0.0, self.impulse_time_remaining - dt)
        if self.impulse_time_remaining <= 0.0:
            self.impulse_force = None
            self.impulse_moment = 0.0


@dataclass
class RuntimeTargets:
    target_x: float = 0.0
    target_z: float = 1.0
    desired_vx: float = 0.0
    shaped_desired_vx: float = 0.0
    position_velocity_correction: float = 0.0
    total_desired_vx: float = 0.0
    desired_vz: float = 0.0
    desired_ax: float = 0.0
    desired_az: float = 0.0
    theta_target: float = 0.0
    theta_target_limited: float = 0.0
    loiter_active: bool = False
    loiter_braking_active: bool = False
    altitude_hold_active: bool = False
    throttle_deadband_active: bool = True
    target_x_rate_cmd: float = 0.0
    target_z_rate_cmd: float = 0.0
    target_capture_event: bool = False
    target_capture_count: int = 0
    target_capture_pending: bool = False
    target_step_event: bool = False

    def capture(self, state: np.ndarray) -> None:
        self.target_x = float(state[0])
        self.target_z = float(state[1])
        self.desired_vx = 0.0
        self.shaped_desired_vx = 0.0
        self.position_velocity_correction = 0.0
        self.total_desired_vx = 0.0
        self.desired_vz = 0.0
        self.desired_ax = 0.0
        self.desired_az = 0.0
        self.theta_target = 0.0
        self.theta_target_limited = 0.0
        self.loiter_braking_active = False
        self.throttle_deadband_active = True
        self.target_x_rate_cmd = 0.0
        self.target_z_rate_cmd = 0.0
        self.target_capture_event = False
        self.target_capture_count = 0
        self.target_capture_pending = False
        self.target_step_event = False


class LoiterInputShaper:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.desired_vx = 0.0
        self._last_accel = 0.0
        self._release_time = 0.0
        self.braking_active = False
        self.capture_pending = False

    def reset(self, desired_vx: float = 0.0) -> None:
        self.desired_vx = float(desired_vx)
        self._last_accel = 0.0
        self._release_time = 0.0
        self.braking_active = False
        self.capture_pending = False

    def mark_target_captured(self) -> None:
        self.capture_pending = False
        self.braking_active = False

    @staticmethod
    def _rate_limit(value: float, target: float, max_delta: float) -> float:
        return float(np.clip(target, value - max_delta, value + max_delta))

    def update(self, stick_x: float, dt: float) -> float:
        if abs(stick_x) > 1e-3:
            self._release_time = 0.0
            self.braking_active = False
            self.capture_pending = False
            target = float(np.clip(stick_x, -1.0, 1.0) * self.cfg.loit_speed_ms)
            accel_limit = self.cfg.loit_acc_max_mss
            jerk_limit = self.cfg.psc_jerk_xy_max_msss
        else:
            self._release_time += dt
            if self.cfg.loit_capture_persistent:
                if abs(self.desired_vx) > self.cfg.loit_capture_desired_vx_threshold_ms:
                    self.capture_pending = True
                self.braking_active = self._release_time >= self.cfg.loit_brk_delay_s and self.capture_pending
            else:
                self.capture_pending = (
                    self._release_time >= self.cfg.loit_brk_delay_s
                    and abs(self.desired_vx) > self.cfg.loit_capture_desired_vx_threshold_ms
                )
                self.braking_active = self.capture_pending
            target = 0.0
            accel_limit = self.cfg.loit_brk_acc_mss if self.braking_active else self.cfg.loit_acc_max_mss
            jerk_limit = self.cfg.loit_brk_jerk_msss
        previous_desired_vx = self.desired_vx
        raw_accel = (target - self.desired_vx) / max(dt, 1e-6)
        accel = self._rate_limit(self._last_accel, raw_accel, max(jerk_limit, 1e-6) * dt)
        accel = float(np.clip(accel, -accel_limit, accel_limit))
        self.desired_vx += accel * dt
        if (
            self.cfg.loit_shaper_clamp_target
            and previous_desired_vx != target
            and (previous_desired_vx - target) * (self.desired_vx - target) <= 0.0
        ):
            self.desired_vx = target
            accel = 0.0
        if target == 0.0 and abs(self.desired_vx) < 0.015 and self.braking_active:
            self.desired_vx = 0.0
        self._last_accel = accel
        return self.desired_vx

def _dummy_mixer_output() -> MixerOutput:
    return MixerOutput(0.0, 0.0, 0.0, 0.0, 0.0, False, False, False, 0.0, 0.0, 0.0)


def _control_output(
    thrust_cmd: float,
    vane_angle_cmd: float,
    ax_target: float = 0.0,
    theta_target: float = 0.0,
    omega_target: float = 0.0,
    desired_moment: float = 0.0,
    rate_error: float = 0.0,
    p: float = 0.0,
    i: float = 0.0,
    d: float = 0.0,
    ff: float = 0.0,
    anti_windup_correction: float = 0.0,
    integrator_inhibited: bool = False,
    mixer: MixerOutput | None = None,
) -> RigidBodyControlOutput:
    return RigidBodyControlOutput(
        thrust_cmd=float(thrust_cmd),
        vane_angle_cmd=float(vane_angle_cmd),
        ax_target=float(ax_target),
        theta_target=float(theta_target),
        omega_target=float(omega_target),
        desired_moment=float(desired_moment),
        rate_error=float(rate_error),
        rate_p=float(p),
        rate_i=float(i),
        rate_d=float(d),
        rate_ff=float(ff),
        anti_windup_correction=float(anti_windup_correction),
        integrator_inhibited=bool(integrator_inhibited),
        mixer=mixer or _dummy_mixer_output(),
    )


class ManualControlSystem:
    def __init__(self, rb_cfg: RigidBodyConfig, controller_cfg: ControllerConfig | None = None):
        self.cfg = rb_cfg
        self.controller_cfg = controller_cfg or ControllerConfig()
        moment_limit = abs(rb_cfg.k_moment) * rb_cfg.hover_thrust * rb_cfg.vane_angle_max
        self.attitude = AttitudeController(rb_cfg, kp_theta=self.controller_cfg.atc_ang_pit_p)
        self.rate = RatePIDController(
            moment_limit=moment_limit,
            kp=self.controller_cfg.atc_rat_pit_p,
            ki=self.controller_cfg.atc_rat_pit_i,
            kd=self.controller_cfg.atc_rat_pit_d,
            kff=self.controller_cfg.atc_rat_pit_ff,
            integrator_limit=self.controller_cfg.atc_rat_pit_imax,
        )
        self.mixer = SingleCopterMixer(rb_cfg.k_moment, rb_cfg.vane_angle_max, rb_cfg.thrust_control_floor)
        self.thrust_curve = ThrottleToThrustModel(rb_cfg)

    def reset(self) -> None:
        self.attitude.reset()
        self.rate.reset()

    def _attitude_rate_mix(
        self,
        state: np.ndarray,
        thrust_cmd: float,
        theta_target: float,
        controller_dt: float,
        ax_target: float = 0.0,
        omega_target: float | None = None,
    ) -> RigidBodyControlOutput:
        if omega_target is None:
            omega_target = self.attitude.compute(theta=float(state[2]), theta_target=theta_target, dt=controller_dt)
        desired_moment, rate_error, p, i, d, ff = self.rate.compute(omega_target, float(state[5]), controller_dt)
        mixer = self.mixer.mix(desired_moment, float(state[6]))
        self.rate.apply_mixer_feedback(desired_moment, mixer, controller_dt)
        return _control_output(
            thrust_cmd,
            mixer.vane_angle_cmd,
            ax_target=ax_target,
            theta_target=theta_target,
            omega_target=omega_target,
            desired_moment=desired_moment,
            rate_error=rate_error,
            p=p,
            i=i,
            d=d,
            ff=ff,
            anti_windup_correction=self.rate.last_anti_windup_correction,
            integrator_inhibited=self.rate.last_integrator_inhibited,
            mixer=mixer,
        )

    def compute(
        self,
        mode: ControlMode,
        state: np.ndarray,
        commands: ManualCommands,
        controller_dt: float | None = None,
        targets: RuntimeTargets | None = None,
    ) -> RigidBodyControlOutput:
        dt = controller_dt if controller_dt is not None else self.cfg.dt
        thrust_cmd = self.thrust_curve.thrust(commands.throttle)
        if mode in (ControlMode.DIRECT, ControlMode.ACTUATOR_LAB):
            return _control_output(thrust_cmd, commands.direct_vane)
        if mode == ControlMode.RATE:
            return self._attitude_rate_mix(state, thrust_cmd, 0.0, dt, omega_target=commands.omega_target)
        if mode == ControlMode.STABILIZE:
            return self._attitude_rate_mix(state, thrust_cmd, commands.theta_target, dt)
        if targets is None:
            targets = RuntimeTargets(float(state[0]), float(state[1]))
        return self.compute_position_hold(mode, state, commands, targets, dt)

    def compute_position_hold(
        self,
        mode: ControlMode,
        state: np.ndarray,
        commands: ManualCommands,
        targets: RuntimeTargets,
        dt: float,
    ) -> RigidBodyControlOutput:
        x, z, theta, vx, vz, _omega, _thrust, _vane = [float(v) for v in state[:8]]
        cfg = self.controller_cfg
        stick_z = float(np.clip(commands.stick_z, -1.0, 1.0))
        if abs(stick_z) <= cfg.thr_dz:
            climb_cmd = 0.0
            targets.throttle_deadband_active = True
        elif stick_z > 0.0:
            climb_cmd = (stick_z - cfg.thr_dz) / max(1.0 - cfg.thr_dz, 1e-6) * cfg.pilot_speed_up_ms
            targets.throttle_deadband_active = False
        else:
            climb_cmd = (stick_z + cfg.thr_dz) / max(1.0 - cfg.thr_dz, 1e-6) * cfg.pilot_speed_dn_ms
            targets.throttle_deadband_active = False
        targets.target_z += climb_cmd * dt
        targets.target_z_rate_cmd = climb_cmd
        targets.altitude_hold_active = True

        z_error = targets.target_z - z
        targets.desired_vz = float(np.clip(cfg.psc_posz_p * z_error + climb_cmd, -cfg.pilot_speed_dn_ms, cfg.pilot_speed_up_ms))
        targets.desired_az = float(np.clip(cfg.psc_velz_p * (targets.desired_vz - vz), -cfg.pilot_accel_z_mss, cfg.pilot_accel_z_mss))
        body_up_z = max(0.2, float(np.cos(theta)))
        thrust_cmd = self.cfg.m * (self.cfg.g + targets.desired_az) / body_up_z
        thrust_cmd = float(np.clip(thrust_cmd, 0.0, self.cfg.T_max))

        if mode == ControlMode.ALT_HOLD:
            targets.loiter_active = False
            targets.loiter_braking_active = False
            theta_target = commands.theta_target
            targets.theta_target = theta_target
            targets.theta_target_limited = theta_target
            ax_target = 0.0
            targets.position_velocity_correction = 0.0
            targets.total_desired_vx = 0.0
        else:
            targets.loiter_active = True
            x_error = targets.target_x - x
            pos_vel = cfg.psc_ne_pos_p * x_error
            desired_vx_total = float(np.clip(targets.desired_vx + pos_vel, -cfg.loit_speed_ms, cfg.loit_speed_ms))
            targets.position_velocity_correction = pos_vel
            targets.total_desired_vx = desired_vx_total
            raw_ax = cfg.psc_ne_vel_p * (desired_vx_total - vx)
            ax_target = float(np.clip(raw_ax, -cfg.psc_accel_xy_max_mss, cfg.psc_accel_xy_max_mss))
            targets.desired_ax = ax_target
            theta_unlimited = float(np.arctan2(ax_target, self.cfg.g))
            angle_limit = min(cfg.loit_angle_max, cfg.atc_angle_max, self.cfg.theta_max)
            theta_target = float(np.clip(theta_unlimited, -angle_limit, angle_limit))
            targets.theta_target = theta_unlimited
            targets.theta_target_limited = theta_target
        return self._attitude_rate_mix(state, thrust_cmd, theta_target, dt, ax_target=ax_target)

    def reset_pid_for_mode_change(self, omega_target: float, omega: float) -> None:
        self.rate.reset(initial_error=omega_target - omega)
        self.attitude.seed_last_omega_target(omega_target)

def _move_toward(value: float, target: float, rate: float, dt: float) -> float:
    delta = rate * dt
    return float(np.clip(target, value - delta, value + delta))


def configure_actuator_lab(rb_cfg: RigidBodyConfig) -> RigidBodyConfig:
    """Return a validated copy configured for manual total-COM experiments."""
    moving_mass = replace(
        rb_cfg.moving_mass,
        enabled=True,
        use_total_com_geometry=True,
        use_legacy_gravity_offset_moment=False,
    )
    configured = replace(rb_cfg, moving_mass=moving_mass)
    configured.validate()
    return configured


class InteractiveApp:
    def __init__(
        self,
        rb_cfg: RigidBodyConfig | None = None,
        ui_cfg: InteractiveSimConfig | None = None,
        controller_cfg: ControllerConfig | None = None,
        *,
        actuator_lab_enabled: bool = False,
    ):
        self.rb_cfg = rb_cfg or RigidBodyConfig(dt=0.005)
        self.ui_cfg = ui_cfg or InteractiveSimConfig(physics_dt=self.rb_cfg.dt, controller_dt=0.01)
        self.controller_cfg = controller_cfg or ControllerConfig()
        self.actuator_lab_enabled = bool(actuator_lab_enabled)
        if self.actuator_lab_enabled:
            mm = self.rb_cfg.moving_mass
            if not (
                mm.enabled
                and mm.use_total_com_geometry
                and not mm.use_legacy_gravity_offset_moment
            ):
                raise ValueError(
                    "ACTUATOR_LAB requires a prevalidated moving-mass total-COM configuration"
                )
        self.rb_cfg.validate()
        self.plant = RigidBodySingleFan2D(self.rb_cfg)
        self.motor = FirstOrderMotor(self.rb_cfg.T_max, self.rb_cfg.motor_time_constant)
        self.servo = VaneServo(
            dt=self.rb_cfg.dt,
            angle_limit=self.rb_cfg.vane_angle_max,
            rate_limit=self.rb_cfg.vane_rate_limit,
            time_constant=self.rb_cfg.servo_time_constant,
            deadband=self.rb_cfg.servo_deadband,
            command_delay=self.rb_cfg.servo_delay,
        )
        self.control = ManualControlSystem(self.rb_cfg, self.controller_cfg)
        self.mode = ControlMode.ACTUATOR_LAB if self.actuator_lab_enabled else ControlMode.DIRECT
        self.thrust_curve = ThrottleToThrustModel(self.rb_cfg)
        self.commands = ManualCommands(throttle=self.thrust_curve.throttle_for_hover())
        self.targets = RuntimeTargets(self.rb_cfg.target_x, self.rb_cfg.target_z)
        self.loiter_shaper = LoiterInputShaper(self.controller_cfg)
        self.disturbance = Disturbance(force=np.zeros(2), moment=0.0)
        self.logger = InteractiveCSVLogger(self.ui_cfg.log_directory)

        self.state = self.plant.reset(self._reset_state("F1", startup=True))
        self.targets.capture(self.state)
        self.sim_time = 0.0
        self.speed = self.ui_cfg.initial_speed
        self.paused = False
        self.step_once = False
        self.slow_motion = False
        self.emergency_cut = False
        self.crash_reason = ""
        self.mode_status = "ACTUATOR LAB ready" if self.actuator_lab_enabled else "ready"
        self.camera_center = np.array([0.0, self.rb_cfg.target_z], dtype=float)
        self.camera_follow = True
        self.measured_real_time_factor = 0.0
        self.trace: list[tuple[float, float]] = []
        self.controller_time_remaining = 0.0
        self.last_control = _control_output(self.rb_cfg.hover_thrust, 0.0)
        self.last_motor = MotorOutput(self.rb_cfg.hover_thrust, 0.0, False)
        self.last_servo = ServoOutput(0.0, 0.0, 0.0, False, False)
        self.last_forces = self.plant.force_moment_breakdown(self.state)

    def _reset_state(self, preset_key: str, *, startup: bool = False) -> np.ndarray:
        preset = self.ui_cfg.presets.get(preset_key, self.ui_cfg.presets["F1"])
        state = np.array(preset.state, dtype=float)
        if not self.actuator_lab_enabled:
            return state
        state = state[:8].copy()
        state[6] = (
            0.45 * self.rb_cfg.hover_thrust
            if preset_key == "F6"
            else self.rb_cfg.hover_thrust
        )
        state[7] = 0.0
        initial_offset = 0.0 if startup else float(
            np.clip(
                self.rb_cfg.moving_mass.initial_offset_m,
                -self.rb_cfg.moving_mass.max_offset_m,
                self.rb_cfg.moving_mass.max_offset_m,
            )
        )
        return np.concatenate([state, np.array([initial_offset, 0.0, 0.0])])

    @property
    def actuator_lab_mass_limit_m(self) -> float:
        return min(
            abs(float(self.ui_cfg.actuator_lab_mass_limit_m)),
            abs(float(self.rb_cfg.moving_mass.max_offset_m)),
        )

    @property
    def actuator_lab_vane_limit_rad(self) -> float:
        return min(
            np.deg2rad(abs(float(self.ui_cfg.actuator_lab_vane_limit_deg))),
            abs(float(self.rb_cfg.vane_angle_max)),
        )

    def adjust_actuator_lab_mass_target(self, direction: float, *, coarse: bool = False) -> float:
        step = (
            self.ui_cfg.actuator_lab_mass_coarse_step_m
            if coarse
            else self.ui_cfg.actuator_lab_mass_step_m
        )
        self.commands.moving_mass_target_m = float(
            np.clip(
                self.commands.moving_mass_target_m + float(direction) * step,
                -self.actuator_lab_mass_limit_m,
                self.actuator_lab_mass_limit_m,
            )
        )
        return self.commands.moving_mass_target_m

    def adjust_actuator_lab_vane_command(self, direction: float, *, coarse: bool = False) -> float:
        step_deg = (
            self.ui_cfg.actuator_lab_vane_coarse_step_deg
            if coarse
            else self.ui_cfg.actuator_lab_vane_step_deg
        )
        self.commands.direct_vane = float(
            np.clip(
                self.commands.direct_vane + np.deg2rad(float(direction) * step_deg),
                -self.actuator_lab_vane_limit_rad,
                self.actuator_lab_vane_limit_rad,
            )
        )
        return self.commands.direct_vane

    def set_mode(self, mode: ControlMode) -> bool:
        if mode == ControlMode.ACTUATOR_LAB and not self.actuator_lab_enabled:
            self.mode_status = "restart with --actuator-lab to use mode 6"
            return False
        if mode == self.mode:
            return True
        previous = self.mode
        if previous == ControlMode.ACTUATOR_LAB:
            self.commands.moving_mass_target_m = 0.0
        if mode == ControlMode.ACTUATOR_LAB:
            self.commands.direct_vane = 0.0
            self.commands.moving_mass_target_m = 0.0
        if mode == ControlMode.STABILIZE:
            self.commands.theta_target = float(wrap_pi(self.state[2]))
            self.commands.omega_target = float(self.state[5])
            self.control.attitude.reset_to_current(float(self.state[2]), float(self.state[5]))
        if mode == ControlMode.ALT_HOLD:
            self.targets.capture(self.state)
            self.targets.altitude_hold_active = True
            self.targets.loiter_active = False
            self.commands.theta_target = float(wrap_pi(self.state[2]))
            self.control.attitude.reset_to_current(float(self.state[2]), float(self.state[5]))
        if mode == ControlMode.LOITER:
            self.targets.capture(self.state)
            self.targets.altitude_hold_active = True
            self.targets.loiter_active = True
            self.loiter_shaper.reset(0.0)
            self.control.attitude.reset_to_current(float(self.state[2]), float(self.state[5]))
        if mode == ControlMode.RATE and previous in {
            ControlMode.DIRECT,
            ControlMode.STABILIZE,
            ControlMode.ACTUATOR_LAB,
        }:
            self.commands.omega_target = float(self.state[5])
        self.control.reset_pid_for_mode_change(self.commands.omega_target, float(self.state[5]))
        self.mode = mode
        self.mode_status = f"{previous.value} -> {mode.value}"
        return True

    def reset(self, preset_key: str = "F1") -> None:
        preset = self.ui_cfg.presets.get(preset_key, self.ui_cfg.presets["F1"])
        self.state = self.plant.reset(self._reset_state(preset_key))
        self.control.reset()
        self.servo.reset()
        self.loiter_shaper.reset()
        self.commands.zero(self.thrust_curve.throttle_for_hover())
        self.targets.capture(self.state)
        self.disturbance = Disturbance(force=np.zeros(2), moment=0.0)
        self.sim_time = 0.0
        self.controller_time_remaining = 0.0
        self.emergency_cut = False
        self.crash_reason = ""
        self.mode_status = f"reset: {preset.name}"
        self.trace.clear()
        self.last_control = _control_output(self.rb_cfg.hover_thrust, 0.0)
        self.last_motor = MotorOutput(float(self.state[6]), 0.0, False)
        self.last_servo = ServoOutput(0.0, 0.0, 0.0, False, False)
        self.last_forces = self.plant.force_moment_breakdown(self.state)

    def handle_events(self, pygame) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_n:
                    self.step_once = True
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_l:
                    if self.logger.enabled:
                        self.logger.stop()
                    else:
                        self.logger.start()
                elif event.key == pygame.K_BACKSPACE:
                    self.commands.zero(self.thrust_curve.throttle_for_hover())
                    self.emergency_cut = False
                    self.mode_status = "manual commands centered"
                elif self.mode == ControlMode.ACTUATOR_LAB and event.key in (pygame.K_a, pygame.K_d):
                    coarse = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)
                    direction = 1.0 if event.key == pygame.K_d else -1.0
                    self.adjust_actuator_lab_vane_command(direction, coarse=coarse)
                elif self.mode == ControlMode.ACTUATOR_LAB and event.key == pygame.K_v:
                    self.commands.direct_vane = 0.0
                elif self.mode == ControlMode.ACTUATOR_LAB and event.key in (pygame.K_f, pygame.K_h):
                    coarse = bool(getattr(event, "mod", 0) & pygame.KMOD_SHIFT)
                    direction = 1.0 if event.key == pygame.K_h else -1.0
                    self.adjust_actuator_lab_mass_target(direction, coarse=coarse)
                elif self.mode == ControlMode.ACTUATOR_LAB and event.key == pygame.K_g:
                    self.commands.moving_mass_target_m = 0.0
                elif event.key == pygame.K_1:
                    self.set_mode(ControlMode.DIRECT)
                elif event.key == pygame.K_2:
                    self.set_mode(ControlMode.RATE)
                elif event.key == pygame.K_3:
                    self.set_mode(ControlMode.STABILIZE)
                elif event.key == pygame.K_4:
                    self.set_mode(ControlMode.ALT_HOLD)
                elif event.key == pygame.K_5:
                    self.set_mode(ControlMode.LOITER)
                elif event.key == pygame.K_6:
                    self.set_mode(ControlMode.ACTUATOR_LAB)
                elif event.key == pygame.K_LEFTBRACKET:
                    self.speed = max(self.ui_cfg.min_speed, self.speed - self.ui_cfg.speed_step)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self.speed = min(self.ui_cfg.max_speed, self.speed + self.ui_cfg.speed_step)
                elif event.key == pygame.K_m:
                    self.slow_motion = not self.slow_motion
                elif event.key == pygame.K_x:
                    self.emergency_cut = not self.emergency_cut
                    self.mode_status = "emergency motor cut" if self.emergency_cut else "motor cut released"
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    self.ui_cfg.pixels_per_meter = min(
                        self.ui_cfg.max_pixels_per_meter,
                        self.ui_cfg.pixels_per_meter * self.ui_cfg.zoom_step,
                    )
                elif event.key == pygame.K_MINUS:
                    self.ui_cfg.pixels_per_meter = max(
                        self.ui_cfg.min_pixels_per_meter,
                        self.ui_cfg.pixels_per_meter / self.ui_cfg.zoom_step,
                    )
                elif event.key == pygame.K_c:
                    self.camera_follow = not self.camera_follow
                    self.mode_status = "camera follow on" if self.camera_follow else "camera follow off"
                elif event.key in (pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6):
                    self.reset(f"F{event.key - pygame.K_F1 + 1}")
                elif event.key == pygame.K_i:
                    self.disturbance.impulse_time_remaining = self.ui_cfg.impulse_duration_s
                    self.disturbance.impulse_force = np.array([self.ui_cfg.disturbance_force_x_N, 0.0])
                elif event.key == pygame.K_o:
                    self.disturbance.impulse_time_remaining = self.ui_cfg.impulse_duration_s
                    self.disturbance.impulse_moment = self.ui_cfg.disturbance_moment_Nm
        return True

    def update_inputs(self, pygame, dt: float) -> None:
        keys = pygame.key.get_pressed()
        pitch_axis = 0.0 if self.mode == ControlMode.ACTUATOR_LAB else (
            float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        )
        climb_axis = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        self.commands.stick_x = pitch_axis
        self.commands.stick_z = climb_axis

        if self.mode not in (ControlMode.ALT_HOLD, ControlMode.LOITER):
            if keys[pygame.K_w]:
                self.commands.throttle += self.ui_cfg.throttle_slew_per_s * dt
            if keys[pygame.K_s]:
                self.commands.throttle -= self.ui_cfg.throttle_slew_per_s * dt
            self.commands.throttle = float(np.clip(self.commands.throttle, 0.0, 1.0))

        direct_target = np.deg2rad(self.ui_cfg.direct_vane_max_deg) * pitch_axis
        theta_target = np.deg2rad(self.ui_cfg.manual_theta_max_deg) * pitch_axis
        omega_target = np.deg2rad(self.ui_cfg.manual_omega_max_deg_s) * pitch_axis
        return_rate = self.ui_cfg.command_return_rate

        if self.mode != ControlMode.ACTUATOR_LAB:
            self.commands.direct_vane = _move_toward(
                self.commands.direct_vane,
                direct_target,
                np.deg2rad(self.ui_cfg.vane_slew_deg_s if pitch_axis else self.ui_cfg.direct_vane_max_deg * return_rate),
                dt,
            )
        self.commands.theta_target = _move_toward(
            self.commands.theta_target,
            theta_target,
            np.deg2rad(self.ui_cfg.theta_target_slew_deg_s if pitch_axis else self.ui_cfg.manual_theta_max_deg * return_rate),
            dt,
        )
        self.commands.omega_target = _move_toward(
            self.commands.omega_target,
            omega_target,
            np.deg2rad(self.ui_cfg.omega_target_slew_deg_s if pitch_axis else self.ui_cfg.manual_omega_max_deg_s * return_rate),
            dt,
        )

        fx = (float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])) * self.ui_cfg.disturbance_force_x_N
        fz = (float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])) * self.ui_cfg.disturbance_force_z_N
        moment = (float(keys[pygame.K_e]) - float(keys[pygame.K_q])) * self.ui_cfg.disturbance_moment_Nm
        self.disturbance.force = np.array([fx, fz], dtype=float)
        self.disturbance.moment = float(moment)

    def _update_loiter_targets(self, dt: float) -> None:
        self.targets.target_capture_event = False
        if self.mode != ControlMode.LOITER:
            self.targets.loiter_active = False
            self.targets.loiter_braking_active = False
            self.targets.target_x_rate_cmd = 0.0
            return
        desired_vx = self.loiter_shaper.update(self.commands.stick_x, dt)
        self.targets.desired_vx = desired_vx
        self.targets.shaped_desired_vx = desired_vx
        self.targets.loiter_braking_active = self.loiter_shaper.braking_active
        self.targets.target_capture_pending = self.loiter_shaper.capture_pending
        self.targets.target_x_rate_cmd = desired_vx
        self.targets.target_x += desired_vx * dt
        capture_ready = (
            self.loiter_shaper.capture_pending
            and abs(desired_vx) <= self.controller_cfg.loit_capture_desired_vx_threshold_ms
            and (
                self.controller_cfg.loit_capture_without_jump
                or abs(float(self.state[3])) <= self.controller_cfg.loit_capture_vx_threshold_ms
            )
        )
        if capture_ready:
            if not self.controller_cfg.loit_capture_without_jump:
                self.targets.target_x = float(self.state[0])
            self.targets.target_capture_event = True
            self.targets.target_capture_count += 1
            self.loiter_shaper.mark_target_captured()
            self.targets.loiter_braking_active = False
            self.targets.target_capture_pending = False
    def physics_step(self, wall_time: float, real_time_factor: float) -> None:
        disturbance_force = self.disturbance.combined_force()
        disturbance_moment = self.disturbance.combined_moment()
        if self.mode == ControlMode.ACTUATOR_LAB:
            self.last_control = self.control.compute(
                self.mode,
                self.state,
                self.commands,
                self.rb_cfg.dt,
                self.targets,
            )
            self.controller_time_remaining = 0.0
        elif self.controller_time_remaining <= 1e-12:
            self._update_loiter_targets(self.ui_cfg.controller_dt)
            self.last_control = self.control.compute(
                self.mode,
                self.state,
                self.commands,
                self.ui_cfg.controller_dt,
                self.targets,
            )
            self.controller_time_remaining += self.ui_cfg.controller_dt
        if self.emergency_cut:
            self.last_control = _control_output(
                0.0,
                self.last_control.vane_angle_cmd,
                ax_target=self.last_control.ax_target,
                theta_target=self.last_control.theta_target,
                omega_target=self.last_control.omega_target,
                mixer=self.last_control.mixer,
            )
        self.last_motor = self.motor.update(float(self.state[6]), self.last_control.thrust_cmd + self.controller_cfg.motor_thrust_bias)
        servo_cmd = self.last_control.vane_angle_cmd + np.deg2rad(self.controller_cfg.servo_bias_deg)
        self.last_servo = self.servo.update(float(self.state[7]), servo_cmd)
        self.state = self.plant.step(
            self.last_motor.thrust_dot,
            self.last_servo.vane_angle_dot,
            disturbance_force,
            disturbance_moment,
            moving_mass_target_m=(
                self.commands.moving_mass_target_m
                if self.actuator_lab_enabled
                else None
            ),
        )
        self.disturbance.tick(self.rb_cfg.dt)
        self.sim_time += self.rb_cfg.dt
        self.controller_time_remaining -= self.rb_cfg.dt
        safety = check_safety(self.state, self.rb_cfg)
        if safety.crashed:
            self.crash_reason = safety.reason
            self.paused = True
            self.mode_status = f"auto-paused: {safety.reason}"
        self.last_forces = self.plant.force_moment_breakdown(self.state, disturbance_force, disturbance_moment)
        if self.logger.enabled:
            self.logger.write(
                interactive_row(
                    self.sim_time,
                    wall_time,
                    self.mode.value,
                    self.state,
                    self.commands.throttle,
                    self.commands.direct_vane,
                    self.commands.stick_x,
                    self.commands.stick_z,
                    self.targets,
                    self.last_control,
                    self.last_motor,
                    self.last_servo,
                    self.last_forces,
                    self.crash_reason,
                    safety.min_body_z,
                    self.rb_cfg.dt,
                    self.ui_cfg.controller_dt,
                    real_time_factor,
                )
            )
        self.trace.append((float(self.state[0]), float(self.state[1])))
        if len(self.trace) > self.ui_cfg.trace_length:
            self.trace.pop(0)

    def render_errors(self, x: float, z: float) -> tuple[float, float]:
        x_err = self.targets.target_x - x if self.mode == ControlMode.LOITER else 0.0
        z_err = self.targets.target_z - z if self.mode in (ControlMode.ALT_HOLD, ControlMode.LOITER) else 0.0
        return float(x_err), float(z_err)

    def vane_visual_geometry(
        self,
        bottom: np.ndarray,
        body_up: np.ndarray,
        body_right: np.ndarray,
        actual_vane: float,
        command_vane: float,
    ) -> dict[str, np.ndarray | float]:
        visual_scale = self.ui_cfg.vane_visual_scale
        hinge = bottom - self.ui_cfg.vane_visual_offset_m * body_up
        neutral_dir = -body_up
        actual_angle = actual_vane * visual_scale
        command_angle = command_vane * visual_scale
        actual_dir = np.cos(actual_angle) * neutral_dir + np.sin(actual_angle) * body_right
        command_dir = np.cos(command_angle) * neutral_dir + np.sin(command_angle) * body_right
        return {
            "hinge": hinge,
            "neutral_dir": neutral_dir,
            "actual_dir": actual_dir,
            "command_dir": command_dir,
            "length": float(self.ui_cfg.vane_visual_length_m),
        }

    def actuator_lab_visual_geometry(
        self,
        body_up: np.ndarray,
        body_right: np.ndarray,
    ) -> dict[str, np.ndarray]:
        total_com = np.asarray(self.state[:2], dtype=float)
        total_com_body = np.array(
            [
                self.last_forces.total_com_body_right_m,
                self.last_forces.total_com_body_up_m,
            ],
            dtype=float,
        )
        fixed_body_origin = (
            total_com
            - total_com_body[0] * body_right
            - total_com_body[1] * body_up
        )
        rail_center = (
            fixed_body_origin
            + self.rb_cfg.moving_mass.moving_mass_body_up_offset_m * body_up
        )
        physical_limit = abs(float(self.rb_cfg.moving_mass.max_offset_m))
        actual_offset = float(self.state[8]) if self.state.shape == (11,) else 0.0
        target_offset = float(self.commands.moving_mass_target_m)
        return {
            "total_com": total_com,
            "fixed_body_origin": fixed_body_origin,
            "rail_start": rail_center - physical_limit * body_right,
            "rail_end": rail_center + physical_limit * body_right,
            "moving_mass_actual": rail_center + actual_offset * body_right,
            "moving_mass_target": rail_center + target_offset * body_right,
        }

    def expected_pitch_direction(self, total_moment: float | None = None) -> str:
        moment = self.last_forces.total_moment if total_moment is None else float(total_moment)
        deadband = abs(float(self.ui_cfg.actuator_lab_moment_deadband_Nm))
        if moment > deadband:
            return "RIGHT / POSITIVE"
        if moment < -deadband:
            return "LEFT / NEGATIVE"
        return "NEAR ZERO"

    def actuator_pair_moment(self) -> float:
        return float(
            self.last_forces.thrust_moment_from_com_offset
            + self.last_forces.vane_moment_about_total_com
        )

    def actuator_pair_direction(self, pair_moment: float | None = None) -> str:
        moment = self.actuator_pair_moment() if pair_moment is None else float(pair_moment)
        return f"PAIR {self.expected_pitch_direction(moment)}"

    def vane_side_force_direction(self, body_right: np.ndarray) -> str:
        component = float(np.dot(self.last_forces.vane_force, body_right))
        if component > 1e-9:
            return "BODY-RIGHT (+)"
        if component < -1e-9:
            return "BODY-LEFT (-)"
        return "NEAR ZERO"

    def actuator_lab_panel_lines(
        self,
        body_right: np.ndarray | None = None,
    ) -> list[str]:
        if body_right is None:
            _body_up, body_right = self.plant.body_axes(float(self.state[2]))
        return [
            "ACTUATOR LAB",
            f"vane command: {np.rad2deg(self.commands.direct_vane):+.2f} deg",
            f"actual vane angle: {np.rad2deg(self.state[7]):+.2f} deg",
            f"vane side-force: {self.vane_side_force_direction(body_right)}",
            f"vane moment: {self.last_forces.vane_moment:+.4f} N m",
            f"moving-mass target: {1000.0 * self.commands.moving_mass_target_m:+.1f} mm",
            f"actual moving-mass offset: {1000.0 * self.last_forces.moving_mass_offset_m:+.1f} mm",
            f"moving-mass velocity: {1000.0 * self.last_forces.moving_mass_velocity_m_s:+.1f} mm/s",
            f"total COM body-right: {1000.0 * self.last_forces.total_com_body_right_m:+.2f} mm",
            f"total COM body-up: {1000.0 * self.last_forces.total_com_body_up_m:+.2f} mm",
            f"thrust-offset moment: {self.last_forces.thrust_moment_from_com_offset:+.4f} N m",
            f"actuator-pair moment: {self.actuator_pair_moment():+.4f} N m",
            f"actuator pair: {self.actuator_pair_direction()}",
            f"damping moment: {self.last_forces.damping_moment:+.4f} N m",
            f"total pitch moment: {self.last_forces.total_moment:+.4f} N m",
            f"expected pitch: {self.expected_pitch_direction()}",
            "theta + = RIGHT; mass offset + = BODY-RIGHT",
        ]

    def render(self, pygame, screen, font, small_font) -> None:
        w, h = screen.get_size()
        screen.fill((18, 20, 24))

        scale = self.ui_cfg.pixels_per_meter
        if self.camera_follow:
            tx = self.targets.target_x if self.mode == ControlMode.LOITER else float(self.state[0])
            tz = self.targets.target_z if self.mode in (ControlMode.ALT_HOLD, ControlMode.LOITER) else float(self.state[1])
            self.camera_center = 0.85 * np.array([self.state[0], self.state[1]], dtype=float) + 0.15 * np.array([tx, tz], dtype=float)
        origin = np.array([w * 0.5, h * 0.58]) - np.array([self.camera_center[0] * scale, -self.camera_center[1] * scale])

        def world_to_screen(p):
            p = np.asarray(p, dtype=float)
            return (int(origin[0] + p[0] * scale), int(origin[1] - p[1] * scale))

        def draw_arrow(start, vec, color, vec_scale=1.0, width_px=3):
            start = np.asarray(start, dtype=float)
            end = start + np.asarray(vec, dtype=float) * vec_scale
            pygame.draw.line(screen, color, world_to_screen(start), world_to_screen(end), width_px)
            pygame.draw.circle(screen, color, world_to_screen(end), 4)

        pygame.draw.line(screen, (95, 95, 95), (0, origin[1]), (w, origin[1]), 2)
        pygame.draw.line(screen, (70, 70, 70), world_to_screen((0, -0.2)), world_to_screen((0, 2.6)), 1)

        if len(self.trace) > 1:
            pygame.draw.lines(screen, (70, 130, 200), False, [world_to_screen(p) for p in self.trace], 1)

        x, z, theta, _vx, _vz, _omega, thrust, vane = self.state[:8]
        x_err, z_err = self.render_errors(float(x), float(z))
        translational_energy = 0.5 * self.rb_cfg.m * (self.state[3] ** 2 + self.state[4] ** 2)
        rotational_energy = 0.5 * self.rb_cfg.Iyy * self.state[5] ** 2
        potential_energy = self.rb_cfg.m * self.rb_cfg.g * z
        total_energy = translational_energy + rotational_energy + potential_energy
        body_up, body_right = self.plant.body_axes(float(theta))
        cg = np.array([x, z])
        fixed_body_origin = cg
        lab_geometry = None
        if self.rb_cfg.moving_mass.use_total_com_geometry:
            lab_geometry = self.actuator_lab_visual_geometry(body_up, body_right)
            fixed_body_origin = lab_geometry["fixed_body_origin"]
        top = fixed_body_origin + self.rb_cfg.l * body_up
        bottom = fixed_body_origin - self.rb_cfg.l * body_up
        half_w = 0.5 * self.rb_cfg.W
        corners = [bottom - half_w * body_right, bottom + half_w * body_right, top + half_w * body_right, top - half_w * body_right]
        if self.ui_cfg.show_target_marker and self.mode in (ControlMode.ALT_HOLD, ControlMode.LOITER):
            target = np.array([self.targets.target_x if self.mode == ControlMode.LOITER else x, self.targets.target_z])
            pygame.draw.circle(screen, (255, 220, 80), world_to_screen(target), 7, 2)
            if self.ui_cfg.show_loiter_error_vector:
                pygame.draw.line(screen, (150, 130, 60), world_to_screen((x, z)), world_to_screen((target[0], z)), 1)
                pygame.draw.line(screen, (150, 130, 60), world_to_screen((target[0], z)), world_to_screen(target), 1)
            if self.ui_cfg.show_desired_accel_arrow:
                draw_arrow(cg, np.array([self.targets.desired_ax, self.targets.desired_az]) / max(self.rb_cfg.g, 1e-6), (255, 220, 80), 0.25, 2)
        pygame.draw.polygon(screen, (60, 145, 220), [world_to_screen(c) for c in corners])
        if self.mode == ControlMode.ACTUATOR_LAB and lab_geometry is not None:
            rail_start = world_to_screen(lab_geometry["rail_start"])
            rail_end = world_to_screen(lab_geometry["rail_end"])
            actual_pos = world_to_screen(lab_geometry["moving_mass_actual"])
            target_pos = world_to_screen(lab_geometry["moving_mass_target"])
            fixed_origin_px = world_to_screen(lab_geometry["fixed_body_origin"])
            total_com_px = world_to_screen(lab_geometry["total_com"])
            pygame.draw.line(screen, (210, 210, 210), rail_start, rail_end, 4)
            pygame.draw.circle(screen, (255, 150, 55), actual_pos, 7)
            pygame.draw.circle(screen, (245, 245, 245), actual_pos, 7, 2)
            pygame.draw.line(screen, (120, 235, 255), (target_pos[0] - 6, target_pos[1] - 6), (target_pos[0] + 6, target_pos[1] + 6), 3)
            pygame.draw.line(screen, (120, 235, 255), (target_pos[0] - 6, target_pos[1] + 6), (target_pos[0] + 6, target_pos[1] - 6), 3)
            pygame.draw.rect(screen, (250, 250, 250), (fixed_origin_px[0] - 4, fixed_origin_px[1] - 4, 8, 8), 2)
            pygame.draw.polygon(
                screen,
                (255, 80, 80),
                [
                    (total_com_px[0], total_com_px[1] - 7),
                    (total_com_px[0] + 7, total_com_px[1]),
                    (total_com_px[0], total_com_px[1] + 7),
                    (total_com_px[0] - 7, total_com_px[1]),
                ],
                2,
            )
            marker_labels = [
                ("MM actual", actual_pos, (18, -48), (255, 230, 205)),
                ("MM target", target_pos, (18, -14), (190, 245, 255)),
                ("total COM", total_com_px, (18, 24), (255, 185, 185)),
            ]
            for label, marker, offset_px, color in marker_labels:
                label_pos = (marker[0] + offset_px[0], marker[1] + offset_px[1])
                pygame.draw.line(
                    screen,
                    color,
                    marker,
                    (label_pos[0] - 3, label_pos[1] + 8),
                    1,
                )
                screen.blit(small_font.render(label, True, color), label_pos)
        else:
            pygame.draw.circle(screen, (255, 80, 80), world_to_screen(cg), 5)
        pygame.draw.circle(screen, (20, 20, 20), world_to_screen(bottom), 5)

        thrust_point = self.last_forces.thrust_application_point if lab_geometry is not None else bottom
        vane_point = self.last_forces.vane_application_point if lab_geometry is not None else bottom
        draw_arrow(thrust_point, self.last_forces.thrust_force / max(self.rb_cfg.hover_thrust, 1e-6), (245, 160, 55), 0.35)
        draw_arrow(vane_point, self.last_forces.vane_force / max(self.rb_cfg.hover_thrust, 1e-6), (190, 90, 230), 0.6)
        draw_arrow(cg, self.last_forces.total_force / max(self.rb_cfg.hover_thrust, 1e-6), (80, 220, 130), 0.25)
        draw_arrow(cg, self.last_forces.disturbance_force / max(self.rb_cfg.hover_thrust, 1e-6), (255, 230, 80), 0.5)

        if self.ui_cfg.show_theta_target_vector:
            target_up = np.array([np.sin(self.last_control.theta_target), np.cos(self.last_control.theta_target)])
            pygame.draw.line(screen, (255, 255, 255), world_to_screen(cg), world_to_screen(cg + 0.45 * target_up), 1)

        if self.ui_cfg.show_vane_overlay:
            displayed_vane_command = (
                self.commands.direct_vane
                if self.mode == ControlMode.ACTUATOR_LAB
                else self.last_control.vane_angle_cmd
            )
            geom = self.vane_visual_geometry(bottom, body_up, body_right, float(vane), displayed_vane_command)
            hinge = geom["hinge"]
            neutral_dir = geom["neutral_dir"]
            actual_dir = geom["actual_dir"]
            cmd_dir = geom["command_dir"]
            length = geom["length"]
            pygame.draw.line(
                screen,
                (235, 235, 235),
                world_to_screen(hinge - 0.35 * length * neutral_dir),
                world_to_screen(hinge + 0.35 * length * neutral_dir),
                3,
            )
            if self.ui_cfg.show_vane_command_ghost:
                pygame.draw.line(screen, (220, 170, 245), world_to_screen(hinge), world_to_screen(hinge + length * cmd_dir), 3)
            pygame.draw.line(screen, (210, 70, 245), world_to_screen(hinge), world_to_screen(hinge + length * actual_dir), 7)
            pygame.draw.circle(screen, (245, 245, 245), world_to_screen(hinge), 4)
            labels = []
            if self.last_control.mixer.saturated or self.last_servo.angle_saturated:
                labels.append("SAT")
            if self.last_servo.rate_saturated:
                labels.append("RATE")
            if self.last_control.mixer.authority_limited:
                labels.append("AUTH")
            limit_text = " ".join(labels)
            vane_text = f"VANE actual={np.rad2deg(vane):.1f}deg cmd={np.rad2deg(displayed_vane_command):.1f}deg {limit_text}".rstrip()
            screen.blit(small_font.render(vane_text, True, (250, 235, 255)), world_to_screen(hinge + 0.28 * body_right - 0.18 * body_up))

        control_help = (
            "LAB A/D vane  F/H moving-mass target  V/G center  Shift=coarse  W/S throttle"
            if self.mode == ControlMode.ACTUATOR_LAB
            else "W/S throttle or climb  A/D mode stick  arrows force  Q/E moment  I/O impulse  X motor cut"
        )
        lines = [
            control_help,
            "1 direct  2 rate  3 stabilize  4 alt_hold  5 loiter  6 actuator_lab  Space pause  N step  R reset  L log",
            f"t={self.sim_time:6.2f}s  requested={self.speed:.2f}x measured={self.measured_real_time_factor:.2f}x{' slow' if self.slow_motion else ''}  mode={self.mode.value}  paused={self.paused}",
            f"x={x: .2f} z={z: .2f}  vx={self.state[3]: .2f} vz={self.state[4]: .2f}",
            f"target_x={self.targets.target_x: .2f} target_z={self.targets.target_z: .2f}  x_err={x_err: .2f} z_err={z_err: .2f} dist={(x_err*x_err+z_err*z_err)**0.5: .2f}",
            f"des_vx={self.targets.desired_vx: .2f} actual_vx={self.state[3]: .2f} des_ax={self.targets.desired_ax: .2f}  des_vz={self.targets.desired_vz: .2f} actual_vz={self.state[4]: .2f} des_az={self.targets.desired_az: .2f}",
            f"hold alt={int(self.targets.altitude_hold_active)} deadband={int(self.targets.throttle_deadband_active)} loiter={int(self.targets.loiter_active)} brake={int(self.targets.loiter_braking_active)} stick_x={self.commands.stick_x:.1f} stick_z={self.commands.stick_z:.1f}",
            f"theta={np.rad2deg(theta): .2f} deg  omega={np.rad2deg(self.state[5]): .1f} deg/s",
            f"throttle_cmd={self.commands.throttle:.2f} thrust={thrust:.2f} N",
            f"theta_t={np.rad2deg(self.last_control.theta_target): .1f} deg  omega_t={np.rad2deg(self.last_control.omega_target): .1f} deg/s",
            f"moment req={self.last_control.desired_moment:.3f} phys_ach={self.last_control.mixer.physically_achievable_moment:.3f} floor_ach={self.last_control.mixer.achievable_moment:.3f}",
            f"vane cmd={np.rad2deg(self.last_control.vane_angle_cmd): .1f} deg actual={np.rad2deg(vane): .1f} deg",
            f"PID P/I/D/FF={self.last_control.rate_p:.3f}/{self.last_control.rate_i:.3f}/{self.last_control.rate_d:.3f}/{self.last_control.rate_ff:.3f} AW={self.last_control.anti_windup_correction:.3f} inhibit={int(self.last_control.integrator_inhibited)}",
            f"dist F=({self.last_forces.disturbance_force[0]:.1f},{self.last_forces.disturbance_force[1]:.1f}) N M={self.last_forces.disturbance_moment:.2f} Nm impulse={self.disturbance.impulse_time_remaining:.2f}s",
            f"energy={total_energy:.2f} J  dt={self.rb_cfg.dt:.4f}s controller_dt={self.ui_cfg.controller_dt:.4f}s",
            f"sat motor={int(self.last_motor.saturated)} servo_angle={int(self.last_servo.angle_saturated)} servo_rate={int(self.last_servo.rate_saturated)} mixer_angle={int(self.last_control.mixer.angle_saturated)} authority={int(self.last_control.mixer.authority_limited)} log={int(self.logger.enabled)}",
            f"status={self.mode_status} crash={self.crash_reason or '-'} camera_follow={int(self.camera_follow)} zoom={self.ui_cfg.pixels_per_meter:.0f}px/m",
            "solid vane=actual, ghost=command, SAT/RATE/AUTH expose limits; visual scaling does not affect physics",
        ]
        y = 12
        for idx, line in enumerate(lines):
            surf = (font if idx < 2 else small_font).render(line, True, (235, 235, 235))
            screen.blit(surf, (12, y))
            y += 24 if idx < 2 else 19

        if self.mode == ControlMode.ACTUATOR_LAB:
            panel_lines = self.actuator_lab_panel_lines(body_right)
            panel_x = max(12, w - 505)
            panel_y = 60
            panel_height = 18 + len(panel_lines) * 20
            pygame.draw.rect(screen, (8, 10, 14), (panel_x, panel_y, 493, panel_height))
            pygame.draw.rect(screen, (120, 235, 255), (panel_x, panel_y, 493, panel_height), 2)
            for idx, line in enumerate(panel_lines):
                text_font = font if idx == 0 else small_font
                color = (120, 235, 255) if idx == 0 else (235, 240, 245)
                screen.blit(text_font.render(line, True, color), (panel_x + 10, panel_y + 8 + idx * 20))

        pygame.display.flip()

    def run(self) -> None:
        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - depends on local environment
            raise SystemExit("pygame is required. Install dependencies with `python -m pip install -r requirements.txt`.") from exc

        pygame.init()
        screen = pygame.display.set_mode((1200, 820))
        pygame.display.set_caption("Single-Fan Rigid-Body Interactive Simulator")
        font = pygame.font.SysFont("consolas", 18)
        small_font = pygame.font.SysFont("consolas", 16)
        clock = pygame.time.Clock()

        running = True
        accumulator = 0.0
        last_wall = perf_counter()
        while running:
            now = perf_counter()
            wall_dt = min(0.05, now - last_wall)
            last_wall = now
            running = self.handle_events(pygame)
            self.update_inputs(pygame, wall_dt)

            factor = self.ui_cfg.slow_motion_speed if self.slow_motion else self.speed
            if not self.paused:
                accumulator += wall_dt * factor
            elif self.step_once:
                accumulator += self.rb_cfg.dt
                self.step_once = False

            sim_before = self.sim_time
            while accumulator >= self.rb_cfg.dt:
                self.physics_step(now, factor)
                accumulator -= self.rb_cfg.dt
            self.measured_real_time_factor = (self.sim_time - sim_before) / max(wall_dt, 1e-6)

            self.render(pygame, screen, font, small_font)
            clock.tick(self.ui_cfg.render_rate)

        self.logger.close()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the interactive single-fan simulator.")
    parser.add_argument("--params", default=None, help="Optional JSON parameter override file.")
    parser.add_argument(
        "--actuator-lab",
        action="store_true",
        help="Start in ACTUATOR_LAB with validated total-COM moving-mass geometry.",
    )
    args = parser.parse_args()
    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(args.params)
    if args.actuator_lab:
        rb_cfg = configure_actuator_lab(rb_cfg)
    InteractiveApp(
        rb_cfg,
        ui_cfg,
        controller_cfg,
        actuator_lab_enabled=args.actuator_lab,
    ).run()


if __name__ == "__main__":
    main()
