from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ManualCommands:
    throttle: float
    direct_vane: float = 0.0
    theta_target: float = 0.0
    omega_target: float = 0.0
    stick_x: float = 0.0
    stick_z: float = 0.0

    def zero(self, hover_throttle: float) -> None:
        self.throttle = hover_throttle
        self.direct_vane = 0.0
        self.theta_target = 0.0
        self.omega_target = 0.0
        self.stick_x = 0.0
        self.stick_z = 0.0


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

    def capture(self, state: np.ndarray) -> None:
        self.target_x = float(state[0])
        self.target_z = float(state[1])
        self.desired_vx = 0.0
        self.desired_vz = 0.0
        self.desired_ax = 0.0
        self.desired_az = 0.0
        self.theta_target = 0.0
        self.theta_target_limited = 0.0
        self.loiter_braking_active = False
        self.throttle_deadband_active = True
        self.target_x_rate_cmd = 0.0
        self.target_z_rate_cmd = 0.0


class LoiterInputShaper:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.desired_vx = 0.0
        self._last_accel = 0.0
        self._release_time = 0.0
        self.braking_active = False

    def reset(self, desired_vx: float = 0.0) -> None:
        self.desired_vx = float(desired_vx)
        self._last_accel = 0.0
        self._release_time = 0.0
        self.braking_active = False

    @staticmethod
    def _rate_limit(value: float, target: float, max_delta: float) -> float:
        return float(np.clip(target, value - max_delta, value + max_delta))

    def update(self, stick_x: float, dt: float) -> float:
        if abs(stick_x) > 1e-3:
            self._release_time = 0.0
            self.braking_active = False
            target = float(np.clip(stick_x, -1.0, 1.0) * self.cfg.loit_speed_ms)
            accel_limit = self.cfg.loit_acc_max_mss
            jerk_limit = self.cfg.psc_jerk_xy_max_msss
        else:
            self._release_time += dt
            self.braking_active = self._release_time >= self.cfg.loit_brk_delay_s and abs(self.desired_vx) > 0.02
            target = 0.0
            accel_limit = self.cfg.loit_brk_acc_mss if self.braking_active else self.cfg.loit_acc_max_mss
            jerk_limit = self.cfg.loit_brk_jerk_msss
        raw_accel = (target - self.desired_vx) / max(dt, 1e-6)
        accel = self._rate_limit(self._last_accel, raw_accel, max(jerk_limit, 1e-6) * dt)
        accel = float(np.clip(accel, -accel_limit, accel_limit))
        self.desired_vx += accel * dt
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
        if mode == ControlMode.DIRECT:
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
        x, z, theta, vx, vz, _omega, _thrust, _vane = [float(v) for v in state]
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
        else:
            targets.loiter_active = True
            x_error = targets.target_x - x
            pos_vel = cfg.psc_ne_pos_p * x_error
            desired_vx_total = float(np.clip(targets.desired_vx + pos_vel, -cfg.loit_speed_ms, cfg.loit_speed_ms))
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


class InteractiveApp:
    def __init__(self, rb_cfg: RigidBodyConfig | None = None, ui_cfg: InteractiveSimConfig | None = None, controller_cfg: ControllerConfig | None = None):
        self.rb_cfg = rb_cfg or RigidBodyConfig(dt=0.005)
        self.ui_cfg = ui_cfg or InteractiveSimConfig(physics_dt=self.rb_cfg.dt, controller_dt=0.01)
        self.controller_cfg = controller_cfg or ControllerConfig()
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
        self.mode = ControlMode.DIRECT
        self.thrust_curve = ThrottleToThrustModel(self.rb_cfg)
        self.commands = ManualCommands(throttle=self.thrust_curve.throttle_for_hover())
        self.targets = RuntimeTargets(self.rb_cfg.target_x, self.rb_cfg.target_z)
        self.loiter_shaper = LoiterInputShaper(self.controller_cfg)
        self.disturbance = Disturbance(force=np.zeros(2), moment=0.0)
        self.logger = InteractiveCSVLogger(self.ui_cfg.log_directory)

        self.state = self.plant.reset(np.array(self.ui_cfg.presets["F1"].state, dtype=float))
        self.targets.capture(self.state)
        self.sim_time = 0.0
        self.speed = self.ui_cfg.initial_speed
        self.paused = False
        self.step_once = False
        self.slow_motion = False
        self.emergency_cut = False
        self.crash_reason = ""
        self.mode_status = "ready"
        self.camera_center = np.array([0.0, self.rb_cfg.target_z], dtype=float)
        self.camera_follow = True
        self.measured_real_time_factor = 0.0
        self.trace: list[tuple[float, float]] = []
        self.controller_time_remaining = 0.0
        self.last_control = _control_output(self.rb_cfg.hover_thrust, 0.0)
        self.last_motor = MotorOutput(self.rb_cfg.hover_thrust, 0.0, False)
        self.last_servo = ServoOutput(0.0, 0.0, 0.0, False, False)
        self.last_forces = self.plant.force_moment_breakdown(self.state)

    def set_mode(self, mode: ControlMode) -> None:
        if mode == self.mode:
            return
        previous = self.mode
        if previous == ControlMode.DIRECT and mode == ControlMode.RATE:
            self.commands.omega_target = float(self.state[5])
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
        if previous == ControlMode.STABILIZE and mode == ControlMode.RATE:
            self.commands.omega_target = float(self.state[5])
        self.control.reset_pid_for_mode_change(self.commands.omega_target, float(self.state[5]))
        self.mode = mode
        self.mode_status = f"{previous.value} -> {mode.value}"
    def reset(self, preset_key: str = "F1") -> None:
        preset = self.ui_cfg.presets.get(preset_key, self.ui_cfg.presets["F1"])
        self.state = self.plant.reset(np.array(preset.state, dtype=float))
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
        pitch_axis = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
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
        if self.mode != ControlMode.LOITER:
            self.targets.loiter_active = False
            self.targets.loiter_braking_active = False
            self.targets.target_x_rate_cmd = 0.0
            return
        desired_vx = self.loiter_shaper.update(self.commands.stick_x, dt)
        self.targets.desired_vx = desired_vx
        self.targets.loiter_braking_active = self.loiter_shaper.braking_active
        self.targets.target_x_rate_cmd = desired_vx
        self.targets.target_x += desired_vx * dt
        if self.loiter_shaper.braking_active and abs(desired_vx) <= 0.02 and abs(float(self.state[3])) <= 0.08:
            self.targets.target_x = float(self.state[0])
    def physics_step(self, wall_time: float, real_time_factor: float) -> None:
        disturbance_force = self.disturbance.combined_force()
        disturbance_moment = self.disturbance.combined_moment()
        if self.controller_time_remaining <= 1e-12:
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

        x, z, theta, _vx, _vz, _omega, thrust, vane = self.state
        translational_energy = 0.5 * self.rb_cfg.m * (self.state[3] ** 2 + self.state[4] ** 2)
        rotational_energy = 0.5 * self.rb_cfg.Iyy * self.state[5] ** 2
        potential_energy = self.rb_cfg.m * self.rb_cfg.g * z
        total_energy = translational_energy + rotational_energy + potential_energy
        body_up, body_right = self.plant.body_axes(float(theta))
        cg = np.array([x, z])
        top = cg + self.rb_cfg.l * body_up
        bottom = cg - self.rb_cfg.l * body_up
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
        pygame.draw.circle(screen, (255, 80, 80), world_to_screen(cg), 5)
        pygame.draw.circle(screen, (20, 20, 20), world_to_screen(bottom), 5)

        if self.ui_cfg.show_vane_overlay:
            length = 0.24
            visual_scale = self.ui_cfg.vane_visual_scale
            neutral_dir = body_right
            actual_dir = np.cos(vane * visual_scale) * body_right + np.sin(vane * visual_scale) * body_up
            cmd_dir = np.cos(self.last_control.vane_angle_cmd * visual_scale) * body_right + np.sin(self.last_control.vane_angle_cmd * visual_scale) * body_up
            hinge = bottom - 0.02 * body_up
            pygame.draw.line(screen, (150, 150, 150), world_to_screen(hinge - 0.12 * neutral_dir), world_to_screen(hinge + 0.12 * neutral_dir), 1)
            if self.ui_cfg.show_vane_command_ghost:
                pygame.draw.line(screen, (210, 160, 235), world_to_screen(hinge), world_to_screen(hinge + length * cmd_dir), 2)
            pygame.draw.line(screen, (190, 90, 230), world_to_screen(hinge), world_to_screen(hinge + length * actual_dir), 5)
            labels = []
            if self.last_control.mixer.saturated or self.last_servo.angle_saturated:
                labels.append("SAT")
            if self.last_servo.rate_saturated:
                labels.append("RATE")
            if self.last_control.mixer.authority_limited:
                labels.append("AUTH")
            vane_text = f"actual {np.rad2deg(vane):.1f} deg cmd {np.rad2deg(self.last_control.vane_angle_cmd):.1f} deg {' '.join(labels)}"
            screen.blit(small_font.render(vane_text, True, (235, 220, 245)), world_to_screen(hinge + 0.18 * body_right - 0.12 * body_up))

        draw_arrow(bottom, self.last_forces.thrust_force / max(self.rb_cfg.hover_thrust, 1e-6), (245, 160, 55), 0.35)
        draw_arrow(bottom, self.last_forces.vane_force / max(self.rb_cfg.hover_thrust, 1e-6), (190, 90, 230), 0.6)
        draw_arrow(cg, self.last_forces.total_force / max(self.rb_cfg.hover_thrust, 1e-6), (80, 220, 130), 0.25)
        draw_arrow(cg, self.last_forces.disturbance_force / max(self.rb_cfg.hover_thrust, 1e-6), (255, 230, 80), 0.5)

        if self.ui_cfg.show_theta_target_vector:
            target_up = np.array([np.sin(self.last_control.theta_target), np.cos(self.last_control.theta_target)])
            pygame.draw.line(screen, (255, 255, 255), world_to_screen(cg), world_to_screen(cg + 0.45 * target_up), 1)

        lines = [
            "W/S throttle or climb  A/D mode stick  arrows force  Q/E moment  I/O impulse  X motor cut",
            "1 direct  2 rate  3 stabilize  4 alt_hold  5 loiter  Space pause  N step  R reset  L log",
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
    args = parser.parse_args()
    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(args.params)
    InteractiveApp(rb_cfg, ui_cfg, controller_cfg).run()


if __name__ == "__main__":
    main()
