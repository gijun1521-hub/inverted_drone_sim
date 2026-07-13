from __future__ import annotations

import csv
from pathlib import Path
from time import strftime

import numpy as np

try:
    from .actuators import MotorOutput, ServoOutput
    from .cascaded_controller import RigidBodyControlOutput
    from .rigid_body_model import ForceMomentBreakdown
except ImportError:  # pragma: no cover - supports direct script execution
    from actuators import MotorOutput, ServoOutput
    from cascaded_controller import RigidBodyControlOutput
    from rigid_body_model import ForceMomentBreakdown


INTERACTIVE_FIELDS = [
    "sim_time", "wall_time", "mode", "actuator_lab_active", "x_cg", "z_cg", "theta", "theta_wrapped", "vx", "vz", "omega",
    "thrust", "vane_angle", "throttle_cmd", "direct_vane_cmd", "theta_target", "omega_target",
    "rate_error", "rate_p", "rate_i", "rate_d", "rate_ff", "desired_moment", "achievable_moment",
    "physically_achievable_moment", "unattainable_moment", "anti_windup_correction", "integrator_inhibited",
    "thrust_cmd", "vane_angle_cmd", "thrust_force_x", "thrust_force_z", "vane_force_x", "vane_force_z",
    "disturbance_force_x", "disturbance_force_z", "disturbance_moment", "total_force_x", "total_force_z",
    "vane_moment", "damping_moment", "moving_mass_moment", "legacy_moving_mass_moment",
    "thrust_moment_from_com_offset", "vane_moment_about_total_com",
    "total_com_body_right_m", "total_com_body_up_m", "total_com_geometry_active", "total_moment",
    "moving_mass_offset_m", "moving_mass_velocity_m_s", "moving_mass_target_m", "moving_mass_saturated",
    "motor_saturated", "servo_angle_saturated",
    "servo_rate_saturated", "mixer_saturated", "mixer_angle_saturated", "mixer_authority_limited",
    "target_x", "target_z", "x_error", "z_error", "altitude_error", "distance_to_target",
    "desired_vx", "actual_vx", "desired_ax", "desired_vz", "actual_vz", "desired_az", "theta_target_limited",
    "loiter_active", "loiter_braking_active", "altitude_hold_active", "throttle_deadband_active",
    "stick_x", "stick_z", "target_x_rate_cmd", "target_z_rate_cmd", "actual_vane_angle",
    "mixer_authority_limited", "servo_angle_saturated", "servo_rate_saturated",
    "PSC.DesPosX", "PSC.PosX", "PSC.DesVelX", "PSC.VelX", "PSC.DesAccX", "PSC.AccX", "PSC.TargetTheta",
    "CTUN.DAlt", "CTUN.Alt", "CTUN.DCRt", "CTUN.CRt", "CTUN.ThO",
    "ATT.DesPitch", "ATT.Pitch", "RATE.DesPitch", "RATE.Pitch", "RATE.POut", "RATE.IOut", "RATE.DOut", "RATE.FFOut",
    "SERVO.VaneCmd", "SERVO.Vane", "MIX.Saturated", "MIX.AuthorityLimited",
    "crash_reason", "min_body_z", "physics_dt", "controller_dt", "real_time_factor",
]
# Preserve order while removing duplicates from legacy/new overlap.
INTERACTIVE_FIELDS = list(dict.fromkeys(INTERACTIVE_FIELDS))


def interactive_row(
    sim_time: float,
    wall_time: float,
    mode: str,
    state: np.ndarray,
    throttle_cmd: float,
    direct_vane_cmd: float,
    stick_x: float,
    stick_z: float,
    targets,
    control: RigidBodyControlOutput,
    motor: MotorOutput,
    servo: ServoOutput,
    forces: ForceMomentBreakdown,
    crash_reason: str,
    min_body_z: float,
    physics_dt: float,
    controller_dt: float,
    real_time_factor: float,
) -> dict[str, float | int | str]:
    x_error = float(targets.target_x - state[0])
    z_error = float(targets.target_z - state[1])
    distance = float((x_error * x_error + z_error * z_error) ** 0.5)
    row = {
        "sim_time": float(sim_time), "wall_time": float(wall_time), "mode": mode,
        "actuator_lab_active": int(mode == "ACTUATOR_LAB"),
        "x_cg": float(state[0]), "z_cg": float(state[1]), "theta": float(state[2]),
        "theta_wrapped": float((state[2] + np.pi) % (2.0 * np.pi) - np.pi),
        "vx": float(state[3]), "vz": float(state[4]), "omega": float(state[5]),
        "thrust": float(state[6]), "vane_angle": float(state[7]), "throttle_cmd": float(throttle_cmd),
        "direct_vane_cmd": float(direct_vane_cmd), "theta_target": float(control.theta_target),
        "omega_target": float(control.omega_target), "rate_error": float(control.rate_error),
        "rate_p": float(control.rate_p), "rate_i": float(control.rate_i), "rate_d": float(control.rate_d),
        "rate_ff": float(control.rate_ff), "desired_moment": float(control.desired_moment),
        "achievable_moment": float(control.mixer.achievable_moment),
        "physically_achievable_moment": float(control.mixer.physically_achievable_moment),
        "unattainable_moment": float(control.mixer.unattainable_moment),
        "anti_windup_correction": float(control.anti_windup_correction),
        "integrator_inhibited": int(control.integrator_inhibited), "thrust_cmd": float(control.thrust_cmd),
        "vane_angle_cmd": float(control.vane_angle_cmd), "thrust_force_x": float(forces.thrust_force[0]),
        "thrust_force_z": float(forces.thrust_force[1]), "vane_force_x": float(forces.vane_force[0]),
        "vane_force_z": float(forces.vane_force[1]), "disturbance_force_x": float(forces.disturbance_force[0]),
        "disturbance_force_z": float(forces.disturbance_force[1]), "disturbance_moment": float(forces.disturbance_moment),
        "total_force_x": float(forces.total_force[0]), "total_force_z": float(forces.total_force[1]),
        "vane_moment": float(forces.vane_moment), "damping_moment": float(forces.damping_moment),
        "moving_mass_moment": float(forces.moving_mass_moment),
        "legacy_moving_mass_moment": float(forces.legacy_moving_mass_moment),
        "thrust_moment_from_com_offset": float(forces.thrust_moment_from_com_offset),
        "vane_moment_about_total_com": float(forces.vane_moment_about_total_com),
        "total_com_body_right_m": float(forces.total_com_body_right_m),
        "total_com_body_up_m": float(forces.total_com_body_up_m),
        "total_com_geometry_active": int(forces.total_com_geometry_active),
        "total_moment": float(forces.total_moment),
        "moving_mass_offset_m": float(forces.moving_mass_offset_m),
        "moving_mass_velocity_m_s": float(forces.moving_mass_velocity_m_s),
        "moving_mass_target_m": float(forces.moving_mass_target_m),
        "moving_mass_saturated": int(forces.moving_mass_saturated),
        "motor_saturated": int(motor.saturated),
        "servo_angle_saturated": int(servo.angle_saturated), "servo_rate_saturated": int(servo.rate_saturated),
        "mixer_saturated": int(control.mixer.saturated), "mixer_angle_saturated": int(control.mixer.angle_saturated),
        "mixer_authority_limited": int(control.mixer.authority_limited),
        "target_x": float(targets.target_x), "target_z": float(targets.target_z), "x_error": x_error,
        "z_error": z_error, "altitude_error": z_error, "distance_to_target": distance,
        "desired_vx": float(targets.desired_vx), "actual_vx": float(state[3]), "desired_ax": float(targets.desired_ax),
        "desired_vz": float(targets.desired_vz), "actual_vz": float(state[4]), "desired_az": float(targets.desired_az),
        "theta_target_limited": float(targets.theta_target_limited), "loiter_active": int(targets.loiter_active),
        "loiter_braking_active": int(targets.loiter_braking_active), "altitude_hold_active": int(targets.altitude_hold_active),
        "throttle_deadband_active": int(targets.throttle_deadband_active), "stick_x": float(stick_x), "stick_z": float(stick_z),
        "target_x_rate_cmd": float(targets.target_x_rate_cmd), "target_z_rate_cmd": float(targets.target_z_rate_cmd),
        "actual_vane_angle": float(state[7]), "PSC.DesPosX": float(targets.target_x), "PSC.PosX": float(state[0]),
        "PSC.DesVelX": float(targets.desired_vx), "PSC.VelX": float(state[3]), "PSC.DesAccX": float(targets.desired_ax),
        "PSC.AccX": float(forces.total_force[0]), "PSC.TargetTheta": float(control.theta_target),
        "CTUN.DAlt": float(targets.target_z), "CTUN.Alt": float(state[1]), "CTUN.DCRt": float(targets.desired_vz),
        "CTUN.CRt": float(state[4]), "CTUN.ThO": float(control.thrust_cmd),
        "ATT.DesPitch": float(control.theta_target), "ATT.Pitch": float(state[2]),
        "RATE.DesPitch": float(control.omega_target), "RATE.Pitch": float(state[5]),
        "RATE.POut": float(control.rate_p), "RATE.IOut": float(control.rate_i), "RATE.DOut": float(control.rate_d),
        "RATE.FFOut": float(control.rate_ff), "SERVO.VaneCmd": float(control.vane_angle_cmd),
        "SERVO.Vane": float(state[7]), "MIX.Saturated": int(control.mixer.saturated),
        "MIX.AuthorityLimited": int(control.mixer.authority_limited), "crash_reason": crash_reason,
        "min_body_z": float(min_body_z), "physics_dt": float(physics_dt), "controller_dt": float(controller_dt),
        "real_time_factor": float(real_time_factor),
    }
    return row


class InteractiveCSVLogger:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.file = None
        self.writer = None
        self.path: Path | None = None

    @property
    def enabled(self) -> bool:
        return self.file is not None

    def start(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"interactive_{strftime('%Y%m%d_%H%M%S')}.csv"
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=INTERACTIVE_FIELDS)
        self.writer.writeheader()
        return self.path

    def stop(self) -> None:
        if self.file is not None:
            self.file.close()
        self.file = None
        self.writer = None

    def write(self, row: dict[str, float | int | str]) -> None:
        if self.writer is not None:
            self.writer.writerow(row)

    def close(self) -> None:
        self.stop()
