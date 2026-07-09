from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from .actuators import MotorOutput, ServoOutput
    from .cascaded_controller import RigidBodyControlOutput
    from .rigid_body_model import ForceMomentBreakdown
except ImportError:  # pragma: no cover - supports direct script execution
    from actuators import MotorOutput, ServoOutput
    from cascaded_controller import RigidBodyControlOutput
    from rigid_body_model import ForceMomentBreakdown


FIELDS = [
    "time",
    "x_cg",
    "z_cg",
    "theta_deg",
    "vx",
    "vz",
    "omega",
    "thrust",
    "vane_angle_deg",
    "target_x",
    "target_z",
    "theta_target_deg",
    "omega_target",
    "ax_target",
    "thrust_cmd",
    "vane_angle_cmd_deg",
    "thrust_force_x",
    "thrust_force_z",
    "vane_force_x",
    "vane_force_z",
    "drag_force_x",
    "drag_force_z",
    "total_force_x",
    "total_force_z",
    "vane_moment",
    "damping_moment",
    "moving_mass_moment",
    "total_moment",
    "moving_mass_offset_m",
    "moving_mass_velocity_m_s",
    "moving_mass_target_m",
    "moving_mass_saturated",
    "x_ddot",
    "z_ddot",
    "theta_ddot",
    "desired_moment",
    "rate_error",
    "rate_p",
    "rate_i",
    "rate_d",
    "rate_ff",
    "mixer_unattainable_moment",
    "motor_saturated",
    "servo_angle_saturated",
    "servo_rate_saturated",
    "mixer_saturated",
]


def make_rigid_body_row(
    time: float,
    state: np.ndarray,
    target_x: float,
    target_z: float,
    control: RigidBodyControlOutput,
    motor: MotorOutput,
    servo: ServoOutput,
    forces: ForceMomentBreakdown,
) -> dict[str, float | int]:
    return {
        "time": float(time),
        "x_cg": float(state[0]),
        "z_cg": float(state[1]),
        "theta_deg": float(np.rad2deg(state[2])),
        "vx": float(state[3]),
        "vz": float(state[4]),
        "omega": float(state[5]),
        "thrust": float(state[6]),
        "vane_angle_deg": float(np.rad2deg(state[7])),
        "target_x": float(target_x),
        "target_z": float(target_z),
        "theta_target_deg": float(np.rad2deg(control.theta_target)),
        "omega_target": float(control.omega_target),
        "ax_target": float(control.ax_target),
        "thrust_cmd": float(control.thrust_cmd),
        "vane_angle_cmd_deg": float(np.rad2deg(control.vane_angle_cmd)),
        "thrust_force_x": float(forces.thrust_force[0]),
        "thrust_force_z": float(forces.thrust_force[1]),
        "vane_force_x": float(forces.vane_force[0]),
        "vane_force_z": float(forces.vane_force[1]),
        "drag_force_x": float(forces.drag_force[0]),
        "drag_force_z": float(forces.drag_force[1]),
        "total_force_x": float(forces.total_force[0]),
        "total_force_z": float(forces.total_force[1]),
        "vane_moment": float(forces.vane_moment),
        "damping_moment": float(forces.damping_moment),
        "moving_mass_moment": float(forces.moving_mass_moment),
        "total_moment": float(forces.total_moment),
        "moving_mass_offset_m": float(forces.moving_mass_offset_m),
        "moving_mass_velocity_m_s": float(forces.moving_mass_velocity_m_s),
        "moving_mass_target_m": float(forces.moving_mass_target_m),
        "moving_mass_saturated": int(forces.moving_mass_saturated),
        "x_ddot": float(forces.x_ddot),
        "z_ddot": float(forces.z_ddot),
        "theta_ddot": float(forces.theta_ddot),
        "desired_moment": float(control.desired_moment),
        "rate_error": float(control.rate_error),
        "rate_p": float(control.rate_p),
        "rate_i": float(control.rate_i),
        "rate_d": float(control.rate_d),
        "rate_ff": float(control.rate_ff),
        "mixer_unattainable_moment": float(control.mixer.unattainable_moment),
        "motor_saturated": int(motor.saturated),
        "servo_angle_saturated": int(servo.angle_saturated),
        "servo_rate_saturated": int(servo.rate_saturated),
        "mixer_saturated": int(control.mixer.saturated),
    }


def save_rigid_body_csv(rows: list[dict[str, float | int]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
