from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from ..actuators import FirstOrderMotor, VaneServo
    from ..config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from ..interactive_logging import INTERACTIVE_FIELDS, interactive_row
    from ..interactive_sim import (
        ControlMode,
        LoiterInputShaper,
        ManualCommands,
        ManualControlSystem,
        RuntimeTargets,
    )
    from ..params import apply_dataclass_overrides, load_interactive_config
    from ..rigid_body_model import RigidBodySingleFan2D
    from ..safety import check_safety
    from ..thrust_curve import ThrottleToThrustModel
except ImportError:  # pragma: no cover - supports top-level script execution
    from actuators import FirstOrderMotor, VaneServo
    from config import ControllerConfig, InteractiveSimConfig, RigidBodyConfig
    from interactive_logging import INTERACTIVE_FIELDS, interactive_row
    from interactive_sim import ControlMode, LoiterInputShaper, ManualCommands, ManualControlSystem, RuntimeTargets
    from params import apply_dataclass_overrides, load_interactive_config
    from rigid_body_model import RigidBodySingleFan2D
    from safety import check_safety
    from thrust_curve import ThrottleToThrustModel


@dataclass(frozen=True)
class LoiterScenarioConfig:
    name: str
    mode: str = "LOITER"
    duration_s: float = 8.0
    settle_time_s: float = 2.0
    stick_start_s: float = 0.5
    stick_end_s: float = 2.0
    stick_x: float = 0.0
    stick_z: float = 0.0
    disturbance_start_s: float = 0.0
    disturbance_duration_s: float = 0.0
    disturbance_force_x: float = 0.0
    disturbance_force_z: float = 0.0
    disturbance_moment: float = 0.0
    initial_x: float = 0.0
    initial_z: float = 1.0
    initial_theta_deg: float = 0.0
    initial_vx: float = 0.0
    initial_vz: float = 0.0
    initial_omega_deg_s: float = 0.0
    target_x: float = 0.0
    target_z: float = 1.0
    capture_current_target: bool = False
    max_final_x_error: float = 0.45
    max_final_z_error: float = 0.35
    max_rms_x_error: float = 1.0
    max_rms_z_error: float = 0.8
    max_theta_deg_limit: float = 55.0
    max_saturation_percent: float = 85.0
    vane_angle_max_deg: float | None = None
    vane_rate_limit_deg_s: float | None = None
    T_max_factor: float | None = None
    moving_mass_enabled: bool = False
    moving_mass_target_m: float = 0.0
    moving_mass_assist_gain_m_per_Nm: float = 0.0
    notes: str = ""


@dataclass(frozen=True)
class LoiterRunResult:
    param_file: str
    scenario: LoiterScenarioConfig
    rows: list[dict[str, float | int | str]]
    metrics: dict[str, float | int | str | bool]
    crashed: bool
    crash_reason: str


def default_loiter_scenarios(duration_s: float | None = None) -> list[LoiterScenarioConfig]:
    scenarios = [
        LoiterScenarioConfig(
            name="stick_move_release",
            duration_s=9.0,
            stick_start_s=0.5,
            stick_end_s=2.2,
            stick_x=0.65,
            capture_current_target=True,
            max_final_x_error=0.55,
            notes="Pilot stick moves target, then releases for braking and hold.",
        ),
        LoiterScenarioConfig(
            name="initial_x_offset_recovery",
            duration_s=8.0,
            initial_x=1.2,
            target_x=0.0,
            max_final_x_error=0.55,
            notes="Starts offset from the hold point with no pilot input.",
        ),
        LoiterScenarioConfig(
            name="horizontal_impulse_recovery",
            duration_s=8.0,
            disturbance_start_s=1.0,
            disturbance_duration_s=0.20,
            disturbance_force_x=8.0,
            max_final_x_error=0.45,
            notes="Short world-frame horizontal force impulse.",
        ),
        LoiterScenarioConfig(
            name="vertical_offset_althold",
            mode="LOITER",
            duration_s=7.0,
            initial_z=1.45,
            target_z=1.0,
            max_final_z_error=0.25,
            notes="Vertical offset recovery using the altitude controller path.",
        ),
        LoiterScenarioConfig(
            name="low_authority_probe",
            duration_s=7.0,
            disturbance_start_s=0.8,
            disturbance_duration_s=0.25,
            disturbance_force_x=7.0,
            vane_angle_max_deg=8.0,
            vane_rate_limit_deg_s=90.0,
            max_final_x_error=1.25,
            max_saturation_percent=100.0,
            notes="Expected to expose saturation or authority limits rather than perfect hold.",
        ),
    ]
    if duration_s is None:
        return scenarios
    return [replace(s, duration_s=duration_s) for s in scenarios]


def authority_stress_scenario(duration_s: float | None = None) -> LoiterScenarioConfig:
    scenario = LoiterScenarioConfig(
        name="authority_stress",
        duration_s=7.0,
        initial_x=1.6,
        initial_z=0.85,
        target_x=0.0,
        target_z=1.2,
        disturbance_start_s=0.8,
        disturbance_duration_s=0.45,
        disturbance_force_x=12.0,
        max_final_x_error=2.1,
        max_final_z_error=0.35,
        max_rms_x_error=3.0,
        max_rms_z_error=1.2,
        max_theta_deg_limit=80.0,
        max_saturation_percent=100.0,
        notes="Authority stress case with low altitude margin, initial x offset, and horizontal impulse.",
    )
    return replace(scenario, duration_s=duration_s) if duration_s is not None else scenario


def moving_mass_pitch_assist_scenario(duration_s: float | None = None) -> LoiterScenarioConfig:
    scenario = LoiterScenarioConfig(
        name="pitch_assist_probe",
        duration_s=7.0,
        initial_x=1.0,
        target_x=0.0,
        moving_mass_enabled=True,
        moving_mass_assist_gain_m_per_Nm=0.025,
        max_final_x_error=0.75,
        max_saturation_percent=100.0,
        notes="Headless probe for comparing vane-only behavior with optional moving-mass pitch assist.",
    )
    return replace(scenario, duration_s=duration_s) if duration_s is not None else scenario


def scenario_by_name(name: str, duration_s: float | None = None) -> LoiterScenarioConfig:
    for scenario in default_loiter_scenarios(duration_s):
        if scenario.name == name:
            return scenario
    if name == "authority_stress":
        return authority_stress_scenario(duration_s)
    if name == "pitch_assist_probe":
        return moving_mass_pitch_assist_scenario(duration_s)
    names = ", ".join(s.name for s in default_loiter_scenarios())
    raise ValueError(f"unknown scenario {name!r}; expected one of: {names}, authority_stress, pitch_assist_probe")


def _apply_overrides(instance, overrides: dict | None):
    if not overrides:
        return instance
    return apply_dataclass_overrides(instance, overrides, "headless override")


def _merge_nested(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _initial_state(cfg: RigidBodyConfig, scenario: LoiterScenarioConfig) -> np.ndarray:
    return np.array(
        [
            scenario.initial_x,
            scenario.initial_z,
            math.radians(scenario.initial_theta_deg),
            scenario.initial_vx,
            scenario.initial_vz,
            math.radians(scenario.initial_omega_deg_s),
            cfg.hover_thrust,
            0.0,
        ],
        dtype=float,
    )


def _update_loiter_targets(
    mode: ControlMode,
    state: np.ndarray,
    commands: ManualCommands,
    targets: RuntimeTargets,
    shaper: LoiterInputShaper,
    dt: float,
) -> None:
    if mode != ControlMode.LOITER:
        targets.loiter_active = False
        targets.loiter_braking_active = False
        targets.target_x_rate_cmd = 0.0
        return
    desired_vx = shaper.update(commands.stick_x, dt)
    targets.desired_vx = desired_vx
    targets.loiter_braking_active = shaper.braking_active
    targets.target_x_rate_cmd = desired_vx
    targets.target_x += desired_vx * dt
    if shaper.braking_active and abs(desired_vx) <= 0.02 and abs(float(state[3])) <= 0.08:
        targets.target_x = float(state[0])


def run_headless_loiter(
    param_path: str | Path | None = "params/loiter_example.json",
    scenario: LoiterScenarioConfig | str | None = None,
    *,
    rb_overrides: dict | None = None,
    ui_overrides: dict | None = None,
    controller_overrides: dict | None = None,
) -> LoiterRunResult:
    scenario_cfg = scenario_by_name(scenario) if isinstance(scenario, str) else scenario
    if scenario_cfg is None:
        scenario_cfg = scenario_by_name("stick_move_release")

    rb_cfg, ui_cfg, controller_cfg = load_interactive_config(param_path)
    scenario_rb_overrides = {
        key: value
        for key, value in {
            "vane_angle_max_deg": scenario_cfg.vane_angle_max_deg,
            "vane_rate_limit_deg_s": scenario_cfg.vane_rate_limit_deg_s,
            "T_max_factor": scenario_cfg.T_max_factor,
        }.items()
        if value is not None
    }
    if scenario_cfg.moving_mass_enabled:
        scenario_rb_overrides["moving_mass"] = {"enabled": True}
    rb_overrides = _merge_nested(scenario_rb_overrides, rb_overrides or {})
    rb_cfg = _apply_overrides(rb_cfg, rb_overrides)
    ui_cfg = _apply_overrides(ui_cfg, ui_overrides)
    controller_cfg = _apply_overrides(controller_cfg, controller_overrides)

    plant = RigidBodySingleFan2D(rb_cfg)
    motor = FirstOrderMotor(rb_cfg.T_max, rb_cfg.motor_time_constant)
    servo = VaneServo(
        dt=rb_cfg.dt,
        angle_limit=rb_cfg.vane_angle_max,
        rate_limit=rb_cfg.vane_rate_limit,
        time_constant=rb_cfg.servo_time_constant,
        deadband=rb_cfg.servo_deadband,
        command_delay=rb_cfg.servo_delay,
    )
    control = ManualControlSystem(rb_cfg, controller_cfg)
    thrust_curve = ThrottleToThrustModel(rb_cfg)
    commands = ManualCommands(throttle=thrust_curve.throttle_for_hover())
    loiter_shaper = LoiterInputShaper(controller_cfg)

    state = plant.reset(_initial_state(rb_cfg, scenario_cfg))
    targets = RuntimeTargets(scenario_cfg.target_x, scenario_cfg.target_z)
    if scenario_cfg.capture_current_target:
        targets.capture(state)
    else:
        targets.target_x = scenario_cfg.target_x
        targets.target_z = scenario_cfg.target_z
    targets.altitude_hold_active = scenario_cfg.mode in {"ALT_HOLD", "LOITER"}
    targets.loiter_active = scenario_cfg.mode == "LOITER"
    mode = ControlMode(scenario_cfg.mode)
    if mode in (ControlMode.ALT_HOLD, ControlMode.LOITER):
        control.attitude.reset_to_current(float(state[2]), float(state[5]))
    control.reset_pid_for_mode_change(commands.omega_target, float(state[5]))

    rows: list[dict[str, float | int | str]] = []
    sim_time = 0.0
    controller_time_remaining = 0.0
    last_control = control.compute(mode, state, commands, ui_cfg.controller_dt, targets)
    last_motor = motor.update(float(state[6]), last_control.thrust_cmd)
    last_servo = servo.update(float(state[7]), last_control.vane_angle_cmd)
    moving_mass_target = float(scenario_cfg.moving_mass_target_m)
    crash_reason = ""
    min_body_z = float("nan")
    steps = int(math.ceil(scenario_cfg.duration_s / rb_cfg.dt))

    for _step in range(steps):
        commands.stick_x = scenario_cfg.stick_x if scenario_cfg.stick_start_s <= sim_time < scenario_cfg.stick_end_s else 0.0
        commands.stick_z = scenario_cfg.stick_z if scenario_cfg.stick_start_s <= sim_time < scenario_cfg.stick_end_s else 0.0
        if scenario_cfg.disturbance_start_s <= sim_time < scenario_cfg.disturbance_start_s + scenario_cfg.disturbance_duration_s:
            disturbance_force = np.array([scenario_cfg.disturbance_force_x, scenario_cfg.disturbance_force_z], dtype=float)
            disturbance_moment = scenario_cfg.disturbance_moment
        else:
            disturbance_force = np.zeros(2, dtype=float)
            disturbance_moment = 0.0

        if controller_time_remaining <= 1e-12:
            _update_loiter_targets(mode, state, commands, targets, loiter_shaper, ui_cfg.controller_dt)
            last_control = control.compute(mode, state, commands, ui_cfg.controller_dt, targets)
            if rb_cfg.moving_mass.enabled:
                if scenario_cfg.moving_mass_assist_gain_m_per_Nm:
                    moving_mass_target = scenario_cfg.moving_mass_assist_gain_m_per_Nm * last_control.desired_moment
                else:
                    moving_mass_target = scenario_cfg.moving_mass_target_m
            controller_time_remaining += ui_cfg.controller_dt

        last_motor = motor.update(float(state[6]), last_control.thrust_cmd + controller_cfg.motor_thrust_bias)
        servo_cmd = last_control.vane_angle_cmd + math.radians(controller_cfg.servo_bias_deg)
        last_servo = servo.update(float(state[7]), servo_cmd)
        state = plant.step(
            last_motor.thrust_dot,
            last_servo.vane_angle_dot,
            disturbance_force,
            disturbance_moment,
            moving_mass_target_m=moving_mass_target,
        )
        sim_time += rb_cfg.dt
        controller_time_remaining -= rb_cfg.dt

        safety = check_safety(state, rb_cfg)
        min_body_z = safety.min_body_z
        if safety.crashed:
            crash_reason = safety.reason
        forces = plant.force_moment_breakdown(state, disturbance_force, disturbance_moment)
        row = interactive_row(
            sim_time,
            sim_time,
            mode.value,
            state,
            commands.throttle,
            commands.direct_vane,
            commands.stick_x,
            commands.stick_z,
            targets,
            last_control,
            last_motor,
            last_servo,
            forces,
            crash_reason,
            min_body_z,
            rb_cfg.dt,
            ui_cfg.controller_dt,
            0.0,
        )
        row.update(
            {
                "time": row["sim_time"],
                "x": row["x_cg"],
                "z": row["z_cg"],
                "thrust_actual": row["thrust"],
                "vane_angle_actual": row["vane_angle"],
            }
        )
        rows.append(row)
        if crash_reason:
            break

    metrics = compute_loiter_metrics(rows, scenario_cfg, str(param_path or "<default>"))
    metrics.update(
        {
            "effective_vane_angle_max_deg": float(rb_cfg.vane_angle_max_deg),
            "effective_vane_angle_max_rad": float(rb_cfg.vane_angle_max),
            "effective_vane_rate_limit_deg_s": float(rb_cfg.vane_rate_limit_deg_s),
            "effective_vane_rate_limit_rad_s": float(rb_cfg.vane_rate_limit),
            "effective_T_max_factor": float(rb_cfg.T_max_factor),
            "effective_T_max_N": float(rb_cfg.T_max),
            "effective_hover_thrust_N": float(rb_cfg.hover_thrust),
            "moving_mass_enabled": bool(rb_cfg.moving_mass.enabled),
            "state_dimension": int(state.size),
            "total_mass_kg": float(rb_cfg.m),
            "effective_moving_mass_kg": float(rb_cfg.moving_mass.mass_kg),
            "effective_moving_mass_max_offset_m": float(rb_cfg.moving_mass.max_offset_m),
            "effective_moving_mass_max_rate_m_s": float(rb_cfg.moving_mass.max_rate_m_s),
            "effective_moving_mass_max_accel_m_s2": float(
                rb_cfg.moving_mass.max_accel_m_s2
            ),
            "effective_moving_mass_body_up_offset_m": float(
                rb_cfg.moving_mass.moving_mass_body_up_offset_m
            ),
            "total_com_geometry_active": bool(rb_cfg.moving_mass.use_total_com_geometry),
            "use_legacy_gravity_offset_moment": bool(
                rb_cfg.moving_mass.use_legacy_gravity_offset_moment
            ),
            "legacy_gravity_offset_active": bool(
                rb_cfg.moving_mass.enabled
                and rb_cfg.moving_mass.use_legacy_gravity_offset_moment
            ),
        }
    )
    return LoiterRunResult(str(param_path or "<default>"), scenario_cfg, rows, metrics, bool(crash_reason), crash_reason)


def _values(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    if not rows:
        return np.array([], dtype=float)
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _percent(rows: list[dict[str, float | int | str]], key: str) -> float:
    vals = _values(rows, key)
    return float(100.0 * np.mean(vals > 0.5)) if vals.size else 0.0


def compute_loiter_metrics(
    rows: list[dict[str, float | int | str]],
    scenario: LoiterScenarioConfig,
    param_file: str = "",
) -> dict[str, float | int | str | bool]:
    if not rows:
        return {
            "param_file": param_file,
            "scenario_name": scenario.name,
            "pass": False,
            "crash_reason": "no rows",
            "duration_s": 0.0,
            "notes": "No time-series rows were produced.",
        }
    x_error = _values(rows, "x_error")
    z_error = _values(rows, "z_error")
    theta = _values(rows, "theta")
    omega = _values(rows, "omega")
    vane_cmd = _values(rows, "vane_angle_cmd")
    vane_actual = _values(rows, "vane_angle_actual")
    thrust_cmd = _values(rows, "thrust_cmd")
    thrust_actual = _values(rows, "thrust_actual")
    moving_mass_offset = _values(rows, "moving_mass_offset_m")
    total_com_body_right = _values(rows, "total_com_body_right_m")
    total_com_body_up = _values(rows, "total_com_body_up_m")
    thrust_moment_from_com_offset = _values(rows, "thrust_moment_from_com_offset")
    vane_moment_about_total_com = _values(rows, "vane_moment_about_total_com")
    legacy_moving_mass_moment = _values(rows, "legacy_moving_mass_moment")
    final = rows[-1]
    crash_reason = str(final.get("crash_reason", ""))
    mixer_saturation_percent = _percent(rows, "mixer_saturated")
    authority_limited_percent = _percent(rows, "mixer_authority_limited")
    servo_rate_saturation_percent = _percent(rows, "servo_rate_saturated")
    saturation_peak = max(mixer_saturation_percent, authority_limited_percent, servo_rate_saturation_percent)
    passed = (
        not crash_reason
        and abs(float(final["x_error"])) <= scenario.max_final_x_error
        and abs(float(final["z_error"])) <= scenario.max_final_z_error
        and float(np.sqrt(np.mean(x_error * x_error))) <= scenario.max_rms_x_error
        and float(np.sqrt(np.mean(z_error * z_error))) <= scenario.max_rms_z_error
        and float(np.rad2deg(np.max(np.abs(theta)))) <= scenario.max_theta_deg_limit
        and saturation_peak <= scenario.max_saturation_percent
    )
    notes = scenario.notes
    if crash_reason:
        notes = f"{notes} Crash: {crash_reason}".strip()
    elif not passed:
        notes = f"{notes} Analytical threshold exceeded.".strip()

    return {
        "param_file": param_file,
        "scenario_name": scenario.name,
        "pass": bool(passed),
        "crash_reason": crash_reason,
        "duration_s": float(final["time"]),
        "max_abs_x_error": float(np.max(np.abs(x_error))),
        "final_abs_x_error": abs(float(final["x_error"])),
        "rms_x_error": float(np.sqrt(np.mean(x_error * x_error))),
        "max_abs_z_error": float(np.max(np.abs(z_error))),
        "final_abs_z_error": abs(float(final["z_error"])),
        "rms_z_error": float(np.sqrt(np.mean(z_error * z_error))),
        "max_theta_deg": float(np.rad2deg(np.max(np.abs(theta)))),
        "rms_theta_deg": float(np.rad2deg(np.sqrt(np.mean(theta * theta)))),
        "final_theta_deg": float(np.rad2deg(float(final["theta"]))),
        "max_omega_deg_s": float(np.rad2deg(np.max(np.abs(omega)))),
        "max_vane_cmd_deg": float(np.rad2deg(np.max(np.abs(vane_cmd)))),
        "max_vane_actual_deg": float(np.rad2deg(np.max(np.abs(vane_actual)))),
        "max_thrust_cmd_N": float(np.max(thrust_cmd)),
        "max_thrust_actual_N": float(np.max(thrust_actual)),
        "motor_saturation_percent": _percent(rows, "motor_saturated"),
        "servo_angle_saturation_percent": _percent(rows, "servo_angle_saturated"),
        "servo_rate_saturation_percent": servo_rate_saturation_percent,
        "mixer_saturation_percent": mixer_saturation_percent,
        "mixer_angle_saturation_percent": _percent(rows, "mixer_angle_saturated"),
        "authority_limited_percent": authority_limited_percent,
        "moving_mass_max_offset_m": float(np.max(np.abs(moving_mass_offset))) if moving_mass_offset.size else 0.0,
        "moving_mass_saturation_percent": _percent(rows, "moving_mass_saturated"),
        "max_abs_total_com_body_right_m": float(np.max(np.abs(total_com_body_right))),
        "max_abs_total_com_body_up_m": float(np.max(np.abs(total_com_body_up))),
        "max_abs_thrust_moment_from_com_offset": float(
            np.max(np.abs(thrust_moment_from_com_offset))
        ),
        "rms_thrust_moment_from_com_offset": float(
            np.sqrt(np.mean(thrust_moment_from_com_offset * thrust_moment_from_com_offset))
        ),
        "max_abs_vane_moment_about_total_com": float(
            np.max(np.abs(vane_moment_about_total_com))
        ),
        "max_abs_legacy_moving_mass_moment": float(
            np.max(np.abs(legacy_moving_mass_moment))
        ),
        "final_x": float(final["x"]),
        "final_z": float(final["z"]),
        "final_vx": float(final["vx"]),
        "final_vz": float(final["vz"]),
        "notes": notes,
    }


def save_loiter_timeseries(rows: Iterable[dict[str, float | int | str]], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys([*INTERACTIVE_FIELDS, "time", "x", "z", "thrust_actual", "vane_angle_actual"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
