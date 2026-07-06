from __future__ import annotations

import numpy as np

try:
    from ..actuators import FirstOrderMotor, VaneServo
    from ..config import RigidBodyConfig
    from ..interactive_sim import ControlMode, ManualCommands, ManualControlSystem
    from ..rigid_body_model import RigidBodySingleFan2D
    from ..safety import check_safety
    from ..singlecopter_mixer import SingleCopterMixer
    from .metrics import ScenarioResult
except ImportError:  # pragma: no cover
    from actuators import FirstOrderMotor, VaneServo
    from config import RigidBodyConfig
    from interactive_sim import ControlMode, ManualCommands, ManualControlSystem
    from rigid_body_model import RigidBodySingleFan2D
    from safety import check_safety
    from singlecopter_mixer import SingleCopterMixer
    from validation.metrics import ScenarioResult


def direct_throttle_step(cfg: RigidBodyConfig) -> ScenarioResult:
    motor = FirstOrderMotor(cfg.T_max, cfg.motor_time_constant)
    up = motor.update(cfg.hover_thrust, 0.7 * cfg.T_max)
    down = motor.update(cfg.hover_thrust, 0.2 * cfg.T_max)
    passed = up.thrust_dot > 0.0 and down.thrust_dot < 0.0
    return ScenarioResult("DIRECT throttle step", passed, "thrust lag sign check", {"up_dot": up.thrust_dot, "down_dot": down.thrust_dot})


def direct_vane_step(cfg: RigidBodyConfig) -> ScenarioResult:
    plant = RigidBodySingleFan2D(cfg)
    pos = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.12])
    neg = pos.copy()
    neg[7] = -0.12
    a = plant.force_moment_breakdown(pos)
    b = plant.force_moment_breakdown(neg)
    passed = a.vane_force[0] > 0.0 and b.vane_force[0] < 0.0 and a.vane_moment < 0.0 and b.vane_moment > 0.0
    return ScenarioResult("DIRECT vane step", passed, "side force and moment signs", {"pos_moment": a.vane_moment, "neg_moment": b.vane_moment})


def rate_pitch_command(cfg: RigidBodyConfig) -> ScenarioResult:
    control = ManualControlSystem(cfg)
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])
    pos = control.compute(ControlMode.RATE, state, ManualCommands(0.4, omega_target=1.0), cfg.dt)
    control.reset()
    neg = control.compute(ControlMode.RATE, state, ManualCommands(0.4, omega_target=-1.0), cfg.dt)
    passed = pos.desired_moment > 0.0 and neg.desired_moment < 0.0
    return ScenarioResult("RATE pitch-rate command", passed, "desired moment follows omega target", {"pos": pos.desired_moment, "neg": neg.desired_moment})


def stabilize_initial_tilt(cfg: RigidBodyConfig) -> ScenarioResult:
    control = ManualControlSystem(cfg)
    pos_state = np.array([0.0, 1.0, np.deg2rad(5), 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0])
    neg_state = pos_state.copy()
    neg_state[2] = np.deg2rad(-5)
    pos = control.compute(ControlMode.STABILIZE, pos_state, ManualCommands(0.4, theta_target=0.0), cfg.dt)
    control.reset()
    neg = control.compute(ControlMode.STABILIZE, neg_state, ManualCommands(0.4, theta_target=0.0), cfg.dt)
    passed = pos.desired_moment < 0.0 and neg.desired_moment > 0.0
    return ScenarioResult("STABILIZE initial tilt recovery", passed, "restoring moment sign", {"pos": pos.desired_moment, "neg": neg.desired_moment})


def external_horizontal_impulse(cfg: RigidBodyConfig) -> ScenarioResult:
    plant = RigidBodySingleFan2D(cfg)
    plant.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))
    plant.step(0.0, 0.0, disturbance_force=np.array([5.0, 0.0]))
    vx_after = plant.state[3]
    terms = plant.force_moment_breakdown(plant.state)
    passed = vx_after > 0.0 and abs(terms.disturbance_force[0]) < 1e-12
    return ScenarioResult("External horizontal impulse", passed, "velocity changes and force disappears", {"vx_after": vx_after})


def external_pitch_impulse(cfg: RigidBodyConfig) -> ScenarioResult:
    plant = RigidBodySingleFan2D(cfg)
    plant.reset(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))
    plant.step(0.0, 0.0, disturbance_moment=0.2)
    omega_after = plant.state[5]
    terms = plant.force_moment_breakdown(plant.state)
    passed = omega_after > 0.0 and abs(terms.disturbance_moment) < 1e-12
    return ScenarioResult("External pitch moment impulse", passed, "angular velocity changes and moment disappears", {"omega_after": omega_after})


def low_thrust_authority(cfg: RigidBodyConfig) -> ScenarioResult:
    mixer = SingleCopterMixer(cfg.k_moment, cfg.vane_angle_max, cfg.thrust_control_floor)
    out = mixer.mix(0.2, thrust=0.0)
    return ScenarioResult("Low thrust authority", out.authority_limited and out.physically_achievable_moment == 0.0, "authority limited at zero thrust", {"unavailable": out.unattainable_moment})


def emergency_motor_cut(cfg: RigidBodyConfig) -> ScenarioResult:
    motor = FirstOrderMotor(cfg.T_max, cfg.motor_time_constant)
    plant = RigidBodySingleFan2D(cfg)
    plant.reset(np.array([0.0, 0.35, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))
    crashed = False
    for _ in range(1000):
        out = motor.update(plant.state[6], 0.0)
        plant.step(out.thrust_dot, 0.0)
        status = check_safety(plant.state, cfg)
        if status.crashed:
            crashed = True
            break
    return ScenarioResult("Emergency motor cut", crashed, "zero thrust command eventually crashes", {"final_z": plant.state[1]})


def nonlinear_vane_model(cfg: RigidBodyConfig) -> ScenarioResult:
    legacy = RigidBodyConfig(vane_model="linear_legacy")
    nonlinear = RigidBodyConfig(vane_model="nonlinear_with_axial_loss")
    state = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.35])
    a = RigidBodySingleFan2D(legacy).force_moment_breakdown(state)
    b = RigidBodySingleFan2D(nonlinear).force_moment_breakdown(state)
    passed = b.axial_force_magnitude < a.axial_force_magnitude
    return ScenarioResult("Nonlinear vane model", passed, "nonlinear model loses axial thrust", {"legacy_axial": a.axial_force_magnitude, "nonlinear_axial": b.axial_force_magnitude})


def long_finite_run(cfg: RigidBodyConfig) -> ScenarioResult:
    plant = RigidBodySingleFan2D(cfg)
    plant.reset(np.array([0.0, 1.2, 0.02, 0.0, 0.0, 0.0, cfg.hover_thrust, 0.0]))
    finite = True
    for _ in range(int(3.0 / cfg.dt)):
        plant.step(0.0, 0.0)
        finite = finite and bool(np.all(np.isfinite(plant.state)))
    return ScenarioResult("Long finite run", finite, "state remains finite", {"final_theta": plant.state[2]})


SCENARIOS = [
    direct_throttle_step,
    direct_vane_step,
    rate_pitch_command,
    stabilize_initial_tilt,
    external_horizontal_impulse,
    external_pitch_impulse,
    low_thrust_authority,
    emergency_motor_cut,
    nonlinear_vane_model,
    long_finite_run,
]
