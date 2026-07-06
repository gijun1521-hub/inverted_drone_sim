from __future__ import annotations

import numpy as np

try:
    from ..actuators import FirstOrderMotor, VaneServo
    from ..config import RigidBodyConfig
    from ..analysis.moving_mass_sweep import sweep_rows
    from ..analysis.compare_actuators import compare_authority
    from ..config import MovingMassConfig
    from ..moving_mass_model import MovingMassSingleFan2D
    from ..interactive_sim import ControlMode, ManualCommands, ManualControlSystem
    from ..rigid_body_model import RigidBodySingleFan2D
    from ..safety import check_safety
    from ..singlecopter_mixer import SingleCopterMixer
    from .metrics import ScenarioResult
except ImportError:  # pragma: no cover
    from actuators import FirstOrderMotor, VaneServo
    from config import RigidBodyConfig
    from analysis.moving_mass_sweep import sweep_rows
    from analysis.compare_actuators import compare_authority
    from config import MovingMassConfig
    from moving_mass_model import MovingMassSingleFan2D
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


def reaction_only_no_external_torque(cfg: RigidBodyConfig) -> ScenarioResult:
    mm = MovingMassSingleFan2D(MovingMassConfig(thrust=0.0, g=0.0))
    mm.reset()
    for _ in range(30):
        mm.step(0.3, thrust=0.0)
    passed = abs(mm.last_breakdown.angular_momentum) < 1e-6
    return ScenarioResult("Analytical reaction-only momentum", passed, "angular momentum conservation", {"angular_momentum": mm.last_breakdown.angular_momentum})


def moving_mass_offset_with_thrust(cfg: RigidBodyConfig) -> ScenarioResult:
    mm = MovingMassSingleFan2D(MovingMassConfig())
    state = mm.reset()
    state[8] = 0.25
    mm.state = state
    mm.step(0.25)
    passed = abs(mm.last_breakdown.cg_offset_moment) > 0.0
    return ScenarioResult("Moving mass offset with thrust", passed, "sustained CG-offset moment", {"cg_offset_moment": mm.last_breakdown.cg_offset_moment})


def return_to_center_reaction(cfg: RigidBodyConfig) -> ScenarioResult:
    mm = MovingMassSingleFan2D(MovingMassConfig(thrust=0.0, g=0.0))
    mm.reset()
    first = 0.0
    for _ in range(10):
        mm.step(0.3, thrust=0.0)
        if abs(mm.last_breakdown.reaction_moment) > abs(first):
            first = mm.last_breakdown.reaction_moment
    mm.state[8] = 0.3
    mm.state[9] = 0.0
    for _ in range(20):
        mm.step(0.0, thrust=0.0)
        if mm.last_breakdown.reaction_moment * first < 0.0:
            break
    second = mm.last_breakdown.reaction_moment
    passed = first * second < 0.0
    return ScenarioResult("Return-to-center reaction", passed, "reverse reaction appears", {"first": first, "second": second})


def vane_vs_moving_mass_authority(cfg: RigidBodyConfig) -> ScenarioResult:
    comp = compare_authority()
    passed = comp.vane_moment > 0.0 and comp.moving_mass_reaction_moment > 0.0 and comp.moving_mass_cg_offset_moment > 0.0
    return ScenarioResult("Vane vs moving-mass authority", passed, "all analytical authority terms positive", comp.__dict__)


def hybrid_authority_margin(cfg: RigidBodyConfig) -> ScenarioResult:
    comp = compare_authority()
    desired = comp.vane_moment * 1.2
    passed = comp.hybrid_total_moment > desired
    return ScenarioResult("Hybrid authority margin", passed, "hybrid proxy exceeds target moment", {"desired": desired, "hybrid": comp.hybrid_total_moment})


def sensitivity_sweep_smoke(cfg: RigidBodyConfig) -> ScenarioResult:
    rows = sweep_rows()
    finite = all(np.isfinite(float(row["moving_mass_to_vane_ratio"])) for row in rows)
    low = [r for r in rows if abs(float(r["moving_mass_ratio"]) - 0.1) < 1e-9]
    high = [r for r in rows if abs(float(r["moving_mass_ratio"]) - 0.7) < 1e-9]
    passed = finite and np.mean([float(r["max_cg_offset"]) for r in high]) > np.mean([float(r["max_cg_offset"]) for r in low])
    return ScenarioResult("Sensitivity sweep smoke", passed, "finite outputs and mass-ratio trend", {"rows": len(rows)})


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
    reaction_only_no_external_torque,
    moving_mass_offset_with_thrust,
    return_to_center_reaction,
    vane_vs_moving_mass_authority,
    hybrid_authority_margin,
    sensitivity_sweep_smoke,
]
