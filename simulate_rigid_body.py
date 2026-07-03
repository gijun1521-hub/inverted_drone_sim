from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .actuators import FirstOrderMotor, VaneServo
    from .cascaded_controller import ArduPilotLikeController
    from .config import RigidBodyConfig
    from .rigid_body_logging import make_rigid_body_row, save_rigid_body_csv
    from .rigid_body_model import RigidBodySingleFan2D
except ImportError:  # pragma: no cover - supports direct script execution
    from actuators import FirstOrderMotor, VaneServo
    from cascaded_controller import ArduPilotLikeController
    from config import RigidBodyConfig
    from rigid_body_logging import make_rigid_body_row, save_rigid_body_csv
    from rigid_body_model import RigidBodySingleFan2D


def run_simulation(
    cfg: RigidBodyConfig | None = None,
    controller: ArduPilotLikeController | None = None,
    initial_state: np.ndarray | None = None,
):
    cfg = cfg or RigidBodyConfig()
    plant = RigidBodySingleFan2D(cfg)
    controller = controller or ArduPilotLikeController(cfg)
    motor = FirstOrderMotor(cfg.T_max, cfg.motor_time_constant)
    servo = VaneServo(
        dt=cfg.dt,
        angle_limit=cfg.vane_angle_max,
        rate_limit=cfg.vane_rate_limit,
        time_constant=cfg.servo_time_constant,
        deadband=cfg.servo_deadband,
        command_delay=cfg.servo_delay,
    )

    state = plant.reset(initial_state)
    controller.reset()
    servo.reset()

    times: list[float] = []
    states: list[np.ndarray] = []
    control_rows = []

    for i in range(int(cfg.max_time / cfg.dt)):
        t = i * cfg.dt
        control = controller.compute(state)
        motor_out = motor.update(state[6], control.thrust_cmd)
        servo_out = servo.update(state[7], control.vane_angle_cmd)
        forces = plant.force_moment_breakdown(state)

        times.append(t)
        states.append(state.copy())
        control_rows.append(
            make_rigid_body_row(
                t,
                state,
                cfg.target_x,
                cfg.target_z,
                control,
                motor_out,
                servo_out,
                forces,
            )
        )

        state = plant.step(motor_out.thrust_dot, servo_out.vane_angle_dot)
        if state[1] < 0.0 or abs(state[2]) > np.deg2rad(85.0):
            break

    return np.asarray(times), np.asarray(states), control_rows, cfg


def main() -> None:
    times, states, rows, cfg = run_simulation()
    results_dir = Path(__file__).resolve().parent / "results"
    csv_path = save_rigid_body_csv(rows, results_dir / "rigid_body_simulation.csv")
    final = states[-1]

    print("Rigid-body single-fan simulation complete")
    print(f"steps: {len(times)}")
    print(f"final x_cg: {final[0]: .3f} m")
    print(f"final z_cg: {final[1]: .3f} m")
    print(f"final theta: {np.rad2deg(final[2]): .3f} deg")
    print(f"final thrust: {final[6]: .3f} N")
    print(f"final vane: {np.rad2deg(final[7]): .3f} deg")
    print(f"mixer saturated frames: {sum(row['mixer_saturated'] for row in rows)}")
    print(f"servo rate saturated frames: {sum(row['servo_rate_saturated'] for row in rows)}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
