# 2D Inverted Drone Simulator

A small Python prototype for a 2D moving-base inverted-pendulum-style drone.

This project intentionally starts with a simple horizontal base acceleration
command, `ax_cmd`, instead of a full aerodynamic vane model. The first goal is
to verify sign conventions, geometry, dynamics, logging, and visualization
before tuning gains or adding hardware-specific effects.

## Control Model

The simulator now follows a moving-base inverted pendulum model:

```python
state = [x, z, theta, vx, vz, omega]
action = [throttle, ax_cmd]
```

`x, z` are the thrust point position. The center of gravity is:

```python
cg_x = x + l * sin(theta)
cg_z = z + l * cos(theta)
```

The attitude is stabilized by horizontally accelerating the thrust point,
similar to a cart-pole:

```python
x_ddot = ax_cmd
z_ddot = T / m - g
theta_ddot = (g * sin(theta) - x_ddot * cos(theta)) / l - damping * omega
```

Sign convention:

- `theta > 0`: CG is right of the thrust point.
- `theta < 0`: CG is left of the thrust point.
- If `theta < 0`, attitude control should initially command `ax_cmd < 0` so
  the thrust point moves left toward the CG.

## Setup

```bash
pip install -r requirements.txt
```

## Run PID Simulation

From this directory:

```bash
python simulate_pid.py
```

Or from the repository root:

```bash
python inverted_drone_sim/simulate_pid.py
```

Or open `../drone_simulation.ipynb` and run the cells from top to bottom.
The notebook also has a scenario dictionary you can edit for initial angle,
target height, side kicks, target changes, damping, and `ax_cmd` limits.

Outputs are written to `results/`:

- `states.png`
- `trajectory.png`
- `pid_animation.gif`
- `simulation.csv`

The CSV includes CG position, control-term breakdowns, `theta_ddot`, and an
`ax_saturated` flag so visual motion can be checked against the internal state.

## Run Tests

From the repository root:

```bash
python -m unittest discover -s tests
```

The tests check geometry sign conventions, upright hover acceleration, and
attitude-only stabilization before position control is trusted.

## Optional Passive Check

```bash
python simulate_passive.py
```

This runs the model with hover throttle and zero horizontal base acceleration so the initial tilt
falls away from upright, confirming the unstable inverted-pendulum term.

## Rigid-Body Single-Fan Model

The moving-base model remains as a conceptual baseline. A second model is
available for more physical single-fan work:

```bash
python simulate_rigid_body.py
```

This model uses the CG as the reference point and keeps actuator states in the
plant state:

```python
state = [x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

It computes real force and moment terms from fan thrust, vane side-force,
gravity, translational drag, angular damping, motor lag, servo lag/rate limits,
and a cascaded ArduPilot-like controller:

```text
position -> theta target -> rate target -> desired moment -> vane angle
altitude -> thrust target -> motor lag
```

Outputs are written to:

- `results/rigid_body_simulation.csv`

## Interactive Real-Time Simulator

Manual flight comes before position hold, gain sweeps, or reinforcement
learning. Run:

```bash
python interactive_sim.py
```

This pygame app uses a fixed-step physics loop independent of rendering. It
keeps keyboard input outside the plant: keys create actuator/controller
commands, while disturbances enter the plant as world-frame forces and pitch
moments.

Controls:

- `1`: direct actuator test
- `2`: rate / acro-like mode
- `3`: stabilize-like mode
- `4`: alt-hold placeholder
- `W/S`: increase/decrease throttle command
- `A/D`: vane, rate, or attitude command depending on mode
- arrow keys: continuous world-frame disturbance force
- `Q/E`: continuous pitch disturbance moment
- `I/O`: short force or pitch-moment impulse
- `X`: emergency motor cut
- `Space`: pause/resume
- `N`: single physics step while paused
- `R`: reset
- `F1`-`F6`: reset presets from `InteractiveSimConfig` in `config.py`
- `L`: start/stop timestamped CSV logging
- `[` / `]`: decrease/increase simulation speed
- `+` / `-`: zoom
- `C`: toggle camera follow
- `M`: toggle slow motion
- `Backspace`: reset manual commands
- `Esc`: quit

Interactive logs are written under `results/interactive_logs/`.

Replay a recorded run with:

```bash
python replay_interactive.py results/interactive_logs/<log>.csv
```

The replay tool creates an animation plus state, force/moment, and controller
term plots under `results/replay/`.

Hardening notes:

- controller shaping receives explicit controller `dt`
- attitude errors use shortest-angle wrapping via `wrap_pi`
- mode transitions seed rate/attitude targets to avoid stale PID kicks
- mixer reports floor-normalized command authority separately from physically
  achievable moment at the actual thrust
- low-thrust saturation feeds rate-PID anti-windup
- the vane model can be `linear_legacy` or `nonlinear_with_axial_loss`
- manual throttle commands go through a pluggable thrust-curve model
- safety checks pause the simulator on ground contact, state limits, or
  non-finite state values
