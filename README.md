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
python -m unittest discover -s inverted_drone_sim/tests
```

The tests check geometry sign conventions, upright hover acceleration, and
attitude-only stabilization before position control is trusted.

## Optional Passive Check

```bash
python simulate_passive.py
```

This runs the model with hover throttle and zero horizontal base acceleration so the initial tilt
falls away from upright, confirming the unstable inverted-pendulum term.
