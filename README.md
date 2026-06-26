# 2D Inverted Drone Simulator

A small Python prototype for a 2D moving-base inverted-pendulum-style drone.

The first milestone is PID stabilization with matplotlib plots and a GIF
animation. The code is kept modular so it can later be used as a Gymnasium
environment for reinforcement learning.

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

## Optional Passive Check

```bash
python simulate_passive.py
```

This runs the model with hover throttle and zero horizontal base acceleration so the initial tilt
falls away from upright, confirming the unstable inverted-pendulum term.
