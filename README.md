# 2D Inverted Drone Simulator

A small Python prototype for a 2D moving-base inverted-pendulum-style drone.

This project intentionally starts with a simple horizontal base acceleration
command, `ax_cmd`, instead of a full aerodynamic vane model. The first goal is
to verify sign conventions, geometry, dynamics, logging, and visualization
before tuning gains or adding hardware-specific effects.

The current direction is analytical rather than experimentally calibrated. The
models use nominal physical assumptions and parameter sweeps to study sign
correctness, conservation laws, relative authority, and feasibility trends. Do
not interpret outputs as exact flight predictions until bench data is added.

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

## Developer Commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python validate_sim.py
python analyze_authority.py
python analyze_moving_mass.py
python interactive_sim.py --params params/default_rigid_body.json
python replay_interactive.py results/interactive_logs/<log>.csv
```

Generated CSV, plots, GIFs, videos, JSON, and HTML under `results/` are ignored
by git. Validation writes `results/validation_summary.csv` and
`results/validation_report.md`; interactive logging writes under
`results/interactive_logs/`.

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
## Interactive LOITER Mode

Run the ArduCopter-inspired LOITER example with:

```bash
python interactive_sim.py --params params/loiter_example.json
```

Modes and controls:

- `1`: DIRECT
- `2`: RATE
- `3`: STABILIZE
- `4`: ALT_HOLD
- `5`: LOITER
- `A/D`: DIRECT commands vane angle; RATE commands pitch rate; STABILIZE and ALT_HOLD command lean angle; LOITER commands horizontal movement speed.
- `W/S`: DIRECT, RATE, and STABILIZE command raw throttle; ALT_HOLD and LOITER command climb/descent rate with a deadband.
- Arrow keys: external force disturbance.
- `Q/E`: pitch moment disturbance.
- `I/O`: impulse disturbance.
- `X`: emergency motor cut.
- `R`: reset.
- `F1-F6`: presets.
- `L`: CSV logging.

Vane visualization:

- Solid vane: actual servo angle.
- Ghost vane: commanded vane angle.
- Neutral line: zero-deflection reference.
- `SAT`: actuator or mixer saturation.
- `RATE`: servo rate limit.
- `AUTH`: mixer authority limited.
- `SAT/RATE/AUTH` only appears when actuator or mixer limits are active.
- `vane_visual_scale` may exaggerate the displayed angle and does not affect physics.
- `vane_visual_length_m` and `vane_visual_offset_m` change only the overlay, not physics.

The mode hierarchy is ArduCopter-inspired: Stabilize is lean-angle control, AltHold adds vertical target control, and Loiter adds horizontal position/velocity control. This is still a simplified 2D research simulator, not exact ArduPilot firmware and not experimentally calibrated.

Analyze a saved interactive log with:

```bash
python analyze_interactive_log.py results/interactive_logs/<log>.csv
```

## Headless LOITER Tuning Comparison

Run deterministic LOITER comparisons without opening pygame:

```bash
python compare_loiter_params.py
python sweep_loiter_authority.py
```

The comparison script runs the sluggish, nominal, and aggressive LOITER
parameter examples across repeatable scenarios and writes:

- `results/analysis/loiter_param_comparison.csv`
- `results/analysis/loiter_param_comparison.md`
- PNG plots when matplotlib is available

The authority sweep writes:

- `results/analysis/loiter_authority_sweep.csv`
- `results/analysis/loiter_authority_sweep.md`
- `results/analysis/loiter_authority_sweep.png` when matplotlib is available

Key metrics:

- `final_abs_x_error`: final horizontal hold error.
- `rms_x_error`: run-level horizontal tracking error.
- `max_theta_deg`: peak pitch demand/response.
- `max_vane_cmd_deg`: peak requested vane angle.
- `mixer_saturation_percent`: percent of samples where mixer output saturated.
- `authority_limited_percent`: requested moment exceeded current thrust/vane authority.
- `servo_rate_saturation_percent`: percent of samples clipped by servo rate limit.

These are analytical indicators, not calibrated real-flight predictions.
Saturation is not always a failure; it is a design signal. Compare results
relatively across parameter sets and scenarios. See
[docs/loiter_tuning_analysis.md](docs/loiter_tuning_analysis.md) for scenario
definitions, metric interpretation, and limitations.

Troubleshooting:

- If `pygame` install fails on Windows, try Python 3.11.
- If the motion looks too perfect, try `params/loiter_sluggish_example.json` or enable the zero-default noise parameters in the controller section.
- If LOITER does not return to target, check saturation, `authority_limited`, thrust-to-weight ratio, vane authority, and gains.
- If the vane is not visible enough, increase `interactive.vane_visual_scale`, `interactive.vane_visual_length_m`, or enable `interactive.show_vane_overlay`.
- Unknown parameter keys are reported by section. Structured files use `rigid_body`, `interactive`, and `controller` sections; old flat rigid-body JSON still works.

See also [docs/arducopter_alignment.md](docs/arducopter_alignment.md).
