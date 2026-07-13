# ACTUATOR LAB

ACTUATOR LAB is an interactive 2D engineering and education mode for commanding the vane angle and moving-mass lateral position independently. The moving-mass control changes position along the rail; it never changes `moving_mass.mass_kg`.

## Start the lab

Use the dedicated 2 kg prototype profile:

```powershell
.venv\Scripts\python.exe interactive_sim.py --params params/moving_mass_prototype_2kg.json --actuator-lab
```

The startup flag creates and validates a copied effective rigid-body configuration before the plant is constructed. It enables the moving-mass actuator and total-COM geometry and disables the legacy gravity-offset moment. The JSON defaults remain unchanged.

The prototype values remain:

- total mass: `2.0 kg`, including the moving mass
- moving mass: `0.5 kg`
- physical rail limit: `+/-0.05 m`
- moving-mass body-up position: `+0.12 m`
- maximum moving-mass rate: `0.20 m/s`
- maximum moving-mass acceleration: `1.0 m/s^2`
- `H = 0.50 m` and `l = 0.25 m`

ACTUATOR LAB starts in the 11-state moving-mass representation with centered actual and target offsets, a zero vane command, and hover thrust. Pressing `6` in a normal 8-state run is refused with a message asking for a restart with `--actuator-lab`; total-COM physics is never toggled during a running simulation.

## Controls

| Key | Command |
| --- | --- |
| `A` / `D` | vane command `-0.5 deg` / `+0.5 deg` |
| `Shift+A` / `Shift+D` | vane command `-2.0 deg` / `+2.0 deg` |
| `V` | center vane command |
| `F` / `H` | moving-mass target `-1 mm` / `+1 mm` |
| `Shift+F` / `Shift+H` | moving-mass target `-5 mm` / `+5 mm` |
| `G` | center moving-mass target |
| `W` / `S` | increase / decrease throttle with the existing manual throttle logic |
| `Backspace` | center both lab commands and reset the existing manual commands |
| `R`, `F1`-`F6` | reset without retaining stale lab commands |
| `Space` | pause / resume |
| `N` | advance one physics step while paused |

The A/D/F/H controls are deterministic `KEYDOWN` increments, including while paused. The command target passes through a braking-aware moving-mass trajectory that always preserves the configured rail, rate, and acceleration limits. Planned fixed-target motion from rest reduces speed according to the remaining stopping distance and settles exactly without overshoot. An arbitrary target change while the mass is already moving quickly can produce physically unavoidable temporary overshoot; in that case the tracker continues acceleration-limited braking instead of teleporting the offset or resetting velocity. A target crossing is clamped exactly only when stopping to zero is possible within the current acceleration step. The vane and thrust still pass through the existing servo and motor dynamics.

## Manual limits and physical limits

The default UI limits are deliberately smaller than the plant limits:

- moving-mass command limit: `+/-10 mm`
- vane command limit: `+/-5 deg`

The effective moving-mass command limit is the smaller of the configured lab limit and `moving_mass.max_offset_m`. These limits constrain only manual commands. They do not change the physical rail limit or physical vane limit.

## Sign convention

- `theta > 0`: right / positive pitch
- `moving_mass_offset > 0`: moving mass on the body-right side
- at `theta = 0` with positive thrust, a positive mass offset produces a positive thrust-offset pitch moment
- a positive vane angle produces body-right side force at the lower vane and therefore a negative pitch moment

Therefore:

- moving mass right plus positive vane produces opposing moment components
- moving mass right plus negative vane produces reinforcing positive moment components

The information panel displays numeric signs, vane side-force direction, individual moments, total COM coordinates, and the expected total pitch direction. It uses `RIGHT / POSITIVE`, `LEFT / NEGATIVE`, or `NEAR ZERO` with a small moment deadband.

The `actuator-pair moment` is defined exactly as the thrust-offset moment plus the vane moment about the instantaneous total COM. Its `PAIR RIGHT / POSITIVE`, `PAIR LEFT / NEGATIVE`, or `PAIR NEAR ZERO` classification uses the same moment deadband. This actuator-only sum is the diagnostic for mass/vane cancellation; it deliberately excludes damping and disturbance moments. The existing `total pitch moment` includes damping and disturbances as well. Evaluate exact actuator cancellation only when moving-mass velocity and body angular velocity are both near zero.

## Visualization

The vehicle overlay distinguishes:

- body-fixed physical rail: solid line
- actual moving mass: filled circle with an `MM actual` label
- moving-mass target: X-shaped ghost with an `MM target` label
- instantaneous total COM: diamond with a `total COM` label
- fixed-body origin: outlined square
- vane side force: force arrow at the vane application point

In total-COM geometry mode, state `x/z` is the instantaneous total-system COM. The renderer derives the fixed-body origin from the reported total COM offset before drawing the body, rail, mass, target, thrust point, and vane point. It does not draw state `x/z` as both the total COM and fixed-body origin.

## Pause-and-step experiment

1. Press `Space` to pause.
2. Press `Shift+H` once to set the moving-mass target to `+5 mm`.
3. Press `N` repeatedly and watch the actual offset brake into `+5 mm` without a large overshoot.
4. Press `D` in `+0.5 deg` increments.
5. Continue single-stepping and observe whether the total moment approaches `NEAR ZERO`.

## Logging and limitations

Existing fields record the control mode, vane command and actual angle, moving-mass target and actual offset, moving-mass velocity, total COM position, thrust-offset moment, vane moment, and total moment. `actuator_lab_active` is the only new field. Replay remains compatible with older CSV files that lack it.

ACTUATOR LAB is an analytical 2D tool, not real-flight validation. The current model retains fixed `Iyy` and does not include position-dependent inertia, reaction kick, moving-mass acceleration reaction, internal momentum coupling, rail/servo reaction forces, 3D dynamics, or real-flight calibration.
