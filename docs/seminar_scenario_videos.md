# Seminar scenario videos

This workflow creates deterministic, headless comparisons from the same plant, controller, actuator, safety, and logging code used by the LOITER analysis. It does not record the pygame window or contain flip maneuvers.

## Run it

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_seminar_videos.py
```

For a reduced-resolution renderer check:

```powershell
.venv\Scripts\python.exe generate_seminar_videos.py --smoke
```

The smoke command still reruns the full four-direction gain selection; only the rendered comparison is shortened. The notebook `notebooks/seminar_scenario_videos.ipynb` runs the same workflow from top to bottom and links every output.

## Controller and physical comparison

The authoritative controller source is `params/loiter_transient_provisional.json`:

- rate PID `P=0.070`, `I=0.000`, `D=0.008`;
- angle P `10.0`;
- horizontal position/velocity P `0.55/0.70`;
- brake delay `0.50 s`, acceleration `1.00 m/s^2`, jerk `3.00 m/s^3`;
- capture velocity thresholds `0.08 m/s` actual and `0.02 m/s` desired;
- persistent capture, shaped-velocity target clamp, and capture-without-target-jump enabled.

The workflow applies only explicit seminar vehicle overrides: total mass `2.0 kg`, physical moving mass `0.5 kg`, body-up position `0.12 m`, rail `+/-0.05 m`, rate `0.2 m/s`, acceleration `1.0 m/s^2`, total-COM geometry enabled, and legacy gravity-offset moment disabled. Canonical profiles are not modified.

“Vane-only” retains the physical 0.5 kg mass but commands and maintains exactly 0 mm offset with zero assist gain. “Moving-mass assist” uses the identical vehicle and controller with the selected `0.1325 m/Nm` gain.

## Gain selection

The generator performs a deterministic staged sweep over four mirrored cases: `+8 N`, `-8 N`, `+1 m`, and `-1 m`. It begins with `0.000`, `0.025`, `0.040`, `0.055`, `0.070`, `0.085`, and `0.100 m/Nm`, expands in `0.005 m/Nm` increments when no coarse candidate passes, then refines in `0.0025 m/Nm` increments around the best accepted value.

Candidates are rejected for non-finite values, crash or ground contact, defined premature pause, a second acceleration lobe after a full pause, early reversal, rail saturation, more than 5% vane saturation, capture target discontinuity, or parameter mismatch. Accepted candidates are scored across all requested tail, position, pitch, vane, mass-travel, tracking, and strict-settling metrics. The exact results and remaining limitations are in `moving_mass_gain_selection.md`.

## Exact rendered scenarios

Both scenarios start at `x=0 m`, `z=1 m`, pitch `0 deg`, run for exactly 8 seconds, and hold `z=1 m`.

- LOITER disturbance recovery holds `x=0 m` and applies `+8 N` in the world x direction from `t=1.5 s` through `t<1.7 s`.
- The absolute target case changes `target_x` directly from `0` to `+1.0 m` at exactly `t=1.0 s` and holds it. No external minimum-jerk or smoothed trajectory replaces the step.

The camera remains fixed at `x=[-1.5,+2.0] m`, `z=[0.0,+2.5] m`. Four synchronized 960x540 panels form a 1920x1080 composite at 30 FPS. H.264/yuv420p MP4 is produced through system FFmpeg or `imageio-ffmpeg`; GIF and PNG remain available if no encoder is present.

## Metrics and outputs

Outputs are written to `results/analysis/seminar_videos/`:

- composite MP4/GIF and thumbnail PNG;
- four individual scenario/variant MP4 files;
- `scenario_metrics.csv` and `scenario_summary.md`;
- `moving_mass_gain_sweep.csv` and `moving_mass_gain_selection.md`;
- `manifest.json` with hashes, controller values, selected gain, sweep fingerprint, render settings, and exact run metrics.

The tail window is the final 2 seconds. Strict settling requires the remainder of the run to stay within `0.05 m` horizontal target error and `0.05 m/s` horizontal speed. An unsettled result is reported honestly at the finite 8-second observation limit. Percentage changes use `100 * (assist - vane_only) / abs(vane_only)`; negative values indicate a reduction.

## Model limits

This deterministic 2D comparison is not hardware or flight-safety validation. It does not cover full 3D coupling, structural flexibility, calibration uncertainty, moving-mass reaction kick, ground effect, or real atmospheric turbulence.
