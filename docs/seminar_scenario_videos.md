# Seminar scenario videos

This workflow creates deterministic, headless comparison videos from the same plant, controller, actuator, safety, and logging code used by the LOITER analysis. It does not record the pygame window and does not contain flip maneuvers.

## Run it

Install the repository dependencies in the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Generate the final 30 FPS artifacts:

```powershell
.venv\Scripts\python.exe generate_seminar_videos.py
```

Run a short reduced-resolution renderer check:

```powershell
.venv\Scripts\python.exe generate_seminar_videos.py --smoke
```

Open `notebooks/seminar_scenario_videos.ipynb`, select the project environment, and use **Run All**. The notebook locates the repository root, displays the selected profiles and assist gain, runs the same CLI, shows `scenario_metrics.csv`, previews the composite MP4 and GIF, and creates a clickable link for every artifact. It contains no hidden manual setup state.

## Video dependencies and fallback

The encoder search order is:

1. a system `ffmpeg` executable on `PATH`;
2. the executable supplied by `imageio-ffmpeg`;
3. GIF and PNG only, with a visible warning in the CLI output and `manifest.json`.

MP4 output is H.264 with `yuv420p` pixel format for PowerPoint compatibility. MP4 success is reported only after the encoder exits successfully and creates a non-empty file.

## Exact scenarios

Both scenarios start at `x=0 m`, `z=1 m`, `theta=0 deg`, run for 8 seconds with the profile physics timestep, hold `z=1 m`, and use LOITER mode.

- **LOITER horizontal-disturbance recovery:** target `x=0 m`; an `+8 N` world-frame horizontal force is applied from `t=1.5 s` through `t<1.7 s`. The magnitude is the existing validated `horizontal_impulse_recovery` magnitude; only its timing is changed for the seminar scenario.
- **+1 m position command and hold:** target `x=0 m` until exactly `t=1.0 s`, then the absolute target becomes `x=+1.0 m` and remains there.

The fixed camera covers `x=[-1.5,+2.0] m` and `z=[0.0,+2.5] m` in every panel. The four 960x540 panels share 30 FPS timestamps and form one 1920x1080 2x2 composite.

## Fair comparison conditions

Every run uses:

- total vehicle mass `2.0 kg`;
- physical moving mass `0.5 kg`;
- moving-mass body-up offset `0.12 m`;
- physical rail limit `+/-0.05 m`;
- total-COM geometry enabled;
- legacy gravity-offset moment disabled;
- identical initial state, duration, physics/controller timesteps, scenario target, disturbance, controller gains, camera, and render settings;
- `ATC_RAT_PIT_P=0.07`, `ATC_RAT_PIT_I=0`, `ATC_RAT_PIT_D=0.008`, `ATC_ANG_PIT_P=10`, `PSC_NE_POS_P=0.5`, and `PSC_NE_VEL_P=0.9`.

The workflow reads, but does not modify, `params/loiter_tuned_vane_only.json` and `params/moving_mass_prototype_2kg_tuned.json`.

“Mass locked at center” does **not** mean that the 0.5 kg mass is removed. The mass remains enabled in the model, contributes to total mass and total-COM geometry, and retains the physical rail parameters. Its initial offset, commanded offset, and actual offset stay at zero, and its proportional assist gain is zero.

The active variant changes only the analysis command law: `moving_mass_target_m = 0.055 m/Nm * desired_pitch_moment_Nm`, followed by the existing rail, rate, and acceleration limits. The `0.055 m/Nm` value is analysis metadata passed explicitly by the scenario runner; it is not added to controller parameters or global defaults.

## Metrics and outputs

Outputs are written to `results/analysis/seminar_videos/`:

- `seminar_scenario_comparison.mp4` and `.gif`;
- `loiter_locked.mp4`, `loiter_assist.mp4`;
- `forward_1m_locked.mp4`, `forward_1m_assist.mp4`;
- `seminar_video_thumbnail.png`;
- `scenario_metrics.csv`, `scenario_summary.md`, and `manifest.json`.

The documented tail window is the final 2 seconds. Settling begins when the disturbance ends (`t=1.7 s`) or when the target steps (`t=1.0 s`), and requires the remainder of the log to stay within `0.05 m` horizontal target error and `0.05 m/s` horizontal speed. An unsettled result uses the available observation-window length as a finite censored settling time and records `settled=false`.

For the disturbance run, `position_overshoot_m` is the peak absolute horizontal recovery excursion after the force ends. For the step run, it is the positive excursion beyond `x=+1.0 m`.

## Model limits

This is a deterministic 2D analytical comparison. It does not model full 3D roll/yaw coupling, swirl, four independent vane aerodynamics, structural flexibility, actuator calibration error, sensor latency beyond configured terms, moving-mass reaction kick, ground effect, or real atmospheric turbulence. The results compare two configurations inside this model and must not be presented as real-flight equivalence, flight-safety evidence, or hardware validation.
