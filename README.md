# 2D 역진자 드론 시뮬레이터

이 저장소는 **2D single-fan / inverted-drone 시뮬레이터**를 만들기 위한 작은 Python 프로토타입입니다.

현재 프로젝트는 크게 두 층으로 나뉩니다.

- `ax_cmd`를 사용하는 단순 이동 기반 역진자 baseline
- 추력, 등가 pitch축 vane, motor lag, servo lag, cascaded LOITER 제어, saturation reporting, authority mapping을 포함한 더 물리적인 2D single-fan rigid-body 모델

현재 목표는 **분석적으로 명확한 시뮬레이터 기반**을 만드는 것입니다. 하드웨어 고유 효과를 넣기 전에 sign convention, geometry, dynamics, logging, visualization, relative authority를 먼저 확인합니다.

현재 방향은 실험 보정 기반이 아니라 분석 기반입니다. 모델은 명목상의 물리 가정과 parameter sweep을 사용해 sign correctness, conservation law, relative authority, feasibility trend를 봅니다. 벤치 테스트 데이터가 추가되기 전까지 결과를 실제 비행 성능 예측으로 해석하면 안 됩니다.

## Baseline Control Model

초기 baseline은 moving-base inverted pendulum 모델을 따릅니다.

```python
state = [x, z, theta, vx, vz, omega]
action = [throttle, ax_cmd]
```

`x, z`는 thrust point 위치입니다. 중심 of gravity, 즉 CG는 다음과 같이 계산합니다.

```python
cg_x = x + l * sin(theta)
cg_z = z + l * cos(theta)
```

자세는 thrust point를 수평 가속시키는 방식으로 안정화합니다. 개념적으로는 cart-pole과 비슷합니다.

```python
x_ddot = ax_cmd
z_ddot = T / m - g
theta_ddot = (g * sin(theta) - x_ddot * cos(theta)) / l - damping * omega
```

Sign convention:

- `theta > 0`: CG가 thrust point의 오른쪽에 있습니다.
- `theta < 0`: CG가 thrust point의 왼쪽에 있습니다.
- `theta < 0`일 때 자세 제어는 초기에 `ax_cmd < 0`을 명령해야 합니다. 즉 thrust point가 CG 쪽인 왼쪽으로 움직여야 합니다.

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

`results/` 아래에 생성되는 CSV, plot, GIF, video, JSON, HTML 파일은 git에서 무시됩니다. Validation은 `results/validation_summary.csv`와 `results/validation_report.md`를 만들고, interactive logging은 `results/interactive_logs/` 아래에 저장됩니다.

## PID Simulation 실행

이 디렉터리에서 실행할 때:

```bash
python simulate_pid.py
```

Repository root에서 실행할 때:

```bash
python inverted_drone_sim/simulate_pid.py
```

또는 `../drone_simulation.ipynb`를 열고 위에서 아래로 cell을 실행하면 됩니다. Notebook에는 초기 각도, 목표 높이, side kick, target change, damping, `ax_cmd` limit 등을 바꿀 수 있는 scenario dictionary도 있습니다.

출력은 `results/`에 저장됩니다.

- `states.png`
- `trajectory.png`
- `pid_animation.gif`
- `simulation.csv`

CSV에는 CG position, control-term breakdown, `theta_ddot`, `ax_saturated` flag가 포함됩니다. 그래서 화면상의 움직임이 내부 상태와 맞는지 확인할 수 있습니다.

## Test 실행

Repository root에서:

```bash
python -m unittest discover -s tests
```

Test는 position control을 신뢰하기 전에 geometry sign convention, upright hover acceleration, attitude-only stabilization을 확인합니다.

## Optional Passive Check

```bash
python simulate_passive.py
```

이 명령은 hover throttle과 zero horizontal base acceleration으로 모델을 실행합니다. 초기 tilt가 upright에서 멀어지는 방향으로 넘어지는지 확인해서, inverted-pendulum의 불안정 항이 제대로 들어갔는지 볼 수 있습니다.

## 현재 2D Single-Fan 모델

Moving-base 모델은 개념 baseline으로 남아 있습니다. 현재 single-fan 작업은 2D rigid-body 모델을 사용합니다.

```bash
python simulate_rigid_body.py
```

이 모델은 CG를 기준점으로 사용하고, actuator state를 plant state 안에 포함합니다.

```python
state = [x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

모델은 fan thrust, vane side-force, gravity, translational drag, angular damping, motor lag, servo lag/rate limit, cascaded ArduCopter-inspired controller로부터 force와 moment를 계산합니다.

```text
position -> theta target -> rate target -> desired moment -> vane angle
altitude -> thrust target -> motor lag
```

출력은 다음 파일에 저장됩니다.

- `results/rigid_body_simulation.csv`

## Interactive Real-Time Simulator

Manual flight는 position hold, gain sweep, reinforcement learning보다 먼저 확인해야 합니다. 실행 명령은 다음과 같습니다.

```bash
python interactive_sim.py
```

이 pygame app은 rendering과 독립된 fixed-step physics loop를 사용합니다. Keyboard input은 plant 밖에서 actuator/controller command를 만들고, disturbance는 world-frame force와 pitch moment로 plant에 들어갑니다.

Controls:

- `1`: direct actuator test
- `2`: rate / acro-like mode
- `3`: stabilize-like mode
- `4`: alt-hold placeholder
- `W/S`: throttle command 증가/감소
- `A/D`: mode에 따라 vane, rate, attitude command 입력
- arrow keys: continuous world-frame disturbance force
- `Q/E`: continuous pitch disturbance moment
- `I/O`: short force 또는 pitch-moment impulse
- `X`: emergency motor cut
- `Space`: pause/resume
- `N`: pause 상태에서 single physics step
- `R`: reset
- `F1`-`F6`: `config.py`의 `InteractiveSimConfig` preset으로 reset
- `L`: timestamped CSV logging start/stop
- `[` / `]`: simulation speed 감소/증가
- `+` / `-`: zoom
- `C`: camera follow toggle
- `M`: slow motion toggle
- `Backspace`: manual command reset
- `Esc`: quit

Interactive log는 `results/interactive_logs/` 아래에 저장됩니다.

기록된 run을 replay하려면:

```bash
python replay_interactive.py results/interactive_logs/<log>.csv
```

Replay tool은 animation, state plot, force/moment plot, controller term plot을 `results/replay/` 아래에 만듭니다.

Hardening notes:

- controller shaping은 명시적인 controller `dt`를 받습니다.
- attitude error는 `wrap_pi`를 통해 shortest-angle wrapping을 사용합니다.
- mode transition은 stale PID kick을 피하기 위해 rate/attitude target을 seed합니다.
- mixer는 floor-normalized command authority와 actual thrust에서 물리적으로 가능한 moment를 따로 보고합니다.
- low-thrust saturation은 rate-PID anti-windup에 반영됩니다.
- vane model은 `linear_legacy` 또는 `nonlinear_with_axial_loss`를 사용할 수 있습니다.
- manual throttle command는 pluggable thrust-curve model을 거칩니다.
- safety check는 ground contact, state limit, non-finite state value가 발생하면 simulator를 pause합니다.

## Interactive LOITER Mode

ArduCopter-inspired LOITER 예제를 실행하려면:

```bash
python interactive_sim.py --params params/loiter_example.json
```

Modes and controls:

- `1`: DIRECT
- `2`: RATE
- `3`: STABILIZE
- `4`: ALT_HOLD
- `5`: LOITER
- `A/D`: DIRECT에서는 vane angle, RATE에서는 pitch rate, STABILIZE와 ALT_HOLD에서는 lean angle, LOITER에서는 horizontal movement speed를 명령합니다.
- `W/S`: DIRECT, RATE, STABILIZE에서는 raw throttle을 명령하고, ALT_HOLD와 LOITER에서는 deadband가 있는 climb/descent rate를 명령합니다.
- Arrow keys: external force disturbance
- `Q/E`: pitch moment disturbance
- `I/O`: impulse disturbance
- `X`: emergency motor cut
- `R`: reset
- `F1-F6`: presets
- `L`: CSV logging

Vane visualization:

- Solid vane: actual servo angle
- Ghost vane: commanded vane angle
- Neutral line: body thrust axis를 따라 downstream 방향으로 그려지는 zero-deflection reference
- `SAT`: actuator 또는 mixer saturation
- `RATE`: servo rate limit
- `AUTH`: mixer authority limited
- `SAT/RATE/AUTH`는 actuator 또는 mixer limit이 active일 때만 표시됩니다.
- `vane_visual_scale`은 표시 각도를 과장할 수 있지만 physics에는 영향을 주지 않습니다.
- `vane_visual_length_m`와 `vane_visual_offset_m`은 overlay만 바꾸며 physics에는 영향을 주지 않습니다.

Mode hierarchy는 ArduCopter-inspired 구조입니다. Stabilize는 lean-angle control이고, AltHold는 vertical target control을 추가하며, Loiter는 horizontal position/velocity control을 추가합니다. 하지만 이것은 여전히 단순화된 2D 연구용 simulator입니다. 정확한 ArduPilot firmware 동작도 아니고, 실험적으로 보정된 모델도 아닙니다.

저장된 interactive log를 분석하려면:

```bash
python analyze_interactive_log.py results/interactive_logs/<log>.csv
```

## Headless LOITER Tuning Comparison

pygame을 열지 않고 deterministic LOITER 비교를 실행하려면:

```bash
python compare_loiter_params.py
python sweep_loiter_authority.py
python sweep_loiter_authority.py --quick
python sweep_loiter_authority.py --scenario all --no-plots
```

Comparison script는 sluggish, nominal, aggressive LOITER parameter example을 반복 가능한 scenario들에서 실행하고 다음 파일을 씁니다.

- `results/analysis/loiter_param_comparison.csv`
- `results/analysis/loiter_param_comparison.md`
- matplotlib이 가능할 경우 PNG plot

Authority sweep은 다음 파일을 씁니다.

- `results/analysis/loiter_authority_sweep.csv`
- `results/analysis/loiter_authority_sweep.md`
- matplotlib이 가능할 경우 `results/analysis/authority_maps/*.png` heatmap과 summary plot

Key metrics:

- `final_abs_x_error`: 최종 horizontal hold error
- `rms_x_error`: run 전체 horizontal tracking error
- `max_theta_deg`: 최대 pitch demand/response
- `max_vane_cmd_deg`: 최대 requested vane angle
- `combined_design_score`: 같은 scenario 안에서 candidate를 비교하기 위한 analytical ranking score
- `mixer_saturation_percent`: mixer output이 saturated된 sample 비율
- `authority_limited_percent`: requested moment가 현재 thrust/vane authority를 초과한 비율
- `servo_rate_saturation_percent`: servo rate limit으로 clipping된 sample 비율

이 값들은 분석용 지표이며, 보정된 실제 비행 예측값이 아닙니다. Saturation이 항상 실패를 의미하는 것은 아니고, 설계 신호로 봐야 합니다. Parameter set과 scenario 사이의 상대 비교에 사용하세요.

자세한 내용:

- [docs/loiter_tuning_analysis.md](docs/loiter_tuning_analysis.md): scenario 정의, metric 해석, 한계
- [docs/vane_authority_mapping.md](docs/vane_authority_mapping.md): 확장된 vane authority mapping workflow

## Four-Vane / ArduPilot SingleCopter 관련 문서

향후 four-vane SingleCopter mixer 준비 내용은 [docs/four_vane_mixer_prep.md](docs/four_vane_mixer_prep.md)에 정리되어 있습니다.

현재 simulator는 여전히 **하나의 등가 2D pitch-axis vane/moment**를 사용합니다. Four-vane type은 이후 작업을 위한 architecture layer일 뿐이며, 아직 full 3D physics가 아닙니다.

ArduPilot SingleCopter의 Motor1/Motor2/Motor3/Motor4 flap naming과 이 repository의 front/right/rear/left convention 관계는 [docs/ardupilot_singlecopter_mapping.md](docs/ardupilot_singlecopter_mapping.md)에 정리되어 있습니다.

Disabled-by-default 2D moving-mass pitch assist는 [docs/moving_mass_pitch_assist.md](docs/moving_mass_pitch_assist.md)에 정리되어 있습니다.

## Troubleshooting

- Windows에서 `pygame` 설치가 실패하면 Python 3.11을 시도하세요.
- 움직임이 너무 이상적으로 보이면 `params/loiter_sluggish_example.json`을 사용하거나 controller section의 zero-default noise parameter를 활성화해 보세요.
- LOITER가 target으로 돌아오지 않으면 saturation, `authority_limited`, thrust-to-weight ratio, vane authority, gain을 확인하세요.
- Vane이 잘 보이지 않으면 `interactive.vane_visual_scale`, `interactive.vane_visual_length_m`를 키우거나 `interactive.show_vane_overlay`를 활성화하세요.
- 알 수 없는 parameter key는 section별로 보고됩니다. Structured file은 `rigid_body`, `interactive`, `controller` section을 사용하고, 예전 flat rigid-body JSON도 계속 동작합니다.

관련 문서: [docs/arducopter_alignment.md](docs/arducopter_alignment.md).
