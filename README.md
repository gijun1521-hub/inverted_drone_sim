# Single Fan Drone 2D Simulator

이 저장소는 **single fan / ducted fan 드론**을 분석하기 위한 Python 기반 2D 시뮬레이터입니다.

목표는 바로 강화학습으로 가는 것이 아니라, 먼저 다음을 차근차근 확인하는 것입니다.

1. 추력편향 vane 제어만으로 pitch 출렁임을 얼마나 줄일 수 있는가?
2. 2D 모델에서 LOITER-like 제어가 어떤 조건에서 안정적인가?
3. vane authority, saturation, servo rate limit이 성능에 어떤 영향을 주는가?
4. moving mass pitch assist가 보조 모멘트로 의미가 있는가?
5. 나중에 scripted flip, reaction kick, trajectory search, PPO/RL로 확장할 수 있는가?

현재 이 저장소는 **분석용 2D simulator**입니다. 실제 Cleo Robotics Dronut 같은 제품의 성능 예측기나, 실제 비행 보정 모델이 아닙니다.

---

## 먼저 읽기

현재 main 기준으로 가능한 것과 아직 아닌 것을 구분해야 합니다.

### 지금 가능한 것

- 2D single-fan rigid-body simulation
- 하나의 등가 pitch-axis vane / moment 모델
- motor lag, servo lag, servo rate limit
- cascaded ArduCopter-inspired 제어 구조
- DIRECT / RATE / STABILIZE / ALT_HOLD / LOITER-like interactive mode
- headless LOITER parameter comparison
- vane authority sweep / saturation 분석
- disabled-by-default 2D moving-mass pitch assist 모델
- four-vane mixer type/interface 준비
- ArduPilot SingleCopter flap naming과 simulator convention 문서화

### 아직 아닌 것

- full 3D dynamics
- roll dynamics
- yaw / swirl physics
- four independent vane physics wiring
- 정확한 ArduPilot firmware 재현
- 실험 보정된 실제 비행 예측
- reaction kick
- flip controller
- reinforcement learning / PPO environment

이 구분이 중요합니다. 지금 결과는 **실제 비행 성능값**이 아니라, 설계 방향을 비교하기 위한 **상대 분석 지표**로 봐야 합니다.

---

## 빠른 시작

### 1. 의존성 설치

```bash
python -m pip install -r requirements.txt
```

### 2. 전체 테스트

```bash
python -m unittest discover -s tests
```

### 3. 기본 검증

```bash
python validate_sim.py
```

### 4. Interactive simulator 실행

```bash
python interactive_sim.py --params params/default_rigid_body.json
```

### 5. LOITER 예제 실행

```bash
python interactive_sim.py --params params/loiter_example.json
```

### 6. Headless 분석 실행

```bash
python compare_loiter_params.py
python sweep_loiter_authority.py --quick
```

생성된 CSV, plot, GIF, video, JSON, HTML 결과물은 `results/` 아래에 저장되고 git에서는 무시됩니다.

---

## 추천 작업 순서

처음 보는 경우 아래 순서로 보면 됩니다.

1. `python -m unittest discover -s tests`
2. `python validate_sim.py`
3. `python interactive_sim.py --params params/default_rigid_body.json`
4. `python interactive_sim.py --params params/loiter_example.json`
5. `python compare_loiter_params.py`
6. `python sweep_loiter_authority.py --quick`
7. `docs/loiter_tuning_analysis.md` 읽기
8. `docs/vane_authority_mapping.md` 읽기
9. `docs/moving_mass_pitch_assist.md` 읽기
10. 그 다음 scripted flip / reaction kick / trajectory search / RL 확장 검토

바로 RL로 가지 말고, deterministic simulation과 분석 결과를 먼저 확인하는 흐름을 권장합니다.

---

## 프로젝트 구조 요약

현재 프로젝트는 크게 두 모델을 포함합니다.

### 1. Moving-base inverted-pendulum baseline

초기 baseline은 thrust point를 수평으로 움직이는 역진자 모델입니다.

```python
state = [x, z, theta, vx, vz, omega]
action = [throttle, ax_cmd]
```

`x, z`는 thrust point 위치입니다. CG는 다음과 같이 계산합니다.

```python
cg_x = x + l * sin(theta)
cg_z = z + l * cos(theta)
```

자세 dynamics는 다음 개념을 따릅니다.

```python
x_ddot = ax_cmd
z_ddot = T / m - g
theta_ddot = (g * sin(theta) - x_ddot * cos(theta)) / l - damping * omega
```

Sign convention:

- `theta > 0`: CG가 thrust point의 오른쪽에 있음
- `theta < 0`: CG가 thrust point의 왼쪽에 있음
- `theta < 0`이면 attitude control은 초기에 `ax_cmd < 0`을 명령해야 함

이 baseline은 실제 single-fan 모델이라기보다, sign convention과 inverted-pendulum intuition을 확인하기 위한 개념 모델입니다.

### 2. 현재 2D single-fan rigid-body 모델

현재 핵심 모델은 CG 기준 2D rigid-body 모델입니다.

```python
state = [x_cg, z_cg, theta, vx, vz, omega, thrust, vane_angle]
```

포함된 항목:

- fan thrust
- vane side-force
- gravity
- translational drag
- angular damping
- motor lag
- servo lag / rate limit
- cascaded controller

제어 흐름은 다음과 같은 구조입니다.

```text
position -> theta target -> rate target -> desired moment -> vane angle
altitude -> thrust target -> motor lag
```

기본 실행:

```bash
python simulate_rigid_body.py
```

출력:

- `results/rigid_body_simulation.csv`

---

## Interactive simulator 사용법

기본 실행:

```bash
python interactive_sim.py
```

LOITER 예제:

```bash
python interactive_sim.py --params params/loiter_example.json
```

이 simulator는 pygame을 사용합니다. Rendering과 physics loop는 분리되어 있고, keyboard input은 actuator/controller command로 변환됩니다. 외란은 world-frame force 또는 pitch moment로 plant에 들어갑니다.

### 모드

- `1`: DIRECT
- `2`: RATE / acro-like
- `3`: STABILIZE-like
- `4`: ALT_HOLD-like
- `5`: LOITER-like

### 주요 조작키

- `W/S`: throttle 또는 climb/descent command
- `A/D`: mode에 따라 vane / pitch rate / lean angle / horizontal speed command
- arrow keys: 외력 disturbance
- `Q/E`: pitch moment disturbance
- `I/O`: impulse disturbance
- `X`: emergency motor cut
- `Space`: pause/resume
- `N`: pause 상태에서 single physics step
- `R`: reset
- `F1-F6`: preset reset
- `L`: CSV logging start/stop
- `[` / `]`: simulation speed 조절
- `+` / `-`: zoom
- `C`: camera follow toggle
- `M`: slow motion toggle
- `Backspace`: manual command reset
- `Esc`: quit

Interactive log는 다음 위치에 저장됩니다.

```text
results/interactive_logs/
```

저장된 log replay:

```bash
python replay_interactive.py results/interactive_logs/<log>.csv
```

Replay 결과는 다음 위치에 저장됩니다.

```text
results/replay/
```

---

## Vane visualization 의미

Interactive simulator에서 vane overlay는 다음 의미를 가집니다.

- Solid vane: 실제 servo angle
- Ghost vane: commanded vane angle
- Neutral line: body thrust axis를 따라 downstream 방향으로 그려지는 zero-deflection reference
- `SAT`: actuator 또는 mixer saturation
- `RATE`: servo rate limit
- `AUTH`: mixer authority limited

`SAT/RATE/AUTH`는 limit이 active일 때만 표시됩니다.

주의:

- `vane_visual_scale`은 표시 각도만 과장합니다.
- `vane_visual_length_m`와 `vane_visual_offset_m`은 overlay만 바꿉니다.
- 위 값들은 physics를 바꾸지 않습니다.

---

## Headless LOITER tuning comparison

pygame을 열지 않고 deterministic 비교를 실행할 수 있습니다.

```bash
python compare_loiter_params.py
python sweep_loiter_authority.py
python sweep_loiter_authority.py --quick
python sweep_loiter_authority.py --scenario all --no-plots
```

`compare_loiter_params.py`는 sluggish / nominal / aggressive LOITER parameter example을 반복 가능한 scenario에서 비교합니다.

출력:

- `results/analysis/loiter_param_comparison.csv`
- `results/analysis/loiter_param_comparison.md`
- matplotlib 사용 가능 시 PNG plot

`sweep_loiter_authority.py`는 vane authority와 saturation 경향을 분석합니다.

출력:

- `results/analysis/loiter_authority_sweep.csv`
- `results/analysis/loiter_authority_sweep.md`
- matplotlib 사용 가능 시 `results/analysis/authority_maps/*.png`

### 주요 metric

- `final_abs_x_error`: 최종 horizontal hold error
- `rms_x_error`: run 전체 horizontal tracking error
- `max_theta_deg`: 최대 pitch demand/response
- `max_vane_cmd_deg`: 최대 requested vane angle
- `combined_design_score`: 같은 scenario 안에서 candidate를 비교하기 위한 analytical ranking score
- `mixer_saturation_percent`: mixer output이 saturation된 sample 비율
- `authority_limited_percent`: requested moment가 현재 thrust/vane authority를 초과한 비율
- `servo_rate_saturation_percent`: servo rate limit으로 clipping된 sample 비율

이 metric들은 실제 비행 성능 예측값이 아닙니다. 같은 scenario 안에서 parameter set 또는 설계 후보를 상대 비교하는 용도로 사용해야 합니다.

Saturation은 항상 실패가 아닙니다. 경우에 따라 “이 설계가 authority 한계에 얼마나 자주 닿는지”를 알려주는 설계 신호입니다.

---

## Moving-mass pitch assist

현재 main에는 **disabled-by-default 2D moving-mass pitch assist** 모델이 들어 있습니다.

문서:

- [docs/moving_mass_pitch_assist.md](docs/moving_mass_pitch_assist.md)

현재 moving mass 모델의 성격:

- 기본값은 disabled
- enabled일 때만 2D plant state가 확장됨
- moving mass state는 offset / velocity / target으로 구성됨
- offset limit, rate limit, acceleration limit을 가짐
- pitch moment는 quasi-static gravity moment로 계산됨

개념적으로는 다음 항입니다.

```text
moving_mass_moment_Nm = mass_kg * gravity * offset
```

주의:

- 아직 reaction kick 모델이 아닙니다.
- 아직 flip controller가 아닙니다.
- 아직 RL이 아닙니다.
- 아직 3D dynamics가 아닙니다.
- 실제 moving-mass hardware calibration이 아닙니다.

---

## Four-vane / ArduPilot SingleCopter 관련

향후 four-vane SingleCopter 구조를 위한 type/interface 준비는 되어 있습니다.

문서:

- [docs/four_vane_mixer_prep.md](docs/four_vane_mixer_prep.md)
- [docs/ardupilot_singlecopter_mapping.md](docs/ardupilot_singlecopter_mapping.md)

ArduPilot SingleCopter 문서 기준의 flap naming은 다음과 같이 simulator convention에 대응시킵니다.

```text
Motor 1: Forward Flap -> front
Motor 2: Right Flap   -> right
Motor 3: Back Flap    -> rear
Motor 4: Left Flap    -> left
Motor 5: Motor        -> scalar fan thrust
```

현재 simulator는 여전히 **하나의 등가 2D pitch-axis vane/moment**를 사용합니다. Four-vane type은 나중에 full wiring을 하기 위한 architecture layer이며, 아직 3D four-vane physics가 아닙니다.

---

## 문서 지도

중요 문서:

- [docs/arducopter_alignment.md](docs/arducopter_alignment.md): ArduCopter-inspired control hierarchy와 simulator alignment
- [docs/loiter_tuning_analysis.md](docs/loiter_tuning_analysis.md): LOITER scenario, metric 해석, 한계
- [docs/vane_authority_mapping.md](docs/vane_authority_mapping.md): vane authority / saturation 분석 workflow
- [docs/four_vane_mixer_prep.md](docs/four_vane_mixer_prep.md): future four-vane mixer architecture 준비
- [docs/ardupilot_singlecopter_mapping.md](docs/ardupilot_singlecopter_mapping.md): ArduPilot SingleCopter flap naming 대응
- [docs/moving_mass_pitch_assist.md](docs/moving_mass_pitch_assist.md): disabled-by-default moving-mass pitch assist 모델
- [docs/moving_mass_comparison_analysis.md](docs/moving_mass_comparison_analysis.md): vane-only와 moving-mass assist variant 비교 분석 workflow

---

## 기타 실행 명령

PID baseline simulation:

```bash
python simulate_pid.py
```

상위 작업 디렉터리에서 실행할 경우:

```bash
python inverted_drone_sim/simulate_pid.py
```

Notebook:

```text
../drone_simulation.ipynb
```

Passive check:

```bash
python simulate_passive.py
```

Authority analysis:

```bash
python analyze_authority.py
```

Moving-mass analysis:

```bash
python analyze_moving_mass.py
```

Interactive log analysis:

```bash
python analyze_interactive_log.py results/interactive_logs/<log>.csv
```

---

## Troubleshooting

- Windows에서 `pygame` 설치가 실패하면 Python 3.11을 시도하세요.
- 움직임이 너무 이상적으로 보이면 `params/loiter_sluggish_example.json`을 사용해 보세요.
- LOITER가 target으로 돌아오지 않으면 saturation, `authority_limited`, thrust-to-weight ratio, vane authority, gain을 확인하세요.
- Vane이 잘 보이지 않으면 `interactive.vane_visual_scale`, `interactive.vane_visual_length_m`, `interactive.show_vane_overlay`를 확인하세요.
- 알 수 없는 parameter key는 section별로 보고됩니다.
- Structured parameter file은 `rigid_body`, `interactive`, `controller` section을 사용합니다.
- 예전 flat rigid-body JSON도 계속 동작합니다.

---

## 개발 방향 메모

이 프로젝트의 안전한 확장 순서는 다음과 같습니다.

1. 2D deterministic simulator 안정화
2. vane authority / saturation 분석
3. moving-mass assist 비교 분석
4. scripted pitch flip baseline
5. reaction kick / inertial moving-mass model
6. CEM 또는 random trajectory search
7. PPO/RL environment

즉, **바로 RL로 가지 않습니다.** 먼저 deterministic model과 분석 결과를 통해 어떤 action space와 reward가 말이 되는지 확인한 뒤 RL로 넘어가는 것이 목표입니다.
