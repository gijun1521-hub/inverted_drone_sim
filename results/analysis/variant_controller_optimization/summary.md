# Mandatory corrected-objective independent refinement

This is deterministic 2D simulation research only. It makes no real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety claim, and it does not modify rigid-body physics.

## Mandatory refinement outcome

The corrected cached-only selections were retained as executable baselines, then each variant received a separate deterministic sampling sequence with at least 12 corrected-eligible parents, 160 adaptive-local candidates, 96 joint candidates, and an additional refinement round. A candidate remains selectable only when all prior hard gates pass, both steps settle, both overshoots are at most 8%, and both steps have at most one meaningful target crossing.

| variant | cached candidates | existing eligible | refinement parents | newly evaluated | refined result | improvement (s) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| vane_only | 124 | 8 | 32 | 416 | improved_refined_selection | 0.8800 |
| moving_mass_assist | 135 | 13 | 12 | 640 | improved_refined_selection | 0.2700 |

Total candidate rows: **1315**. Selected validation cases per rerun: **46**. Two-rerun deterministic validation: **True**. All finalist gates passed: **True**.

## Historical previous selections

The prior 11-12% results are retained below as historical output and are not final under the corrected objective.

| variant | previous candidate | worst overshoot | crossings (+/-) | corrected class |
| --- | --- | ---: | ---: | --- |
| vane_only | `vane_only:atc_rat_pit_p=0.1135603516:atc_rat_pit_d=0.02711947874:atc_ang_pit_p=28.37744:psc_ne_pos_p=0.7824344023:psc_ne_vel_p=0.9622915101:moving_mass_assist_gain_m_per_Nm=0` | 0.111986 | 1/1 | overshoot_ineligible |
| moving_mass_assist | `moving_mass_assist:atc_rat_pit_p=0.1196431641:atc_rat_pit_d=0.02469923182:atc_ang_pit_p=18.13904:psc_ne_pos_p=0.7170058309:psc_ne_vel_p=0.9594815928:moving_mass_assist_gain_m_per_Nm=0.02670550751` | 0.119429 | 1/1 | overshoot_ineligible |

## Corrected cached-only selections before refinement

| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) | settle (s) | overshoot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 0.13908691 | 0.01670096 | 24.856640 | 0.61457726 | 0.98948159 | 0.00000000 | 8.7800 | 0.069420 |
| moving_mass_assist | 0.13908691 | 0.01670096 | 24.856640 | 0.61457726 | 0.98948159 | 0.02729249 | 7.8300 | 0.067300 |

## Corrected independently refined selections

| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 0.13947083 | 0.01719345 | 28.831537 | 0.62862063 | 0.98495586 | 0.00000000 |
| moving_mass_assist | 0.13995564 | 0.01662736 | 28.296659 | 0.62532858 | 0.98911427 | 0.01868041 |

## Independent-optimum comparison

The five independently refined controller values converged to different values. No equality or difference constraint was imposed; each variant used its own deterministic sequence and ranking.

Round-by-round candidate counts, before/after settling times, preferred-band candidates, parent distances, local-boundary flags, and continuation decisions are recorded in `boundary_diagnostics.csv`. Separate response-neighborhood tables are exported under `refinement_neighborhoods/`.

| variant | settle + / - (s) | rise + / - (s) | worst overshoot | crossings + / - | worst final error (m) | effort index | vane / servo / mixer sat. (%) | mass limiter max (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 7.9000 / 7.9000 | 2.4550 / 2.4550 | 0.069968 | 1 / 1 | 0.00231982 | 2.095663 | 0.000 / 0.000 / 0.000 | 0.000 |
| moving_mass_assist | 7.5600 / 7.5600 | 2.4400 / 2.4400 | 0.070767 | 1 / 1 | 0.00267567 | 2.608819 | 0.000 / 0.000 / 0.000 | 12.292 |

## Exact selection rationale

The executable optimizer uses this strict lexicographic order: (1) every pre-existing hard gate, (2) both steps settle, (3) both step overshoots <= 0.08, (4) both target-crossing counts <= 1, (5) shortest worst settling time, (6) smallest worst distance from the 0.02-0.05 preferred overshoot band, (7) shortest worst rise time, (8) smallest worst final error, (9) lower actuator effort and limiter use, and (10) better symmetry and robustness. Zero overshoot remains eligible but has a 0.02 band-distance penalty. The 0.12 overshoot hard failure and pre-existing two-crossing hard failure remain unchanged. The refined candidate replaces the cached-only baseline only when its worst settling time improves by at least 0.10 s; otherwise the baseline is preserved and the neighborhood is reported converged.

The Pareto CSV retains valid tradeoffs rather than collapsing them into one weighted score. The shared-objective comparison CSV evaluates the PR #24 shared references and both independently optimized controllers with the same screening objectives.
