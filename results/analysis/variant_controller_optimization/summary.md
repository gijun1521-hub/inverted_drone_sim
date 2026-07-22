# Corrected independent variant-controller optimization

This is deterministic 2D simulation research only. It makes no real-flight, Pixhawk, Raspberry Pi, HIL, or hardware-safety claim, and it does not modify rigid-body physics.

## Corrected reranking outcome

All cached simulation metrics were reranked before any new simulation. A candidate is selectable only when all prior hard gates pass, both steps settle, both overshoots are at most 8%, and both steps have at most one meaningful target crossing.

| variant | cached candidates | existing eligible | newly evaluated | corrected result |
| --- | ---: | ---: | ---: | --- |
| vane_only | 124 | 8 | 0 | selected |
| moving_mass_assist | 135 | 13 | 0 | selected |

Total candidate rows: **259**. Selected validation cases per rerun: **46**. Two-rerun deterministic validation: **True**. All finalist gates passed: **True**.

## Historical previous selections

The prior 11-12% results are retained below as historical output and are not final under the corrected objective.

| variant | previous candidate | worst overshoot | crossings (+/-) | corrected class |
| --- | --- | ---: | ---: | --- |
| vane_only | `vane_only:atc_rat_pit_p=0.1135603516:atc_rat_pit_d=0.02711947874:atc_ang_pit_p=28.37744:psc_ne_pos_p=0.7824344023:psc_ne_vel_p=0.9622915101:moving_mass_assist_gain_m_per_Nm=0` | 0.111986 | 1/1 | overshoot_ineligible |
| moving_mass_assist | `moving_mass_assist:atc_rat_pit_p=0.1196431641:atc_rat_pit_d=0.02469923182:atc_ang_pit_p=18.13904:psc_ne_pos_p=0.7170058309:psc_ne_vel_p=0.9594815928:moving_mass_assist_gain_m_per_Nm=0.02670550751` | 0.119429 | 1/1 | overshoot_ineligible |

## Final selected controllers

| variant | Rate P | Rate D | Angle P | Position P | Velocity P | mass gain (m/Nm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 0.13908691 | 0.01670096 | 24.856640 | 0.61457726 | 0.98948159 | 0.00000000 |
| moving_mass_assist | 0.13908691 | 0.01670096 | 24.856640 | 0.61457726 | 0.98948159 | 0.02729249 |

| variant | settle + / - (s) | rise + / - (s) | worst overshoot | crossings + / - | worst final error (m) | effort index | vane / servo / mixer sat. (%) | mass limiter max (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vane_only | 8.7800 / 8.7800 | 2.4750 / 2.4750 | 0.069420 | 1 / 1 | 0.00112479 | 1.755729 | 0.000 / 0.000 / 0.000 | 0.000 |
| moving_mass_assist | 7.8300 / 7.8300 | 2.5100 / 2.5100 | 0.067300 | 1 / 1 | 0.00523997 | 2.353194 | 0.000 / 0.000 / 0.000 | 12.833 |

## Exact selection rationale

The executable optimizer uses this strict lexicographic order: (1) every pre-existing hard gate, (2) both steps settle, (3) both step overshoots <= 0.08, (4) both target-crossing counts <= 1, (5) shortest worst settling time, (6) smallest worst distance from the 0.02-0.05 preferred overshoot band, (7) shortest worst rise time, (8) smallest worst final error, (9) lower actuator effort and limiter use, and (10) better symmetry and robustness. Zero overshoot remains eligible but has a 0.02 band-distance penalty. The 0.12 overshoot hard failure and pre-existing two-crossing hard failure remain unchanged.

The Pareto CSV retains valid tradeoffs rather than collapsing them into one weighted score. The shared-objective comparison CSV evaluates the PR #24 shared references and both independently optimized controllers with the same screening objectives.
