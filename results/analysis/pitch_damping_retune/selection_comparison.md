# Previous Stage 3C rank 1 versus boundary-extended local rank 1

Stage 0 remains **FAILED / NON-ACCEPTABLE** and normalization-only.

No near-equivalent lower-effort tie-break was used. The final controller is the valid raw-score local rank-1 point from the targeted extension.

| metric | previous P=0.09000, D=0.01950 | selected P=0.09375, D=0.02100 | selected change |
| --- | ---: | ---: | ---: |
| raw aggregate score | 0.396902568460 | 0.391509724441 | -1.359% |
| scenario mean score | 0.262214908688 | 0.258414981966 | -1.449% |
| worst-scenario score | 0.268866157393 | 0.265322061898 | -1.318% |
| Tail RMS pitch (deg) | 0.580047616 | 0.525546418 | -9.396% |
| Tail RMS pitch rate (deg/s) | 2.507062065 | 2.372223592 | -5.378% |
| Tail RMS vx (m/s) | 0.030709111 | 0.030239088 | -1.531% |
| Tail path (m) | 0.063884133 | 0.063139595 | -1.165% |
| Final absolute error (m) | 0.012403504 | 0.017361590 | 39.973% |
| Position overshoot (m) | 0.332599945 | 0.330149368 | -0.737% |
| Recovery excursion (m) | 0.588281373 | 0.585183881 | -0.527% |
| Strict settling time (s) | 5.143571429 | 5.134285714 | -0.181% |
| Vane RMS (deg) | 0.712389941 | 0.732413779 | 2.811% |
| Vane total variation (deg) | 33.536981401 | 34.517019467 | 2.922% |
| Vane command-rate RMS (deg/s) | 28.485461639 | 30.716419702 | 7.832% |
| Meaningful vane sign changes | 0.571428571 | 0.857142857 | 50.000% |
| Vane variation rate (deg/s) | 3.156566461 | 3.251910828 | 3.021% |
| Tail high-frequency vane energy (deg^2) | 0.000432438 | 0.000563649 | 30.342% |
| Vane zero-crossing frequency (Hz) | 0.059523810 | 0.083333333 | 40.000% |
| Vane saturation (%) | 0.000000000 | 0.000000000 | 0.000% |
| Servo-rate saturation (%) | 0.000000000 | 0.000000000 | 0.000% |
| Mixer saturation (%) | 0.000000000 | 0.000000000 | 0.000% |

Both controllers pass every hard gate in all seven full-duration scenarios. The selected point is interior on both audited P and D upper axes after one D-axis extension round.
