# Raw-score rank 1 versus selected low-control-effort candidate

Stage 0 is a **FAILED / NON-ACCEPTABLE baseline used for normalization only**. It is not a validated controller.

The final controller is **the selected robust low-control-effort candidate under the predeclared near-equivalence rule**. It is raw-score rank 15, not rank 1, and it is not described as the mathematical raw-score optimum.

The predeclared inclusive rule is `valid raw aggregate score <= raw-score best + 0.010000`. The best raw score is `0.396902568`, the limit is `0.406902568`, and `19` of `120` valid Stage 3C candidates are inside the band.

The selected score is `0.406150436` (absolute penalty `0.009247867`, relative penalty `2.330%`). Relative to rank 1, it reduces mean vane RMS by `2.554%`, vane total variation by `4.278%`, and vane command-rate RMS by `5.372%`.

| metric | raw-score rank 1 | selected rank 15 | selected change |
| --- | ---: | ---: | ---: |
| raw aggregate score | 0.396902568 | 0.406150436 | 2.330% |
| scenario mean score | 0.262214909 | 0.268382452 | 2.352% |
| worst-scenario score | 0.268866157 | 0.277601514 | 3.249% |
| Tail RMS pitch (deg) | 0.580047616 | 0.646350172 | 11.431% |
| Tail RMS pitch rate (deg/s) | 2.507062065 | 2.605516404 | 3.927% |
| Tail RMS vx (m/s) | 0.030709111 | 0.035597665 | 15.919% |
| Tail path (m) | 0.063884133 | 0.075808424 | 18.665% |
| Final absolute error (m) | 0.012403504 | 0.008168487 | -34.144% |
| Position overshoot (m) | 0.332599945 | 0.338402286 | 1.745% |
| Recovery excursion (m) | 0.588281373 | 0.595711035 | 1.263% |
| Strict settling time (s) | 5.143571429 | 4.978571429 | -3.208% |
| Vane RMS (deg) | 0.712389941 | 0.694195216 | -2.554% |
| Vane total variation (deg) | 33.536981401 | 32.102370172 | -4.278% |
| Vane command-rate RMS (deg/s) | 28.485461639 | 26.955244974 | -5.372% |
| Meaningful vane sign changes | 0.571428571 | 0.285714286 | -50.000% |
| Vane variation rate (deg/s) | 3.156566461 | 3.020625735 | -4.307% |
| Tail high-frequency vane energy (deg^2) | 0.000432438 | 0.000330236 | -23.634% |
| Vane zero-crossing frequency (Hz) | 0.059523810 | 0.023809524 | -60.000% |
| Vane saturation (%) | 0.000000000 | 0.000000000 | 0.000% |
| Servo-rate saturation (%) | 0.000000000 | 0.000000000 | 0.000% |
| Mixer saturation (%) | 0.000000000 | 0.000000000 | 0.000% |

Both candidates pass every physical and behavioral hard gate in all seven full-duration scenarios. All mirrored symmetry fractions are zero for both candidates. Detailed per-scenario metrics and every hard-gate field are preserved in `selection_comparison.json` and `selection_comparison.csv`.

The selected candidate has modestly higher tail pitch/velocity/path residuals than raw rank 1, but the absolute residuals remain small, every transient/safety gate passes, final-error and settling measures are mixed or improved, and the selected candidate remains substantially better than the failed Stage 0 normalization baseline. These differences are disclosed rather than hidden by the aggregate score.
