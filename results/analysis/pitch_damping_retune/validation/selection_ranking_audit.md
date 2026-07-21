# Selection-ranking audit — publication blocked

The completed corrected-source workflow selected Rate P/I/D `0.08500000 / 0.00000000 / 0.01850000` with Angle P `25.00000000` using the documented near-equivalence rule: candidates within `max(0.01, 1% of best score)` are tie-broken by lower mean vane RMS, lower Rate D, lower vane total variation, better symmetry, then score.

This candidate passes all physical and behavioral hard gates, but it is **raw-score rank 15** in the final Stage 3C crosscheck with score `0.406150435755185`. The raw-score rank-1 Stage 3C point is Rate P `0.09000000`, Rate D `0.01950000`, Angle P `25.00000000`, score `0.396902568459614`. The lowest valid score across all staged outputs is the Stage 3A point Rate P `0.09500000`, Rate D `0.02000000`, Angle P `25.00000000`, score `0.394627695506957`.

Because the final audit requires confirmation that the selected candidate ranks first, the provisional profile is not approved for commit, push, or Draft PR publication pending an explicit choice between raw-score-first selection and the documented near-equivalent actuator-effort tie-break.
