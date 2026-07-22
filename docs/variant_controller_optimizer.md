# Variant controller optimizer

This workflow independently tunes the Vane-only and Moving-mass-assist
controllers in the deterministic 2D simulation. It does not modify the
rigid-body physics and makes no real-flight, Pixhawk, Raspberry Pi, HIL, or
hardware-safety claim.

Run both searches with:

```bash
python optimize_variant_controllers.py --variant both --workers 4 --resume
```

The default search definition is
`params/variant_controller_search_space.json`. Use `--search-space PATH` to
run a reviewed alternative. The JSON controls the parameter bounds, response
targets, deterministic seed, sampling counts, refinement widths, and boundary
extension policy.

The search is not a Cartesian grid. It evaluates fixed historical anchors and
axis boundaries, a deterministic Halton coarse design, coordinate refinements
around the best valid candidates, and a final joint refinement. If the best
valid point is exactly on a searched boundary, the workflow extends that side
and evaluates the new slab. A zero lower bound on Moving-mass gain is retained
as a physical boundary.

Each candidate first runs the mirrored +1/-1 m steps and +8/-8 N LOITER
disturbances. Candidate tasks are isolated in worker processes. Only the parent
process updates the fingerprinted atomic cache, so `--resume` is safe and
worker completion order cannot change ranking. Selection is lexicographic:

1. all hard gates pass;
2. both step cases settle;
3. shortest worst-case settling time;
4. shortest worst-case rise time;
5. overshoot closest to the configured preferred band;
6. smallest steady-state error;
7. lower actuator effort and limiter use;
8. stronger mirrored and scenario robustness.

The selected controllers then run the existing seven controller scenarios and
the 32 final event-aware LOITER cases twice. Results are written only under
`results/analysis/variant_controller_optimization/`. The PR #24 shared-controller
artifacts remain read-only references, and a preservation audit records their
before/after hashes.

Verify the final artifact manifest with:

```bash
python optimize_variant_controllers.py --verify-manifest
```

`pareto_front.csv` exposes valid tradeoffs without hiding them in a single
weighted score. `shared-objective comparison.csv` separately labels the PR #24
shared-controller actuator-isolation references and the two independently
optimized results.
