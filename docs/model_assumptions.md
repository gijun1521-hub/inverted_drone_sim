# Model Assumptions

## Moving-Base Baseline

The original moving-base inverted-pendulum model is kept as a conceptual
baseline and regression test. It directly commands horizontal base acceleration
and is not intended to represent the final physical single-fan actuator chain.

## Single Rigid-Body Vane Model

`RigidBodySingleFan2D` uses the center of gravity as the state reference. The
main thrust acts along the body-up axis and is intentionally applied through the
CG in the single-rigid-body formulation, so main thrust creates no pitch moment
by itself. The vane force acts below the CG and creates the pitch moment.

Two vane models are available:

- `linear_legacy`: preserves the earlier linear side-force approximation.
- `nonlinear_with_axial_loss`: uses `k_side * thrust * sin(vane_angle)` and an
  axial efficiency loss. These coefficients are placeholders, not experimentally
  validated values.

Wind is modeled as constant world-frame velocity for drag relative to air, plus
an optional finite-duration gust force/moment for headless validation. No ground
contact bounce, duct aerodynamics, motor voltage sag, or 3D coupling is included
yet.
