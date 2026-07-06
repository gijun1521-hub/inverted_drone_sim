# Analytical Modeling Strategy

This simulator is not experimentally calibrated yet. The current research phase
uses analytical models to explore sign correctness, conservation laws,
parameter sensitivity, and relative control authority.

## Why Analytical Models

Bench data is valuable later, but requiring fitted coefficients now would make
the simulator look more certain than it is. Analytical models keep assumptions
visible and make it easier to compare vane-only and moving-mass concepts.

## Assumed Quantities

- Mass and mass distribution
- Geometry and actuator application points
- Pitch inertia
- Thrust-to-weight ratio
- Vane lift slope or effectiveness
- Servo limits, rate limits, and lag
- Moving mass ratio
- Moving mass travel, rate, and acceleration limits

## Swept Quantities

- Moving mass ratio and inertia ratio
- q travel, rate, and acceleration limits
- Thrust-to-weight ratio
- Vane angle limit
- Vane area to duct area ratio
- Requested moment and authority margin

## Safe Conclusions

- Sign correctness
- Relative control authority
- Sensitivity to mass ratio and inertia ratio
- Saturation and authority trends
- Approximate feasibility of control concepts

## Unsafe Conclusions

- Exact PID gains
- Exact flight time
- Exact disturbance rejection
- Exact aerodynamic efficiency
- Exact experimental response
