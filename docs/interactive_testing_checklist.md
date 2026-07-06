# Interactive Testing Checklist

## DIRECT Mode

- Increase throttle and confirm upward thrust increases after motor lag.
- At upright hover, confirm thrust near weight produces approximate hover.
- Command positive and negative vane and confirm opposite side-force directions.
- Confirm vane moment direction matches the telemetry.
- Reduce thrust and confirm the low-thrust authority warning appears.

## RATE Mode

- Command positive and negative pitch-rate targets.
- Confirm angular velocity responds in the commanded direction.
- Hold a disturbance moment with `Q/E`, then release it and confirm the force or
  moment is no longer applied while accumulated velocity remains.
- Watch mixer and servo saturation flags.

## STABILIZE Mode

- Start from `F2` and `F3` initial tilts.
- Confirm restoring moment and smooth mode transitions.
- Inject force and moment disturbances and observe recovery.

## Moving Mass Future Tests

- Verify q command direction.
- Verify body reaction direction during q acceleration.
- Verify total CG moves in the expected direction.
- Verify thrust-offset moment direction after CG shift.
