# Authority Analysis

The analytical authority tools compare three moment sources:

```text
M_total = M_vane + M_reaction + M_cg_offset
```

## Vane Moment

The vane provides aerodynamic side force below the CG. It is useful for fast
damping and rate control, but authority falls with thrust and vane saturation.

## Moving-Mass Reaction Moment

Reaction moment comes from accelerating the moving mass. It is fast but cannot
be continuous without moving the mass back and creating a reverse reaction.

## Moving-Mass CG-Offset Moment

When the mass remains offset, the total CG shifts and thrust can create a
sustained moment. This effect is range-limited by q travel and depends strongly
on thrust and mass ratio.

## Hybrid Behavior

The intended future strategy is:

- Vanes handle high-frequency rate damping.
- Moving mass handles low-frequency trim or CG bias.
- Reaction can assist fast initial tilt but must respect q limits.
- Unavailable moment must be reported instead of hidden by gain tuning.
