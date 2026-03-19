# MuJoCo Chain Track Design Lessons

Hard-won lessons from building a contact-based tank track in MuJoCo.

## Collision pair compatibility

| Pair              | Works? |
|-------------------|--------|
| box-box           | YES    |
| box-capsule       | YES    |
| box-sphere        | YES    |
| box-cylinder      | YES    |
| capsule-capsule   | YES    |
| sphere-cylinder   | YES    |
| capsule-cylinder  | NO     |
| cylinder-cylinder | NO     |

## Sprocket geometry

- **Hub** (cylinder): collision-enabled, long in Z to span full track width. The chain side links rest on it.
- **Pins** (capsules along Z): short, only extend between the side chains. They poke between cross-links to engage the chain. Keep these as short as possible to avoid unwanted contact with side chains.
- `N_PINS = round(2*pi*R / link_pitch)` so valleys between pins align with link spacing.

## Chain link design (dual-chain)

Each link body has:
- Left side box at `z = -CHAIN_Z`
- Right side box at `z = +CHAIN_Z`
- Cross-link tread plate spanning between them
- Optional roller capsule at the joint point (capsule-capsule contact with pins works)

## Constraint approach

- All links as independent freejoint bodies + equality connect constraints to form the closed loop.
- **Do not** use deep kinematic trees (parent-child chain of 20+ bodies) — they blow up immediately.
- Good starting params: `solref="0.002 1"`, `solimp="0.97 0.999 0.0001 0.5 2"`
- `solref=0.0005` is too stiff — amplifies interpenetration into explosions.
- `noslip_iterations=200` is sufficient.

## Spawn placement

- Chain center must be offset outward from the pin circle: `chain_r = SPROCKET_R + PIN_RADIUS + SIDE_THICK`
- Spawning at the same radius as the pins causes interpenetration -> 10,000N contact forces -> explosion.
- Always run `--debug` headless first to check contact forces at step 0 before trying GUI.

## Stadium path parametrization

- Left arc: theta goes `pi/2 -> 3*pi/2` (wrapping LEFT of left sprocket). **NOT** `pi/2 -> -pi/2` which wraps the wrong side.
- Right arc: theta goes `-pi/2 -> pi/2` (correct).
- Tangent angle for left arc (CCW traversal): `theta + pi/2`.

## Stability tips

- Warmup: no drive for 0.5s, ramp over 0.5s to prevent initial shock.
- Drive both sprockets simultaneously to avoid pulling chain off one side.
- Contact forces should be <50N steady state for ~30g links. Anything >500N = geometry problem.
- Gravity of -3 m/s^2 causes ~15cm sag on straight sections.

## Debugging

`--debug` mode prints per-step: ncon, max contact force, eq violation, max qvel, stadium tracking error.

Healthy indicators:
- `ncon > 0` after settling (chain is touching sprockets)
- `cf < 50N` (no jamming)
- `eq_v < 2mm`
- `trk < 5cm`

Red flags:
- `ncon = 0` after settling = chain fell off
- `cf > 500N` = interpenetration or geometry clash
- `eq_v > 10mm` = constraints too soft or forces too high
