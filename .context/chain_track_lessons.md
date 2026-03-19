# Chain Track Simulation Lessons

## What Works (Current Approach)

**Kinematic tree chain + continuously-updated engagement constraints.**

- Chain links connected by **hinge joints in a kinematic tree** (link_0 = freejoint root, links 1-29 = nested children with hinge joints)
- One `connect` equality constraint closes the loop (link_29 → link_0)
- Sprocket engagement: one `connect` constraint per link per sprocket, **anchor updated every timestep** to follow the link along the arc
- Engagement/disengagement is **position-based**: link within ±0.10 of SPROCKET_R and on the correct side of the sprocket
- Anchor radius = SPROCKET_R (not actual distance)

### Critical Details

1. **Loop closure anchor must be recomputed after setting initial shape.** MuJoCo auto-computes `eq_data[idx, 3:6]` (anchor2) from the initial body positions at XML parse time. If you later change qpos to bend the chain into a stadium, the auto-computed anchor2 is wrong (off by meters). Fix: after `mj_forward`, recompute anchor2 from the current world positions.

2. **Pre-seed engagement at t=0.** Use the known stadium geometry to determine which links start on arcs and set their engagement anchors immediately. This avoids force spikes from late engagement.

3. **Continuously update engagement anchors.** Fixed-angle engagement (set anchor once, hold it) causes the chain to stall — the sprocket can't rotate past ~170 deg before the hinge chain locks up. Updating the anchor every step lets links slide along the arc while staying at radius R.

4. **Use SPROCKET_R for anchor radius, not actual distance.** Using `dist` allows links to engage at wrong radii (e.g., 0.53 instead of 0.35), causing chain deformation. Use the ideal radius and tighten the engagement distance check to ±0.10.

5. **ARC_HALF = π × 0.40 works.** Smaller arc engagement zone allows links to disengage and hand off sooner, preventing chain lockup.

## Failed Approaches

### 1. Contact-based sprocket drive
- Sprocket teeth physically mesh with chain gaps via contact
- Failed: MuJoCo constraint solver can't maintain chain integrity under contact forces
- Pin constraint violations up to 0.5m

### 2. Freejoint chain + equality constraints for everything
- All links as freejoints, connect constraints for pins AND engagement
- `connect` = ball joint (3 translational DOF, 0 rotational) → lateral wobble
- Dual Z-offset constraints to emulate hinge → doubled constraint count
- Solver overwhelmed with 120+ equality constraints

### 3. Fixed-angle engagement (set anchor once, hold it)
- Sprocket rotates ~170 deg then stalls — hinge chain acts as rigid brake
- Links locked at fixed sprocket angles can't advance
- Chain speed drops to 0 after initial rotation

### 4. Engagement using actual distance instead of SPROCKET_R
- Links engage at wrong radii when they drift from ideal path
- Drive/idler speed mismatch (84 deg divergence)
- Chain deforms: track error 0.37m

## Key MuJoCo Facts

- `eq_data[i, 0:3]` = anchor in body1 frame, `eq_data[i, 3:6]` = anchor in body2 frame
- `data.eq_active` (not `model.eq_active`) in MuJoCo 3.6+
- `connect` constraint = ball joint (pins a point, no rotational constraint)
- MuJoCo auto-computes anchor2 at XML parse time from initial body positions — stale if you change qpos later
- Collision: `(ct1 & ca2) || (ct2 & ca1)` — both directions checked
- `contype=0 conaffinity=0` must be explicit on visual-only geoms (default inherits from `<default>`)
- Cylinder collision: box-cylinder works, capsule-cylinder does NOT
- Velocity actuators (`<velocity kv="...">`) for constant-speed drives
- Kinematic tree + closed loop = one equality constraint for loop closure
- Adding engagement equality constraints to kinematic tree bodies is fine IF anchors are updated every step (no constraint loop buildup)

## Parameters That Work

```
N_LINKS = 30
SPROCKET_R = 0.35
HALF_SPAN = 1.4
TIMESTEP = 0.002
TARGET_VEL = 1.0 rad/s
ARC_HALF = π × 0.40
kv = 50 (velocity servo gain)
Engagement distance tolerance: ±0.10 from SPROCKET_R
Solver: Newton, iterations=300, tolerance=1e-10, noslip=10
Constraint: solref=0.005 1, solimp=0.95 0.99 0.001
```

## Resulting Performance

- Chain speed: ~0.33 m/s sustained
- Track error: 0.017 mean, 0.06 max
- Constraint violation: 0.001
- Drive/idler angle difference: <3 deg (well coupled)
- Stable over 10+ seconds, no explosions

## Next Steps

- Build full tank: two mirrored tracks, hull, ground plane, differential steering
- Sprockets as children of hull body
- Ground contact on tread links (contype/conaffinity groups to avoid link-link self-collision)
