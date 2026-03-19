# Tank Track Simulation in MuJoCo

## Architecture

**Kinematic tree chain + fixed-anchor apex-only engagement + hub collision.**

- Chain links: hinge joints in a kinematic tree (link_0 = freejoint root, rest nested)
- One `connect` equality constraint closes the loop
- Track in XZ plane, hinge axis Y, gravity -Z, floor at z=0
- Sprockets are children of the hull body
- Chain link trees are independent free bodies, coupled to hull only via engagement constraints

## Sprocket Engagement

Engagement constraints pin a link's center to a point on the sprocket at radius R. **Anchor is set once and held fixed** — this creates real constraint forces that transfer sprocket torque to the chain and ultimately to the ground.

### Apex-Only Engagement
Only engage ~2-3 links at the deepest point of each arc (drive: `lx < sx - R*0.5`, idler: `lx > sx + R*0.5`). Links near the tangent transition points are NOT engaged — they ride on the sprocket hub via collision instead. This gives a stable engagement count (~3 per sprocket) and eliminates force discontinuities from rapid constraint toggling.

### Disengagement
Angle-based: store local angle at engagement, compute `world = local - spr_angle` each step, disengage when deviation from arc center exceeds ARC_HALF (π×0.40).

### Hub Collision
Sprocket cylinders have collision enabled (`contype=1 conaffinity=2`) so chain links physically ride over them during transitions in/out of the engagement zone.

## Key Implementation Details

### Loop Closure Anchor
MuJoCo auto-computes the second anchor (`eq_data[idx, 3:6]`) from body positions at XML parse time. After bending the chain via qpos, this is wrong by meters. Recompute after `mj_forward`:
```python
w1 = data.xpos[bid1] + data.xmat[bid1].reshape(3,3) @ model.eq_data[idx, 0:3]
model.eq_data[idx, 3:6] = data.xmat[bid2].reshape(3,3).T @ (w1 - data.xpos[bid2])
```

### Pre-seed Engagement at t=0
Determine which links start on arcs from stadium geometry. Set anchors at actual distance (not SPROCKET_R) for zero initial correction force. Runtime engagement uses SPROCKET_R with ±0.10 tolerance.

### Y-axis Hinge Convention
Positive Y rotation = X toward -Z. Negate all angles from `atan2(dz, dx)`:
- Hinge qpos: `-delta`
- Link_0 quat: `(cos(-ang/2), 0, sin(-ang/2), 0)`
- Engagement: `local = world + spr_angle`
- Disengagement: `world = local - spr_angle`

## Collision Groups
| Body | contype | conaffinity | Collides with |
|------|---------|-------------|---------------|
| Links | 2 | 1 | Ground, hubs |
| Ground | 1 | 2 | Links |
| Sprocket hubs | 1 | 2 | Links |
| Hull | 0 | 0 | Nothing |

## What Doesn't Work

- **Continuously-updated anchors**: zero constraint force, sprocket spins freely, chain doesn't advance
- **Engaging all arc links**: 10+ per sprocket, rapid toggling (eng=13→19→13), force discontinuities, hull oscillates
- **Fixed anchors + large ARC_HALF (>π×0.5)**: chain stalls at ~170 deg, hinge tree locks up
- **Freejoint chain + connect constraints for pins**: ball joints wobble, solver overwhelmed
- **Contact-based sprocket teeth**: solver can't maintain chain integrity
- **High sprocket height**: bottom treads float above ground, no traction

## Performance

### Timestep
dt=0.004: same quality as 0.002 at 3x speed. dt=0.008 degrades constraints.

### Rendering (WSL2)
- Shadows: 56% of render time. Disable with `<quality shadowsize="0"/>`.
- Long capsule geoms (grid lines): 8x slower. Use checker texture.
- Offscreen: ~50-100ms/frame on WSL2 regardless of backend. GPU passthrough only for GUI viewer.

### GUI Loop
Cap steps per frame to avoid death spiral (sim slower than realtime → tries to catch up → falls further behind → infinite loop).

## MuJoCo Facts
- `eq_data[i, 0:3]` = anchor in body1, `[3:6]` = body2
- `data.eq_active` not `model.eq_active` (MuJoCo 3.6)
- `connect` = ball joint (position only, no rotation)
- Collision check: `(ct1 & ca2) || (ct2 & ca1)`
- Box-cylinder collision works. Capsule-cylinder does NOT.
- `contype=0 conaffinity=0` must be explicit on visual geoms

## Parameters
```
N_LINKS=30  SPROCKET_R=0.35  HALF_SPAN=1.4  TIMESTEP=0.004
TARGET_VEL=1.0 rad/s  kv=200  Hull=20kg  ARC_HALF=π×0.40
Engagement: apex-only, dist tolerance ±0.10
Solver: Newton, iter=300, tol=1e-10, noslip=10
Constraint: solref=0.005 1, solimp=0.95 0.99 0.001
SPROCKET_Z = SPROCKET_R + 0.02
```
