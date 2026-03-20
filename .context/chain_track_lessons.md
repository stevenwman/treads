# Tank Track Simulation in MuJoCo

## Architecture

**Kinematic tree chain + fixed-anchor engagement (max 2 per sprocket) + hub collision + hinge limits.**

- Chain links: hinge joints in a kinematic tree (link_0 = freejoint root, rest nested)
- One `connect` equality constraint closes the loop
- Track in XZ plane, hinge axis Y, gravity -Z, floor at z=0
- Sprockets are children of the hull body
- Chain link trees are independent free bodies, coupled to hull only via engagement constraints

## Sprocket Engagement

Engagement constraints pin a link's center to a point on the sprocket at radius R. **Anchor is set once and held fixed** — this creates real constraint forces that transfer sprocket torque to the chain and ultimately to the ground.

### Max 2 Per Sprocket
Hard cap of 2 engaged links per sprocket. Only engage links closest to the horizontal center line (drive: `lx < sx - R*0.85`, idler: `lx > sx + R*0.85`). Both seed and runtime enforce this cap via `_count_engaged()` check before engaging. This prevents overconstraining — 3+ links fighting each other causes artifacts.

### Disengagement
Angle-based: store local angle at engagement, compute `world = local - spr_angle` each step, disengage when deviation from arc center exceeds ARC_HALF (π×0.40).

### Hub Collision
Sprocket cylinders have collision enabled (`contype=1 conaffinity=2`) so chain links physically ride over them during transitions in/out of the engagement zone.

### Hinge Angle Limits
Chain hinge joints have `limited="true" range="-1.2 1.2"` (±69 deg). Without limits, links can flip 180+ degrees under load (especially at sprocket transitions), permanently breaking the chain. The max normal angle on the arc is ~42 deg, so 69 deg gives headroom.

## Key Implementation Details

### Loop Closure Anchor
MuJoCo auto-computes the second anchor (`eq_data[idx, 3:6]`) from body positions at XML parse time. After bending the chain via qpos, this is wrong by meters. Recompute after `mj_forward`:
```python
w1 = data.xpos[bid1] + data.xmat[bid1].reshape(3,3) @ model.eq_data[idx, 0:3]
model.eq_data[idx, 3:6] = data.xmat[bid2].reshape(3,3).T @ (w1 - data.xpos[bid2])
```

### Pre-seed Engagement at t=0
Use the same position check as runtime (not stadium arc-length). Collect candidates, sort by closeness to horizontal center line (`|dz|`), take top 2. Use actual distance for anchor (zero initial force).

### Split Constraint Stiffness
Contacts and equality constraints need different stiffness:
- **Contacts** (ground/obstacles): stiff `solref="0.004 1"` — prevents penetration
- **Equality constraints** (engagement/loop): soft `solref="0.01 1"` — forgiving under obstacle loads

Using the same solref for both causes either penetration (too soft for contacts) or explosions (too stiff for engagement).

## Y-axis Hinge Convention (XZ Plane)

**Y-axis hinge positive = X toward -Z** (opposite to CCW in XZ). Negate all angles:
- Initial hinge qpos: `qpos = -delta`
- Link_0 quaternion: `(cos(-ang/2), 0, sin(-ang/2), 0)`
- Engagement local angle: `local = world + spr_angle`
- Disengagement world angle: `world = local - spr_angle`
- Anchor: `(R*cos(local), 0, R*sin(local))`

## Collision Groups
| Body | contype | conaffinity | Collides with |
|------|---------|-------------|---------------|
| Links | 2 | 1 | Ground, hubs, obstacles |
| Ground/obstacles | 1 | 2 | Links |
| Sprocket hubs | 1 | 2 | Links |
| Hull | 0 | 0 | Nothing |

## Solver & Integrator

**`integrator="implicitfast"`** is critical for contact-rich stability. Handles velocity-dependent contact forces implicitly, prevents energy injection that causes explosions.

**`cone="elliptic"`** with Newton solver — more stable friction model for many simultaneous contacts.

**`impratio="2"`** — friction constraint is 2x stiffer than normal, prevents slip without increasing friction coefficient.

## What Doesn't Work

- **Continuously-updated anchors**: zero constraint force, chain doesn't advance
- **3+ links engaged per sprocket**: overconstraining causes artifacts
- **No hinge limits**: links flip 180+ deg under load, chain snaps permanently
- **Same solref for contacts and equality**: either penetration or explosions
- **solref < 2×timestep**: creates maximum stiffness, huge forces
- **Large ARC_HALF (>π×0.5) with fixed anchors**: chain stalls
- **High sprocket height**: bottom treads float, no traction
- **Grid line geoms**: destroy renderer (8x slower)
- **Shadows**: 56% of render time, disable with `shadowsize="0"`

## Parameters
```
N_LINKS=30  SPROCKET_R=0.35  HALF_SPAN=1.4  TIMESTEP=0.002
TARGET_VEL=1.0 rad/s  kv=500  Hull=20kg  ARC_HALF=π×0.40
MAX_ENG_PER_SPROCKET=2
Engagement: lx < sx - R*0.85 (drive), lx > sx + R*0.85 (idler), dist ±0.10
Hinge limits: ±1.2 rad (±69 deg)
Hinge damping: 0.2 (chain), 2.0 (sprockets)
Solver: implicitfast, Newton, elliptic cone, iter=300, tol=1e-8, noslip=30, impratio=2
Contact: solref=0.004 1, solimp=0.95 0.99 0.001, condim=3, friction=1.0
Equality: solref=0.01 1, solimp=0.9 0.95 0.001
SPROCKET_Z = SPROCKET_R + 0.02
```

## MuJoCo Facts
- `eq_data[i, 0:3]` = anchor in body1, `[3:6]` = body2
- `data.eq_active` not `model.eq_active` (MuJoCo 3.6)
- `connect` = ball joint (position only, no rotation)
- Auto-computed anchor2 is stale if qpos changes after XML parse
- Collision: `(ct1 & ca2) || (ct2 & ca1)` — both directions
- Box-cylinder works. Capsule-cylinder does NOT.
- Box-box can generate up to 8 contact points per pair
- `contype=0 conaffinity=0` must be explicit on visual geoms
- Velocity actuators (`<velocity kv="...">`) for constant-speed drives
- `solref[0]` must be ≥ 2×timestep (MuJoCo clamps to this minimum)

## Rendering (WSL2)
- Shadows: 56% of render time → `<quality shadowsize="0"/>`
- Long capsule geoms: 8x slower → use checker texture
- Offscreen: ~50-100ms/frame regardless of backend
- GUI viewer uses Windows GPU via WSLg (fast), offscreen Renderer is CPU (slow)
- GUI death spiral: cap MAX_STEPS_PER_FRAME, reset wall clock when falling behind
