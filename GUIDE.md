# Tank Track Simulation — How It Works

A MuJoCo simulation of tank-style tracked vehicles. This guide explains the
core ideas so you can read the code without prior MuJoCo experience.

## The problem

A tank track is a loop of rigid links (the "chain") wrapped around sprocket
wheels. The drive sprocket is powered by the engine and pulls the chain; the
idler sprocket on the other end keeps the chain taut. The bottom of the chain
presses against the ground and propels the vehicle.

Simulating this is tricky because:

1. **The chain is a closed loop**, but physics engines model open kinematic
   trees (parent → child → grandchild). There's no built-in way to say "the
   last link connects back to the first."
2. **Links must engage and disengage from sprockets** as the chain moves —
   they latch on when they enter the arc, ride along as the sprocket turns,
   then release when they exit.
3. **Contact is everywhere** — links touching the ground, links touching
   sprocket hubs, links touching each other.

## Our approach

### Chain as a kinematic tree + loop-closure constraint

Instead of N free-floating links, we build **one kinematic tree per track
side**:

```
link_0 (freejoint — 6 DOF root)
  └─ link_1 (hinge)
       └─ link_2 (hinge)
            └─ ...
                 └─ link_N-1 (hinge)
```

Each hinge joint lets adjacent links bend relative to each other (like a
real chain). Then we add a **connect equality constraint** from link_N-1
back to link_0 to close the loop. MuJoCo enforces this constraint at each
timestep, keeping the chain circular.

Why not use N separate free bodies? Because MuJoCo's constraint solver
works much better with kinematic trees. Free bodies + many constraints =
slow and wobbly. A kinematic tree + one loop constraint = fast and stable.

### Stadium-shaped path

The track follows a "stadium" shape (rectangle with semicircular ends):

```
    ←── top (links move left) ──────────────
   ╭─────────────────────────────────────────╮
   │  drive                           idler  │
   │  sprocket                     sprocket  │
   ╰─────────────────────────────────────────╯
    ───────── bottom (links move right) ──→
```

We parameterize this as a function `stadium_point(arc_length)` that returns
the (x, z, angle) at any distance along the path. This is used to:

- Place the chain links in their initial positions
- Set the initial hinge angles so the chain starts in the right shape

See `shared/geometry.py` for the math.

### Sprocket engagement

This is the key insight of the simulation. We don't simulate gear teeth.
Instead, we use **MuJoCo equality constraints** that we turn on and off:

1. **At startup**, we pre-create one `connect` constraint for every
   (link, engaging sprocket) pair. All start **inactive**.

2. **Each timestep**, the `EngagementManager` checks every link:
   - Is the link wrapping around a sprocket's arc zone? → **Engage**: compute
     where the link sits in the sprocket's local frame, set the constraint
     anchor there, activate it. The link is now pinned to the sprocket.
   - Has an engaged link rotated past the arc boundary? → **Disengage**:
     deactivate the constraint. The link is free again.

3. The drive sprocket has a velocity motor. When it turns, engaged links
   get dragged along → the whole chain moves → bottom links push against
   the ground → the tank moves forward.

See `shared/engagement.py` for the implementation.

### Why "engage only at the apex"?

We only engage links that are past the deepest point of the arc (directly
behind the drive sprocket, or directly in front of the idler). And we cap
it to 2 links per sprocket at a time.

If we engaged every link touching the sprocket, the constraint solver would
fight itself — too many constraints pulling in slightly different directions.
Two links at the apex is enough to transmit torque without instability.

## Code structure

```
track_synthesis/
├── shared/                     # reusable building blocks
│   ├── config.py               # TankConfig, Sprocket dataclasses
│   ├── geometry.py             # stadium path math
│   ├── xml_builder.py          # generates MuJoCo XML
│   ├── engagement.py           # EngagementManager
│   └── simulation.py           # MuJoCo setup + run modes
├── tank_3spr/                  # 3-sprocket variant
│   └── __main__.py             # just config + run()
├── tank_4spr/                  # 4-sprocket variant
│   └── __main__.py             # just config + run()
├── examples/
│   └── single_track.py         # simplest possible example (no hull)
```

### Data flow

```
TankConfig                          # you define the parameters
    │
    ▼
build_tank_xml(config) ──→ XML string
    │
    ▼
MjModel + MjData                   # MuJoCo creates the physics world
    │
    ▼
SimLookups.from_model()            # pre-cache all body/joint/actuator IDs
    │
    ▼
set_initial_chain_shape()          # bend hinges into stadium shape
    │
    ▼
EngagementManager.seed()           # activate constraints for links on arcs
    │
    ▼
 ┌──────── simulation loop ────────┐
 │  set motor velocities           │
 │  engagement.update()            │
 │  mj_step()                      │
 │  viewer.sync()                  │
 └─────────────────────────────────┘
```

### Adding a new tank variant

Create a new folder with a `__main__.py` that defines a `TankConfig`:

```python
from shared import TankConfig, Sprocket, run

CONFIG = TankConfig(
    name="my_tank",
    sprocket_radius=0.30,
    n_links=24,
    half_span=1.0,
    sprockets=[
        Sprocket("drive", x_offset=-1.0, engages_chain=True, color="0.7 0.2 0.2 1"),
        Sprocket("idler", x_offset=1.0,  engages_chain=True, has_tensioner=True, color="0.2 0.2 0.7 1"),
    ],
)

if __name__ == "__main__":
    run(CONFIG)
```

That's it. All the XML generation, engagement logic, and rendering reuse
the shared modules.

## Running

```bash
python -m tank_3spr              # 3-sprocket interactive viewer
python -m tank_3spr --debug      # headless diagnostics
python -m tank_3spr --record     # save MP4

python -m tank_4spr              # 4-sprocket variant

python examples/single_track.py  # simplest example — one chain, no hull
```

## Key MuJoCo concepts used

| Concept | What it does here |
|---|---|
| **freejoint** | Gives link_0 and the hull full 6-DOF freedom |
| **hinge joint** | Connects adjacent chain links (bend in XZ plane) |
| **slide joint** | Idler tensioner — spring pulls it outward to keep chain taut |
| **connect constraint** | Loop closure (last→first link) and sprocket engagement |
| **eq_active** | Runtime flag to enable/disable constraints (engagement on/off) |
| **velocity actuator** | Motors on sprocket hinges — we set target angular velocity |
| **contype/conaffinity** | Collision filtering — links hit ground, not the hull |
