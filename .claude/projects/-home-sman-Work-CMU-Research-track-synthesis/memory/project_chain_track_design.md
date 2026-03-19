---
name: chain_track_design_lessons
description: Hard-won lessons from building a contact-based tank track in MuJoCo — collision pairs, constraint tuning, sprocket geometry, spawn placement
type: project
---

## MuJoCo Chain Track Design Lessons

### Collision pair compatibility
- box-box, box-capsule, box-sphere, box-cylinder, capsule-capsule, sphere-cylinder: **YES**
- capsule-cylinder, cylinder-cylinder: **NO** — use capsules or boxes instead

### Sprocket geometry
- Hub (cylinder): collision-enabled, long in Z to span full track width — chain side links rest on it
- Pins (capsules along Z): short, only extend between the side chains — engage cross-links/rollers
- N_PINS should match chain pitch: `round(2π * R / link_pitch)` so valleys align with links
- Don't confuse "pins long for support" (that's the hub's job) vs "pins short for engagement" (pins poke between cross-links)

### Chain link design (dual-chain)
- Each link body has: left side box, right side box, cross-link tread plate, optional roller capsule at joint
- Side chains at z = ±CHAIN_Z, cross-links span between them
- Rollers (capsules at joint points) mesh with sprocket pins (capsule-capsule contact works)

### Constraint approach
- All links as independent freejoint bodies + equality connect constraints to close the loop
- Deep kinematic trees (parent-child chain of 20+ bodies) blow up immediately — avoid
- solref ~0.002, solimp ~0.97/0.999 is a good starting point
- solref=0.0005 is too stiff — amplifies interpenetration into explosions
- noslip_iterations=200 is fine, 500 is overkill

### Spawn placement
- Chain center must be offset outward: `chain_r = SPROCKET_R + PIN_RADIUS + SIDE_THICK`
- Spawning at the same radius as the pins causes interpenetration → 10,000N forces → explosion
- Always run `--debug` first to check contact forces at step 0 before trying GUI

### Stadium path
- Left arc must go π/2 → 3π/2 (wrapping LEFT of left sprocket), NOT π/2 → -π/2 (which wraps right)
- Right arc goes -π/2 → π/2 (correct)
- Tangent angle for left arc: `theta + π/2` (counter-clockwise traversal)

### Stability
- Warmup period (no drive for 0.5s, ramp over 0.5s) prevents initial shock
- Drive both sprockets to avoid one side pulling chain off the other
- Contact forces should be <50N steady state for ~30g links — anything >500N means geometry problem
- Gravity of -3 m/s² causes ~15cm sag on straight sections — acceptable or reduce gravity

### Debugging
- `--debug` mode: headless, prints per-step: ncon, max contact force, eq violation, max qvel, stadium tracking error
- Key health indicators: ncon > 0 (chain touching sprockets), cf < 50N (no jamming), eq_v < 2mm, trk < 5cm
- ncon = 0 after settling means chain fell off sprockets

**Why:** Each of these was discovered through a sim explosion or visual failure. Following these saves hours of trial and error.
**How to apply:** Reference when building or modifying any MuJoCo chain/track/belt mechanism.
