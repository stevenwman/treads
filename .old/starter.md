# Tank track simulation in MuJoCo — implementation spec

## Goal

Simulate a driveable tank with two track chains that make real frictional contact with the ground. The track links are driven around a stadium-shaped loop by kinematic "ghost" mocap bodies, while the links themselves are dynamic rigid bodies that interact with terrain via MuJoCo's contact physics. This gives us realistic ground reaction forces without needing to model sprocket teeth or closed-loop chain kinematics.

## Architecture overview

```
For each track (left and right):
  - N mocap "ghost" bodies trace a stadium path relative to the chassis
  - N real rigid-body track links (freejoint) make ground contact
  - Soft weld constraints couple each link to its ghost
  - Ghost positions are updated every timestep from chassis pose + stadium offset

Force path:
  motor command → ghost positions advance along stadium
  → weld constraints drag links → links contact ground
  → ground friction → constraint forces feed back to chassis
```

## Key design decisions

- **Each track link is a separate freejoint body** in the worldbody (flat tree, not nested). No parent-child chain between links.
- **No link-to-link pin joints.** The links are independent. The mocap ghosts enforce the track shape — the links just follow along. This avoids all closed-loop kinematic headaches.
- **The stadium path is computed in chassis-local frame** then transformed to world frame each timestep. This means the ghosts move with the chassis automatically.
- **Weld constraints are soft** (tunable spring-damper via `solref` and `solimp`). This lets links deflect when they hit bumps, then spring back to the ideal path.

## Stadium parametric curve

The track path is a stadium (discorectangle): two straight segments connected by semicircular caps.

Parameters:

- `half_length`: half the straight segment length (center-to-center of sprocket/idler)
- `radius`: semicircle radius (sprocket/idler wheel radius)
- `y_offset`: lateral offset from chassis center (track width position)

```python
import numpy as np

def stadium_parametric(s, half_length, radius):
    """
    Given arc-length parameter s (0 to perimeter), return (x, z) in chassis-local frame.
    x = longitudinal, z = vertical. The bottom run is at z = -radius, top run at z = +radius.
    """
    flat = 2 * half_length
    arc = np.pi * radius
    perimeter = 2 * flat + 2 * arc
    s = s % perimeter

    if s < flat:
        # Bottom run: left to right
        x = -half_length + s
        z = -radius
        angle = 0.0
    elif s < flat + arc:
        # Right semicircle (around sprocket)
        theta = (s - flat) / radius - np.pi / 2
        x = half_length + radius * np.cos(theta)
        z = radius * np.sin(theta)
        angle = theta + np.pi / 2
    elif s < 2 * flat + arc:
        # Top run: right to left
        x = half_length - (s - flat - arc)
        z = radius
        angle = np.pi
    else:
        # Left semicircle (around idler)
        theta = (s - 2 * flat - arc) / radius + np.pi / 2
        x = -half_length + radius * np.cos(theta)
        z = radius * np.sin(theta)
        angle = theta + np.pi / 2

    return x, z, angle


def stadium_perimeter(half_length, radius):
    return 2 * (2 * half_length) + 2 * np.pi * radius
```

The `angle` output is the tangent angle, used to set the orientation of each link (it should be tangent to the path).

## Model parameters

```python
# Chassis
CHASSIS_HALF_SIZE = [1.5, 0.5, 0.2]  # x, y, z half-extents
CHASSIS_MASS = 1000.0  # kg

# Track geometry
HALF_LENGTH = 1.2       # half the sprocket-to-idler distance
RADIUS = 0.18           # sprocket/idler radius
Y_OFFSET = 0.55         # lateral offset from chassis center
Z_OFFSET = 0.0          # vertical offset of track center from chassis center

# Track links
N_LINKS = 50            # links per track (so 100 total)
LINK_HALF_SIZE = [0.04, 0.035, 0.008]  # x, y, z half-extents per link
LINK_MASS = 0.3         # kg per link
LINK_FRICTION = "1.5 0.005 0.001"  # sliding, torsional, rolling

# Weld constraint tuning
WELD_SOLREF = "0.02 0.8"    # timeconst, dampratio — stiffer = tighter tracking
WELD_SOLIMP = "0.9 0.95 0.001"  # dmin, dmax, width

# Simulation
TIMESTEP = 0.002
TRACK_SPEED = 3.0  # m/s linear track speed (controlled by user input)
```

## MJCF XML generation

Write a Python script `generate_tank.py` that:

1. Computes the initial position and orientation of each track link along the stadium
2. Generates the full MJCF XML
3. Saves it to `tank.xml`

### XML structure

```xml
<mujoco model="tank">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="50"/>

  <default>
    <geom condim="3"/>
    <joint damping="0.01"/>
  </default>

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>
    <material name="ground_mat" texture="grid" texrepeat="10 10"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="50 50 1" material="ground_mat"
          friction="1.0 0.005 0.001"/>

    <!-- Chassis -->
    <body name="chassis" pos="0 0 0.5">
      <freejoint name="chassis_free"/>
      <geom type="box" size="1.5 0.5 0.2" mass="1000"
            rgba="0.3 0.35 0.3 1"/>
      <!-- Visual-only sprocket and idler cylinders (no contact) -->
      <geom name="sprocket_L" type="cylinder" size="0.18 0.04"
            pos="1.2 0.55 0" euler="90 0 0"
            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>
      <geom name="sprocket_R" type="cylinder" size="0.18 0.04"
            pos="1.2 -0.55 0" euler="90 0 0"
            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>
      <geom name="idler_L" type="cylinder" size="0.15 0.04"
            pos="-1.2 0.55 0" euler="90 0 0"
            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>
      <geom name="idler_R" type="cylinder" size="0.15 0.04"
            pos="-1.2 -0.55 0" euler="90 0 0"
            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>
    </body>

    <!-- ============================================ -->
    <!-- GENERATED: Mocap ghosts and track links      -->
    <!-- ============================================ -->
    <!-- For each track (left, right) and each link i: -->

    <!-- Mocap ghost (kinematic, no contact) -->
    <!--
    <body mocap="true" name="ghost_L_0" pos="x y z">
      <geom type="box" size="0.04 0.035 0.008"
            contype="0" conaffinity="0" rgba="0 1 0 0.15"/>
    </body>
    -->

    <!-- Real track link (dynamic, makes ground contact) -->
    <!--
    <body name="link_L_0" pos="x y z" quat="w x y z">
      <freejoint name="link_L_0_free"/>
      <geom type="box" size="0.04 0.035 0.008" mass="0.3"
            friction="1.5 0.005 0.001" rgba="0.25 0.25 0.25 1"/>
    </body>
    -->

  </worldbody>

  <!-- ============================================ -->
  <!-- GENERATED: Weld constraints ghost → link     -->
  <!-- ============================================ -->
  <equality>
    <!--
    <weld body1="ghost_L_0" body2="link_L_0"
          solref="0.02 0.8" solimp="0.9 0.95 0.001"/>
    -->
  </equality>

  <!-- Contact exclusions: links shouldn't collide with chassis -->
  <contact>
    <!--
    <exclude body1="chassis" body2="link_L_0"/>
    -->
  </contact>

</mujoco>
```

### Initial placement

For each link `i` on each track (left/right):

```python
perimeter = stadium_perimeter(HALF_LENGTH, RADIUS)
link_spacing = perimeter / N_LINKS

for i in range(N_LINKS):
    s = i * link_spacing
    local_x, local_z, angle = stadium_parametric(s, HALF_LENGTH, RADIUS)

    # Position in world frame (chassis starts at origin)
    pos = [local_x, Y_OFFSET, local_z + CHASSIS_Z]  # for left track
    # For right track, negate Y_OFFSET

    # Orientation: rotate around Y axis by `angle`
    # Convert angle to quaternion (rotation about Y axis)
    quat = angle_to_quat_y(angle)
```

The `angle_to_quat_y` helper:

```python
def angle_to_quat_y(angle):
    """Quaternion for rotation about Y axis by `angle` radians."""
    return [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
```

## Simulation control script

Write `run_tank.py`:

```python
import mujoco
import mujoco.viewer
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("tank.xml")
data = mujoco.MjData(model)

# Cache body and mocap IDs
chassis_id = model.body("chassis").id
ghost_mocap_ids = {}  # dict: "ghost_L_0" -> mocap index
for side in ["L", "R"]:
    for i in range(N_LINKS):
        name = f"ghost_{side}_{i}"
        body_id = model.body(name).id
        ghost_mocap_ids[name] = model.body(name).mocapid[0]

# Track state
track_offset = 0.0  # cumulative arc-length offset (driven by input)
left_speed = 0.0
right_speed = 0.0

def update_ghosts():
    """Recompute all ghost positions from chassis pose + stadium offset."""
    chassis_pos = data.xpos[chassis_id]
    chassis_mat = data.xmat[chassis_id].reshape(3, 3)

    perimeter = stadium_perimeter(HALF_LENGTH, RADIUS)
    spacing = perimeter / N_LINKS

    for side in ["L", "R"]:
        y_sign = 1.0 if side == "L" else -1.0
        speed = left_speed if side == "L" else right_speed
        offset = track_offset  # or separate offsets per side for steering

        for i in range(N_LINKS):
            s = (speed * data.time + i * spacing) % perimeter
            local_x, local_z, angle = stadium_parametric(s, HALF_LENGTH, RADIUS)

            # Local position relative to chassis
            local_pos = np.array([local_x, y_sign * Y_OFFSET, local_z + Z_OFFSET])

            # Transform to world frame
            world_pos = chassis_pos + chassis_mat @ local_pos

            # Set mocap position
            mid = ghost_mocap_ids[f"ghost_{side}_{i}"]
            data.mocap_pos[mid] = world_pos

            # Set mocap orientation (chassis rotation * local track tangent)
            # The link should be tangent to the stadium and aligned with chassis yaw
            local_quat = angle_to_quat_y(angle)
            world_quat = quat_multiply(chassis_quat(data, chassis_id), local_quat)
            data.mocap_quat[mid] = world_quat


def key_callback(keycode):
    """WASD controls: W/S = forward/back, A/D = steer."""
    global left_speed, right_speed
    base_speed = 3.0
    turn_diff = 1.5

    if keycode == ord('w'):  # forward
        left_speed = base_speed
        right_speed = base_speed
    elif keycode == ord('s'):  # backward
        left_speed = -base_speed
        right_speed = -base_speed
    elif keycode == ord('a'):  # turn left
        left_speed = base_speed - turn_diff
        right_speed = base_speed + turn_diff
    elif keycode == ord('d'):  # turn right
        left_speed = base_speed + turn_diff
        right_speed = base_speed - turn_diff
    else:
        left_speed = 0.0
        right_speed = 0.0


# Main simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        update_ghosts()
        mujoco.mj_step(model, data)
        viewer.sync()
```

## Per-side track speed for differential steering

Each track side needs its own cumulative arc-length offset to allow differential steering (different speeds on left vs right track). Track this as:

```python
left_arc = 0.0   # cumulative arc position for left track
right_arc = 0.0  # cumulative arc position for right track

# Each timestep:
left_arc += left_speed * model.opt.timestep
right_arc += right_speed * model.opt.timestep

# Then for link i on left track:
s = (left_arc + i * spacing) % perimeter
```

## Force readout

To measure ground reaction forces on the chassis (e.g. for force feedback or control):

Option A — read constraint forces from `data.efc_force` for the weld constraints.

Option B — attach a force/torque sensor to the chassis:

```xml
<sensor>
  <force name="chassis_force" site="chassis_center"/>
  <torque name="chassis_torque" site="chassis_center"/>
</sensor>
```

Then read `data.sensordata`.

## Collision filtering

Track links should collide with the ground but NOT with:

- The chassis
- The visual-only sprocket/idler geoms
- Other track links (they're independent, no pin joints)

Use `contype`/`conaffinity` bit masks:

```
Ground plane:         contype=1  conaffinity=1
Chassis geom:         contype=2  conaffinity=2
Track link geoms:     contype=1  conaffinity=1  (collides with ground)
Sprocket/idler geoms: contype=0  conaffinity=0  (no collision)
Ghost geoms:          contype=0  conaffinity=0  (no collision)
```

To prevent links from colliding with each other, set a unique group or use `<exclude>` pairs. The simplest approach: give all link geoms `contype=4 conaffinity=1` and the ground `contype=1 conaffinity=4`. This way links collide with ground but not with each other or the chassis.

Detailed bit mask plan:

```
Ground:   contype=0b001  conaffinity=0b010   (ground "offers" to type 2)
Links:    contype=0b010  conaffinity=0b001   (links "offer" to type 1)
Chassis:  contype=0b100  conaffinity=0b100   (chassis-to-chassis only, or 0 if not needed)
```

A contact is generated when `(geomA.contype & geomB.conaffinity) || (geomB.contype & geomA.conaffinity)` is nonzero. With the above:

- Ground × Link: `(1 & 1) = 1` → contact ✓
- Link × Link: `(2 & 1) = 0` and `(2 & 1) = 0` → no contact ✓
- Link × Chassis: `(2 & 4) = 0` and `(4 & 1) = 0` → no contact ✓

## Terrain

For testing on rough terrain, add box obstacles or a heightfield:

```xml
<body pos="3 0 0.05">
  <geom type="box" size="0.5 2 0.05" friction="1.0 0.005 0.001"
        contype="1" conaffinity="2" rgba="0.5 0.4 0.3 1"/>
</body>
```

## Summary of files to generate

| File               | Purpose                                                  |
| ------------------ | -------------------------------------------------------- |
| `generate_tank.py` | Generates `tank.xml` with all links, ghosts, constraints |
| `tank.xml`         | The MuJoCo model (auto-generated, don't hand-edit)       |
| `run_tank.py`      | Simulation loop with ghost updates and keyboard control  |
| `stadium.py`       | Stadium parametric curve and helper functions            |

## Tuning guide

| Parameter               | Effect                            | Start value |
| ----------------------- | --------------------------------- | ----------- |
| `solref[0]` (timeconst) | Lower = stiffer tracking          | 0.02        |
| `solref[1]` (dampratio) | Higher = more damped              | 0.8         |
| `solimp[0]` (dmin)      | Min impedance                     | 0.9         |
| `solimp[1]` (dmax)      | Max impedance                     | 0.95        |
| `N_LINKS`               | More = smoother track, slower sim | 50          |
| `LINK_FRICTION`         | Higher = more traction            | 1.5         |
| `LINK_MASS`             | Heavier = more momentum, more sag | 0.3         |
| `timestep`              | Smaller = more stable, slower     | 0.002       |

If the track links vibrate or bounce: decrease `solref[0]` or increase damping.
If the links lag behind the ghosts: decrease `solref[0]` (stiffer spring).
If the simulation is unstable: decrease `timestep` or soften `solimp`.
