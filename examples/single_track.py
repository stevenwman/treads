"""Single-track chain example -- the "hello world" of track simulation.

One chain loop wrapping around two sprockets (drive + idler) in the XZ plane.
No hull, no obstacles, no tank -- just the bare chain mechanism.

Run:  python examples/single_track.py

Architecture:
    Reuses shared/ modules (geometry, engagement) but with a simpler config
    and XML builder. Good starting point before diving into the full tank.
"""

# -- Path setup so we can import from shared/ --
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math, time
from dataclasses import dataclass
import mujoco
import mujoco.viewer as mjv

from shared.geometry import stadium_point, normalize_angle
from shared.engagement import EngagementManager as _BaseEngagementManager


# ============================================================================
# Single-side engagement (the base class loops over left+right sides)
# ============================================================================

class SingleTrackEngagement(_BaseEngagementManager):
    """EngagementManager that only processes one side ("left").

    The base class iterates SIDES = [("left",1), ("right",-1)].  We override
    seed() and update() to iterate only over [("left",1)] since this demo
    has a single track with no "right" side bodies in the model.
    """

    def seed(self, model, data, lookups):
        c = self.config
        for spr in c.engaging_sprockets:
            bid = lookups.sprocket_body_ids[("left", spr.name)]
            sx, sz = data.xpos[bid][0], data.xpos[bid][2]
            candidates = []
            for i in range(c.n_links):
                lx, lz = data.xpos[lookups.link_body_ids["left"][i]][[0, 2]]
                dx, dz = lx - sx, lz - sz
                dist = math.sqrt(dx*dx + dz*dz)
                if self._is_on_arc(lx, sx, spr.name, dist):
                    candidates.append((abs(dz), i, dx, dz, dist))
            candidates.sort()
            for _, i, dx, dz, dist in candidates[:c.max_engaged_per_sprocket]:
                self._engage_link(model, data, lookups, "left", i,
                                  spr.name, dx, dz, dist)
        mujoco.mj_forward(model, data)

    def update(self, model, data, lookups):
        c = self.config
        for i in range(c.n_links):
            lx, lz = data.xpos[lookups.link_body_ids["left"][i]][[0, 2]]
            for spr in c.engaging_sprockets:
                key = ("left", i, spr.name)
                eq_idx = lookups.engagement_eq_ids.get(key)
                if eq_idx is None:
                    continue
                bid = lookups.sprocket_body_ids[("left", spr.name)]
                sx, sz = data.xpos[bid][0], data.xpos[bid][2]
                jid = lookups.sprocket_joint_ids[("left", spr.name)]
                spr_angle = data.qpos[model.jnt_qposadr[jid]]
                dx, dz = lx - sx, lz - sz
                dist = math.sqrt(dx*dx + dz*dz)
                if key in self._engaged:
                    self._maybe_disengage(data, key, eq_idx, spr.name, spr_angle)
                elif (self._is_on_arc(lx, sx, spr.name, dist)
                      and self._count_engaged("left", spr.name)
                          < c.max_engaged_per_sprocket):
                    self._engage_link(model, data, lookups, "left", i,
                                      spr.name, dx, dz, dist)
                else:
                    data.eq_active[eq_idx] = 0


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Sprocket:
    """One sprocket wheel. Mirrors shared.config.Sprocket."""
    name: str               # "drive" or "idler"
    x_offset: float         # X position relative to center
    engages_chain: bool = True
    has_tensioner: bool = False
    color: str = "0.2 0.6 0.2 1"


@dataclass
class SingleTrackConfig:
    """All parameters for the single-track demo.

    Stripped-down version of TankConfig -- no hull, no track gauge, just
    the chain geometry and physics.
    """
    name: str = "single_track"

    # Track shape
    sprocket_radius: float = 0.20       # radius of both sprockets
    n_links: int = 30                   # chain links in the loop
    half_span: float = 0.60             # half drive-to-idler distance

    # Link geometry
    link_thickness: float = 0.02        # half-height of each link box
    link_width: float = 0.10            # half-width of each link box

    # Physics
    timestep: float = 0.002
    target_velocity: float = 1.0        # drive motor target (rad/s)
    tension_stiffness: float = 80.0     # idler tensioner spring
    hinge_damping: float = 0.2          # inter-link hinge damping
    hinge_range: tuple = (-70, 70)      # inter-link hinge limits (degrees)
    sprocket_hinge_damping: float = 2.0

    # Engagement -- how links latch onto sprockets
    arc_half: float = None
    max_engaged_per_sprocket: int = 2
    engagement_distance_tol: float = 0.10
    engagement_arc_threshold: float = 0.85

    def __post_init__(self):
        if self.arc_half is None:
            self.arc_half = math.pi * 0.40
        # Drive sprocket (left/red), idler sprocket (right/blue)
        self.sprockets = [
            Sprocket("drive", -self.half_span, color="0.7 0.2 0.2 1"),
            Sprocket("idler",  self.half_span,
                     has_tensioner=True, color="0.2 0.2 0.7 1"),
        ]

    # Derived geometry (same formulas as TankConfig)
    @property
    def hub_radius(self):
        return self.sprocket_radius - 0.04

    @property
    def hub_half_width(self):
        return self.link_width + 0.02

    @property
    def perimeter(self):
        """Stadium path = two semicircles + two straights."""
        return 2 * math.pi * self.sprocket_radius + 4 * self.half_span

    @property
    def link_pitch(self):
        """Center-to-center spacing between adjacent links."""
        return self.perimeter / self.n_links

    @property
    def sprocket_z(self):
        """Height of sprocket centers above ground."""
        return self.sprocket_radius + 0.02

    @property
    def engaging_sprockets(self):
        return [s for s in self.sprockets if s.engages_chain]


# ============================================================================
# XML Builder
# ============================================================================

def build_xml(config):
    """Generate MJCF XML for the single-track scene.

    Structure: ground plane, 2 sprocket bodies with hinge joints,
    one chain of nested link bodies, velocity actuators, and equality
    constraints (loop closure + engagement).
    """
    c = config
    x = []  # accumulator for XML lines

    # -- Header: physics options, defaults, assets --
    x.append(f'<mujoco model="{c.name}">')
    x.append(f'  <option timestep="{c.timestep}" gravity="0 0 -9.81"'
             f' integrator="implicitfast" solver="Newton" cone="elliptic"'
             f' iterations="300" tolerance="1e-8" noslip_iterations="30"'
             f' impratio="2"/>')
    x.append('  <size nconmax="1000" njmax="3000"/>')
    x.append('  <visual><global offwidth="1280" offheight="720"/>'
             '<quality shadowsize="0" offsamples="1"/></visual>')
    x.append('  <default>')
    x.append('    <geom friction="1.0 0.005 0.005" condim="3" margin="0.002"'
             '     solref="0.004 1" solimp="0.95 0.99 0.001"/>')
    x.append('    <equality solref="0.01 1" solimp="0.9 0.95 0.001"/>')
    x.append('  </default>')
    x.append('  <asset>')
    x.append('    <texture name="grid" type="2d" builtin="checker"'
             ' width="512" height="512"'
             ' rgb1="0.4 0.45 0.4" rgb2="0.45 0.5 0.45"/>')
    x.append('    <material name="grid_mat" texture="grid"'
             ' texrepeat="10 10" texuniform="true"/>')
    x.append('  </asset>')

    # -- Worldbody --
    x.append('  <worldbody>')
    # Ground
    x.append('    <geom name="floor" type="plane" size="5 5 0.1"'
             ' material="grid_mat" contype="1" conaffinity="2"/>')
    # Cameras
    x.append('    <camera name="overview" pos="0 -2.5 1.5"'
             ' xyaxes="1 0 0 0 0.4 1" fovy="50"/>')
    x.append('    <camera name="top" pos="0 0 3"'
             ' xyaxes="1 0 0 0 1 0" fovy="50"/>')
    x.append('    <camera name="side" pos="-2 0 0.5"'
             ' xyaxes="0 -1 0 0 0 1" fovy="50"/>')
    x.append('    <light pos="0 -2 4" dir="0 0.3 -0.7"'
             ' diffuse="1 1 1" specular="0.3 0.3 0.3"/>')

    # -- Sprockets: two bodies at y=0, separated along X --
    for spr in c.sprockets:
        x.append(f'    <body name="{spr.name}_spr"'
                 f' pos="{spr.x_offset} 0 {c.sprocket_z}">')
        if spr.has_tensioner:
            # Spring-loaded slide keeps chain taut
            x.append(f'      <joint name="{spr.name}_slide" type="slide"'
                     f' axis="1 0 0" stiffness="{c.tension_stiffness}"'
                     f' damping="80" range="-0.05 0.3"/>')
        # Hinge -- sprocket rotates around Y axis
        x.append(f'      <joint name="{spr.name}_hinge" type="hinge"'
                 f' axis="0 1 0" damping="{c.sprocket_hinge_damping}"/>')
        # Hub cylinder -- links ride on this surface
        x.append(f'      <geom type="cylinder"'
                 f' size="{c.hub_radius} {c.hub_half_width}"'
                 f' euler="90 0 0" rgba="{spr.color}"'
                 f' contype="1" conaffinity="2"/>')
        x.append(f'    </body>')

    # -- Chain links: nested body tree --
    # Link 0 has a freejoint (6-DOF root). Links 1..N-1 are children
    # connected by Y-axis hinges -- one efficient kinematic tree.
    half_len = c.link_pitch / 2 - 0.005
    lo, hi = c.hinge_range
    x0, z0, a0 = stadium_point(0, c)
    qw, qy = math.cos(-a0/2), math.sin(-a0/2)

    x.append(f'    <body name="link_0" pos="{x0:.6f} 0 {c.sprocket_z+z0:.6f}"'
             f' quat="{qw:.6f} 0 {qy:.6f} 0">')
    x.append(f'      <freejoint name="link_0_jnt"/>')
    x.append(f'      <geom type="box"'
             f' size="{half_len:.4f} {c.link_width} {c.link_thickness}"'
             f' rgba="1 0.2 0.2 1" mass="0.05"'
             f' contype="2" conaffinity="1"/>')
    for i in range(1, c.n_links):
        x.append(f'      <body name="link_{i}" pos="{c.link_pitch:.6f} 0 0">')
        x.append(f'        <joint name="hinge_{i}" type="hinge" axis="0 1 0"'
                 f' pos="{-c.link_pitch/2:.6f} 0 0" damping="{c.hinge_damping}"'
                 f' limited="true" range="{lo} {hi}"/>')
        x.append(f'        <geom type="box"'
                 f' size="{half_len:.4f} {c.link_width} {c.link_thickness}"'
                 f' rgba="0.9 0.6 0.1 1" mass="0.05"'
                 f' contype="2" conaffinity="1"/>')
    for _ in range(c.n_links):
        x.append('    </body>')

    x.append('  </worldbody>')

    # -- Actuators: velocity motors on each sprocket hinge --
    x.append('  <actuator>')
    for spr in c.sprockets:
        x.append(f'    <velocity name="{spr.name}_motor"'
                 f' joint="{spr.name}_hinge"'
                 f' kv="500" ctrllimited="true" ctrlrange="-5 5"/>')
    x.append('  </actuator>')

    # -- Equality constraints --
    x.append('  <equality>')
    # Loop closure: last link connects back to first
    x.append(f'    <connect name="loop"'
             f' body1="link_{c.n_links-1}" body2="link_0"'
             f' anchor="{c.link_pitch/2:.6f} 0 0"/>')
    # Engagement: one constraint per (link, sprocket) pair, initially off.
    # The engagement manager activates these at runtime to simulate meshing.
    for i in range(c.n_links):
        for spr in c.engaging_sprockets:
            x.append(f'    <connect name="eng_{i}_{spr.name}"'
                     f' body1="{spr.name}_spr" body2="link_{i}"'
                     f' anchor="0 0 0" active="false"/>')
    x.append('  </equality>')
    x.append('</mujoco>')

    return '\n'.join(x)


# ============================================================================
# Simulation Lookups -- MuJoCo ID tables built once at startup
# ============================================================================

@dataclass
class SingleTrackLookups:
    """Pre-computed MuJoCo IDs for fast access during simulation.

    Keyed by ("left", ...) to match what the engagement manager expects.
    There is no "right" side -- we just use "left" as a label.
    """
    engagement_eq_ids: dict
    link_body_ids: dict
    sprocket_body_ids: dict
    sprocket_joint_ids: dict
    motor_actuator_ids: dict
    chain_hinge_ids: dict

    @classmethod
    def from_model(cls, model, config):
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        S = "left"  # our single track pretends to be the "left" side

        # Parse engagement constraint names: "eng_{link}_{sprocket}"
        eq_ids = {}
        for idx in range(model.neq):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
            if name and name.startswith("eng_"):
                parts = name[4:].split("_")
                eq_ids[(S, int(parts[0]), "_".join(parts[1:]))] = idx

        spr_b, spr_j, mot = {}, {}, {}
        for spr in config.sprockets:
            spr_b[(S, spr.name)] = bid(f"{spr.name}_spr")
            spr_j[(S, spr.name)] = jid(f"{spr.name}_hinge")
            mot[(S, spr.name)]   = aid(f"{spr.name}_motor")

        return cls(
            engagement_eq_ids=eq_ids,
            link_body_ids={S: [bid(f"link_{i}") for i in range(config.n_links)]},
            sprocket_body_ids=spr_b,
            sprocket_joint_ids=spr_j,
            motor_actuator_ids=mot,
            chain_hinge_ids={S: [jid(f"hinge_{i}")
                                 for i in range(1, config.n_links)]},
        )


# ============================================================================
# Initialization
# ============================================================================

def _set_initial_chain_shape(model, data, lookups, config):
    """Bend the flat kinematic tree into the stadium loop shape.

    Each inter-link hinge gets a qpos = angle change between consecutive
    stadium path points. Then fix loop-closure anchor to match.
    """
    for idx, joint_id in enumerate(lookups.chain_hinge_ids["left"]):
        i = idx + 1
        _, _, a_curr = stadium_point(i * config.link_pitch, config)
        _, _, a_prev = stadium_point((i-1) * config.link_pitch, config)
        data.qpos[model.jnt_qposadr[joint_id]] = -normalize_angle(a_curr - a_prev)
    mujoco.mj_forward(model, data)

    # Fix loop-closure anchor so last link meets first cleanly
    for idx in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if name == "loop":
            b1, b2 = model.eq_obj1id[idx], model.eq_obj2id[idx]
            r1 = data.xmat[b1].reshape(3, 3)
            world_pt = data.xpos[b1] + r1 @ model.eq_data[idx, 0:3]
            r2 = data.xmat[b2].reshape(3, 3)
            model.eq_data[idx, 3:6] = r2.T @ (world_pt - data.xpos[b2])
    mujoco.mj_forward(model, data)


# ============================================================================
# Simulation
# ============================================================================

def create_simulation(config):
    """Build model, bend chain, seed engagement. Returns (model, data, lookups, engagement)."""
    xml = build_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lookups = SingleTrackLookups.from_model(model, config)
    _set_initial_chain_shape(model, data, lookups, config)

    engagement = SingleTrackEngagement(config)
    engagement.seed(model, data, lookups)
    return model, data, lookups, engagement


def step_simulation(model, data, lookups, engagement, config):
    """Advance one timestep: set motor targets, update engagement, step physics."""
    ramp = min(1.0, data.time / 1.0)   # smooth ramp-up over first second
    vel = config.target_velocity * ramp
    for spr in config.sprockets:
        data.ctrl[lookups.motor_actuator_ids[("left", spr.name)]] = vel
    engagement.update(model, data, lookups)
    mujoco.mj_step(model, data)


def run_gui(config=None):
    """Launch the interactive MuJoCo viewer with real-time stepping."""
    if config is None:
        config = SingleTrackConfig()

    print(f"Single-track demo: {config.n_links} links, "
          f"R={config.sprocket_radius}m, span={2*config.half_span}m")
    print(f"Perimeter={config.perimeter:.3f}m, pitch={config.link_pitch:.4f}m")

    model, data, lookups, engagement = create_simulation(config)

    with mjv.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        sim_start = data.time
        frame_count = 0
        last_report = wall_start

        while viewer.is_running():
            t0 = time.perf_counter()
            # Step simulation to match wall-clock time
            target_time = sim_start + (t0 - wall_start)
            n_steps = 0
            while data.time < target_time and n_steps < 10:
                step_simulation(model, data, lookups, engagement, config)
                n_steps += 1
            if data.time < target_time:  # sim can't keep up -- reset clock
                wall_start, sim_start = t0, data.time

            t1 = time.perf_counter()
            viewer.sync()
            t2 = time.perf_counter()

            frame_count += 1
            if t2 - last_report > 2.0:
                fps = frame_count / (t2 - last_report)
                print(f"FPS={fps:.1f}  sim={1000*(t1-t0):.1f}ms "
                      f"({n_steps} steps)  t={data.time:.1f}s")
                frame_count, last_report = 0, t2


if __name__ == "__main__":
    run_gui()
