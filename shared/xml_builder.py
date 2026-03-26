"""Builds MuJoCo MJCF XML for a tank with two tracked drives.

The XML describes:
    - A ground plane with obstacles (ramp, bump, step)
    - A hull body with sprocket wheels as children
    - Two chain tracks (left/right), each a kinematic tree of linked boxes
    - Velocity actuators on every sprocket
    - Equality constraints for loop closure and sprocket engagement
"""
import math
from .config import SIDES
from .geometry import stadium_point


def build_tank_xml(config):
    """Generate complete MJCF XML string for the tank scene.

    Args:
        config: TankConfig with all geometry and physics parameters.

    Returns:
        XML string ready for mujoco.MjModel.from_xml_string().
    """
    xml = []

    _add_header(xml, config)
    _add_world_open(xml)
    _add_cameras_and_lights(xml)
    _add_ground_and_obstacles(xml)
    _add_hull_with_sprockets(xml, config)
    _add_chain_links(xml, config)
    xml.append('  </worldbody>')
    _add_actuators(xml, config)
    _add_equality_constraints(xml, config)
    xml.append('</mujoco>')

    return '\n'.join(xml)


# ---------------------------------------------------------------------------
# Private helpers — each one appends XML lines for one section
# ---------------------------------------------------------------------------

def _add_header(xml, config):
    """Simulation options, size limits, visual quality, default physics."""
    xml.append(f'<mujoco model="{config.name}">')
    xml.append(
        f'  <option timestep="{config.timestep}" gravity="0 0 -9.81"'
        f' integrator="implicitfast" solver="Newton" cone="elliptic"'
        f' iterations="300" tolerance="1e-8" noslip_iterations="30"'
        f' impratio="2"/>')
    xml.append('  <size nconmax="2000" njmax="6000"/>')
    xml.append('  <visual>'
               '<global offwidth="1920" offheight="1080"/>'
               '<quality shadowsize="0" offsamples="1"/>'
               '</visual>')
    xml.append('  <default>')
    xml.append('    <geom friction="1.0 0.005 0.005" condim="3" margin="0.002"'
               '     solref="0.004 1" solimp="0.95 0.99 0.001"/>')
    xml.append('    <equality solref="0.01 1" solimp="0.9 0.95 0.001"/>')
    xml.append('  </default>')
    xml.append('  <asset>')
    xml.append('    <texture name="grid" type="2d" builtin="checker"'
               ' width="512" height="512"'
               ' rgb1="0.4 0.45 0.4" rgb2="0.45 0.5 0.45"/>')
    xml.append('    <material name="grid_mat" texture="grid"'
               ' texrepeat="10 10" texuniform="true"/>')
    xml.append('  </asset>')


def _add_world_open(xml):
    xml.append('  <worldbody>')


def _add_cameras_and_lights(xml):
    xml.append('    <camera name="overview" pos="0 -5 3"'
               ' xyaxes="1 0 0 0 0.3 1" fovy="50"/>')
    xml.append('    <camera name="top" pos="0 0 6"'
               ' xyaxes="1 0 0 0 1 0" fovy="60"/>')
    xml.append('    <camera name="side" pos="-4 0 1.5"'
               ' xyaxes="0 -1 0 0 0 1" fovy="50"/>')
    xml.append('    <light pos="0 -3 6" dir="0 0.3 -0.7"'
               ' diffuse="1 1 1" specular="0.3 0.3 0.3"/>')


def _add_ground_and_obstacles(xml):
    """Ground plane, RGB axis markers, and terrain obstacles."""
    # Ground
    xml.append('    <geom name="floor" type="plane" size="20 20 0.1"'
               ' material="grid_mat" contype="1" conaffinity="2"/>')
    # RGB axis markers (visual only)
    for axis, color in [("0.5 0 0.002", "1 0 0 0.8"),
                        ("0 0.5 0.002", "0 1 0 0.8"),
                        ("0 0 0.5",     "0 0 1 0.8")]:
        xml.append(f'    <geom type="capsule" fromto="0 0 0.002 {axis}"'
                   f' size="0.008" rgba="{color}" contype="0" conaffinity="0"/>')
    # Ramp: gentle 8-degree slope, 3 cm rise
    xml.append('    <geom name="ramp" type="box" size="1.0 1.5 0.03"'
               ' pos="4 0 0.03" euler="0 -8 0" rgba="0.6 0.4 0.3 1"'
               ' contype="1" conaffinity="2"/>')
    # Bump: two wedges forming a ridge, 5 cm tall
    xml.append('    <geom name="bump_up" type="box" size="0.5 1.5 0.05"'
               ' pos="6.7 0 0.05" euler="0 -15 0" rgba="0.5 0.35 0.3 1"'
               ' contype="1" conaffinity="2"/>')
    xml.append('    <geom name="bump_dn" type="box" size="0.5 1.5 0.05"'
               ' pos="7.7 0 0.05" euler="0 15 0" rgba="0.5 0.35 0.3 1"'
               ' contype="1" conaffinity="2"/>')
    # Step: ramp up + flat top + ramp down, 8 cm tall
    xml.append('    <geom name="step_up" type="box" size="0.6 1.5 0.08"'
               ' pos="9.5 0 0.08" euler="0 -12 0" rgba="0.55 0.4 0.35 1"'
               ' contype="1" conaffinity="2"/>')
    xml.append('    <geom name="step_top" type="box" size="1.0 1.5 0.08"'
               ' pos="10.7 0 0.08" rgba="0.55 0.4 0.35 1"'
               ' contype="1" conaffinity="2"/>')
    xml.append('    <geom name="step_dn" type="box" size="0.6 1.5 0.08"'
               ' pos="12.3 0 0.08" euler="0 12 0" rgba="0.55 0.4 0.35 1"'
               ' contype="1" conaffinity="2"/>')


def _add_hull_with_sprockets(xml, config):
    """Hull body with tracking cameras and all sprocket wheels."""
    c = config
    xml.append(f'    <body name="hull" pos="0 0 {c.hull_z}">')
    xml.append(f'      <freejoint name="hull_jnt"/>')
    # Tracking cameras attached to hull
    xml.append(f'      <camera name="tracking" pos="-3 -3 2.5"'
               f' xyaxes="1 -1 0 0.3 0.3 1" fovy="45" mode="track"/>')
    xml.append(f'      <camera name="front" pos="8 -3.5 2"'
               f' xyaxes="0.5 1 0 -0.2 0.1 1" fovy="45" mode="track"/>')
    # Hull box (visual only — no collision)
    xml.append(f'      <geom name="hull_box" type="box"'
               f' size="{c.hull_half_x} {c.hull_half_y} {c.hull_half_z}"'
               f' rgba="0.3 0.35 0.3 1" mass="{c.hull_mass}"'
               f' contype="0" conaffinity="0"/>')

    # Sprocket wheels — children of the hull, one set per side
    dz = c.sprocket_z - c.hull_z  # Z offset from hull center to sprocket center
    for side, y_sign in SIDES:
        y_offset = y_sign * c.track_gauge / 2
        for spr in c.sprockets:
            full_name = f"{side}_{spr.name}"
            xml.append(f'      <body name="{full_name}_spr"'
                       f' pos="{spr.x_offset} {y_offset} {dz}">')
            if spr.has_tensioner:
                xml.append(f'        <joint name="{full_name}_slide"'
                           f' type="slide" axis="1 0 0"'
                           f' stiffness="{c.tension_stiffness}"'
                           f' damping="80" range="-0.05 0.3"/>')
            xml.append(f'        <joint name="{full_name}_hinge"'
                       f' type="hinge" axis="0 1 0"'
                       f' damping="{c.sprocket_hinge_damping}"/>')
            # Hub cylinder — collision enabled so links ride on it
            xml.append(f'        <geom type="cylinder"'
                       f' size="{c.hub_radius} {c.hub_half_width}"'
                       f' euler="90 0 0" rgba="{spr.color}"'
                       f' contype="1" conaffinity="2"/>')
            xml.append(f'      </body>')

    xml.append('    </body>')  # end hull


def _add_chain_links(xml, config):
    """Two chain tracks (left/right), each a kinematic tree of box links.

    Link 0 has a freejoint (6-DOF root). Links 1..N-1 are nested children,
    each connected by a Y-axis hinge. This forms a single kinematic tree
    per side that MuJoCo can solve efficiently.
    """
    c = config
    half_len = c.link_pitch / 2 - 0.005
    lo, hi = c.hinge_range

    for side, y_sign in SIDES:
        y_offset = y_sign * c.track_gauge / 2

        # Link 0: positioned on the track path, with a freejoint
        x0, z0, angle0 = stadium_point(0, c)
        world_z = c.sprocket_z + z0
        # Quaternion for Y-axis rotation by -angle0
        qw = math.cos(-angle0 / 2)
        qy = math.sin(-angle0 / 2)

        xml.append(f'    <body name="{side}_link_0"'
                   f' pos="{x0:.6f} {y_offset} {world_z:.6f}"'
                   f' quat="{qw:.6f} 0 {qy:.6f} 0">')
        xml.append(f'      <freejoint name="{side}_link_0_jnt"/>')
        xml.append(f'      <geom type="box"'
                   f' size="{half_len:.4f} {c.link_width} {c.link_thickness}"'
                   f' rgba="1 0.2 0.2 1" mass="0.05"'
                   f' contype="2" conaffinity="1"/>')

        # Links 1..N-1: each nested inside the previous one
        for i in range(1, c.n_links):
            xml.append(f'      <body name="{side}_link_{i}"'
                       f' pos="{c.link_pitch:.6f} 0 0">')
            xml.append(f'        <joint name="{side}_hinge_{i}"'
                       f' type="hinge" axis="0 1 0"'
                       f' pos="{-c.link_pitch / 2:.6f} 0 0"'
                       f' damping="{c.hinge_damping}"'
                       f' limited="true" range="{lo} {hi}"/>')
            xml.append(f'        <geom type="box"'
                       f' size="{half_len:.4f} {c.link_width} {c.link_thickness}"'
                       f' rgba="0.9 0.6 0.1 1" mass="0.05"'
                       f' contype="2" conaffinity="1"/>')

        # Close all nested <body> tags
        for _ in range(c.n_links):
            xml.append('    </body>')


def _add_actuators(xml, config):
    """Velocity motors on every sprocket hinge."""
    xml.append('  <actuator>')
    for side, _ in SIDES:
        for spr in config.sprockets:
            name = f"{side}_{spr.name}"
            xml.append(f'    <velocity name="{name}_motor"'
                       f' joint="{name}_hinge"'
                       f' kv="500" ctrllimited="true" ctrlrange="-5 5"/>')
    xml.append('  </actuator>')


def _add_equality_constraints(xml, config):
    """Loop closure + sprocket engagement constraints.

    Loop closure: connects the last link back to the first, forming a loop.
    Engagement:   one constraint per (link, engaging_sprocket) pair, initially
                  inactive. The EngagementManager activates/deactivates these
                  at runtime to simulate chain-sprocket meshing.
    """
    xml.append('  <equality>')
    for side, _ in SIDES:
        # Loop closure — last link connects back to first link
        xml.append(f'    <connect name="{side}_loop"'
                   f' body1="{side}_link_{config.n_links - 1}"'
                   f' body2="{side}_link_0"'
                   f' anchor="{config.link_pitch / 2:.6f} 0 0"/>')
        # Engagement constraints (one per link per engaging sprocket)
        for i in range(config.n_links):
            for spr in config.engaging_sprockets:
                name = f"{side}_{spr.name}"
                xml.append(f'    <connect name="{side}_eng_{i}_{spr.name}"'
                           f' body1="{name}_spr"'
                           f' body2="{side}_link_{i}"'
                           f' anchor="0 0 0" active="false"/>')
    xml.append('  </equality>')
