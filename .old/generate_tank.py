#!/usr/bin/env python3
"""Generate tank.xml MuJoCo model with ghost-driven track links."""

import numpy as np
from stadium import stadium_parametric, stadium_perimeter, angle_to_quat_y

# === Parameters ===

# Chassis (compact, squarish aspect ratio)
CHASSIS_HALF_SIZE = [0.3, 0.2, 0.08]
CHASSIS_MASS = 20.0
CHASSIS_Z = 0.16

# Track geometry
HALF_LENGTH = 0.28
RADIUS = 0.08
Y_OFFSET = 0.22
Z_OFFSET = 0.0

# Track links (5 per side, 10 total)
N_LINKS = 5
LINK_HALF_SIZE = [0.10, 0.04, 0.015]
LINK_MASS = 0.2
LINK_FRICTION = "1.5 0.005 0.001"

# Weld constraint tuning
WELD_SOLREF = "0.002 1.0"
WELD_SOLIMP = "0.95 0.99 0.001"

# Simulation
TIMESTEP = 0.002


def fmt(v, prec=6):
    """Format a float, stripping trailing zeros."""
    return f"{v:.{prec}f}".rstrip("0").rstrip(".")


def fmt_pos(pos):
    return " ".join(fmt(p) for p in pos)


def fmt_quat(q):
    return " ".join(fmt(c) for c in q)


def generate():
    perimeter = stadium_perimeter(HALF_LENGTH, RADIUS)
    spacing = perimeter / N_LINKS

    lines = []
    w = lines.append  # shorthand

    w('<mujoco model="tank">')
    w(f'  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" solver="Newton" iterations="100"/>')
    w("")
    w("  <default>")
    w('    <geom condim="3"/>')
    w('    <joint damping="0.01"/>')
    w("  </default>")
    w("")
    w("  <asset>")
    w('    <texture type="2d" name="grid" builtin="checker" width="512" height="512"')
    w('             rgb1="0.2 0.2 0.2" rgb2="0.3 0.3 0.3"/>')
    w('    <material name="ground_mat" texture="grid" texrepeat="10 10"/>')
    w("  </asset>")
    w("")
    w("  <worldbody>")
    w("    <!-- Ground for track links (with friction) -->")
    w('    <geom type="plane" size="50 50 1" material="ground_mat"')
    w('          friction="1.0 0.005 0.001" condim="3" contype="1" conaffinity="2"/>')
    w("    <!-- Ground for chassis (frictionless, condim=1 on both sides) -->")
    w('    <geom type="plane" size="50 50 1"')
    w('          condim="1" contype="8" conaffinity="4" rgba="0 0 0 0"/>')
    w("")
    w("    <!-- Chassis -->")
    w(f'    <body name="chassis" pos="0 0 {fmt(CHASSIS_Z)}">')
    w('      <freejoint name="chassis_free"/>')
    w(f'      <geom type="box" size="{fmt_pos(CHASSIS_HALF_SIZE)}" mass="{fmt(CHASSIS_MASS)}"')
    w('            condim="1" contype="4" conaffinity="8" rgba="0.3 0.35 0.3 1"/>')
    w("      <!-- Sensor site -->")
    w('      <site name="chassis_center" pos="0 0 0" size="0.01"/>')
    w("      <!-- Visual-only sprocket and idler cylinders -->")
    w(f'      <geom name="sprocket_L" type="cylinder" size="{fmt(RADIUS)} 0.03"')
    w(f'            pos="{fmt(HALF_LENGTH)} {fmt(Y_OFFSET)} 0" euler="90 0 0"')
    w('            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>')
    w(f'      <geom name="sprocket_R" type="cylinder" size="{fmt(RADIUS)} 0.03"')
    w(f'            pos="{fmt(HALF_LENGTH)} {fmt(-Y_OFFSET)} 0" euler="90 0 0"')
    w('            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>')
    w(f'      <geom name="idler_L" type="cylinder" size="{fmt(RADIUS * 0.8)} 0.03"')
    w(f'            pos="{fmt(-HALF_LENGTH)} {fmt(Y_OFFSET)} 0" euler="90 0 0"')
    w('            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>')
    w(f'      <geom name="idler_R" type="cylinder" size="{fmt(RADIUS * 0.8)} 0.03"')
    w(f'            pos="{fmt(-HALF_LENGTH)} {fmt(-Y_OFFSET)} 0" euler="90 0 0"')
    w('            contype="0" conaffinity="0" rgba="0.4 0.4 0.4 1"/>')
    w("    </body>")
    w("")

    # Generate ghost mocap bodies and track links
    link_size_str = fmt_pos(LINK_HALF_SIZE)
    ghost_names = []
    link_names = []

    for side in ["L", "R"]:
        y_sign = 1.0 if side == "L" else -1.0
        w(f"    <!-- {'Left' if side == 'L' else 'Right'} track -->")

        for i in range(N_LINKS):
            s = i * spacing
            local_x, local_z, angle = stadium_parametric(s, HALF_LENGTH, RADIUS)

            # World position (chassis starts at origin, height CHASSIS_Z)
            px = local_x
            py = y_sign * Y_OFFSET
            pz = local_z + Z_OFFSET + CHASSIS_Z

            quat = angle_to_quat_y(angle)

            ghost_name = f"ghost_{side}_{i}"
            link_name = f"link_{side}_{i}"
            ghost_names.append(ghost_name)
            link_names.append(link_name)

            # Ghost mocap body
            w(f'    <body mocap="true" name="{ghost_name}" pos="{fmt_pos([px, py, pz])}" quat="{fmt_quat(quat)}">')
            w(f'      <geom type="box" size="{link_size_str}"')
            w('            contype="0" conaffinity="0" rgba="0 1 0 0.15"/>')
            w("    </body>")

            # Real track link
            w(f'    <body name="{link_name}" pos="{fmt_pos([px, py, pz])}" quat="{fmt_quat(quat)}">')
            w(f'      <freejoint name="{link_name}_free"/>')
            w(f'      <geom type="box" size="{link_size_str}" mass="{fmt(LINK_MASS)}"')
            w(f'            friction="{LINK_FRICTION}" contype="2" conaffinity="1" rgba="0.25 0.25 0.25 1"/>')
            w("    </body>")

        w("")

    # Terrain obstacles — ramps (tilted boxes) so treads can climb them
    # Two geoms per obstacle: one for links (friction), one for chassis (frictionless)
    w("    <!-- Ramp obstacle -->")
    w('    <body pos="2 0 0.02" euler="0 15 0">')
    w('      <geom type="box" size="0.25 1 0.015" friction="1.5 0.005 0.001"')
    w('            condim="3" contype="1" conaffinity="2" rgba="0.5 0.4 0.3 1"/>')
    w('      <geom type="box" size="0.25 1 0.015"')
    w('            condim="1" contype="8" conaffinity="4" rgba="0 0 0 0"/>')
    w("    </body>")
    w("    <!-- Small bump -->")
    w('    <body pos="4 0 0.01">')
    w('      <geom type="box" size="0.15 1 0.01" friction="1.5 0.005 0.001"')
    w('            condim="3" contype="1" conaffinity="2" rgba="0.5 0.4 0.3 1"/>')
    w('      <geom type="box" size="0.15 1 0.01"')
    w('            condim="1" contype="8" conaffinity="4" rgba="0 0 0 0"/>')
    w("    </body>")

    w("  </worldbody>")
    w("")

    # Weld constraints
    w("  <equality>")
    for ghost_name, link_name in zip(ghost_names, link_names):
        w(f'    <weld body1="{ghost_name}" body2="{link_name}"')
        w(f'          solref="{WELD_SOLREF}" solimp="{WELD_SOLIMP}"/>')
    w("  </equality>")
    w("")

    # Contact exclusions (chassis <-> links)
    w("  <contact>")
    for link_name in link_names:
        w(f'    <exclude body1="chassis" body2="{link_name}"/>')
    w("  </contact>")
    w("")

    # Sensors
    w("  <sensor>")
    w('    <force name="chassis_force" site="chassis_center"/>')
    w('    <torque name="chassis_torque" site="chassis_center"/>')
    w("  </sensor>")
    w("")
    w("</mujoco>")

    xml = "\n".join(lines)
    with open("tank.xml", "w") as f:
        f.write(xml)
    print(f"Generated tank.xml ({len(lines)} lines, {2 * N_LINKS} links, {2 * N_LINKS} ghosts)")


if __name__ == "__main__":
    generate()
