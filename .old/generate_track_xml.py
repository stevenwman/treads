"""
Generate a MuJoCo XML with N slider-crank instances for a tank track.

Each instance is a copy of the single slider-crank mechanism with unique names
and a phase offset applied at runtime. All cranks share the same O2 pivot.

Also generates N tread plate bodies (freejoint) to be positioned at runtime.

Usage:
    uv run python generate_track_xml.py          # default N=20
    uv run python generate_track_xml.py --n 30   # custom N
"""

import argparse
from pathlib import Path

# Same parameters as the single slider-crank
a = 2.91648
b = 4.615711
px = 3.181522
py = 0.004651
slider_angle = -3.137783
offset = -0.073295
gx = 3.118361
gy = -0.034922

import math
sa = math.sin(slider_angle)
ca = math.cos(slider_angle)
rail_base_x = gx + offset * (-sa)
rail_base_y = gy + offset * ca


def generate_xml(n=20):
    lines = []
    lines.append(f'<mujoco model="tank_track_N{n}">')
    lines.append('  <option timestep="0.002" gravity="0 0 0" integrator="implicit" solver="Newton" iterations="200">')
    lines.append('    <flag contact="disable"/>')
    lines.append('  </option>')
    lines.append('')
    lines.append('  <compiler angle="radian" autolimits="true"/>')
    lines.append('')
    lines.append('  <default>')
    lines.append('    <geom contype="0" conaffinity="0" density="1000"/>')
    lines.append('    <joint type="hinge" axis="0 0 1" damping="1" armature="0.1"/>')
    lines.append('  </default>')
    lines.append('')
    lines.append('  <worldbody>')

    # Ground pivot marker
    lines.append(f'    <geom type="sphere" size="0.15" pos="{gx} {gy} 0" rgba="0.15 0.15 0.15 1" name="O2_mark"/>')
    lines.append('')

    # Slider rail visual (shared)
    lines.append(f'    <geom type="box" size="6 0.03 0.02"')
    lines.append(f'          pos="{rail_base_x} {rail_base_y} -0.1"')
    lines.append(f'          rgba="0.5 0.5 0.5 0.3" name="slider_rail"/>')
    lines.append('')

    # N slider-crank instances
    for i in range(n):
        mech_alpha = "0.3"
        lines.append(f'    <!-- Mechanism {i} -->')
        lines.append(f'    <body name="crank_body_{i}" pos="{gx} {gy} 0">')
        lines.append(f'      <joint name="crank_joint_{i}"/>')
        lines.append(f'      <geom type="capsule" fromto="0 0 0 {a} 0 0" size="0.05" rgba="0.9 0.25 0.25 {mech_alpha}" name="crank_geom_{i}"/>')
        lines.append(f'      <body name="rod_body_{i}" pos="{a} 0 0">')
        lines.append(f'        <joint name="rod_joint_{i}"/>')
        lines.append(f'        <geom type="capsule" fromto="0 0 0 {b} 0 0" size="0.05" rgba="0.25 0.25 0.9 {mech_alpha}" name="rod_geom_{i}"/>')
        lines.append(f'        <site name="tracer_P_{i}" pos="{px} {py} 0" size="0.12" rgba="1 0 0 1" type="sphere"/>')
        lines.append(f'        <body name="rod_tip_{i}" pos="{b} 0 0"/>')
        lines.append(f'      </body>')
        lines.append(f'    </body>')
        lines.append('')
        lines.append(f'    <body name="slider_body_{i}" pos="{rail_base_x} {rail_base_y} 0">')
        lines.append(f'      <joint name="slider_joint_{i}" type="slide" axis="-0.99999271 -0.00381683 0" damping="1" armature="0.1"/>')
        lines.append(f'      <geom type="box" size="0.12 0.08 0.05" rgba="0.9 0.6 0.15 {mech_alpha}" name="slider_geom_{i}"/>')
        lines.append(f'    </body>')
        lines.append('')

    # N tread plate bodies (freejoint)
    for i in range(n):
        lines.append(f'    <!-- Tread plate {i} -->')
        lines.append(f'    <body name="tread_{i}" pos="0 0 0">')
        lines.append(f'      <freejoint name="tread_joint_{i}"/>')
        # Tread plate: flat box, dark gray
        # Width ~ half the arc gap between consecutive tracer points
        lines.append(f'      <geom type="box" size="0.5 0.15 0.04" rgba="0.25 0.25 0.25 1" name="tread_geom_{i}"/>')
        lines.append(f'    </body>')
        lines.append('')

    lines.append('  </worldbody>')
    lines.append('')

    # Equality constraints (one per mechanism)
    lines.append('  <equality>')
    for i in range(n):
        lines.append(f'    <connect name="close_loop_{i}" body1="rod_tip_{i}" body2="slider_body_{i}" anchor="0 0 0"/>')
    lines.append('  </equality>')
    lines.append('')

    # Sensors for tracer positions
    lines.append('  <sensor>')
    for i in range(n):
        lines.append(f'    <framepos name="tracer_pos_{i}" objtype="site" objname="tracer_P_{i}"/>')
    lines.append('  </sensor>')
    lines.append('')

    # Keyframe: all zeros
    total_dof = n * 3 + n * 7  # 3 per mechanism + 7 per freejoint tread
    qpos_str = " ".join(["0"] * total_dof)
    lines.append('  <keyframe>')
    lines.append(f'    <key name="home" qpos="{qpos_str}"/>')
    lines.append('  </keyframe>')

    lines.append('</mujoco>')

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tank track MuJoCo XML")
    parser.add_argument("--n", type=int, default=20, help="Number of slider-crank instances")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    xml_str = generate_xml(args.n)

    out_path = Path(args.output) if args.output else Path(__file__).parent / "tank_track.xml"
    out_path.write_text(xml_str)
    print(f"Generated {out_path} with N={args.n} mechanisms + {args.n} tread plates")
    print(f"  Total qpos: {args.n * 10} ({args.n}*3 mechanism + {args.n}*7 tread)")
