#!/usr/bin/env python3
"""Generate a MuJoCo XML model for a tank track driven by N slider-crank mechanisms."""

import argparse
import math
import textwrap

# Slider-crank parameters (same for all N instances)
PARAMS = dict(
    a=2.91648,
    b=4.615711,
    px=3.181522,
    py=0.004651,
    slider_angle=-3.137783,
    offset=-0.073295,
    gx=3.118361,
    gy=-0.034922,
)


def generate_xml(N: int = 20, show_mechanism: bool = False) -> str:
    """Return MuJoCo XML string for N slider-crank mechanisms sharing one crankshaft."""

    bodies = []
    # Each mechanism instance i gets:
    #   - crank arm body (freejoint)  [optional visibility]
    #   - connecting rod body (freejoint)  [optional visibility]
    #   - tracer point body (freejoint)
    # Each tread plate between consecutive tracers (freejoint)

    vis = "1" if show_mechanism else "0"

    for i in range(N):
        tag = f"_{i:02d}"

        # Crank arm body
        bodies.append(f"""\
    <body name="crank{tag}" pos="0 0 0">
      <freejoint name="crank_jnt{tag}"/>
      <geom name="crank_geom{tag}" type="capsule" size="0.03" fromto="0 0 0 1 0 0"
            rgba="0.85 0.2 0.2 0.8" contype="0" conaffinity="0"
            group="{2 if not show_mechanism else 0}"/>
      <site name="crank_site{tag}" size="0.04" rgba="0.85 0.2 0.2 1"/>
    </body>""")

        # Connecting rod body
        bodies.append(f"""\
    <body name="rod{tag}" pos="0 0 0">
      <freejoint name="rod_jnt{tag}"/>
      <geom name="rod_geom{tag}" type="capsule" size="0.025" fromto="0 0 0 1 0 0"
            rgba="0.2 0.4 0.85 0.8" contype="0" conaffinity="0"
            group="{2 if not show_mechanism else 0}"/>
    </body>""")

        # Tracer point body (always visible)
        bodies.append(f"""\
    <body name="tracer{tag}" pos="0 0 0">
      <freejoint name="tracer_jnt{tag}"/>
      <geom name="tracer_geom{tag}" type="sphere" size="0.06"
            rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0"/>
    </body>""")

    # Tread plates between consecutive tracers
    for i in range(N):
        tag = f"_{i:02d}"
        bodies.append(f"""\
    <body name="tread{tag}" pos="0 0 0">
      <freejoint name="tread_jnt{tag}"/>
      <geom name="tread_geom{tag}" type="box" size="0.3 0.04 0.15"
            rgba="0.35 0.3 0.25 1" contype="0" conaffinity="0"/>
    </body>""")

    bodies_str = "\n".join(bodies)

    xml = textwrap.dedent(f"""\
    <mujoco model="tank_track">
      <option gravity="0 0 0" integrator="Euler" timestep="0.002"/>

      <visual>
        <headlight diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
        <global offwidth="1280" offheight="720"/>
      </visual>

      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.85 0.85 0.85"
                 rgb2="0.7 0.7 0.7" width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10" reflectance="0.1"/>
      </asset>

      <worldbody>
        <!-- Ground plane -->
        <geom type="plane" size="15 15 0.1" material="grid_mat" pos="0 0 -2"
              contype="0" conaffinity="0"/>

        <!-- Crank pivot marker at O2 -->
        <site name="pivot_O2" pos="{PARAMS['gx']} 0 {PARAMS['gy']}" size="0.1"
              rgba="0.1 0.7 0.2 1" type="sphere"/>

        <!-- Slider rail visualization -->
        <geom type="capsule" size="0.01"
              fromto="{PARAMS['gx'] - 8} 0 {PARAMS['gy'] + PARAMS['offset']} {PARAMS['gx'] + 8} 0 {PARAMS['gy'] + PARAMS['offset']}"
              rgba="0.5 0.5 0.5 0.2" contype="0" conaffinity="0"
              group="2"/>

    {bodies_str}
      </worldbody>
    </mujoco>
    """)
    return xml


def main():
    parser = argparse.ArgumentParser(description="Generate tank track MuJoCo XML")
    parser.add_argument("-n", "--num-treads", type=int, default=20, help="Number of tread mechanisms (default: 20)")
    parser.add_argument("--show-mechanism", action="store_true", help="Show crank arms and connecting rods")
    parser.add_argument("-o", "--output", default="tank_track_mujoco.xml", help="Output XML file")
    args = parser.parse_args()

    xml = generate_xml(N=args.num_treads, show_mechanism=args.show_mechanism)
    with open(args.output, "w") as f:
        f.write(xml)
    print(f"Generated {args.output} with N={args.num_treads} mechanisms")


if __name__ == "__main__":
    main()
