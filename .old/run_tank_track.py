"""
Visualize N slider-crank mechanisms forming a tank track in MuJoCo.

Each mechanism is driven analytically with a phase offset of 2*pi/N.
Tread plates are positioned between consecutive tracer points.

Usage:
    uv run python generate_track_xml.py          # generate XML first
    uv run python run_tank_track.py               # interactive viewer
    uv run python run_tank_track.py --n 30        # custom N (must match generated XML)
"""

import argparse
import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time

from run_slider_crank import solve_slider_crank, gx, gy

N_DEFAULT = 20


def euler_to_quat(yaw):
    """Convert a Z-axis rotation (yaw) to a quaternion [w, x, y, z]."""
    return np.array([
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ])


def run_interactive(n=N_DEFAULT):
    xml_path = Path(__file__).parent / "tank_track.xml"
    if not xml_path.exists():
        print(f"XML not found at {xml_path}. Run generate_track_xml.py first.")
        return

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Disable all equality constraints (analytical driving)
    for i in range(model.neq):
        model.eq_active0[i] = 0

    crank_speed = 1.5  # rad/s
    phase_step = 2 * math.pi / n

    # qpos layout: [mech_0 (3), mech_1 (3), ..., mech_{n-1} (3), tread_0 (7), tread_1 (7), ..., tread_{n-1} (7)]
    mech_qpos_start = 0
    tread_qpos_start = n * 3

    # Initialize
    theta = 0.0
    tracer_points = np.zeros((n, 2))
    for i in range(n):
        sol = solve_slider_crank(theta + i * phase_step)
        if sol is not None:
            data.qpos[mech_qpos_start + i * 3: mech_qpos_start + i * 3 + 3] = sol["qpos"]
            tracer_points[i] = sol["P"]

    # Set initial tread positions
    for i in range(n):
        j = (i + 1) % n
        mid_x = (tracer_points[i, 0] + tracer_points[j, 0]) / 2
        mid_y = (tracer_points[i, 1] + tracer_points[j, 1]) / 2
        dx = tracer_points[j, 0] - tracer_points[i, 0]
        dy = tracer_points[j, 1] - tracer_points[i, 1]
        yaw = math.atan2(dy, dx)
        quat = euler_to_quat(yaw)

        tread_offset = tread_qpos_start + i * 7
        data.qpos[tread_offset: tread_offset + 3] = [mid_x, mid_y, 0.0]
        data.qpos[tread_offset + 3: tread_offset + 7] = quat

    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Top-down XY camera
        viewer.cam.lookat[:] = [gx, gy, 0.0]
        viewer.cam.distance = 20.0
        viewer.cam.elevation = -90.0
        viewer.cam.azimuth = 90.0

        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            theta = crank_speed * t

            # Solve all N mechanisms
            for i in range(n):
                sol = solve_slider_crank(theta + i * phase_step)
                if sol is not None:
                    data.qpos[mech_qpos_start + i * 3: mech_qpos_start + i * 3 + 3] = sol["qpos"]
                    tracer_points[i] = sol["P"]

            # Position tread plates between consecutive tracer points
            for i in range(n):
                j = (i + 1) % n
                mid_x = (tracer_points[i, 0] + tracer_points[j, 0]) / 2
                mid_y = (tracer_points[i, 1] + tracer_points[j, 1]) / 2
                dx = tracer_points[j, 0] - tracer_points[i, 0]
                dy = tracer_points[j, 1] - tracer_points[i, 1]
                yaw = math.atan2(dy, dx)
                quat = euler_to_quat(yaw)

                tread_offset = tread_qpos_start + i * 7
                data.qpos[tread_offset: tread_offset + 3] = [mid_x, mid_y, 0.0]
                data.qpos[tread_offset + 3: tread_offset + 7] = quat

            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tank track viewer")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Number of mechanisms (must match generated XML)")
    args = parser.parse_args()
    run_interactive(args.n)
