"""
Simulate and visualize a slider-crank linkage in MuJoCo.

The coupler point on the connecting rod traces a stadium curve.
Joint angles are computed analytically and set via qpos (kinematic approach)
to avoid issues with MuJoCo's soft equality constraints.

Usage:
    uv run python run_slider_crank.py              # interactive viewer
    uv run python run_slider_crank.py --headless    # headless: record trace and plot
"""

import argparse
import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time


XML_PATH = Path(__file__).parent / "slider_crank_mujoco.xml"

# Optimized slider-crank parameters (approach B)
a = 2.91648           # crank length O2->A
b = 4.615711          # connecting rod length A->B
px = 3.181522         # coupler point offset x in rod frame
py = 0.004651         # coupler point offset y in rod frame
slider_angle = -3.137783  # slider direction angle (approx -pi)
offset = -0.073295    # perpendicular offset of slider rail from O2
gx = 3.118361         # crank pivot x
gy = -0.034922        # crank pivot y

# Precompute slider direction trig
sa = math.sin(slider_angle)
ca = math.cos(slider_angle)

# Rail base point (O2 + offset * perpendicular_to_slider_direction)
# Perpendicular direction: (-sin(slider_angle), cos(slider_angle))
rail_base_x = gx + offset * (-sa)
rail_base_y = gy + offset * ca


def solve_slider_crank(theta):
    """
    Given crank angle theta, solve the slider-crank kinematics.
    Returns dict with A, B, P positions and qpos, or None if no solution.
    """
    # Crank tip
    Ax = gx + a * math.cos(theta)
    Ay = gy + a * math.sin(theta)

    # Project A onto slider line to find foot point and perpendicular distance
    # d_perp = (A - O2) . perp_hat - offset
    # where perp_hat = (-sin(slider_angle), cos(slider_angle))
    d_perp = (Ax - gx) * (-sa) + (Ay - gy) * ca - offset

    # Distance along slider direction from foot to B
    val = b * b - d_perp * d_perp
    if val < 0:
        return None
    d_along = math.sqrt(val)

    # Foot point (projection of A onto slider line)
    foot_x = Ax + d_perp * sa
    foot_y = Ay - d_perp * ca

    # Slider point B (take the + branch for d_along)
    Bx = foot_x + d_along * ca
    By = foot_y + d_along * sa

    # Connecting rod angle in world frame
    theta3 = math.atan2(By - Ay, Bx - Ax)

    # Coupler tracing point P
    cos3 = math.cos(theta3)
    sin3 = math.sin(theta3)
    Px = Ax + px * cos3 - py * sin3
    Py = Ay + px * sin3 + py * cos3

    # Joint angles for MuJoCo:
    # crank_joint: absolute angle of crank from its initial orientation
    q_crank = theta
    # rod_joint: angle of rod relative to crank
    q_rod = theta3 - theta
    # slider_joint: displacement of slider body along slide axis from rail_base
    # Slider displacement = signed distance of B from rail_base along slider direction
    q_slider = (Bx - rail_base_x) * ca + (By - rail_base_y) * sa

    return {
        "A": np.array([Ax, Ay]),
        "B": np.array([Bx, By]),
        "P": np.array([Px, Py]),
        "qpos": np.array([q_crank, q_rod, q_slider]),
    }


def run_interactive():
    """Launch the interactive MuJoCo viewer with kinematic animation."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Disable equality constraints since we drive kinematics analytically
    for i in range(model.neq):
        model.eq_active0[i] = 0

    # Initial config
    theta = 0.0
    sol = solve_slider_crank(theta)
    if sol is not None:
        data.qpos[:] = sol["qpos"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    crank_speed = 1.5  # rad/s

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Camera looking at XY plane from above
        viewer.cam.lookat[:] = [gx, gy, 0.0]
        viewer.cam.distance = 20.0
        viewer.cam.elevation = -90.0
        viewer.cam.azimuth = 90.0

        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            theta = crank_speed * t

            sol = solve_slider_crank(theta)
            if sol is not None:
                data.qpos[:] = sol["qpos"]
                data.qvel[:] = 0

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.002)


def run_headless(n_revolutions=2, n_points=500):
    """Run headless, record tracer path, and plot against target stadium."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Disable equality constraints
    for i in range(model.neq):
        model.eq_active0[i] = 0

    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "tracer_pos"
    )
    sensor_adr = model.sensor_adr[sensor_id]

    theta_start = 0.0
    d_theta = 2 * math.pi * n_revolutions / n_points

    trace = []
    trace_analytical = []

    for i in range(n_points):
        theta = theta_start + i * d_theta
        sol = solve_slider_crank(theta)
        if sol is not None:
            data.qpos[:] = sol["qpos"]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            pos = data.sensordata[sensor_adr: sensor_adr + 3].copy()
            trace.append(pos[:2])
            trace_analytical.append(sol["P"])

    trace = np.array(trace)
    trace_analytical = np.array(trace_analytical)

    print(f"Recorded {len(trace)} points over {n_revolutions} revolutions")
    print(f"  Trace X range: [{trace[:, 0].min():.3f}, {trace[:, 0].max():.3f}]")
    print(f"  Trace Y range: [{trace[:, 1].min():.3f}, {trace[:, 1].max():.3f}]")

    # Verify MuJoCo positions match analytical
    err = np.linalg.norm(trace - trace_analytical, axis=1)
    print(f"  Max MuJoCo vs analytical error: {err.max():.2e}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stadium import TARGET

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(TARGET[:, 0], TARGET[:, 1], "k--", lw=2, label="Target stadium")
    # Plot only one revolution for clarity
    n_one_rev = n_points // n_revolutions
    ax.plot(
        trace[:n_one_rev, 0], trace[:n_one_rev, 1],
        "r-", lw=1.2, alpha=0.8, label="Coupler trace (1 rev)"
    )
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Slider-Crank: Coupler Trace vs Target Stadium")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(__file__).parent / "slider_crank_trace.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slider-crank linkage viewer"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run headless and plot tracer path"
    )
    args = parser.parse_args()

    if args.headless:
        run_headless()
    else:
        run_interactive()
