"""
Simulate and visualize a Stephenson-III six-bar linkage in MuJoCo.

Since this mechanism has extreme link ratios that make MuJoCo's soft
equality constraints numerically challenging, we use a kinematic approach:
compute joint angles analytically at each crank position and set them
directly via qpos, using mj_forward for rendering.

Usage:
    uv run python run_linkage.py              # interactive viewer
    uv run python run_linkage.py --headless   # headless: record trace and plot
"""

import argparse
import math
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time


XML_PATH = Path(__file__).parent / "linkage_mujoco.xml"

# Linkage parameters
A_LEN = 4.0499   # crank O2->A
B_LEN = 4.8983   # coupler A->B
C_LEN = 2.0489   # rocker O4->B
E_LEN = 3.0571   # dyad link B->C
F_LEN = 2.9938   # dyad link O6->C
PX, PY = 1.0946, 2.4234  # tracer offset in link_e frame

O2 = np.array([0.5584, 1.4180])
O4 = np.array([1.7381, 1.4180])
O6 = np.array([-0.0037, 5.0])


def circle_intersect(c1, r1, c2, r2, branch=0):
    """Find intersection of two circles. branch=0 or 1 for the two solutions."""
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    D = math.sqrt(dx * dx + dy * dy)
    if D > r1 + r2 + 1e-10 or D < abs(r1 - r2) - 1e-10:
        return None
    a = (r1 * r1 - r2 * r2 + D * D) / (2 * D)
    h_sq = r1 * r1 - a * a
    if h_sq < 0:
        h_sq = 0
    h = math.sqrt(h_sq)
    mx = c1[0] + a * dx / D
    my = c1[1] + a * dy / D
    if branch == 0:
        return np.array([mx + h * dy / D, my - h * dx / D])
    else:
        return np.array([mx - h * dy / D, my + h * dx / D])


def solve_linkage(theta_crank, prev_B=None, prev_C=None):
    """
    Given crank angle, solve for all joint positions.
    Returns dict with A, B, C, P, qpos or None if no solution.
    Uses prev_B, prev_C to maintain branch consistency across steps.
    Default branches (B=0, C=1) produce the stadium-matching coupler curve.
    """
    A = O2 + A_LEN * np.array([math.cos(theta_crank), math.sin(theta_crank)])

    # Solve four-bar: B at intersection of circle(A, B_LEN) and circle(O4, C_LEN)
    B0 = circle_intersect(A, B_LEN, O4, C_LEN, branch=0)
    B1 = circle_intersect(A, B_LEN, O4, C_LEN, branch=1)

    if B0 is None and B1 is None:
        return None

    # Pick branch closest to previous B, default to branch 0
    if prev_B is not None and B0 is not None and B1 is not None:
        B = B0 if np.linalg.norm(B0 - prev_B) < np.linalg.norm(B1 - prev_B) else B1
    elif B0 is not None:
        B = B0
    else:
        B = B1

    # Solve second dyad: C at intersection of circle(B, E_LEN) and circle(O6, F_LEN)
    C0 = circle_intersect(B, E_LEN, O6, F_LEN, branch=0)
    C1 = circle_intersect(B, E_LEN, O6, F_LEN, branch=1)

    if C0 is None and C1 is None:
        return None

    # Pick branch closest to previous C, default to branch 1 (stadium config)
    if prev_C is not None and C0 is not None and C1 is not None:
        C = C0 if np.linalg.norm(C0 - prev_C) < np.linalg.norm(C1 - prev_C) else C1
    elif C1 is not None:
        C = C1
    else:
        C = C0

    # Compute world angles
    theta_b_world = math.atan2(B[1] - A[1], B[0] - A[0])
    theta_c_world = math.atan2(B[1] - O4[1], B[0] - O4[0])
    theta_e_world = math.atan2(C[1] - B[1], C[0] - B[0])
    theta_f_world = math.atan2(C[1] - O6[1], C[0] - O6[0])

    # Joint angles (relative to parent in tree)
    q_crank = theta_crank
    q_coupler = theta_b_world - theta_crank
    q_rocker = theta_c_world
    q_link_e = theta_e_world - theta_c_world
    q_link_f = theta_f_world

    # Tracer P in world
    cos_e = math.cos(theta_e_world)
    sin_e = math.sin(theta_e_world)
    P = B + np.array([PX * cos_e - PY * sin_e, PX * sin_e + PY * cos_e])

    return {
        "A": A, "B": B, "C": C, "P": P,
        "qpos": np.array([q_crank, q_coupler, q_rocker, q_link_e, q_link_f]),
    }


def run_interactive():
    """Launch the interactive MuJoCo viewer with kinematic animation."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Disable equality constraints since we're doing kinematic control
    for i in range(model.neq):
        model.eq_active0[i] = 0

    # Initial config
    theta = math.radians(60)
    sol = solve_linkage(theta)
    data.qpos[:] = sol["qpos"]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    crank_speed = 1.5  # rad/s
    prev_B = sol["B"]
    prev_C = sol["C"]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to look at XY plane from above
        viewer.cam.lookat[:] = [1.0, 3.0, 0.0]
        viewer.cam.distance = 15.0
        viewer.cam.elevation = -90.0
        viewer.cam.azimuth = 90.0

        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            theta = math.radians(60) + crank_speed * t

            sol = solve_linkage(theta, prev_B, prev_C)
            if sol is not None:
                data.qpos[:] = sol["qpos"]
                data.qvel[:] = 0
                prev_B = sol["B"]
                prev_C = sol["C"]

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.002)


def run_headless(n_revolutions=2, n_points=500):
    """Run headless, record tracer path, and plot it with the target stadium."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Disable equality constraints
    for i in range(model.neq):
        model.eq_active0[i] = 0

    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "tracer_pos"
    )
    sensor_adr = model.sensor_adr[sensor_id]

    theta_start = math.radians(60)
    d_theta = 2 * math.pi * n_revolutions / n_points

    trace = []
    trace_analytical = []
    prev_B = None
    prev_C = None

    for i in range(n_points):
        theta = theta_start + i * d_theta
        sol = solve_linkage(theta, prev_B, prev_C)
        if sol is not None:
            data.qpos[:] = sol["qpos"]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)

            pos = data.sensordata[sensor_adr : sensor_adr + 3].copy()
            trace.append(pos[:2])
            trace_analytical.append(sol["P"])
            prev_B = sol["B"]
            prev_C = sol["C"]

    trace = np.array(trace)
    trace_analytical = np.array(trace_analytical)

    print(f"Recorded {len(trace)} points over {n_revolutions} revolutions")
    print(f"  Trace X range: [{trace[:,0].min():.3f}, {trace[:,0].max():.3f}]")
    print(f"  Trace Y range: [{trace[:,1].min():.3f}, {trace[:,1].max():.3f}]")

    # Verify MuJoCo positions match analytical
    err = np.linalg.norm(trace - trace_analytical, axis=1)
    print(f"  Max MuJoCo vs analytical error: {err.max():.2e}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stadium import stadium_points

    target = stadium_points(half_length=2.0, radius=0.8, n=200)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(target[:, 0], target[:, 1], "k--", lw=2, label="Target stadium")
    ax.plot(trace[:, 0], trace[:, 1], "r-", lw=1.2, alpha=0.8, label="Tracer path")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Stephenson-III Six-Bar: Tracer Path vs Target Stadium")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(__file__).parent / "linkage_trace.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stephenson-III six-bar linkage viewer"
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
