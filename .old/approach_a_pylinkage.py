"""
Approach A: Four-bar and six-bar linkage synthesis using pylinkage.

Uses pylinkage's joint system for forward kinematics with a coupler
tracing point (Fixed joint). Optimization via scipy differential_evolution
for full control over all parameters (ground pivots, link lengths, coupler
point offset). Also attempts pylinkage's built-in PSO for comparison.

Target: stadium curve with half_length=2.0, radius=0.8, centered at origin.
"""

import warnings
import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import pylinkage as pl

from stadium import TARGET, curve_error

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Forward kinematics via pylinkage joints
# ---------------------------------------------------------------------------

def trace_four_bar(params, n_points=200):
    """
    Trace the coupler curve of a four-bar linkage using pylinkage.

    params: [crank_r, coupler_len, rocker_len, ground_len,
             cp_dist, cp_angle, gx, gy]

    O2 = (gx, gy)             -- crank ground pivot
    O4 = (gx + ground_len, gy) -- rocker ground pivot
    A  = crank tip (rotates around O2 at radius crank_r)
    B  = coupler-rocker joint (Revolute: distance0=coupler_len from A,
                                         distance1=rocker_len from O4)
    P  = coupler tracing point (Fixed: distance=cp_dist from A,
                                       angle=cp_angle relative to A->B)
    """
    crank_r, coupler_len, rocker_len, ground_len, cp_dist, cp_angle, gx, gy = params

    if any(v <= 0 for v in [crank_r, coupler_len, rocker_len, ground_len, cp_dist]):
        return None

    # Grashof: shortest + longest <= sum of other two (full crank rotation)
    lengths = sorted([crank_r, coupler_len, rocker_len, ground_len])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    angle_step = 2.0 * np.pi / n_points

    O2 = pl.Static(gx, gy, name="O2")
    O4 = pl.Static(gx + ground_len, gy, name="O4")

    A = pl.Crank(
        x=gx + crank_r, y=gy,
        joint0=O2, distance=crank_r, angle=angle_step, name="A",
    )

    B = pl.Revolute(
        x=gx + crank_r + coupler_len * 0.5,
        y=gy + coupler_len * 0.5,
        joint0=A, joint1=O4,
        distance0=coupler_len, distance1=rocker_len, name="B",
    )

    P = pl.Fixed(
        x=gx + crank_r + cp_dist * np.cos(cp_angle),
        y=gy + cp_dist * np.sin(cp_angle),
        joint0=A, joint1=B,
        distance=cp_dist, angle=cp_angle, name="P",
    )

    linkage = pl.Linkage(
        joints=(O2, O4, A, B, P),
        order=(A, B, P),
    )

    try:
        path = []
        for coords in linkage.step(iterations=n_points, dt=1):
            px, py = coords[4]  # P is index 4
            if px is None or py is None:
                return None
            path.append([px, py])
        return np.array(path)
    except Exception:
        return None


def trace_six_bar_watt(params, n_points=200):
    """
    Trace the coupler curve of a Watt-I six-bar using pylinkage.

    Base four-bar (O2-A-B-O4) plus a dyad (C-D-O6).
    C is a Fixed point on the base coupler (relative to A->B).
    D is a Revolute connecting C and O6 via two link lengths.
    Q is the tracing point, Fixed relative to C->D.

    params: [crank_r, coupler_len, rocker_len, ground_len,
             c_dist, c_angle,      -- point C on base coupler
             cd_len, d_o6_len,     -- dyad link lengths
             q_dist, q_angle,      -- tracing point Q relative to C->D
             o6x, o6y,             -- additional ground pivot
             gx, gy]               -- base ground pivot O2
    """
    (crank_r, coupler_len, rocker_len, ground_len,
     c_dist, c_angle, cd_len, d_o6_len,
     q_dist, q_angle, o6x, o6y, gx, gy) = params

    if any(v <= 0 for v in [crank_r, coupler_len, rocker_len, ground_len,
                             c_dist, cd_len, d_o6_len, q_dist]):
        return None

    lengths = sorted([crank_r, coupler_len, rocker_len, ground_len])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    angle_step = 2.0 * np.pi / n_points

    O2 = pl.Static(gx, gy, name="O2")
    O4 = pl.Static(gx + ground_len, gy, name="O4")
    O6 = pl.Static(o6x, o6y, name="O6")

    A = pl.Crank(
        x=gx + crank_r, y=gy,
        joint0=O2, distance=crank_r, angle=angle_step, name="A",
    )

    B = pl.Revolute(
        x=gx + crank_r + coupler_len * 0.5,
        y=gy + coupler_len * 0.5,
        joint0=A, joint1=O4,
        distance0=coupler_len, distance1=rocker_len, name="B",
    )

    C = pl.Fixed(
        x=gx + crank_r + c_dist * np.cos(c_angle),
        y=gy + c_dist * np.sin(c_angle),
        joint0=A, joint1=B,
        distance=c_dist, angle=c_angle, name="C",
    )

    D = pl.Revolute(
        x=(gx + crank_r + o6x) / 2,
        y=(gy + o6y) / 2,
        joint0=C, joint1=O6,
        distance0=cd_len, distance1=d_o6_len, name="D",
    )

    Q = pl.Fixed(
        x=(gx + crank_r + o6x) / 2 + q_dist * np.cos(q_angle),
        y=(gy + o6y) / 2 + q_dist * np.sin(q_angle),
        joint0=C, joint1=D,
        distance=q_dist, angle=q_angle, name="Q",
    )

    linkage = pl.Linkage(
        joints=(O2, O4, O6, A, B, C, D, Q),
        order=(A, B, C, D, Q),
    )

    try:
        path = []
        for coords in linkage.step(iterations=n_points, dt=1):
            qx, qy = coords[7]  # Q is index 7
            if qx is None or qy is None:
                return None
            path.append([qx, qy])
        return np.array(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full joint tracing (for animation)
# ---------------------------------------------------------------------------

def trace_four_bar_joints(params, n_points=100):
    """
    Trace ALL joint positions of a four-bar linkage over one full crank rotation.
    Returns a list of dicts, one per step, with keys: O2, O4, A, B, P.
    Each value is (x, y). Returns None on failure.
    """
    crank_r, coupler_len, rocker_len, ground_len, cp_dist, cp_angle, gx, gy = params

    if any(v <= 0 for v in [crank_r, coupler_len, rocker_len, ground_len, cp_dist]):
        return None

    lengths = sorted([crank_r, coupler_len, rocker_len, ground_len])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    angle_step = 2.0 * np.pi / n_points

    O2 = pl.Static(gx, gy, name="O2")
    O4 = pl.Static(gx + ground_len, gy, name="O4")
    A = pl.Crank(x=gx + crank_r, y=gy, joint0=O2, distance=crank_r,
                 angle=angle_step, name="A")
    B = pl.Revolute(x=gx + crank_r + coupler_len * 0.5,
                    y=gy + coupler_len * 0.5,
                    joint0=A, joint1=O4,
                    distance0=coupler_len, distance1=rocker_len, name="B")
    P = pl.Fixed(x=gx + crank_r + cp_dist * np.cos(cp_angle),
                 y=gy + cp_dist * np.sin(cp_angle),
                 joint0=A, joint1=B,
                 distance=cp_dist, angle=cp_angle, name="P")

    linkage = pl.Linkage(joints=(O2, O4, A, B, P), order=(A, B, P))

    try:
        frames = []
        for coords in linkage.step(iterations=n_points, dt=1):
            c = {
                "O2": coords[0], "O4": coords[1],
                "A": coords[2], "B": coords[3], "P": coords[4],
            }
            if any(v[0] is None or v[1] is None for v in c.values()):
                return None
            frames.append(c)
        return frames
    except Exception:
        return None


def trace_six_bar_joints(params, n_points=100):
    """
    Trace ALL joint positions of a six-bar Watt-I linkage over one full rotation.
    Returns a list of dicts with keys: O2, O4, O6, A, B, C, D, Q.
    """
    (crank_r, coupler_len, rocker_len, ground_len,
     c_dist, c_angle, cd_len, d_o6_len,
     q_dist, q_angle, o6x, o6y, gx, gy) = params

    if any(v <= 0 for v in [crank_r, coupler_len, rocker_len, ground_len,
                             c_dist, cd_len, d_o6_len, q_dist]):
        return None

    lengths = sorted([crank_r, coupler_len, rocker_len, ground_len])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    angle_step = 2.0 * np.pi / n_points

    O2 = pl.Static(gx, gy, name="O2")
    O4 = pl.Static(gx + ground_len, gy, name="O4")
    O6 = pl.Static(o6x, o6y, name="O6")
    A = pl.Crank(x=gx + crank_r, y=gy, joint0=O2, distance=crank_r,
                 angle=angle_step, name="A")
    B = pl.Revolute(x=gx + crank_r + coupler_len * 0.5,
                    y=gy + coupler_len * 0.5,
                    joint0=A, joint1=O4,
                    distance0=coupler_len, distance1=rocker_len, name="B")
    C = pl.Fixed(x=gx + crank_r + c_dist * np.cos(c_angle),
                 y=gy + c_dist * np.sin(c_angle),
                 joint0=A, joint1=B,
                 distance=c_dist, angle=c_angle, name="C")
    D = pl.Revolute(x=(gx + crank_r + o6x) / 2, y=(gy + o6y) / 2,
                    joint0=C, joint1=O6,
                    distance0=cd_len, distance1=d_o6_len, name="D")
    Q = pl.Fixed(x=(gx + crank_r + o6x) / 2 + q_dist * np.cos(q_angle),
                 y=(gy + o6y) / 2 + q_dist * np.sin(q_angle),
                 joint0=C, joint1=D,
                 distance=q_dist, angle=q_angle, name="Q")

    linkage = pl.Linkage(joints=(O2, O4, O6, A, B, C, D, Q),
                         order=(A, B, C, D, Q))

    try:
        frames = []
        for coords in linkage.step(iterations=n_points, dt=1):
            c = {
                "O2": coords[0], "O4": coords[1], "O6": coords[2],
                "A": coords[3], "B": coords[4], "C": coords[5],
                "D": coords[6], "Q": coords[7],
            }
            if any(v[0] is None or v[1] is None for v in c.values()):
                return None
            frames.append(c)
        return frames
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GIF animation
# ---------------------------------------------------------------------------

def make_gif(best_label, best_x, target, filename="approach_a_result.gif",
             n_frames=100, fps=8):
    """Generate an animated GIF showing the linkage mechanism in motion."""
    print(f"\n  Generating GIF animation ({n_frames} frames, {fps} fps) ...")

    is_four_bar = (best_label == "four_bar")
    if is_four_bar:
        frames = trace_four_bar_joints(best_x, n_points=n_frames)
    else:
        frames = trace_six_bar_joints(best_x, n_points=n_frames)

    if frames is None or len(frames) < 2:
        print("  WARNING: Could not trace joint positions for GIF.")
        return

    # Collect all coordinates to set axis limits
    all_xs, all_ys = [], []
    for f in frames:
        for v in f.values():
            all_xs.append(v[0])
            all_ys.append(v[1])
    for p in target:
        all_xs.append(p[0])
        all_ys.append(p[1])
    margin = 0.8
    xmin, xmax = min(all_xs) - margin, max(all_xs) + margin
    ymin, ymax = min(all_ys) - margin, max(all_ys) + margin

    fig, ax = plt.subplots(figsize=(8, 8))

    # Target curve (background, closed)
    tgt = np.vstack([target, target[0]])
    ax.plot(tgt[:, 0], tgt[:, 1], color="lightblue", lw=3, zorder=1,
            label="Target stadium")

    # Artists that will be updated each frame
    if is_four_bar:
        # Ground link (dashed)
        ground_line, = ax.plot([], [], "k--", lw=1.5, zorder=2)
        # Crank link O2->A
        crank_line, = ax.plot([], [], "k-", lw=3, zorder=3)
        # Coupler link A->B
        coupler_line, = ax.plot([], [], "b-", lw=3, zorder=3)
        # Rocker link B->O4
        rocker_line, = ax.plot([], [], "k-", lw=3, zorder=3)
        # Ground pivots (triangles)
        ground_pivots, = ax.plot([], [], "k^", ms=12, zorder=5)
        # Moving joints (circles)
        moving_joints, = ax.plot([], [], "ko", ms=8, zorder=5,
                                 markerfacecolor="white", markeredgewidth=2)
        # Coupler tracing point
        trace_pt, = ax.plot([], [], "ro", ms=10, zorder=6)
        # Trailing path
        trail_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.7, zorder=2)
    else:
        # Six-bar: base four-bar + dyad
        ground_line, = ax.plot([], [], "k--", lw=1.5, zorder=2)
        ground_line2, = ax.plot([], [], "k--", lw=1.5, zorder=2)
        crank_line, = ax.plot([], [], "k-", lw=3, zorder=3)
        coupler_line, = ax.plot([], [], "b-", lw=3, zorder=3)
        rocker_line, = ax.plot([], [], "k-", lw=3, zorder=3)
        dyad_link1, = ax.plot([], [], "m-", lw=3, zorder=3)
        dyad_link2, = ax.plot([], [], "m-", lw=3, zorder=3)
        ground_pivots, = ax.plot([], [], "k^", ms=12, zorder=5)
        moving_joints, = ax.plot([], [], "ko", ms=8, zorder=5,
                                 markerfacecolor="white", markeredgewidth=2)
        trace_pt, = ax.plot([], [], "ro", ms=10, zorder=6)
        trail_line, = ax.plot([], [], "r-", lw=1.5, alpha=0.7, zorder=2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"Approach A: {best_label} linkage mechanism", fontweight="bold")

    trail_x, trail_y = [], []

    def update(i):
        f = frames[i]
        if is_four_bar:
            o2 = f["O2"]; o4 = f["O4"]
            a = f["A"]; b = f["B"]; p = f["P"]

            ground_line.set_data([o2[0], o4[0]], [o2[1], o4[1]])
            crank_line.set_data([o2[0], a[0]], [o2[1], a[1]])
            coupler_line.set_data([a[0], b[0]], [a[1], b[1]])
            rocker_line.set_data([b[0], o4[0]], [b[1], o4[1]])
            ground_pivots.set_data([o2[0], o4[0]], [o2[1], o4[1]])
            moving_joints.set_data([a[0], b[0]], [a[1], b[1]])
            trace_pt.set_data([p[0]], [p[1]])

            trail_x.append(p[0])
            trail_y.append(p[1])
            trail_line.set_data(trail_x, trail_y)
        else:
            o2 = f["O2"]; o4 = f["O4"]; o6 = f["O6"]
            a = f["A"]; b = f["B"]; c = f["C"]; d = f["D"]; q = f["Q"]

            ground_line.set_data([o2[0], o4[0]], [o2[1], o4[1]])
            ground_line2.set_data([o2[0], o6[0]], [o2[1], o6[1]])
            crank_line.set_data([o2[0], a[0]], [o2[1], a[1]])
            coupler_line.set_data([a[0], b[0]], [a[1], b[1]])
            rocker_line.set_data([b[0], o4[0]], [b[1], o4[1]])
            dyad_link1.set_data([c[0], d[0]], [c[1], d[1]])
            dyad_link2.set_data([d[0], o6[0]], [d[1], o6[1]])
            ground_pivots.set_data([o2[0], o4[0], o6[0]],
                                   [o2[1], o4[1], o6[1]])
            moving_joints.set_data([a[0], b[0], c[0], d[0]],
                                   [a[1], b[1], c[1], d[1]])
            trace_pt.set_data([q[0]], [q[1]])

            trail_x.append(q[0])
            trail_y.append(q[1])
            trail_line.set_data(trail_x, trail_y)

        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=True,
                         repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  GIF saved to {filename}")


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

def objective_4bar(x):
    curve = trace_four_bar(x, n_points=200)
    return curve_error(curve, TARGET)

def objective_6bar(x):
    curve = trace_six_bar_watt(x, n_points=200)
    return curve_error(curve, TARGET)


# ---------------------------------------------------------------------------
# pylinkage built-in PSO (four-bar only, link-length constraints)
# ---------------------------------------------------------------------------

def run_pylinkage_pso():
    """Use pylinkage's native PSO to optimize link lengths of a four-bar."""
    print("\n--- pylinkage built-in PSO (four-bar, fixed ground pivots) ---")

    # Fix ground pivots; let PSO vary link lengths + coupler offset
    gx, gy = 0.0, 0.0
    ground_len = 4.0
    angle_step = 2.0 * np.pi / 96

    O2 = pl.Static(gx, gy, name="O2")
    O4 = pl.Static(gx + ground_len, gy, name="O4")
    A = pl.Crank(x=gx + 1.0, y=gy, joint0=O2, distance=1.0,
                 angle=angle_step, name="A")
    B = pl.Revolute(x=gx + 2.5, y=gy + 1.5, joint0=A, joint1=O4,
                    distance0=3.0, distance1=3.0, name="B")
    P = pl.Fixed(x=gx + 2.0, y=gy + 1.0, joint0=A, joint1=B,
                 distance=2.0, angle=0.5, name="P")

    linkage = pl.Linkage(
        joints=(O2, O4, A, B, P),
        order=(A, B, P),
    )

    # Constraints: Crank(r) + Revolute(r0,r1) + Fixed(r,angle) = 1+2+2 = 5
    init_constraints = list(linkage.get_num_constraints())
    print(f"  Initial constraints: {init_constraints}")

    @pl.kinematic_minimization
    def fitness(linkage, params, init_pos, loci, **kwargs):
        coupler_path = []
        for step_coords in loci:
            x, y = step_coords[4]
            if x is None or y is None:
                return float("inf")
            coupler_path.append([x, y])
        coupler_path = np.array(coupler_path)
        if len(coupler_path) < 10:
            return float("inf")
        return curve_error(coupler_path, TARGET)

    lower = np.array([0.3, 0.5, 0.5, 0.3, -np.pi])
    upper = np.array([4.0, 8.0, 6.0, 6.0, np.pi])
    bounds = (lower, upper)

    try:
        results = pl.particle_swarm_optimization(
            eval_func=fitness,
            linkage=linkage,
            bounds=bounds,
            dimensions=5,
            n_particles=40,
            iters=80,
            order_relation=min,
            verbose=True,
        )
        best = results[0]
        print(f"  PSO best score: {best.score:.6f}")
        print(f"  Best constraints: {best.dimensions}")
        return best.score, best.dimensions
    except Exception as e:
        print(f"  PSO failed: {e}")
        return float("inf"), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Approach A: pylinkage-based linkage synthesis")
    print("Target: stadium (half_length=2.0, radius=0.8)")
    print(f"Target curve: {len(TARGET)} points")
    print("=" * 60)

    # ---- pylinkage PSO (four-bar, fixed ground pivots) ----
    try:
        pso_score, pso_constraints = run_pylinkage_pso()
    except Exception as e:
        print(f"  pylinkage PSO skipped: {e}")
        pso_score = float("inf")

    # ---- scipy DE: four-bar ----
    print("\n--- scipy DE: four-bar (full 8-param search) ---")
    bounds_4 = [
        (0.3, 4.0),          # crank_r
        (1.0, 8.0),          # coupler_len
        (0.5, 6.0),          # rocker_len
        (1.0, 8.0),          # ground_len
        (0.3, 6.0),          # cp_dist
        (-np.pi, np.pi),     # cp_angle
        (-3.0, 3.0),         # gx
        (-3.0, 3.0),         # gy
    ]

    res4 = differential_evolution(
        objective_4bar, bounds_4,
        maxiter=300, popsize=25, tol=1e-8, seed=42,
        disp=True, workers=-1,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    print(f"  Four-bar DE best error: {res4.fun:.6f}")

    # Multi-seed improvement
    best4_err, best4_x = res4.fun, res4.x
    for seed in [123, 789, 2024]:
        r = differential_evolution(
            objective_4bar, bounds_4,
            maxiter=200, popsize=20, tol=1e-8, seed=seed,
            disp=False, workers=-1,
            mutation=(0.5, 1.5), recombination=0.9, polish=True,
        )
        if r.fun < best4_err:
            best4_err, best4_x = r.fun, r.x
            print(f"    Seed {seed} improved to {best4_err:.6f}")

    # Nelder-Mead polish
    nm = minimize(objective_4bar, best4_x, method="Nelder-Mead",
                  options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12})
    if nm.fun < best4_err:
        best4_err, best4_x = nm.fun, nm.x
        print(f"  Nelder-Mead polished to {best4_err:.6f}")

    curve4 = trace_four_bar(best4_x, n_points=400)
    names4 = ["crank_r", "coupler_len", "rocker_len", "ground_len",
              "cp_dist", "cp_angle", "gx", "gy"]
    print("  Best four-bar parameters:")
    for n, v in zip(names4, best4_x):
        print(f"    {n:15s} = {v:.6f}")

    # ---- scipy DE: six-bar Watt-I ----
    print("\n--- scipy DE: six-bar Watt-I (14-param search) ---")
    bounds_6 = [
        (0.3, 4.0),          # crank_r
        (1.0, 8.0),          # coupler_len
        (0.5, 6.0),          # rocker_len
        (1.0, 8.0),          # ground_len
        (0.3, 5.0),          # c_dist
        (-np.pi, np.pi),     # c_angle
        (0.5, 6.0),          # cd_len
        (0.5, 6.0),          # d_o6_len
        (0.3, 5.0),          # q_dist
        (-np.pi, np.pi),     # q_angle
        (-5.0, 5.0),         # o6x
        (-5.0, 5.0),         # o6y
        (-3.0, 3.0),         # gx
        (-3.0, 3.0),         # gy
    ]

    res6 = differential_evolution(
        objective_6bar, bounds_6,
        maxiter=400, popsize=30, tol=1e-8, seed=42,
        disp=True, workers=-1,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    print(f"  Six-bar DE best error: {res6.fun:.6f}")

    best6_err, best6_x = res6.fun, res6.x
    for seed in [123, 789, 2024]:
        r = differential_evolution(
            objective_6bar, bounds_6,
            maxiter=250, popsize=25, tol=1e-8, seed=seed,
            disp=False, workers=-1,
            mutation=(0.5, 1.5), recombination=0.9, polish=True,
        )
        if r.fun < best6_err:
            best6_err, best6_x = r.fun, r.x
            print(f"    Seed {seed} improved to {best6_err:.6f}")

    nm6 = minimize(objective_6bar, best6_x, method="Nelder-Mead",
                   options={"maxiter": 15000, "xatol": 1e-12, "fatol": 1e-12})
    if nm6.fun < best6_err:
        best6_err, best6_x = nm6.fun, nm6.x
        print(f"  Nelder-Mead polished to {best6_err:.6f}")

    curve6 = trace_six_bar_watt(best6_x, n_points=400)
    names6 = ["crank_r", "coupler_len", "rocker_len", "ground_len",
              "c_dist", "c_angle", "cd_len", "d_o6_len",
              "q_dist", "q_angle", "o6x", "o6y", "gx", "gy"]
    print("  Best six-bar parameters:")
    for n, v in zip(names6, best6_x):
        print(f"    {n:15s} = {v:.6f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  pylinkage PSO (4-bar, fixed pivots): {pso_score:.6f}")
    print(f"  scipy DE four-bar:                   {best4_err:.6f}")
    print(f"  scipy DE six-bar Watt-I:             {best6_err:.6f}")

    if best4_err <= best6_err:
        best_label = "four_bar"
        best_err = best4_err
        best_x = best4_x
        best_curve = curve4
    else:
        best_label = "six_bar_watt"
        best_err = best6_err
        best_x = best6_x
        best_curve = curve6

    print(f"\n  BEST: {best_label}  error = {best_err:.6f}")
    print(f"  params = {best_x}")

    # Save coupler curve
    if best_curve is not None:
        np.savetxt("approach_a_coupler_curve.csv", best_curve,
                   delimiter=",", header="x,y", comments="")
        print("\n  Coupler curve saved to approach_a_coupler_curve.csv")

    # ---- Visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Target (closed for display)
    tgt = np.vstack([TARGET, TARGET[0]])

    ax = axes[0]
    ax.plot(tgt[:, 0], tgt[:, 1], "b-", lw=2, label="Target stadium")
    if curve4 is not None:
        c = np.vstack([curve4, curve4[0]])
        ax.plot(c[:, 0], c[:, 1], "r--", lw=1.5,
                label=f"Four-bar (err={best4_err:.4f})")
    ax.set_title("Four-bar coupler curve")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(tgt[:, 0], tgt[:, 1], "b-", lw=2, label="Target stadium")
    if curve6 is not None:
        c = np.vstack([curve6, curve6[0]])
        ax.plot(c[:, 0], c[:, 1], "g--", lw=1.5,
                label=f"Six-bar (err={best6_err:.4f})")
    ax.set_title("Six-bar Watt-I coupler curve")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(tgt[:, 0], tgt[:, 1], "b-", lw=2.5, label="Target stadium")
    if best_curve is not None:
        c = np.vstack([best_curve, best_curve[0]])
        color = "r" if best_label == "four_bar" else "g"
        ax.plot(c[:, 0], c[:, 1], f"{color}--", lw=1.5,
                label=f"Best: {best_label} (err={best_err:.4f})")
    ax.set_title(f"Best: {best_label}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Approach A: pylinkage linkage synthesis for stadium curve",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("approach_a_result.png", dpi=150, bbox_inches="tight")
    print("  Visualization saved to approach_a_result.png")

    # ---- Animated GIF ----
    make_gif(best_label, best_x, TARGET, filename="approach_a_result.gif",
             n_frames=100, fps=8)

    print("\nDone.")


if __name__ == "__main__":
    main()
