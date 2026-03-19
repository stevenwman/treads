#!/usr/bin/env python3
"""
Approach B: Pure scipy linkage synthesis for stadium curve.

Implements forward kinematics for three topologies (four-bar, slider-crank,
six-bar Watt-I), optimises each with differential_evolution + local refinement,
and picks the best overall result.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from stadium import TARGET, curve_error

# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def four_bar_coupler_curve(params, n_points=200):
    """
    Four-bar linkage coupler curve.
    params: [a, b, c, d, px, py, gx, gy]
    """
    a, b, c, d, px, py, gx, gy = params

    # Grashof check: shortest + longest <= sum of other two
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2x, O2y = gx, gy
    O4x, O4y = gx + d, gy

    points = np.empty((n_points, 2))
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    for i, theta2 in enumerate(thetas):
        Ax = O2x + a * np.cos(theta2)
        Ay = O2y + a * np.sin(theta2)

        dx = O4x - Ax
        dy = O4y - Ay
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > b + c or dist < abs(b - c):
            return None

        cos_alpha = (b * b + dist * dist - c * c) / (2 * b * dist)
        if abs(cos_alpha) > 1:
            return None

        ang_to_O4 = np.arctan2(dy, dx)
        alpha = np.arccos(np.clip(cos_alpha, -1, 1))
        theta3 = ang_to_O4 + alpha

        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)
        points[i, 0] = Ax + px * cos3 - py * sin3
        points[i, 1] = Ay + px * sin3 + py * cos3

    return points


def slider_crank_coupler_curve(params, n_points=200):
    """
    Slider-crank coupler curve.
    params: [a, b, px, py, slider_angle, offset, gx, gy]
    """
    a, b, px, py, slider_angle, offset, gx, gy = params
    sa = np.sin(slider_angle)
    ca = np.cos(slider_angle)

    points = np.empty((n_points, 2))
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    for i, theta in enumerate(thetas):
        Ax = gx + a * np.cos(theta)
        Ay = gy + a * np.sin(theta)

        # Perpendicular distance from A to slider line
        d_perp = (Ax - gx) * (-sa) + (Ay - gy) * ca - offset
        d_along_sq = b * b - d_perp * d_perp
        if d_along_sq < 0:
            return None
        d_along = np.sqrt(d_along_sq)

        # Foot of perpendicular from A onto slider line
        foot_x = Ax + d_perp * sa
        foot_y = Ay - d_perp * ca
        Bx = foot_x + d_along * ca
        By = foot_y + d_along * sa

        theta3 = np.arctan2(By - Ay, Bx - Ax)
        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)

        points[i, 0] = Ax + px * cos3 - py * sin3
        points[i, 1] = Ay + px * sin3 + py * cos3

    return points


def six_bar_watt_coupler_curve(params, n_points=200):
    """
    Watt-I six-bar linkage coupler curve.
    params: [a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy]
    """
    a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy = params

    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2x, O2y = gx, gy
    O4x, O4y = gx + d, gy

    points = np.empty((n_points, 2))
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    for i, theta2 in enumerate(thetas):
        # Base four-bar
        Ax = O2x + a * np.cos(theta2)
        Ay = O2y + a * np.sin(theta2)

        dx = O4x - Ax
        dy = O4y - Ay
        dist = np.sqrt(dx * dx + dy * dy)

        if dist > b + c or dist < abs(b - c):
            return None

        cos_alpha = (b * b + dist * dist - c * c) / (2 * b * dist)
        if abs(cos_alpha) > 1:
            return None

        ang = np.arctan2(dy, dx)
        alpha = np.arccos(np.clip(cos_alpha, -1, 1))
        theta3 = ang + alpha

        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)

        # Dyad attachment on coupler
        P1x = Ax + px1 * cos3 - py1 * sin3
        P1y = Ay + px1 * sin3 + py1 * cos3

        # Solve dyad: link e from P1 to D, link f from D to O6
        dx2 = O6x - P1x
        dy2 = O6y - P1y
        dist2 = np.sqrt(dx2 * dx2 + dy2 * dy2)

        if dist2 > e + f or dist2 < abs(e - f):
            return None

        cos_beta = (e * e + dist2 * dist2 - f * f) / (2 * e * dist2)
        if abs(cos_beta) > 1:
            return None

        # Tracing point on coupler of base four-bar
        points[i, 0] = Ax + px2 * cos3 - py2 * sin3
        points[i, 1] = Ay + px2 * sin3 + py2 * cos3

    return points


# ---------------------------------------------------------------------------
# Vectorised error (faster than the loop version in stadium.py for optimiser)
# ---------------------------------------------------------------------------

def fast_curve_error(coupler_curve, target_curve):
    """Bidirectional mean-of-min-squared-distances (same metric as stadium.py)."""
    if coupler_curve is None:
        return 1e12
    # Forward: coupler -> target
    # Use broadcasting: (N,1,2) - (1,M,2) -> (N,M,2) -> sum over last -> (N,M)
    diff_fwd = coupler_curve[:, None, :] - target_curve[None, :, :]
    sq_fwd = np.sum(diff_fwd * diff_fwd, axis=2)
    forward = np.sum(np.min(sq_fwd, axis=1))

    # Backward: target -> coupler
    backward = np.sum(np.min(sq_fwd, axis=0))

    return (forward + backward) / (len(coupler_curve) + len(target_curve))


# ---------------------------------------------------------------------------
# Objective wrappers
# ---------------------------------------------------------------------------

def objective_four_bar(x):
    curve = four_bar_coupler_curve(x, n_points=200)
    return fast_curve_error(curve, TARGET)


def objective_slider_crank(x):
    curve = slider_crank_coupler_curve(x, n_points=200)
    return fast_curve_error(curve, TARGET)


def objective_six_bar(x):
    curve = six_bar_watt_coupler_curve(x, n_points=200)
    return fast_curve_error(curve, TARGET)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

bounds_4bar = [
    (0.3, 5.0),   # a (crank)
    (1.0, 8.0),   # b (coupler)
    (0.5, 6.0),   # c (rocker)
    (1.0, 8.0),   # d (ground)
    (-4.0, 4.0),  # px
    (-4.0, 4.0),  # py
    (-3.0, 3.0),  # gx
    (-3.0, 3.0),  # gy
]

bounds_slider = [
    (0.3, 5.0),        # a (crank)
    (1.0, 8.0),        # b (connecting rod)
    (-4.0, 4.0),       # px
    (-4.0, 4.0),       # py
    (-np.pi, np.pi),   # slider_angle
    (-3.0, 3.0),       # offset
    (-3.0, 3.0),       # gx
    (-3.0, 3.0),       # gy
]

bounds_6bar = [
    (0.3, 5.0),   # a
    (1.0, 8.0),   # b
    (0.5, 6.0),   # c
    (1.0, 8.0),   # d
    (0.5, 6.0),   # e
    (0.5, 6.0),   # f
    (-4.0, 4.0),  # px1
    (-4.0, 4.0),  # py1
    (-4.0, 4.0),  # px2
    (-4.0, 4.0),  # py2
    (-5.0, 5.0),  # O6x
    (-5.0, 5.0),  # O6y
    (-3.0, 3.0),  # gx
    (-3.0, 3.0),  # gy
]

# ---------------------------------------------------------------------------
# Optimiser helper
# ---------------------------------------------------------------------------

def run_topology(name, objective, bounds, seeds, maxiter, popsize):
    """Run DE with multiple seeds, then locally refine the best."""
    print(f"\n{'='*60}")
    print(f"  Topology: {name}")
    print(f"  {len(seeds)} seeds, maxiter={maxiter}, popsize={popsize}")
    print(f"{'='*60}")

    best_result = None

    for seed in seeds:
        print(f"  seed={seed} ... ", end="", flush=True)
        res = differential_evolution(
            objective,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-8,
            seed=seed,
            workers=1,
            updating="deferred",
            disp=False,
        )
        print(f"error={res.fun:.6f}")
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    # Local refinement with Nelder-Mead (derivative-free, respects None returns)
    print(f"  Best DE error: {best_result.fun:.6f}  ->  refining ...", flush=True)
    refined = minimize(
        objective,
        best_result.x,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10},
    )
    if refined.fun < best_result.fun:
        best_result = refined
        print(f"  Refined error: {refined.fun:.6f}")
    else:
        print(f"  Refinement did not improve (still {best_result.fun:.6f})")

    print(f"  Final best error: {best_result.fun:.6f}")
    print(f"  Params: {np.array2string(best_result.x, precision=6)}")
    return best_result


# ---------------------------------------------------------------------------
# Joint-position functions (return all joint locations per crank angle)
# ---------------------------------------------------------------------------

def four_bar_joints(params, theta2):
    """Return dict of joint positions for a single crank angle, or None."""
    a, b, c, d, px, py, gx, gy = params
    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])
    A = O2 + a * np.array([np.cos(theta2), np.sin(theta2)])

    dx, dy = O4[0] - A[0], O4[1] - A[1]
    dist = np.hypot(dx, dy)
    if dist > b + c or dist < abs(b - c):
        return None

    cos_alpha = (b * b + dist * dist - c * c) / (2 * b * dist)
    if abs(cos_alpha) > 1:
        return None

    ang_to_O4 = np.arctan2(dy, dx)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    theta3 = ang_to_O4 + alpha

    cos3, sin3 = np.cos(theta3), np.sin(theta3)
    B = A + b * np.array([cos3, sin3])
    # Actually B should satisfy |B - O4| == c. Recompute B from A along coupler:
    B = np.array([A[0] + b * cos3, A[1] + b * sin3])
    P = np.array([A[0] + px * cos3 - py * sin3,
                   A[1] + px * sin3 + py * cos3])

    return {"O2": O2, "O4": O4, "A": A, "B": B, "P": P}


def slider_crank_joints(params, theta):
    """Return dict of joint positions for a single crank angle, or None."""
    a, b, px, py, slider_angle, offset, gx, gy = params
    sa, ca = np.sin(slider_angle), np.cos(slider_angle)

    O2 = np.array([gx, gy])
    A = O2 + a * np.array([np.cos(theta), np.sin(theta)])

    d_perp = (A[0] - gx) * (-sa) + (A[1] - gy) * ca - offset
    d_along_sq = b * b - d_perp * d_perp
    if d_along_sq < 0:
        return None
    d_along = np.sqrt(d_along_sq)

    foot = np.array([A[0] + d_perp * sa, A[1] - d_perp * ca])
    B = foot + d_along * np.array([ca, sa])

    theta3 = np.arctan2(B[1] - A[1], B[0] - A[0])
    cos3, sin3 = np.cos(theta3), np.sin(theta3)
    P = np.array([A[0] + px * cos3 - py * sin3,
                   A[1] + px * sin3 + py * cos3])

    # Slider rail endpoints for drawing
    rail_center = O2 + offset * np.array([-sa, ca])
    rail_start = rail_center - 8 * np.array([ca, sa])
    rail_end = rail_center + 8 * np.array([ca, sa])

    return {"O2": O2, "A": A, "B": B, "P": P,
            "rail_start": rail_start, "rail_end": rail_end,
            "slider_angle": slider_angle, "offset": offset}


def six_bar_joints(params, theta2):
    """Return dict of joint positions for a single crank angle, or None."""
    a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy = params
    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])
    O6 = np.array([O6x, O6y])
    A = O2 + a * np.array([np.cos(theta2), np.sin(theta2)])

    dx, dy = O4[0] - A[0], O4[1] - A[1]
    dist = np.hypot(dx, dy)
    if dist > b + c or dist < abs(b - c):
        return None

    cos_alpha = (b * b + dist * dist - c * c) / (2 * b * dist)
    if abs(cos_alpha) > 1:
        return None

    ang = np.arctan2(dy, dx)
    alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    theta3 = ang + alpha
    cos3, sin3 = np.cos(theta3), np.sin(theta3)

    B = np.array([A[0] + b * cos3, A[1] + b * sin3])

    # Dyad attachment P1 on coupler
    P1 = np.array([A[0] + px1 * cos3 - py1 * sin3,
                    A[1] + px1 * sin3 + py1 * cos3])

    # Dyad: link e from P1 to D, link f from D to O6
    dx2, dy2 = O6[0] - P1[0], O6[1] - P1[1]
    dist2 = np.hypot(dx2, dy2)
    if dist2 > e + f or dist2 < abs(e - f):
        return None
    cos_beta = (e * e + dist2 * dist2 - f * f) / (2 * e * dist2)
    if abs(cos_beta) > 1:
        return None
    ang2 = np.arctan2(dy2, dx2)
    beta = np.arccos(np.clip(cos_beta, -1, 1))
    theta5 = ang2 + beta
    D = P1 + e * np.array([np.cos(theta5), np.sin(theta5)])

    # Tracing point on base four-bar coupler
    P = np.array([A[0] + px2 * cos3 - py2 * sin3,
                   A[1] + px2 * sin3 + py2 * cos3])

    return {"O2": O2, "O4": O4, "O6": O6, "A": A, "B": B,
            "P1": P1, "D": D, "P": P}


# ---------------------------------------------------------------------------
# GIF animation
# ---------------------------------------------------------------------------

def animate_mechanism(params, topology, filename, n_frames=100, fps=8):
    """
    Produce a GIF showing the linkage mechanism in motion for one full
    crank rotation.  Works for 'four_bar', 'slider_crank', 'six_bar'.
    """
    # Choose the right joint function and curve function
    joint_func_map = {
        "four_bar": four_bar_joints,
        "slider_crank": slider_crank_joints,
        "six_bar": six_bar_joints,
    }
    curve_func_map = {
        "four_bar": four_bar_coupler_curve,
        "slider_crank": slider_crank_coupler_curve,
        "six_bar": six_bar_watt_coupler_curve,
    }

    joint_func = joint_func_map[topology]
    curve_func = curve_func_map[topology]

    # Pre-compute full coupler curve for axis limits
    full_curve = curve_func(params, n_points=400)

    thetas = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    # Pre-compute all frames to find axis limits and verify feasibility
    all_joints = []
    for theta in thetas:
        j = joint_func(params, theta)
        all_joints.append(j)

    # Collect all coordinates for axis limits
    all_coords = []
    if full_curve is not None:
        all_coords.append(full_curve)
    all_coords.append(TARGET)
    for j in all_joints:
        if j is None:
            continue
        for key, val in j.items():
            if isinstance(val, np.ndarray) and val.shape == (2,):
                all_coords.append(val.reshape(1, 2))
    if len(all_coords) == 0:
        print(f"  WARNING: no valid frames for {topology}, skipping GIF.")
        return
    all_coords = np.vstack(all_coords)
    xmin, xmax = all_coords[:, 0].min() - 0.5, all_coords[:, 0].max() + 0.5
    ymin, ymax = all_coords[:, 1].min() - 0.5, all_coords[:, 1].max() + 0.5

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{topology.replace('_', '-')} linkage", fontsize=13)

    # Target curve (background)
    ax.plot(TARGET[:, 0], TARGET[:, 1], color="lightskyblue", lw=2.5,
            zorder=1, label="Target stadium")

    # Artists that will be updated each frame
    trail_x, trail_y = [], []
    (trail_line,) = ax.plot([], [], "r-", lw=1.0, alpha=0.6, zorder=2)
    (coupler_dot,) = ax.plot([], [], "ro", ms=6, zorder=5, label="Coupler point")

    # Links and joints — we'll redraw these each frame
    link_lines = []
    joint_markers = []

    def _clear_artists():
        for a in link_lines:
            a.remove()
        link_lines.clear()
        for a in joint_markers:
            a.remove()
        joint_markers.clear()

    def _draw_link(p1, p2, **kwargs):
        defaults = dict(color="steelblue", lw=2.5, solid_capstyle="round", zorder=3)
        defaults.update(kwargs)
        line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **defaults)
        link_lines.append(line)

    def _draw_ground(pt):
        m, = ax.plot(pt[0], pt[1], "^", color="darkgreen", ms=10, zorder=4)
        joint_markers.append(m)

    def _draw_joint(pt):
        m, = ax.plot(pt[0], pt[1], "o", color="navy", ms=6, zorder=4)
        joint_markers.append(m)

    def update(frame_idx):
        _clear_artists()
        j = all_joints[frame_idx]
        if j is None:
            coupler_dot.set_data([], [])
            return

        P = j["P"]
        trail_x.append(P[0])
        trail_y.append(P[1])
        trail_line.set_data(trail_x, trail_y)
        coupler_dot.set_data([P[0]], [P[1]])

        if topology == "four_bar":
            _draw_link(j["O2"], j["A"])
            _draw_link(j["A"], j["B"])
            _draw_link(j["B"], j["O4"])
            _draw_link(j["O2"], j["O4"], ls="--", lw=1.5, color="gray")
            _draw_ground(j["O2"])
            _draw_ground(j["O4"])
            _draw_joint(j["A"])
            _draw_joint(j["B"])

        elif topology == "slider_crank":
            _draw_link(j["O2"], j["A"])
            _draw_link(j["A"], j["B"])
            _draw_link(j["rail_start"], j["rail_end"],
                       ls="--", lw=1.0, color="gray")
            _draw_ground(j["O2"])
            _draw_joint(j["A"])
            # Slider block as a square marker
            m, = ax.plot(j["B"][0], j["B"][1], "s", color="darkorange",
                         ms=8, zorder=4)
            joint_markers.append(m)

        elif topology == "six_bar":
            # Base four-bar
            _draw_link(j["O2"], j["A"])
            _draw_link(j["A"], j["B"])
            _draw_link(j["B"], j["O4"])
            _draw_link(j["O2"], j["O4"], ls="--", lw=1.5, color="gray")
            # Dyad
            _draw_link(j["P1"], j["D"], color="coral", lw=2.0)
            _draw_link(j["D"], j["O6"], color="coral", lw=2.0)
            _draw_ground(j["O2"])
            _draw_ground(j["O4"])
            _draw_ground(j["O6"])
            _draw_joint(j["A"])
            _draw_joint(j["B"])
            _draw_joint(j["P1"])
            _draw_joint(j["D"])

    ax.legend(loc="upper right", fontsize=8)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps,
                         blit=False, repeat=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  GIF saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds = [42, 123, 7]

    # --- Four-bar ---
    res_4bar = run_topology(
        "Four-bar",
        objective_four_bar,
        bounds_4bar,
        seeds=seeds,
        maxiter=500,
        popsize=30,
    )

    # --- Slider-crank ---
    res_slider = run_topology(
        "Slider-crank",
        objective_slider_crank,
        bounds_slider,
        seeds=seeds,
        maxiter=500,
        popsize=30,
    )

    # --- Six-bar Watt-I ---
    res_6bar = run_topology(
        "Six-bar Watt-I",
        objective_six_bar,
        bounds_6bar,
        seeds=seeds,
        maxiter=800,
        popsize=50,
    )

    # --- Compare ---
    results = {
        "Four-bar": res_4bar,
        "Slider-crank": res_slider,
        "Six-bar Watt-I": res_6bar,
    }

    curve_funcs = {
        "Four-bar": four_bar_coupler_curve,
        "Slider-crank": slider_crank_coupler_curve,
        "Six-bar Watt-I": six_bar_watt_coupler_curve,
    }

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    best_name = None
    best_err = float("inf")
    for name, res in results.items():
        tag = ""
        if res.fun < best_err:
            best_err = res.fun
            best_name = name
        print(f"  {name:20s}  error = {res.fun:.6f}")
    print(f"\n  >>> Best topology: {best_name} (error={best_err:.6f})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (name, res) in zip(axes, results.items()):
        curve = curve_funcs[name](res.x, n_points=400)
        ax.plot(TARGET[:, 0], TARGET[:, 1], "k--", lw=1.5, label="Target stadium")
        if curve is not None:
            ax.plot(curve[:, 0], curve[:, 1], "r-", lw=1.2, label="Coupler curve")
        ax.set_title(f"{name}\nerror = {res.fun:.6f}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Approach B — Linkage Synthesis for Stadium Curve", fontsize=14)
    fig.tight_layout()
    out_path = "/home/sman/Work/CMU/Research/track_synthesis/approach_b_result.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Figure saved to {out_path}")

    # --- GIF animations ---
    base = "/home/sman/Work/CMU/Research/track_synthesis/"

    topo_map = {
        "Four-bar": "four_bar",
        "Slider-crank": "slider_crank",
        "Six-bar Watt-I": "six_bar",
    }
    gif_names = {
        "Four-bar": "approach_b_four_bar.gif",
        "Slider-crank": "approach_b_slider_crank.gif",
        "Six-bar Watt-I": "approach_b_six_bar.gif",
    }

    print(f"\n{'='*60}")
    print("  Generating GIF animations ...")
    print(f"{'='*60}")

    for name, res in results.items():
        gif_path = base + gif_names[name]
        print(f"\n  {name} -> {gif_path}")
        animate_mechanism(res.x, topo_map[name], gif_path,
                          n_frames=100, fps=8)

    # Best overall
    best_gif = base + "approach_b_result.gif"
    print(f"\n  Best ({best_name}) -> {best_gif}")
    animate_mechanism(results[best_name].x, topo_map[best_name], best_gif,
                      n_frames=100, fps=8)

    print("\n  Done!")


if __name__ == "__main__":
    main()
