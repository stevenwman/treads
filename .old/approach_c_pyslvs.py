#!/usr/bin/env python3
"""
Approach C: Pyslvs-inspired linkage synthesis for stadium curve.

Since pyslvs fails to compile on Python 3.12, this implements the same concepts
from scratch:
  - Multiple topologies: four-bar, six-bar Watt-I, six-bar Stephenson-III
  - Differential evolution + local refinement (scipy)
  - Mechanism expression representation inspired by Pyslvs/PMKS
  - Bidirectional curve error metric from stadium.py

Target: stadium with half_length=2.0, radius=0.8, centered at origin.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import os

from stadium import stadium_points, curve_error, TARGET

# ---------------------------------------------------------------------------
# Vectorized error metric (much faster than the loop version in stadium.py)
# ---------------------------------------------------------------------------

def fast_curve_error(coupler_curve, target_curve):
    """Bidirectional mean-of-min-squared-distances using cdist."""
    if coupler_curve is None or len(coupler_curve) < 10:
        return 1e12
    D = cdist(coupler_curve, target_curve, metric="sqeuclidean")
    forward = np.mean(np.min(D, axis=1))
    backward = np.mean(np.min(D, axis=0))
    return (forward + backward) / 2.0


# ---------------------------------------------------------------------------
# Forward kinematics: Four-bar linkage
# ---------------------------------------------------------------------------

def four_bar_coupler_curve(params, n_points=200):
    """
    Four-bar linkage coupler curve.
    params: [a, b, c, d, px, py, gx, gy]
      a=crank, b=coupler, c=rocker, d=ground link
      (px,py)=coupler point offset in coupler frame
      (gx,gy)=left ground pivot O2;  O4 = (gx+d, gy)
    """
    a, b, c, d, px, py, gx, gy = params

    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return None

    # Grashof check: shortest + longest <= sum of other two
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2x, O2y = gx, gy
    O4x, O4y = gx + d, gy

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2x + a * np.cos(theta2)
    Ay = O2y + a * np.sin(theta2)

    dx = O4x - Ax
    dy = O4y - Ay
    dist = np.sqrt(dx**2 + dy**2)

    # Check triangle inequality for all positions
    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = (b**2 + dist**2 - c**2) / (2 * b * dist)
    cos_alpha = np.clip(cos_alpha, -1, 1)

    ang_to_O4 = np.arctan2(dy, dx)
    alpha = np.arccos(cos_alpha)
    theta3 = ang_to_O4 + alpha

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    cpx = Ax + px * cos3 - py * sin3
    cpy = Ay + px * sin3 + py * cos3

    return np.column_stack([cpx, cpy])


# ---------------------------------------------------------------------------
# Forward kinematics: Six-bar Watt-I
# ---------------------------------------------------------------------------

def six_bar_watt_coupler_curve(params, n_points=200):
    """
    Watt-I six-bar: base four-bar + dyad (e, f) attached to coupler.
    params: [a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy]
    Tracing point is (px2, py2) on the coupler of the base four-bar.
    The dyad constrains coupler motion via link e from (px1,py1) to ground O6 through link f.
    """
    a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy = params

    if a <= 0 or b <= 0 or c <= 0 or d <= 0 or e <= 0 or f <= 0:
        return None

    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2x, O2y = gx, gy
    O4x, O4y = gx + d, gy

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2x + a * np.cos(theta2)
    Ay = O2y + a * np.sin(theta2)

    dx = O4x - Ax
    dy = O4y - Ay
    dist = np.sqrt(dx**2 + dy**2)

    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = np.clip((b**2 + dist**2 - c**2) / (2 * b * dist), -1, 1)
    ang_to_O4 = np.arctan2(dy, dx)
    theta3 = ang_to_O4 + np.arccos(cos_alpha)

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    # Point P1 on coupler (dyad attachment)
    P1x = Ax + px1 * cos3 - py1 * sin3
    P1y = Ay + px1 * sin3 + py1 * cos3

    # Dyad: link e from P1 to D, link f from D to O6
    dx2 = O6x - P1x
    dy2 = O6y - P1y
    dist2 = np.sqrt(dx2**2 + dy2**2)

    if np.any(dist2 > e + f) or np.any(dist2 < abs(e - f)):
        return None

    # Coupler tracing point on base four-bar coupler
    cpx = Ax + px2 * cos3 - py2 * sin3
    cpy = Ay + px2 * sin3 + py2 * cos3

    return np.column_stack([cpx, cpy])


# ---------------------------------------------------------------------------
# Forward kinematics: Six-bar Stephenson-III
# ---------------------------------------------------------------------------

def six_bar_stephenson_coupler_curve(params, n_points=200):
    """
    Stephenson-III six-bar: base four-bar whose rocker drives a second dyad.
    params: [a, b, c, d, e, f, px, py, O6x, O6y, gx, gy]

    Base four-bar: O2-A-B-O4 with links a,b,c,d.
    Second dyad: link e from B to C (tracing point), link f from C to O6.
    (px, py) is the offset of the tracing point on link e, in the frame of e.
    """
    a, b, c, d, e, f, px, py, O6x, O6y, gx, gy = params

    if a <= 0 or b <= 0 or c <= 0 or d <= 0 or e <= 0 or f <= 0:
        return None

    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2x, O2y = gx, gy
    O4x, O4y = gx + d, gy

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2x + a * np.cos(theta2)
    Ay = O2y + a * np.sin(theta2)

    dx = O4x - Ax
    dy = O4y - Ay
    dist = np.sqrt(dx**2 + dy**2)

    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = np.clip((b**2 + dist**2 - c**2) / (2 * b * dist), -1, 1)
    ang_to_O4 = np.arctan2(dy, dx)
    theta3 = ang_to_O4 + np.arccos(cos_alpha)

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    # B = end of coupler on rocker side
    Bx = Ax + b * cos3
    By = Ay + b * sin3

    # Second dyad: e from B to C, f from C to O6
    dx2 = O6x - Bx
    dy2 = O6y - By
    dist2 = np.sqrt(dx2**2 + dy2**2)

    if np.any(dist2 > e + f) or np.any(dist2 < abs(e - f)):
        return None

    cos_beta = np.clip((e**2 + dist2**2 - f**2) / (2 * e * dist2), -1, 1)
    ang2 = np.arctan2(dy2, dx2)
    theta5 = ang2 + np.arccos(cos_beta)

    cos5 = np.cos(theta5)
    sin5 = np.sin(theta5)

    # Tracing point on link e at offset (px, py)
    cpx = Bx + px * cos5 - py * sin5
    cpy = By + px * sin5 + py * cos5

    return np.column_stack([cpx, cpy])


# ---------------------------------------------------------------------------
# Joint-position functions (return ALL joint positions at each crank angle)
# ---------------------------------------------------------------------------

def four_bar_joint_positions(params, n_points=100):
    """
    Return joint positions for all crank angles.
    Returns dict with arrays of shape (n_points,2) for each joint/point,
    or None if the linkage is invalid.
    Keys: O2, O4, A, B, P  (ground pivots, crank tip, coupler-rocker, coupler point)
    """
    a, b, c, d, px, py, gx, gy = params
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return None
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2[0] + a * np.cos(theta2)
    Ay = O2[1] + a * np.sin(theta2)

    dx = O4[0] - Ax
    dy = O4[1] - Ay
    dist = np.sqrt(dx**2 + dy**2)

    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = np.clip((b**2 + dist**2 - c**2) / (2 * b * dist), -1, 1)
    ang_to_O4 = np.arctan2(dy, dx)
    theta3 = ang_to_O4 + np.arccos(cos_alpha)

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    Bx = Ax + b * cos3
    By = Ay + b * sin3
    Px = Ax + px * cos3 - py * sin3
    Py = Ay + px * sin3 + py * cos3

    return {
        "O2": np.tile(O2, (n_points, 1)),
        "O4": np.tile(O4, (n_points, 1)),
        "A": np.column_stack([Ax, Ay]),
        "B": np.column_stack([Bx, By]),
        "P": np.column_stack([Px, Py]),
    }


def six_bar_watt_joint_positions(params, n_points=100):
    """
    Return joint positions for Watt-I six-bar at each crank angle.
    Keys: O2, O4, O6, A, B, P1, D, P
    """
    a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy = params
    if a <= 0 or b <= 0 or c <= 0 or d <= 0 or e <= 0 or f <= 0:
        return None
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])
    O6 = np.array([O6x, O6y])

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2[0] + a * np.cos(theta2)
    Ay = O2[1] + a * np.sin(theta2)

    dxv = O4[0] - Ax
    dyv = O4[1] - Ay
    dist = np.sqrt(dxv**2 + dyv**2)
    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = np.clip((b**2 + dist**2 - c**2) / (2 * b * dist), -1, 1)
    ang_to_O4 = np.arctan2(dyv, dxv)
    theta3 = ang_to_O4 + np.arccos(cos_alpha)

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    Bx = Ax + b * cos3
    By = Ay + b * sin3

    # P1 on coupler (dyad attachment)
    P1x = Ax + px1 * cos3 - py1 * sin3
    P1y = Ay + px1 * sin3 + py1 * cos3

    # Dyad: link e from P1 to D, link f from D to O6
    dx2 = O6[0] - P1x
    dy2 = O6[1] - P1y
    dist2 = np.sqrt(dx2**2 + dy2**2)
    if np.any(dist2 > e + f) or np.any(dist2 < abs(e - f)):
        return None

    cos_beta = np.clip((e**2 + dist2**2 - f**2) / (2 * e * dist2), -1, 1)
    ang2 = np.arctan2(dy2, dx2)
    theta5 = ang2 + np.arccos(cos_beta)

    Dx = P1x + e * np.cos(theta5)
    Dy = P1y + e * np.sin(theta5)

    # Coupler tracing point (px2, py2) on base four-bar coupler
    Px = Ax + px2 * cos3 - py2 * sin3
    Py = Ay + px2 * sin3 + py2 * cos3

    return {
        "O2": np.tile(O2, (n_points, 1)),
        "O4": np.tile(O4, (n_points, 1)),
        "O6": np.tile(O6, (n_points, 1)),
        "A": np.column_stack([Ax, Ay]),
        "B": np.column_stack([Bx, By]),
        "P1": np.column_stack([P1x, P1y]),
        "D": np.column_stack([Dx, Dy]),
        "P": np.column_stack([Px, Py]),
    }


def six_bar_stephenson_joint_positions(params, n_points=100):
    """
    Return joint positions for Stephenson-III six-bar at each crank angle.
    Keys: O2, O4, O6, A, B, C, P
    """
    a, b, c, d, e, f, px, py, O6x, O6y, gx, gy = params
    if a <= 0 or b <= 0 or c <= 0 or d <= 0 or e <= 0 or f <= 0:
        return None
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])
    O6 = np.array([O6x, O6y])

    theta2 = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    Ax = O2[0] + a * np.cos(theta2)
    Ay = O2[1] + a * np.sin(theta2)

    dxv = O4[0] - Ax
    dyv = O4[1] - Ay
    dist = np.sqrt(dxv**2 + dyv**2)
    if np.any(dist > b + c) or np.any(dist < abs(b - c)):
        return None

    cos_alpha = np.clip((b**2 + dist**2 - c**2) / (2 * b * dist), -1, 1)
    ang_to_O4 = np.arctan2(dyv, dxv)
    theta3 = ang_to_O4 + np.arccos(cos_alpha)

    cos3 = np.cos(theta3)
    sin3 = np.sin(theta3)

    Bx = Ax + b * cos3
    By = Ay + b * sin3

    # Second dyad: e from B to C, f from C to O6
    dx2 = O6[0] - Bx
    dy2 = O6[1] - By
    dist2 = np.sqrt(dx2**2 + dy2**2)
    if np.any(dist2 > e + f) or np.any(dist2 < abs(e - f)):
        return None

    cos_beta = np.clip((e**2 + dist2**2 - f**2) / (2 * e * dist2), -1, 1)
    ang2 = np.arctan2(dy2, dx2)
    theta5 = ang2 + np.arccos(cos_beta)

    cos5 = np.cos(theta5)
    sin5 = np.sin(theta5)

    Cx = Bx + e * cos5
    Cy = By + e * sin5

    # Tracing point on link e
    Px = Bx + px * cos5 - py * sin5
    Py = By + px * sin5 + py * cos5

    return {
        "O2": np.tile(O2, (n_points, 1)),
        "O4": np.tile(O4, (n_points, 1)),
        "O6": np.tile(O6, (n_points, 1)),
        "A": np.column_stack([Ax, Ay]),
        "B": np.column_stack([Bx, By]),
        "C": np.column_stack([Cx, Cy]),
        "P": np.column_stack([Px, Py]),
    }


# Map topology name -> joint position function
JOINT_FUNCS = {
    "four_bar": four_bar_joint_positions,
    "six_bar_watt": six_bar_watt_joint_positions,
    "six_bar_stephenson": six_bar_stephenson_joint_positions,
}


# ---------------------------------------------------------------------------
# GIF animation generation
# ---------------------------------------------------------------------------

def generate_linkage_gif(topo_name, params, filename, n_frames=100, fps=8):
    """
    Generate a GIF animation showing the linkage mechanism in motion.
    Shows: target stadium (light blue), linkage links, joints, and coupler trace.
    """
    joint_func = JOINT_FUNCS[topo_name]
    joints = joint_func(params, n_points=n_frames)
    if joints is None:
        print(f"  [GIF] Cannot generate GIF for {topo_name}: invalid linkage")
        return

    # Compute axis limits from all joint positions
    all_x, all_y = [], []
    for key, arr in joints.items():
        all_x.append(arr[:, 0])
        all_y.append(arr[:, 1])
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    # Also include target
    all_x = np.concatenate([all_x, TARGET[:, 0]])
    all_y = np.concatenate([all_y, TARGET[:, 1]])
    margin = 0.5
    xlim = (all_x.min() - margin, all_x.max() + margin)
    ylim = (all_y.min() - margin, all_y.max() + margin)

    fig, ax = plt.subplots(figsize=(8, 6))

    def draw_frame(i):
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{topo_name}  (frame {i+1}/{n_frames})", fontsize=11)

        # Target stadium curve (light blue, background)
        tgt_closed = np.vstack([TARGET, TARGET[0]])
        ax.plot(tgt_closed[:, 0], tgt_closed[:, 1], color="lightskyblue",
                lw=2.5, zorder=1, label="Target stadium")

        # Coupler trace built up to this frame
        P = joints["P"]
        trail = np.vstack([P[:i+1], P[0:1]]) if i == n_frames - 1 else P[:i+1]
        ax.plot(trail[:, 0], trail[:, 1], "r-", lw=1.0, alpha=0.7, zorder=2,
                label="Coupler trace")

        # -- Draw linkage at frame i --
        link_color = "dimgray"
        link_lw = 2.5

        if topo_name == "four_bar":
            O2 = joints["O2"][i]
            O4 = joints["O4"][i]
            A = joints["A"][i]
            B = joints["B"][i]
            Pt = joints["P"][i]
            # Ground link (dashed)
            ax.plot([O2[0], O4[0]], [O2[1], O4[1]], "--",
                    color="gray", lw=1.5, zorder=3)
            # Crank O2-A
            ax.plot([O2[0], A[0]], [O2[1], A[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Coupler A-B
            ax.plot([A[0], B[0]], [A[1], B[1]], "-",
                    color="steelblue", lw=link_lw, zorder=4)
            # Rocker B-O4
            ax.plot([B[0], O4[0]], [B[1], O4[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Coupler point attachment A-P and B-P (thin)
            ax.plot([A[0], Pt[0]], [A[1], Pt[1]], "-",
                    color="steelblue", lw=1.2, alpha=0.5, zorder=3)
            ax.plot([B[0], Pt[0]], [B[1], Pt[1]], "-",
                    color="steelblue", lw=1.2, alpha=0.5, zorder=3)
            # Ground pivots (triangles)
            ax.plot(*O2, "^", color="darkgreen", ms=10, zorder=6)
            ax.plot(*O4, "^", color="darkgreen", ms=10, zorder=6)
            # Moving joints (circles)
            ax.plot(*A, "o", color="navy", ms=7, zorder=6)
            ax.plot(*B, "o", color="navy", ms=7, zorder=6)
            # Coupler tracing point (red dot)
            ax.plot(*Pt, "o", color="red", ms=8, zorder=7)

        elif topo_name == "six_bar_watt":
            O2 = joints["O2"][i]
            O4 = joints["O4"][i]
            O6 = joints["O6"][i]
            A = joints["A"][i]
            B = joints["B"][i]
            P1 = joints["P1"][i]
            D = joints["D"][i]
            Pt = joints["P"][i]
            # Ground links (dashed)
            ax.plot([O2[0], O4[0]], [O2[1], O4[1]], "--",
                    color="gray", lw=1.5, zorder=3)
            # Crank O2-A
            ax.plot([O2[0], A[0]], [O2[1], A[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Coupler A-B
            ax.plot([A[0], B[0]], [A[1], B[1]], "-",
                    color="steelblue", lw=link_lw, zorder=4)
            # Rocker B-O4
            ax.plot([B[0], O4[0]], [B[1], O4[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Coupler extension to P1
            ax.plot([A[0], P1[0]], [A[1], P1[1]], "-",
                    color="steelblue", lw=1.5, alpha=0.6, zorder=3)
            ax.plot([B[0], P1[0]], [B[1], P1[1]], "-",
                    color="steelblue", lw=1.5, alpha=0.6, zorder=3)
            # Dyad: P1-D, D-O6
            ax.plot([P1[0], D[0]], [P1[1], D[1]], "-",
                    color="darkorange", lw=link_lw, zorder=4)
            ax.plot([D[0], O6[0]], [D[1], O6[1]], "-",
                    color="darkorange", lw=link_lw, zorder=4)
            # Ground pivots
            ax.plot(*O2, "^", color="darkgreen", ms=10, zorder=6)
            ax.plot(*O4, "^", color="darkgreen", ms=10, zorder=6)
            ax.plot(*O6, "^", color="darkgreen", ms=10, zorder=6)
            # Moving joints
            ax.plot(*A, "o", color="navy", ms=7, zorder=6)
            ax.plot(*B, "o", color="navy", ms=7, zorder=6)
            ax.plot(*P1, "o", color="navy", ms=6, zorder=6)
            ax.plot(*D, "o", color="darkorange", ms=7, zorder=6)
            # Coupler tracing point
            ax.plot(*Pt, "o", color="red", ms=8, zorder=7)

        elif topo_name == "six_bar_stephenson":
            O2 = joints["O2"][i]
            O4 = joints["O4"][i]
            O6 = joints["O6"][i]
            A = joints["A"][i]
            B = joints["B"][i]
            C = joints["C"][i]
            Pt = joints["P"][i]
            # Ground links (dashed)
            ax.plot([O2[0], O4[0]], [O2[1], O4[1]], "--",
                    color="gray", lw=1.5, zorder=3)
            # Crank O2-A
            ax.plot([O2[0], A[0]], [O2[1], A[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Coupler A-B
            ax.plot([A[0], B[0]], [A[1], B[1]], "-",
                    color="steelblue", lw=link_lw, zorder=4)
            # Rocker B-O4
            ax.plot([B[0], O4[0]], [B[1], O4[1]], "-",
                    color=link_color, lw=link_lw, zorder=4)
            # Second dyad: B-C (link e), C-O6 (link f)
            ax.plot([B[0], C[0]], [B[1], C[1]], "-",
                    color="darkorange", lw=link_lw, zorder=4)
            ax.plot([C[0], O6[0]], [C[1], O6[1]], "-",
                    color="darkorange", lw=link_lw, zorder=4)
            # Coupler point on link e (thin connection)
            ax.plot([B[0], Pt[0]], [B[1], Pt[1]], "-",
                    color="darkorange", lw=1.2, alpha=0.5, zorder=3)
            # Ground pivots
            ax.plot(*O2, "^", color="darkgreen", ms=10, zorder=6)
            ax.plot(*O4, "^", color="darkgreen", ms=10, zorder=6)
            ax.plot(*O6, "^", color="darkgreen", ms=10, zorder=6)
            # Moving joints
            ax.plot(*A, "o", color="navy", ms=7, zorder=6)
            ax.plot(*B, "o", color="navy", ms=7, zorder=6)
            ax.plot(*C, "o", color="darkorange", ms=7, zorder=6)
            # Coupler tracing point
            ax.plot(*Pt, "o", color="red", ms=8, zorder=7)

    # Build frames as images
    from PIL import Image
    import io
    frames = []
    for i in range(n_frames):
        draw_frame(i)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # Save GIF
    duration_ms = int(1000 / fps)
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  [GIF] Saved {filename}  ({n_frames} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# Topology definitions: name, kinematic function, bounds, param names
# ---------------------------------------------------------------------------

TOPOLOGIES = {
    "four_bar": {
        "func": four_bar_coupler_curve,
        "param_names": ["a", "b", "c", "d", "px", "py", "gx", "gy"],
        "bounds": [
            (0.3, 5.0),   # a (crank)
            (1.0, 8.0),   # b (coupler)
            (0.5, 6.0),   # c (rocker)
            (1.0, 8.0),   # d (ground)
            (-4.0, 4.0),  # px
            (-4.0, 4.0),  # py
            (-3.0, 3.0),  # gx
            (-3.0, 3.0),  # gy
        ],
    },
    "six_bar_watt": {
        "func": six_bar_watt_coupler_curve,
        "param_names": [
            "a", "b", "c", "d", "e", "f",
            "px1", "py1", "px2", "py2",
            "O6x", "O6y", "gx", "gy",
        ],
        "bounds": [
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
        ],
    },
    "six_bar_stephenson": {
        "func": six_bar_stephenson_coupler_curve,
        "param_names": [
            "a", "b", "c", "d", "e", "f",
            "px", "py", "O6x", "O6y", "gx", "gy",
        ],
        "bounds": [
            (0.3, 5.0),   # a
            (1.0, 8.0),   # b
            (0.5, 6.0),   # c
            (1.0, 8.0),   # d
            (0.5, 6.0),   # e
            (0.5, 6.0),   # f
            (-4.0, 4.0),  # px
            (-4.0, 4.0),  # py
            (-5.0, 5.0),  # O6x
            (-5.0, 5.0),  # O6y
            (-3.0, 3.0),  # gx
            (-3.0, 3.0),  # gy
        ],
    },
}


# ---------------------------------------------------------------------------
# Objective wrapper
# ---------------------------------------------------------------------------

def make_objective(topo_name):
    func = TOPOLOGIES[topo_name]["func"]

    def objective(x):
        curve = func(x, n_points=200)
        return fast_curve_error(curve, TARGET)

    return objective


# ---------------------------------------------------------------------------
# Multi-start optimisation for one topology
# ---------------------------------------------------------------------------

def optimize_topology(topo_name, n_starts=4, maxiter=300, popsize=25, seed0=42):
    """Run DE multiple times, then locally refine the best results."""
    topo = TOPOLOGIES[topo_name]
    objective = make_objective(topo_name)
    bounds = topo["bounds"]

    print(f"\n{'='*60}")
    print(f"  Topology: {topo_name}  ({len(bounds)} params, {n_starts} starts)")
    print(f"{'='*60}")

    results = []
    for i in range(n_starts):
        seed = seed0 + i * 7
        t0 = time.time()
        try:
            res = differential_evolution(
                objective,
                bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-9,
                seed=seed,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                updating="deferred",
                workers=1,
            )
            dt = time.time() - t0
            print(f"  start {i+1}/{n_starts}  seed={seed:4d}  "
                  f"error={res.fun:.6f}  iters={res.nit:4d}  {dt:.1f}s")
            results.append(res)
        except Exception as exc:
            print(f"  start {i+1}/{n_starts}  seed={seed:4d}  FAILED: {exc}")

    if not results:
        return None, float("inf"), None

    # Sort by error and locally refine top 3
    results.sort(key=lambda r: r.fun)
    best_error = float("inf")
    best_x = None
    best_curve = None

    for j, res in enumerate(results[:3]):
        try:
            ref = minimize(
                objective, res.x, method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 5000, "ftol": 1e-12},
            )
            curve = TOPOLOGIES[topo_name]["func"](ref.x, n_points=200)
            err = fast_curve_error(curve, TARGET)
            if err < best_error:
                best_error = err
                best_x = ref.x
                best_curve = curve
            print(f"  refined #{j+1}: {res.fun:.6f} -> {err:.6f}")
        except Exception:
            if res.fun < best_error:
                best_error = res.fun
                best_x = res.x
                best_curve = TOPOLOGIES[topo_name]["func"](res.x, n_points=200)

    print(f"  >> best error for {topo_name}: {best_error:.6f}")
    return best_x, best_error, best_curve


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_results(all_results, filename="approach_c_result.png"):
    """Plot target vs best coupler curve for each topology and overall best."""
    n_topos = len(all_results)
    fig, axes = plt.subplots(1, n_topos + 1, figsize=(6 * (n_topos + 1), 5))
    if n_topos + 1 == 1:
        axes = [axes]

    # Individual topology plots
    for idx, (name, info) in enumerate(all_results.items()):
        ax = axes[idx]
        ax.plot(TARGET[:, 0], TARGET[:, 1], "b-", lw=1.5, label="Target stadium")
        if info["curve"] is not None:
            c = info["curve"]
            # Close the curve for plotting
            c_closed = np.vstack([c, c[0]])
            ax.plot(c_closed[:, 0], c_closed[:, 1], "r--", lw=1.2,
                    label=f"Coupler (err={info['error']:.4f})")
        ax.set_title(name, fontsize=11)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Overall best
    best_name = min(all_results, key=lambda k: all_results[k]["error"])
    best = all_results[best_name]
    ax = axes[-1]
    ax.plot(TARGET[:, 0], TARGET[:, 1], "b-", lw=2, label="Target stadium")
    if best["curve"] is not None:
        c = best["curve"]
        c_closed = np.vstack([c, c[0]])
        ax.plot(c_closed[:, 0], c_closed[:, 1], "r-", lw=1.5,
                label=f"Best: {best_name}\nerror={best['error']:.4f}")
    ax.set_title("BEST OVERALL", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nPlot saved to {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Approach C: Pyslvs-inspired linkage synthesis (pure Python fallback)")
    print(f"Target: stadium  half_length=2.0  radius=0.8  ({len(TARGET)} pts)")
    print(f"Error metric: bidirectional mean-of-min-squared-distances")

    all_results = {}

    for topo_name in TOPOLOGIES:
        # Fewer starts / iters for six-bar to keep runtime reasonable
        if "six_bar" in topo_name:
            n_starts, maxiter, popsize = 3, 250, 30
        else:
            n_starts, maxiter, popsize = 5, 400, 30

        best_x, best_error, best_curve = optimize_topology(
            topo_name,
            n_starts=n_starts,
            maxiter=maxiter,
            popsize=popsize,
        )

        param_names = TOPOLOGIES[topo_name]["param_names"]
        param_dict = {}
        if best_x is not None:
            param_dict = {n: float(v) for n, v in zip(param_names, best_x)}

        all_results[topo_name] = {
            "params": param_dict,
            "error": best_error,
            "curve": best_curve,
        }

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, info in sorted(all_results.items(), key=lambda kv: kv[1]["error"]):
        print(f"  {name:30s}  error = {info['error']:.6f}")
        if info["params"]:
            for k, v in info["params"].items():
                print(f"    {k:6s} = {v:+.4f}")

    best_name = min(all_results, key=lambda k: all_results[k]["error"])
    best = all_results[best_name]
    print(f"\n  ** Best topology: {best_name}  error={best['error']:.6f} **")

    # Max deviation for best result
    if best["curve"] is not None:
        D = cdist(best["curve"], TARGET, metric="sqeuclidean")
        max_dev = np.sqrt(np.max(np.min(D, axis=1)))
        print(f"  ** Max point deviation: {max_dev:.4f} **")

    # Plot
    plot_results(all_results)

    # Generate GIF animations
    print("\nGenerating GIF animations...")

    # Individual topology GIFs
    for topo_name, info in all_results.items():
        if info["params"] and info["curve"] is not None:
            param_names = TOPOLOGIES[topo_name]["param_names"]
            param_array = np.array([info["params"][n] for n in param_names])
            gif_file = f"approach_c_{topo_name}.gif"
            generate_linkage_gif(topo_name, param_array, gif_file,
                                 n_frames=100, fps=8)

    # Overall best GIF
    if best["params"] and best["curve"] is not None:
        param_names = TOPOLOGIES[best_name]["param_names"]
        param_array = np.array([best["params"][n] for n in param_names])
        generate_linkage_gif(best_name, param_array, "approach_c_result.gif",
                             n_frames=100, fps=8)

    return all_results


if __name__ == "__main__":
    main()
