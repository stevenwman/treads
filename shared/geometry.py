"""Stadium-shaped track path geometry.

The chain follows a "stadium" (rectangle with semicircular ends):

       ←── top straight (links travel left) ───
      ╭──────────────────────────────────────────╮
      │ drive                              idler │
      │ sprocket                        sprocket │
      ╰──────────────────────────────────────────╯
       ──→ bottom straight (links travel right) ─

Coordinate system:
    X = forward (left-to-right on the diagram above)
    Z = up
    Y = lateral (into the page)

The path starts at the top-right (idler side) and goes counter-clockwise.
"""
import math


def normalize_angle(angle):
    """Wrap an angle to the range [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def stadium_point(arc_length, config):
    """Get position and tangent angle at a point along the stadium path.

    Args:
        arc_length: Distance along the path from the top-right corner.
                    Wraps around automatically.
        config:     TankConfig with sprocket_radius, half_span, perimeter.

    Returns:
        (x, z, angle) where x and z are relative to the sprocket center
        height, and angle is the tangent direction in the XZ plane.
    """
    R = config.sprocket_radius
    H = config.half_span
    P = config.perimeter

    s = arc_length % P
    straight = 2 * H
    arc = math.pi * R

    # --- Top straight: moving from idler (+X) toward drive (-X) ---
    if s < straight:
        return H - s, R, math.pi

    s -= straight

    # --- Drive arc: left semicircle, top to bottom ---
    if s < arc:
        theta = math.pi / 2 + s / R
        return -H + R * math.cos(theta), R * math.sin(theta), theta + math.pi / 2

    s -= arc

    # --- Bottom straight: moving from drive (-X) toward idler (+X) ---
    if s < straight:
        return -H + s, -R, 0.0

    s -= straight

    # --- Idler arc: right semicircle, bottom to top ---
    theta = -math.pi / 2 + s / R
    return H + R * math.cos(theta), R * math.sin(theta), theta + math.pi / 2
