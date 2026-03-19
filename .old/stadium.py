"""Stadium curve generation and error metrics for linkage synthesis."""

import numpy as np


def stadium_points(half_length, radius, n=200):
    """
    Generate n evenly-spaced points along a stadium curve.
    Stadium = two straight segments + two semicircular caps.
    Centered at origin.
    """
    flat = 2 * half_length
    arc = np.pi * radius
    perimeter = 2 * flat + 2 * arc
    points = []
    for i in range(n):
        s = (i / n) * perimeter
        if s < flat:
            x = -half_length + s
            y = -radius
        elif s < flat + arc:
            theta = (s - flat) / radius - np.pi / 2
            x = half_length + radius * np.cos(theta)
            y = radius * np.sin(theta)
        elif s < 2 * flat + arc:
            x = half_length - (s - flat - arc)
            y = radius
        else:
            theta = (s - 2 * flat - arc) / radius + np.pi / 2
            x = -half_length + radius * np.cos(theta)
            y = radius * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def curve_error(coupler_curve, target_curve):
    """
    Bidirectional sum-of-min-squared-distances.
    Measures how well two curves match regardless of parameterization.
    """
    if coupler_curve is None:
        return 1e12

    # Forward: for each coupler point, distance to nearest target point
    forward = 0.0
    for p in coupler_curve:
        dists = np.sum((target_curve - p) ** 2, axis=1)
        forward += np.min(dists)

    # Backward: for each target point, distance to nearest coupler point
    backward = 0.0
    for t in target_curve:
        dists = np.sum((coupler_curve - t) ** 2, axis=1)
        backward += np.min(dists)

    return (forward + backward) / (len(coupler_curve) + len(target_curve))


def fourier_descriptors(path, n_harmonics=15):
    """
    Compute elliptic Fourier descriptors for a closed curve.
    path: array of shape (N, 2)
    Returns: array of shape (n_harmonics, 4) — [an, bn, cn, dn] per harmonic
    """
    N = len(path)
    T = np.arange(N) / N * 2 * np.pi

    descriptors = []
    for k in range(1, n_harmonics + 1):
        an = (2 / N) * np.sum(path[:, 0] * np.cos(k * T))
        bn = (2 / N) * np.sum(path[:, 0] * np.sin(k * T))
        cn = (2 / N) * np.sum(path[:, 1] * np.cos(k * T))
        dn = (2 / N) * np.sum(path[:, 1] * np.sin(k * T))
        descriptors.append([an, bn, cn, dn])

    return np.array(descriptors)


# Default target for all optimizers
TARGET = stadium_points(half_length=2.0, radius=0.8, n=200)
