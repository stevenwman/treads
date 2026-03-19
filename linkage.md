# Linkage synthesis for stadium curve — implementation spec

## Goal

Synthesize a planar linkage mechanism whose coupler point traces a path as close as possible to a stadium curve (discorectangle). Try both four-bar and six-bar topologies, compare results, and output the best mechanism parameters. This will eventually be used in a MuJoCo tank track simulation.

## Context

MotionGen (Stony Brook University) is the state-of-the-art tool for this but is **NOT open source** — it's proprietary. However, the underlying math is fully published, and there are open-source alternatives.

## Recommended approach: optimization-based path synthesis

The most practical approach is:
1. Define a forward kinematic solver (given linkage params → trace coupler curve)
2. Define an error metric (distance between coupler curve and target stadium)
3. Wrap in a global optimizer (differential evolution or PSO)
4. Try both four-bar and six-bar topologies

This is the approach used by most published research and all open-source tools.

---

## Approach A: Use pylinkage (recommended starting point)

`pylinkage` is an open-source Python library purpose-built for this. Install with `pip install pylinkage`.

### How pylinkage works

- You define joints (Crank, Revolute, Prismatic, etc.) and arrange them into a Linkage
- You define a fitness function that scores how well the coupler traces your target
- You call `particle_swarm_optimization()` which varies the link dimensions to minimize your fitness
- The optimization parameters are the "geometric constraints" (link lengths, pivot positions)

### Four-bar example structure

```python
import pylinkage as pl
import numpy as np

# Define joints
crank = pl.Crank(
    0, 1,                    # joint IDs for the two links it connects
    joint0=(0, 0),           # position of ground pivot
    angle=0.0,               # starting angle
    distance=1.0             # crank length
)

# Coupler point + closing joint
pin = pl.Revolute(
    3, 2,                    # joint IDs
    joint0=crank,            # connected to crank output
    joint1=(3, 0),           # connected to ground pivot at (3,0)
    distance0=3,             # coupler length
    distance1=1              # rocker length
)

my_linkage = pl.Linkage(
    joints=(crank, pin),
    order=(crank, pin)       # evaluation order
)
```

### Fitness function for stadium matching

```python
def stadium_points(half_length, radius, n=200):
    """Generate n points along a stadium curve."""
    points = []
    flat = 2 * half_length
    arc = np.pi * radius
    perimeter = 2 * flat + 2 * arc
    for i in range(n):
        s = (i / n) * perimeter
        if s < flat:
            points.append((-half_length + s, -radius))
        elif s < flat + arc:
            theta = (s - flat) / radius - np.pi / 2
            points.append((half_length + radius * np.cos(theta), radius * np.sin(theta)))
        elif s < 2 * flat + arc:
            points.append((half_length - (s - flat - arc), radius))
        else:
            theta = (s - 2 * flat - arc) / radius + np.pi / 2
            points.append((-half_length + radius * np.cos(theta), radius * np.sin(theta)))
    return np.array(points)

TARGET = stadium_points(half_length=2.0, radius=0.8, n=200)

init_pos = my_linkage.get_coords()

@pl.kinematic_minimizastion
def fitness_func(loci, **_kwargs):
    """
    Minimize distance between coupler path and target stadium.
    loci is a list of (x, y) positions traced by each joint.
    """
    # Get the coupler point trajectory (last joint's path)
    coupler_path = np.array(loci[-1])  # shape (N, 2)
    if len(coupler_path) < 10:
        return float('inf')

    # For each coupler point, find distance to nearest stadium point
    total_error = 0.0
    for cp in coupler_path:
        dists = np.linalg.norm(TARGET - cp, axis=1)
        total_error += np.min(dists) ** 2

    # Also penalize if coupler path doesn't cover the full stadium
    # (check that the path spans a reasonable portion)
    for tp in TARGET[::10]:  # sample every 10th target point
        dists = np.linalg.norm(coupler_path - tp, axis=1)
        total_error += np.min(dists) ** 2

    return total_error

# Run optimization
bounds = pl.generate_bounds(my_linkage.get_num_constraints())
score, position, coord = pl.particle_swarm_optimization(
    eval_func=fitness_func,
    linkage=my_linkage,
    bounds=bounds,
    order_relation=min,
)[0]
```

### Limitations of pylinkage

- Primarily designed for four-bar linkages
- Six-bar requires manual joint definition (more complex but doable)
- PSO can be slow for high-dimensional problems
- No built-in support for prismatic (slider) joints in optimization (simulation only)

---

## Approach B: Pure scipy from scratch (more control, handles six-bar)

Build the forward kinematics and optimizer yourself. This is more work but gives full control over topology.

### Four-bar forward kinematics

A four-bar linkage with link lengths a (crank), b (coupler), c (rocker), d (ground), coupler point at offset (px, py) from the coupler line:

```python
import numpy as np
from scipy.optimize import fsolve, differential_evolution

def four_bar_coupler_curve(params, n_points=200):
    """
    params: [a, b, c, d, px, py, ground_x, ground_y]
      a = crank length
      b = coupler length
      c = rocker length
      d = ground length
      px, py = coupler point offset (in coupler-local frame)
      ground_x, ground_y = position of left ground pivot (O2)
    
    Returns array of (x, y) coupler point positions, or None if invalid.
    """
    a, b, c, d, px, py, gx, gy = params

    # Validate Grashof condition
    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None  # Non-Grashof, crank can't fully rotate

    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])

    points = []
    for i in range(n_points):
        theta2 = (i / n_points) * 2 * np.pi

        # Crank endpoint A
        Ax = O2[0] + a * np.cos(theta2)
        Ay = O2[1] + a * np.sin(theta2)

        # Solve for coupler angle theta3
        # Vector from A to O4
        dx = O4[0] - Ax
        dy = O4[1] - Ay
        dist = np.sqrt(dx**2 + dy**2)

        if dist > b + c or dist < abs(b - c):
            return None  # Can't close the loop at this position

        # Law of cosines for angle at A
        cos_alpha = (b**2 + dist**2 - c**2) / (2 * b * dist)
        if abs(cos_alpha) > 1:
            return None

        ang_to_O4 = np.atan2(dy, dx)
        alpha = np.acos(np.clip(cos_alpha, -1, 1))
        theta3 = ang_to_O4 + alpha  # Choose one assembly mode

        # Coupler point in world frame
        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)
        cpx = Ax + px * cos3 - py * sin3
        cpy = Ay + px * sin3 + py * cos3

        points.append([cpx, cpy])

    return np.array(points)
```

### Six-bar forward kinematics (Watt-I topology)

A Watt-I six-bar adds a dyad (two more links) to the coupler of a four-bar. The coupler of the four-bar carries a ternary link, and an additional binary link connects a point on this ternary link to a new ground pivot.

```python
def six_bar_watt_coupler_curve(params, n_points=200):
    """
    Watt-I six-bar linkage.
    
    params: [a, b, c, d,           # base four-bar
             e, f,                   # added dyad link lengths
             px1, py1,               # attachment point on coupler (for dyad)
             px2, py2,               # coupler tracing point
             O6x, O6y,              # new ground pivot for dyad
             gx, gy]                # base ground pivot O2
    
    The base four-bar is links a,b,c,d with ground at (gx,gy).
    Link e connects point (px1,py1) on the coupler to joint D.
    Link f connects joint D to ground pivot O6.
    The tracing point is (px2,py2) on the coupler (or on link e).
    """
    a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy = params

    lengths = sorted([a, b, c, d])
    if lengths[0] + lengths[3] > lengths[1] + lengths[2]:
        return None

    O2 = np.array([gx, gy])
    O4 = np.array([gx + d, gy])
    O6 = np.array([O6x, O6y])

    points = []
    for i in range(n_points):
        theta2 = (i / n_points) * 2 * np.pi

        # Solve base four-bar
        Ax = O2[0] + a * np.cos(theta2)
        Ay = O2[1] + a * np.sin(theta2)

        dx = O4[0] - Ax
        dy = O4[1] - Ay
        dist = np.sqrt(dx**2 + dy**2)

        if dist > b + c or dist < abs(b - c):
            return None

        cos_alpha = (b**2 + dist**2 - c**2) / (2 * b * dist)
        if abs(cos_alpha) > 1:
            return None

        ang = np.atan2(dy, dx)
        alpha = np.acos(np.clip(cos_alpha, -1, 1))
        theta3 = ang + alpha

        cos3 = np.cos(theta3)
        sin3 = np.sin(theta3)

        # Point P1 on coupler (attachment for dyad link e)
        P1x = Ax + px1 * cos3 - py1 * sin3
        P1y = Ay + px1 * sin3 + py1 * cos3

        # Solve the second dyad: link e from P1 to D, link f from D to O6
        dx2 = O6[0] - P1x
        dy2 = O6[1] - P1y
        dist2 = np.sqrt(dx2**2 + dy2**2)

        if dist2 > e + f or dist2 < abs(e - f):
            return None

        cos_beta = (e**2 + dist2**2 - f**2) / (2 * e * dist2)
        if abs(cos_beta) > 1:
            return None

        ang2 = np.atan2(dy2, dx2)
        beta = np.acos(np.clip(cos_beta, -1, 1))
        theta5 = ang2 + beta  # angle of link e

        # Coupler tracing point (on the coupler of the base four-bar)
        cpx = Ax + px2 * cos3 - py2 * sin3
        cpy = Ay + px2 * sin3 + py2 * cos3

        points.append([cpx, cpy])

    return np.array(points)
```

### Error metric

Use bidirectional Hausdorff-like distance for curve matching:

```python
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
        dists = np.sum((target_curve - p)**2, axis=1)
        forward += np.min(dists)

    # Backward: for each target point, distance to nearest coupler point
    backward = 0.0
    for t in target_curve:
        dists = np.sum((coupler_curve - t)**2, axis=1)
        backward += np.min(dists)

    return (forward + backward) / (len(coupler_curve) + len(target_curve))
```

### Optimization with differential evolution

```python
def objective_four_bar(x):
    """x = [a, b, c, d, px, py, gx, gy]"""
    curve = four_bar_coupler_curve(x, n_points=200)
    return curve_error(curve, TARGET)

def objective_six_bar(x):
    """x = [a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy]"""
    curve = six_bar_watt_coupler_curve(x, n_points=200)
    return curve_error(curve, TARGET)

# Four-bar bounds: [a, b, c, d, px, py, gx, gy]
bounds_4bar = [
    (0.3, 5.0),   # a (crank)
    (1.0, 8.0),   # b (coupler)
    (0.5, 6.0),   # c (rocker)
    (1.0, 8.0),   # d (ground)
    (-4.0, 4.0),  # px (coupler point x offset)
    (-4.0, 4.0),  # py (coupler point y offset)
    (-3.0, 3.0),  # gx (ground pivot x)
    (-3.0, 3.0),  # gy (ground pivot y)
]

# Six-bar bounds: [a, b, c, d, e, f, px1, py1, px2, py2, O6x, O6y, gx, gy]
bounds_6bar = [
    (0.3, 5.0),   # a
    (1.0, 8.0),   # b
    (0.5, 6.0),   # c
    (1.0, 8.0),   # d
    (0.5, 6.0),   # e (dyad link 1)
    (0.5, 6.0),   # f (dyad link 2)
    (-4.0, 4.0),  # px1 (dyad attachment x)
    (-4.0, 4.0),  # py1 (dyad attachment y)
    (-4.0, 4.0),  # px2 (tracing point x)
    (-4.0, 4.0),  # py2 (tracing point y)
    (-5.0, 5.0),  # O6x (new ground pivot x)
    (-5.0, 5.0),  # O6y (new ground pivot y)
    (-3.0, 3.0),  # gx
    (-3.0, 3.0),  # gy
]

# Run four-bar optimization
result_4bar = differential_evolution(
    objective_four_bar,
    bounds_4bar,
    maxiter=500,
    popsize=30,
    tol=1e-8,
    seed=42,
    disp=True,
    workers=-1,  # parallel
)

# Run six-bar optimization
result_6bar = differential_evolution(
    objective_six_bar,
    bounds_6bar,
    maxiter=1000,
    popsize=50,  # larger pop for more dimensions
    tol=1e-8,
    seed=42,
    disp=True,
    workers=-1,
)

print(f"Four-bar best error: {result_4bar.fun:.6f}")
print(f"Six-bar best error: {result_6bar.fun:.6f}")
```

---

## Approach C: Use Pyslvs (open source, full synthesis pipeline)

Pyslvs is an open-source planar linkage synthesis system. Install: `pip install pyslvs`

It has built-in:
- Mechanism expression parser
- Forward kinematics solver (based on Sketch Solve / Solvespace)
- Structural synthesis (graph enumeration of valid topologies)
- Dimensional synthesis with DE, PSO, firefly, and RGA algorithms
- Support for revolute AND prismatic joints

### Pyslvs dimensional synthesis for path generation

```python
from pyslvs import (
    parse_vpoints, t_config, expr_solving, example_list,
    FPlanar, norm_path
)
from pyslvs.metaheuristics import AlgorithmType, algorithm, default

# Define your target path as list of (x, y) tuples
target_path = [(x, y) for x, y in stadium_points(2.0, 0.8, 60)]

# Pyslvs uses mechanism "expressions" in PMKS format
# Example four-bar: "M[J[R, color[Green], P[0.0, 0.0], L[ground, L1]],
#   J[R, color[Green], P[d, 0.0], L[ground, L3]],
#   J[R, color[Green], P[ax, ay], L[L1, L2]],
#   J[R, color[Green], P[bx, by], L[L2, L3]]]"
# The synthesis system will find the right dimensions.

# See Pyslvs docs for setting up dimensional synthesis:
# https://pyslvs-ui.readthedocs.io/en/stable/synthesis.html
```

The Pyslvs GUI is actually the easiest way to do this interactively — it lets you:
1. Define or import a target path
2. Choose a mechanism topology (four-bar, six-bar Watt, six-bar Stephenson)
3. Set bounds on link lengths
4. Run optimization with choice of algorithm
5. Visualize results and compare

---

## Approach D: Fourier descriptor matching (atlas-based)

An alternative to optimization: decompose both the target stadium and candidate coupler curves into Fourier descriptors, then match in frequency space. This is what four-bar-rs (Rust tool) does.

### Fourier descriptor basics

```python
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
        an = (2/N) * np.sum(path[:, 0] * np.cos(k * T))
        bn = (2/N) * np.sum(path[:, 0] * np.sin(k * T))
        cn = (2/N) * np.sum(path[:, 1] * np.cos(k * T))
        dn = (2/N) * np.sum(path[:, 1] * np.sin(k * T))
        descriptors.append([an, bn, cn, dn])

    return np.array(descriptors)

def fourier_distance(fd1, fd2):
    """L2 distance between two sets of Fourier descriptors."""
    return np.sqrt(np.sum((fd1 - fd2)**2))
```

This can be used as the error metric instead of point-wise distance. Advantage: invariant to parameterization, rotation, and scale. Used by the state-of-the-art papers.

---

## Stadium curve definition

```python
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
```

---

## Strategy for getting the best stadium approximation

### Step 1: Four-bar baseline

Run `differential_evolution` on the four-bar objective with the stadium target. This will give a baseline "best smooth approximation." Expect the result to be an oblong blob that bulges at the straights and pinches at the arcs. Record the error.

### Step 2: Slider-crank variant

Replace the four-bar with a slider-crank (one revolute joint becomes a prismatic joint). The forward kinematics changes — the piston follows a straight line, which might help with the straight segments of the stadium. The slider-crank has fewer free parameters (7 instead of 8) but a different curve family.

```python
def slider_crank_coupler_curve(params, n_points=200):
    """
    params: [a, b, px, py, slider_angle, offset, gx, gy]
      a = crank length
      b = connecting rod length
      px, py = coupler point offset
      slider_angle = angle of slider axis (0 = horizontal)
      offset = perpendicular offset of slider from crank pivot
      gx, gy = crank pivot position
    """
    a, b, px, py, slider_angle, offset, gx, gy = params
    sa, ca = np.sin(slider_angle), np.cos(slider_angle)

    points = []
    for i in range(n_points):
        theta = (i / n_points) * 2 * np.pi
        Ax = gx + a * np.cos(theta)
        Ay = gy + a * np.sin(theta)

        # Slider position: project A onto slider axis, solve for B
        # B is on the slider line at distance b from A
        # Slider line: point (gx + offset * sin(sa), gy - offset * cos(sa))
        # direction (ca, sa)
        
        # Distance from A to slider line
        d_perp = (Ax - gx) * (-sa) + (Ay - gy) * ca - offset
        d_along_sq = b**2 - d_perp**2
        if d_along_sq < 0:
            return None
        d_along = np.sqrt(d_along_sq)

        # B position on slider
        foot_x = Ax + d_perp * sa
        foot_y = Ay - d_perp * ca
        Bx = foot_x + d_along * ca
        By = foot_y + d_along * sa

        # Coupler angle
        theta3 = np.atan2(By - Ay, Bx - Ax)
        cos3, sin3 = np.cos(theta3), np.sin(theta3)

        cpx = Ax + px * cos3 - py * sin3
        cpy = Ay + px * sin3 + py * cos3
        points.append([cpx, cpy])

    return np.array(points)
```

### Step 3: Six-bar (Watt-I or Stephenson)

Run the six-bar optimizer. This has 14 parameters, so it needs more population and iterations. The six-bar coupler curve can be up to degree 14, which can approximate the stadium much more closely — especially the flat segments.

### Step 4: Six-bar with prismatic joint

A six-bar where one joint is prismatic (slider). The slider naturally produces straight-line motion, which is perfect for the flat runs of the stadium. This is probably the topology most likely to succeed.

### Step 5: Multi-start and refinement

For each topology:
1. Run differential evolution 10 times with different seeds
2. Take the top 5 results
3. Refine each with `scipy.optimize.minimize` (L-BFGS-B or Nelder-Mead) for local polishing
4. Pick the global best

### Step 6: Visualization and comparison

For each result, plot:
- The target stadium
- The best coupler curve overlaid
- The mechanism itself (links, pivots, coupler point)
- Error heatmap (deviation at each point around the curve)

---

## Output format

The final output should be:

```python
{
    "topology": "four_bar" | "slider_crank" | "six_bar_watt" | "six_bar_stephenson",
    "params": {...},        # all link lengths, pivot positions, etc.
    "error": float,         # bidirectional curve error
    "max_deviation": float, # worst-case point deviation
    "coupler_curve": [[x,y], ...],  # the actual traced curve
    "mechanism": {          # for visualization/MuJoCo export
        "ground_pivots": [[x,y], ...],
        "link_lengths": [a, b, c, ...],
        "coupler_point": [px, py],
    }
}
```

---

## Dependencies

```
pip install numpy scipy matplotlib pylinkage
```

Optional for Pyslvs GUI:
```
pip install pyslvs-ui
```

## Files to create

| File | Purpose |
|------|---------|
| `stadium.py` | Stadium curve generation and Fourier descriptors |
| `four_bar.py` | Four-bar forward kinematics and optimizer |
| `slider_crank.py` | Slider-crank forward kinematics and optimizer |
| `six_bar.py` | Six-bar (Watt-I) forward kinematics and optimizer |
| `optimize_all.py` | Run all topologies, compare, pick best |
| `visualize.py` | Plot results, overlay curves, show mechanisms |
| `export_mujoco.py` | Convert best linkage to MuJoCo XML |

## Key references

- Ge, Purwar et al. "A Task-Driven Approach to Unified Synthesis of Planar Four-Bar Linkages Using Algebraic Fitting of a Pencil of G-Manifolds" ASME JCISE 2017 — the G-manifold method (MotionGen's core algorithm)
- Bulatovic & Djordjevic "Optimal synthesis of a path generator six-bar linkage" JMST 2012 — six-bar path synthesis with DE
- pylinkage docs: https://hugofara.github.io/pylinkage/
- Pyslvs docs: https://pyslvs-ui.readthedocs.io/en/stable/
- four-bar-rs (Rust, Fourier descriptors): https://github.com/KmolYuan/four-bar-rs