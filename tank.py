"""
Tank with two tracked drives using dynamic equality constraint switching.

Two mirrored chain tracks (left/right) connected to a rigid hull.
Differential steering: different left/right track speeds to turn.
Ground plane for driving.

Usage:
    uv run python tank.py              # GUI viewer
    uv run python tank.py --debug      # headless diagnostics
    uv run python tank.py --record     # save GIF
"""

import argparse
import math
import numpy as np
import mujoco

# ── Track Parameters ────────────────────────────────────────────────────────

N_LINKS = 30
SPROCKET_R = 0.35         # sprocket pitch radius
HALF_SPAN = 1.4           # half distance between drive/idler centers (along X)
LINK_THICK = 0.02         # link box half-height (radial)
LINK_WIDTH = 0.10         # link box half-depth (lateral, along Y for the tank)
TIMESTEP = 0.002
TENSION_K = 80.0
TARGET_VEL = 1.0          # rad/s target sprocket speed

HUB_R = SPROCKET_R - 0.04
HUB_Z = 0.12             # hub half-length (lateral)

PERIMETER = 2 * math.pi * SPROCKET_R + 2 * (2 * HALF_SPAN)
LINK_PITCH = PERIMETER / N_LINKS

# Tank geometry
TRACK_GAUGE = 1.2         # distance between left and right track centers (Y)
HULL_HALF_X = HALF_SPAN + 0.2
HULL_HALF_Y = TRACK_GAUGE / 2 - 0.05
HULL_HALF_Z = 0.15        # hull thickness
HULL_Z_OFFSET = 0.0       # hull center height above sprocket center

# Sprocket layout per track: (local_name, x_offset, z_offset_in_track_frame)
# In the tank frame: X = forward, Y = lateral, Z = up
# In track frame: the stadium is in the XZ plane (X = forward, Z = up)
SPROCKET_DEFS = [
    ("drive", -HALF_SPAN),
    ("idler",  HALF_SPAN),
    ("mid",    0.0),
]

# Constraint parameters
Z_OFF = LINK_WIDTH * 0.5
ARC_HALF = math.pi * 0.55


def _normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def stadium_point(s):
    """Return (x, z, angle) for arc-length s along the stadium in XZ plane.
    X = forward, Z = up.
    """
    s = s % PERIMETER
    top_len = 2 * HALF_SPAN
    arc_len = math.pi * SPROCKET_R

    if s < top_len:
        x = HALF_SPAN - s
        return x, SPROCKET_R, math.pi

    s -= top_len
    if s < arc_len:
        theta = math.pi / 2 + s / SPROCKET_R
        x = -HALF_SPAN + SPROCKET_R * math.cos(theta)
        z = SPROCKET_R * math.sin(theta)
        return x, z, theta + math.pi / 2

    s -= arc_len
    if s < top_len:
        x = -HALF_SPAN + s
        return x, -SPROCKET_R, 0.0

    s -= top_len
    theta = -math.pi / 2 + s / SPROCKET_R
    x = HALF_SPAN + SPROCKET_R * math.cos(theta)
    z = SPROCKET_R * math.sin(theta)
    return x, z, theta + math.pi / 2


def build_xml():
    """Generate MJCF XML for the full tank."""
    L = []
    a = L.append

    a('<mujoco model="tank">')
    a(f'  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"'
      f' iterations="500" solver="Newton" tolerance="1e-12" noslip_iterations="20"/>')
    a('  <size nconmax="2000" njmax="6000"/>')
    a('  <default>')
    a('    <geom friction="0.8 0.01 0.01" condim="4" margin="0.01"/>')
    a('    <equality solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    a('  </default>')
    a('  <worldbody>')
    a('    <camera name="overview" pos="0 -5 3" xyaxes="1 0 0 0 0.5 1" fovy="50"/>')
    a('    <camera name="top" pos="0 0 6" xyaxes="1 0 0 0 1 0" fovy="60"/>')
    a('    <light pos="0 -3 5" dir="0 0.5 -1" diffuse="1 1 1"/>')
    a('    <light pos="2 2 5" dir="-0.3 -0.3 -1" diffuse="0.5 0.5 0.5"/>')

    # Ground plane
    a('    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.4 0.5 0.4 1"'
      '     contype="1" conaffinity="3" pos="0 0 0"/>')

    # Hull — the main body of the tank, elevated so tracks clear the ground
    hull_z = SPROCKET_R + 0.15  # hull center height (extra clearance for initial settle)
    a(f'    <body name="hull" pos="0 0 {hull_z}">')
    a(f'      <freejoint name="hull_jnt"/>')
    a(f'      <geom name="hull_body" type="box"'
      f' size="{HULL_HALF_X} {HULL_HALF_Y} {HULL_HALF_Z}"'
      f' rgba="0.3 0.35 0.3 1" mass="20.0"'
      f' contype="0" conaffinity="0"/>')

    # Sprockets are children of the hull
    for side, y_sign in [("left", 1), ("right", -1)]:
        y_offset = y_sign * TRACK_GAUGE / 2

        for spr_name, x_off in SPROCKET_DEFS:
            full_name = f"{side}_{spr_name}"
            a(f'      <body name="{full_name}_sprocket" pos="{x_off} {y_offset} 0">')
            if spr_name == "idler":
                a(f'        <joint name="{full_name}_slide" type="slide" axis="1 0 0"'
                  f' stiffness="{TENSION_K}" damping="80" range="-0.05 0.3"/>')
            a(f'        <joint name="{full_name}_hinge" type="hinge" axis="0 1 0" damping="0.2"/>')
            # Hub — mid gets collision for chain support
            if spr_name == "mid":
                a(f'        <geom type="cylinder" size="{HUB_R} {HUB_Z}"'
                  f' euler="90 0 0" rgba="0.2 0.6 0.2 1"'
                  f' contype="0" conaffinity="0"/>')
            else:
                color = "0.7 0.2 0.2 1" if spr_name == "drive" else "0.2 0.2 0.7 1"
                a(f'        <geom type="cylinder" size="{HUB_R} {HUB_Z}"'
                  f' euler="90 0 0" rgba="{color}"'
                  f' contype="0" conaffinity="0"/>')
            # Engagement point markers — ring of spheres at pitch radius
            if spr_name != "mid":
                n_markers = 12
                for mi in range(n_markers):
                    theta = 2 * math.pi * mi / n_markers
                    mx = SPROCKET_R * math.cos(theta)
                    mz = SPROCKET_R * math.sin(theta)
                    a(f'        <geom type="sphere" size="0.012" pos="{mx:.4f} 0 {mz:.4f}"'
                      f' rgba="1 1 0 0.6" contype="0" conaffinity="0"/>')
            a(f'      </body>')

    a('    </body>')  # end hull

    # Chain links — kinematic tree per track
    # Link 0: freejoint root, links 1..N-1: children with hinge joints
    # Loop closure (link N-1 → link 0) via equality constraint
    half_len = LINK_PITCH / 2 - 0.005

    for side, y_sign in [("left", 1), ("right", -1)]:
        y_offset = y_sign * TRACK_GAUGE / 2

        # Link 0 — root with freejoint at its stadium position
        lx0, lz0, ang0 = stadium_point(0)
        qw0 = math.cos(ang0 / 2)
        qy0 = math.sin(ang0 / 2)
        a(f'    <body name="{side}_link_0" pos="{lx0:.6f} {y_offset:.6f} {hull_z + lz0:.6f}"'
          f' quat="{qw0:.6f} 0 {qy0:.6f} 0">')
        a(f'      <freejoint name="{side}_link_0_jnt"/>')
        a(f'      <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
          f' rgba="1.0 0.2 0.2 1" mass="0.05" contype="2" conaffinity="1"/>')
        a(f'      <geom type="sphere" size="0.015" pos="{LINK_PITCH/2:.4f} 0 0"'
          f' rgba="0 1 0 1" contype="0" conaffinity="0"/>')
        a(f'      <geom type="sphere" size="0.015" pos="{-LINK_PITCH/2:.4f} 0 0"'
          f' rgba="0 0 1 1" contype="0" conaffinity="0"/>')
        a(f'      <geom type="sphere" size="0.02" pos="0 0 0"'
          f' rgba="1 0 1 1" contype="0" conaffinity="0"/>')

        # Links 1..N-1 — nested children with hinge joints
        for i in range(1, N_LINKS):
            # Child body placed at LINK_PITCH along parent's X axis (= parent's fwd edge + child's half)
            # Hinge at child's backward edge (-LINK_PITCH/2, 0, 0)
            color = "0.9 0.6 0.1 1"
            a(f'      <body name="{side}_link_{i}" pos="{LINK_PITCH:.6f} 0 0">')
            a(f'        <joint name="{side}_hinge_{i}" type="hinge" axis="0 1 0"'
              f' pos="{-LINK_PITCH/2:.6f} 0 0" damping="0.05"/>')
            a(f'        <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
              f' rgba="{color}" mass="0.05" contype="2" conaffinity="1"/>')
            a(f'        <geom type="sphere" size="0.015" pos="{LINK_PITCH/2:.4f} 0 0"'
              f' rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
            a(f'        <geom type="sphere" size="0.015" pos="{-LINK_PITCH/2:.4f} 0 0"'
              f' rgba="0 0 1 0.5" contype="0" conaffinity="0"/>')
            a(f'        <geom type="sphere" size="0.02" pos="0 0 0"'
              f' rgba="1 0 1 0.5" contype="0" conaffinity="0"/>')

        # Close all nested bodies (N-1 closing tags for links 1..N-1, plus 1 for link 0)
        for i in range(N_LINKS):
            a(f'    </body>')

    a('  </worldbody>')

    # Actuators — velocity servos
    a('  <actuator>')
    for side in ["left", "right"]:
        for spr_name, _ in SPROCKET_DEFS:
            full_name = f"{side}_{spr_name}"
            a(f'    <velocity name="{full_name}_motor" joint="{full_name}_hinge"'
              f' kv="200" ctrllimited="true" ctrlrange="-5 5"/>')
    a('  </actuator>')

    # Equality constraints
    a('  <equality>')

    pin_z = LINK_WIDTH * 0.5
    z_off = LINK_WIDTH * 0.5

    for side in ["left", "right"]:
        # Loop closure: link_(N-1) fwd edge → link_0 bwd edge
        a(f'    <connect name="{side}_loop_close"'
          f' body1="{side}_link_{N_LINKS-1}" body2="{side}_link_0"'
          f' anchor="{LINK_PITCH/2:.6f} 0 0" active="true"/>')

        # Sprocket engagement constraints
        for i in range(N_LINKS):
            for spr_name, _ in SPROCKET_DEFS:
                full_name = f"{side}_{spr_name}"
                for s, z in [("L", -z_off), ("R", z_off)]:
                    a(f'    <connect name="{side}_eng_{i}_{spr_name}_{s}"'
                      f' body1="{full_name}_sprocket" body2="{side}_link_{i}"'
                      f' anchor="0 0 0"'
                      f' solref="0.005 1" solimp="0.95 0.99 0.001" active="false"/>')

    a('  </equality>')
    a('</mujoco>')
    return '\n'.join(L)


def init_lookups(model):
    """Build lookup tables for both tracks."""
    eng_ids = {}  # (side, link_i, spr_name, lr) -> eq_idx
    for idx in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if nm is None:
            continue
        for side in ["left", "right"]:
            prefix = f"{side}_eng_"
            if nm.startswith(prefix):
                rest = nm[len(prefix):]  # e.g., "5_drive_L"
                parts = rest.split("_")
                link_i = int(parts[0])
                lr = parts[-1]
                spr_name = "_".join(parts[1:-1])
                eng_ids[(side, link_i, spr_name, lr)] = idx

    link_bids = {}
    for side in ["left", "right"]:
        link_bids[side] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_{i}")
            for i in range(N_LINKS)
        ]

    spr_bids = {}
    jnt_ids = {}
    for side in ["left", "right"]:
        for spr_name, _ in SPROCKET_DEFS:
            full = f"{side}_{spr_name}"
            spr_bids[(side, spr_name)] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{full}_sprocket")
            jnt_ids[(side, spr_name)] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{full}_hinge")

    # Actuator indices
    act_ids = {}
    for side in ["left", "right"]:
        for spr_name, _ in SPROCKET_DEFS:
            full = f"{side}_{spr_name}"
            act_ids[(side, spr_name)] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{full}_motor")

    # Hinge joint IDs for chain links
    chain_jnt_ids = {}
    for side in ["left", "right"]:
        chain_jnt_ids[side] = []
        for i in range(1, N_LINKS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hinge_{i}")
            chain_jnt_ids[side].append((i, jid))

    return eng_ids, link_bids, spr_bids, jnt_ids, act_ids, chain_jnt_ids


def set_initial_chain_angles(model, data, chain_jnt_ids):
    """Set hinge angles so the chain forms the stadium shape."""
    for side in ["left", "right"]:
        for i, jid in chain_jnt_ids[side]:
            # Angle difference between link i and link i-1
            _, _, ang_i = stadium_point(i * LINK_PITCH)
            _, _, ang_prev = stadium_point((i - 1) * LINK_PITCH)
            # The hinge angle is the relative rotation
            delta = _normalize_angle(ang_i - ang_prev)
            # Negate because Y-axis hinge rotates X toward -Z (opposite convention)
            data.qpos[model.jnt_qposadr[jid]] = -delta
    mujoco.mj_forward(model, data)


# Per-side engagement state: {(side, link_i, spr_name): local_angle}
_engaged = {}


def update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    """Toggle sprocket constraints for both tracks."""
    for side in ["left", "right"]:
        for i in range(N_LINKS):
            # Link world position
            bid = link_bids[side][i]
            lx = data.xpos[bid][0]
            lz = data.xpos[bid][2]  # Z is up in tank frame

            for spr_name, _ in SPROCKET_DEFS:
                key = (side, i, spr_name)

                if spr_name == "mid":
                    continue

                sbid = spr_bids[(side, spr_name)]
                sx = data.xpos[sbid][0]
                sz = data.xpos[sbid][2]
                spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[(side, spr_name)]]]

                if key in _engaged:
                    # Check angle-based disengagement
                    local_angle = _engaged[key]
                    # Y-axis hinge: world = local - sprocket_angle
                    world_angle = _normalize_angle(local_angle - spr_angle)

                    if spr_name == "drive":
                        off = abs(_normalize_angle(world_angle - math.pi))
                    elif spr_name == "idler":
                        off = abs(_normalize_angle(world_angle))

                    if off > ARC_HALF:
                        del _engaged[key]
                        for lr in ("L", "R"):
                            eq_idx = eng_ids.get((side, i, spr_name, lr))
                            if eq_idx is not None:
                                data.eq_active[eq_idx] = 0
                else:
                    # Check position-based engagement
                    dx, dz = lx - sx, lz - sz
                    dist = math.sqrt(dx * dx + dz * dz)

                    on_arc = False
                    if spr_name == "drive":
                        on_arc = lx < sx + LINK_PITCH * 0.3
                    elif spr_name == "idler":
                        on_arc = lx > sx - LINK_PITCH * 0.3

                    on_arc = on_arc and (SPROCKET_R * 0.3 < dist < SPROCKET_R * 2.0)

                    if on_arc:
                        world_angle = math.atan2(dz, dx)
                        local_angle = _normalize_angle(world_angle + spr_angle)

                        spr_ax = SPROCKET_R * math.cos(local_angle)
                        spr_az = SPROCKET_R * math.sin(local_angle)

                        for lr, y in [("L", -Z_OFF), ("R", Z_OFF)]:
                            eq_idx = eng_ids.get((side, i, spr_name, lr))
                            if eq_idx is None:
                                continue
                            # Sprocket anchor in local frame (XZ plane, Y lateral)
                            model.eq_data[eq_idx, 0] = spr_ax
                            model.eq_data[eq_idx, 1] = y
                            model.eq_data[eq_idx, 2] = spr_az
                            # Link anchor
                            model.eq_data[eq_idx, 3] = 0.0
                            model.eq_data[eq_idx, 4] = y
                            model.eq_data[eq_idx, 5] = 0.0
                            data.eq_active[eq_idx] = 1

                        _engaged[key] = local_angle


def step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids,
             left_vel=TARGET_VEL, right_vel=TARGET_VEL):
    """One simulation step."""
    t = data.time
    ramp = min(1.0, t)

    for spr_name, _ in SPROCKET_DEFS:
        li = act_ids[("left", spr_name)]
        ri = act_ids[("right", spr_name)]
        data.ctrl[li] = left_vel * ramp
        data.ctrl[ri] = right_vel * ramp

    # Let the chain settle before engaging sprockets
    if t > 0.5:
        update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)
    mujoco.mj_step(model, data)


def run_gui():
    """Launch interactive MuJoCo viewer."""
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, chain_jnt_ids = init_lookups(model)
    set_initial_chain_angles(model, data, chain_jnt_ids)

    import mujoco.viewer as mjv
    with mjv.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)
            viewer.sync()


def run_debug():
    """Run headless diagnostics."""
    xml = build_xml()
    with open('/home/sman/Work/CMU/Research/track_synthesis/tank.xml', 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, chain_jnt_ids = init_lookups(model)
    set_initial_chain_angles(model, data, chain_jnt_ids)
    hull_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hull")

    for step in range(5000):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        max_cf = 0.0
        for c in range(data.ncon):
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f)
            max_cf = max(max_cf, np.linalg.norm(f))

        max_ev = np.max(np.abs(data.efc_pos[:data.nefc])) if data.nefc > 0 else 0.0
        max_v = np.max(np.abs(data.qvel))

        hull_pos = data.xpos[hull_bid]
        hull_vel = data.cvel[hull_bid]

        n_eng = sum(1 for idx in eng_ids.values() if data.eq_active[idx]) // 2

        if step < 10 or step % 200 == 0 or max_v > 200 or np.any(np.isnan(data.qpos)):
            t = data.time
            print(f"step {step:4d}  t={t:.2f}  "
                  f"ncon={data.ncon:3d}  cf={max_cf:7.1f}  "
                  f"v={max_v:6.1f}  eng={n_eng:2d}  "
                  f"hull=({hull_pos[0]:.2f},{hull_pos[1]:.2f},{hull_pos[2]:.2f})")

        if np.any(np.isnan(data.qpos)):
            print(">>> NaN, stopping")
            break


def run_record():
    """Record GIF."""
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, chain_jnt_ids = init_lookups(model)
    set_initial_chain_angles(model, data, chain_jnt_ids)

    renderer = mujoco.Renderer(model, width=800, height=600)
    frames = []

    for step in range(5000):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)
        if step % 10 == 0:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())

    import imageio
    out = "/home/sman/Work/CMU/Research/track_synthesis/tank.gif"
    imageio.mimsave(out, frames, fps=30, loop=0)
    print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()

    if args.debug:
        run_debug()
    elif args.record:
        run_record()
    else:
        run_gui()
