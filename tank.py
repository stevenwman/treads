"""
Tank with two tracked drives. XZ track plane, gravity -Z, floor at z=0.

Two mirrored chain tracks (left/right) connected to a rigid hull.
Sprockets are children of the hull. Chain links are free kinematic trees.
Differential steering via left/right velocity control.

Usage:
    uv run python tank.py              # GUI
    uv run python tank.py --debug      # headless diagnostics
"""

import argparse
import math
import numpy as np
import mujoco

# ── Track parameters (same as chain_track.py) ──────────────────────────────

N_LINKS = 30
SPROCKET_R = 0.35
HALF_SPAN = 1.4
LINK_THICK = 0.02
LINK_WIDTH = 0.10
TIMESTEP = 0.002
TARGET_VEL = 1.0
TENSION_K = 80.0
ARC_HALF = math.pi * 0.40

HUB_R = SPROCKET_R - 0.04
HUB_HALF_Y = LINK_WIDTH + 0.02

PERIMETER = 2 * math.pi * SPROCKET_R + 2 * (2 * HALF_SPAN)
LINK_PITCH = PERIMETER / N_LINKS

# ── Tank geometry ───────────────────────────────────────────────────────────

TRACK_GAUGE = 1.2         # Y distance between left/right track centers
HULL_HALF_X = HALF_SPAN + 0.2
HULL_HALF_Y = TRACK_GAUGE / 2 - 0.05
HULL_HALF_Z = 0.12
SPROCKET_Z = SPROCKET_R + 0.02  # sprocket center height — low so bottom treads press into ground
HULL_Z = SPROCKET_Z + 0.15      # hull center height (above sprockets)

# Per-track sprocket defs: (local_name, x_offset)
SPR_DEFS = [
    ("drive", -HALF_SPAN),
    ("idler",  HALF_SPAN),
    ("mid",    0.0),
]

SIDES = [("left", 1), ("right", -1)]


def _norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def stadium_point(s):
    """(x_local, z_local, angle) relative to sprocket center height."""
    s = s % PERIMETER
    top = 2 * HALF_SPAN
    arc = math.pi * SPROCKET_R
    if s < top:
        return HALF_SPAN - s, SPROCKET_R, math.pi
    s -= top
    if s < arc:
        th = math.pi / 2 + s / SPROCKET_R
        return -HALF_SPAN + SPROCKET_R * math.cos(th), SPROCKET_R * math.sin(th), th + math.pi / 2
    s -= arc
    if s < top:
        return -HALF_SPAN + s, -SPROCKET_R, 0.0
    s -= top
    th = -math.pi / 2 + s / SPROCKET_R
    return HALF_SPAN + SPROCKET_R * math.cos(th), SPROCKET_R * math.sin(th), th + math.pi / 2


def build_xml():
    L = []
    a = L.append

    a('<mujoco model="tank">')
    a(f'  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"'
      f' integrator="implicitfast" solver="Newton" cone="elliptic"'
      f' iterations="300" tolerance="1e-8" noslip_iterations="30"'
      f' impratio="2"/>')
    a('  <size nconmax="2000" njmax="6000"/>')
    a('  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="0" offsamples="1"/></visual>')
    a('  <default>')
    a('    <geom friction="1.0 0.005 0.005" condim="3" margin="0.002"'
      '     solref="0.004 1" solimp="0.95 0.99 0.001"/>')
    a('    <equality solref="0.01 1" solimp="0.9 0.95 0.001"/>')
    a('  </default>')
    # Checkerboard ground texture
    a('  <asset>')
    a('    <texture name="grid" type="2d" builtin="checker" width="512" height="512"'
      '     rgb1="0.4 0.45 0.4" rgb2="0.45 0.5 0.45"/>')
    a('    <material name="grid_mat" texture="grid" texrepeat="10 10" texuniform="true"/>')
    a('  </asset>')
    a('  <worldbody>')
    a('    <camera name="overview" pos="0 -5 3" xyaxes="1 0 0 0 0.3 1" fovy="50"/>')
    a('    <camera name="top" pos="0 0 6" xyaxes="1 0 0 0 1 0" fovy="60"/>')
    a('    <camera name="side" pos="-4 0 1.5" xyaxes="0 -1 0 0 0 1" fovy="50"/>')
    a('    <light pos="0 -3 6" dir="0 0.3 -0.7" diffuse="1 1 1" specular="0.3 0.3 0.3"/>')
    # Ground
    a('    <geom name="floor" type="plane" size="20 20 0.1" material="grid_mat"'
      '     contype="1" conaffinity="2"/>')
    # RGB axes
    a('    <geom type="capsule" fromto="0 0 0.002 0.5 0 0.002" size="0.008" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0.002 0 0.5 0.002" size="0.008" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0.002 0 0 0.5" size="0.008" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>')
    # Obstacles — angled approach faces so treads can climb
    # Ramp: gentle 8deg slope, 3cm rise
    a('    <geom name="ramp" type="box" size="1.0 1.5 0.03" pos="4 0 0.03"'
      '     euler="0 -8 0" rgba="0.6 0.4 0.3 1" contype="1" conaffinity="2"/>')
    # Bump: two wedges forming a ridge, 5cm tall
    a('    <geom name="bump_up" type="box" size="0.5 1.5 0.05" pos="6.7 0 0.05"'
      '     euler="0 -15 0" rgba="0.5 0.35 0.3 1" contype="1" conaffinity="2"/>')
    a('    <geom name="bump_dn" type="box" size="0.5 1.5 0.05" pos="7.7 0 0.05"'
      '     euler="0 15 0" rgba="0.5 0.35 0.3 1" contype="1" conaffinity="2"/>')
    # Step: ramp up + flat top + ramp down, 8cm tall
    a('    <geom name="step_up" type="box" size="0.6 1.5 0.08" pos="9.5 0 0.08"'
      '     euler="0 -12 0" rgba="0.55 0.4 0.35 1" contype="1" conaffinity="2"/>')
    a('    <geom name="step_top" type="box" size="1.0 1.5 0.08" pos="10.7 0 0.08"'
      '     rgba="0.55 0.4 0.35 1" contype="1" conaffinity="2"/>')
    a('    <geom name="step_dn" type="box" size="0.6 1.5 0.08" pos="12.3 0 0.08"'
      '     euler="0 12 0" rgba="0.55 0.4 0.35 1" contype="1" conaffinity="2"/>')

    # ── Hull ──
    a(f'    <body name="hull" pos="0 0 {HULL_Z}">')
    a(f'      <freejoint name="hull_jnt"/>')
    # Isometric tracking camera attached to hull
    a(f'      <camera name="tracking" pos="-3 -3 2.5" xyaxes="1 -1 0 0.3 0.3 1"'
      f' fovy="45" mode="track"/>')
    a(f'      <camera name="front" pos="4 0 0.3" xyaxes="0 1 0 0 0 1"'
      f' fovy="60" mode="track"/>')
    a(f'      <geom name="hull_box" type="box" size="{HULL_HALF_X} {HULL_HALF_Y} {HULL_HALF_Z}"'
      f' rgba="0.3 0.35 0.3 1" mass="20.0" contype="0" conaffinity="0"/>')

    # Sprockets as children of hull
    for side, y_sign in SIDES:
        y_off = y_sign * TRACK_GAUGE / 2
        for spr_name, x_off in SPR_DEFS:
            full = f"{side}_{spr_name}"
            # Sprocket center is at hull height, offset in X and Y
            # Z offset from hull center: SPROCKET_Z - HULL_Z
            dz = SPROCKET_Z - HULL_Z
            a(f'      <body name="{full}_spr" pos="{x_off} {y_off} {dz}">')
            if spr_name == "idler":
                a(f'        <joint name="{full}_slide" type="slide" axis="1 0 0"'
                  f' stiffness="{TENSION_K}" damping="80" range="-0.05 0.3"/>')
            a(f'        <joint name="{full}_hinge" type="hinge" axis="0 1 0" damping="2.0"/>')
            col = {"drive": "0.7 0.2 0.2 1", "idler": "0.2 0.2 0.7 1", "mid": "0.2 0.6 0.2 1"}[spr_name]
            # Hub: collision enabled so chain links ride on it during transitions
            a(f'        <geom type="cylinder" size="{HUB_R} {HUB_HALF_Y}" euler="90 0 0"'
              f' rgba="{col}" contype="1" conaffinity="2"/>')
            a(f'      </body>')

    a('    </body>')  # end hull

    # ── Chain links (independent kinematic trees per side) ──
    half_len = LINK_PITCH / 2 - 0.005

    for side, y_sign in SIDES:
        y_off = y_sign * TRACK_GAUGE / 2

        # Link 0: freejoint root
        xl, zl, ang = stadium_point(0)
        wz = SPROCKET_Z + zl
        qw = math.cos(-ang / 2)
        qy = math.sin(-ang / 2)

        a(f'    <body name="{side}_link_0" pos="{xl:.6f} {y_off} {wz:.6f}"'
          f' quat="{qw:.6f} 0 {qy:.6f} 0">')
        a(f'      <freejoint name="{side}_link_0_jnt"/>')
        a(f'      <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
          f' rgba="1 0.2 0.2 1" mass="0.05" contype="2" conaffinity="1"/>')

        # Links 1..N-1: nested children
        for i in range(1, N_LINKS):
            a(f'      <body name="{side}_link_{i}" pos="{LINK_PITCH:.6f} 0 0">')
            a(f'        <joint name="{side}_hinge_{i}" type="hinge" axis="0 1 0"'
              f' pos="{-LINK_PITCH/2:.6f} 0 0" damping="0.2"'
              f' limited="true" range="-1.2 1.2"/>')
            a(f'        <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
              f' rgba="0.9 0.6 0.1 1" mass="0.05" contype="2" conaffinity="1"/>')

        # Close nested bodies
        for _ in range(N_LINKS):
            a('    </body>')

    a('  </worldbody>')

    # Actuators
    a('  <actuator>')
    for side, _ in SIDES:
        for spr_name, _ in SPR_DEFS:
            full = f"{side}_{spr_name}"
            a(f'    <velocity name="{full}_motor" joint="{full}_hinge"'
              f' kv="500" ctrllimited="true" ctrlrange="-5 5"/>')
    a('  </actuator>')

    # Equality constraints
    a('  <equality>')
    for side, _ in SIDES:
        # Loop closure
        a(f'    <connect name="{side}_loop" body1="{side}_link_{N_LINKS-1}" body2="{side}_link_0"'
          f' anchor="{LINK_PITCH/2:.6f} 0 0"/>')
        # Sprocket engagement
        for i in range(N_LINKS):
            for spr_name, _ in SPR_DEFS:
                if spr_name == "mid":
                    continue
                full = f"{side}_{spr_name}"
                a(f'    <connect name="{side}_eng_{i}_{spr_name}"'
                  f' body1="{full}_spr" body2="{side}_link_{i}"'
                  f' anchor="0 0 0" active="false"/>')
    a('  </equality>')
    a('</mujoco>')
    return '\n'.join(L)


def init_lookups(model):
    eng_ids = {}  # (side, link_i, spr_name) -> eq_idx
    for idx in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if nm is None:
            continue
        for side, _ in SIDES:
            pref = f"{side}_eng_"
            if nm.startswith(pref):
                rest = nm[len(pref):]
                parts = rest.split("_")
                eng_ids[(side, int(parts[0]), "_".join(parts[1:]))] = idx

    link_bids = {}
    for side, _ in SIDES:
        link_bids[side] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_{i}")
                           for i in range(N_LINKS)]

    spr_bids = {}
    jnt_ids = {}
    act_ids = {}
    for side, _ in SIDES:
        for spr_name, _ in SPR_DEFS:
            full = f"{side}_{spr_name}"
            spr_bids[(side, spr_name)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{full}_spr")
            jnt_ids[(side, spr_name)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{full}_hinge")
            act_ids[(side, spr_name)] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{full}_motor")

    hinge_jids = {}
    for side, _ in SIDES:
        hinge_jids[side] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hinge_{i}")
                            for i in range(1, N_LINKS)]

    return eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids


def set_initial_shape(model, data, hinge_jids):
    for side, _ in SIDES:
        for idx, jid in enumerate(hinge_jids[side]):
            i = idx + 1
            _, _, ang_i = stadium_point(i * LINK_PITCH)
            _, _, ang_prev = stadium_point((i - 1) * LINK_PITCH)
            data.qpos[model.jnt_qposadr[jid]] = -_norm_angle(ang_i - ang_prev)
    mujoco.mj_forward(model, data)

    # Fix loop closure anchors
    for idx in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if nm and nm.endswith("_loop"):
            bid1, bid2 = model.eq_obj1id[idx], model.eq_obj2id[idx]
            w1 = data.xpos[bid1] + data.xmat[bid1].reshape(3, 3) @ model.eq_data[idx, 0:3]
            model.eq_data[idx, 3:6] = data.xmat[bid2].reshape(3, 3).T @ (w1 - data.xpos[bid2])
    mujoco.mj_forward(model, data)


_engaged = {}


MAX_ENG_PER_SPROCKET = 2


def _count_engaged(side, spr_name):
    """Count how many links are currently engaged to this sprocket."""
    return sum(1 for (s, i, sp) in _engaged if s == side and sp == spr_name)


def seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    """Seed engagement using the same position check as runtime, capped at 2 per sprocket."""
    for side, _ in SIDES:
        for spr_name, _ in SPR_DEFS:
            if spr_name == "mid":
                continue

            sbid = spr_bids[(side, spr_name)]
            sx, sz = data.xpos[sbid][0], data.xpos[sbid][2]

            # Collect candidates with their distance from the horizontal center line
            candidates = []
            for i in range(N_LINKS):
                bid = link_bids[side][i]
                lx, lz = data.xpos[bid][0], data.xpos[bid][2]
                dx, dz = lx - sx, lz - sz
                dist = math.sqrt(dx * dx + dz * dz)

                on_arc = False
                if spr_name == "drive":
                    on_arc = lx < sx - SPROCKET_R * 0.85
                elif spr_name == "idler":
                    on_arc = lx > sx + SPROCKET_R * 0.85
                on_arc = on_arc and abs(dist - SPROCKET_R) < 0.10

                if on_arc:
                    # Sort by closeness to horizontal (small |dz| = closer to center line)
                    candidates.append((abs(dz), i, dx, dz, dist))

            # Take the 2 closest to horizontal
            candidates.sort()
            for _, i, dx, dz, dist in candidates[:MAX_ENG_PER_SPROCKET]:
                key = (side, i, spr_name)
                eq_idx = eng_ids.get(key)
                if eq_idx is None:
                    continue

                world_angle = math.atan2(dz, dx)
                spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[(side, spr_name)]]]
                local_angle = _norm_angle(world_angle + spr_angle)

                model.eq_data[eq_idx, 0] = dist * math.cos(local_angle)
                model.eq_data[eq_idx, 1] = 0.0
                model.eq_data[eq_idx, 2] = dist * math.sin(local_angle)
                model.eq_data[eq_idx, 3:6] = 0.0
                data.eq_active[eq_idx] = 1
                _engaged[key] = local_angle

    mujoco.mj_forward(model, data)


def update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    for side, _ in SIDES:
        for i in range(N_LINKS):
            lx = data.xpos[link_bids[side][i]][0]
            lz = data.xpos[link_bids[side][i]][2]

            for spr_name, _ in SPR_DEFS:
                if spr_name == "mid":
                    continue
                key = (side, i, spr_name)
                eq_idx = eng_ids.get(key)
                if eq_idx is None:
                    continue

                sbid = spr_bids[(side, spr_name)]
                sx, sz = data.xpos[sbid][0], data.xpos[sbid][2]
                spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[(side, spr_name)]]]
                dx, dz = lx - sx, lz - sz
                dist = math.sqrt(dx * dx + dz * dz)

                # Only engage the 2 links closest to the horizontal center line
                # Drive: directly left of sprocket (lx < sx - R*0.85)
                # Idler: directly right (lx > sx + R*0.85)
                on_arc = False
                if spr_name == "drive":
                    on_arc = lx < sx - SPROCKET_R * 0.85
                elif spr_name == "idler":
                    on_arc = lx > sx + SPROCKET_R * 0.85
                on_arc = on_arc and abs(dist - SPROCKET_R) < 0.10

                if key in _engaged:
                    # Already engaged — check angle-based disengagement
                    local_angle = _engaged[key]
                    # Y-axis: world = local - spr_angle
                    world_angle = _norm_angle(local_angle - spr_angle)
                    if spr_name == "drive":
                        off = abs(_norm_angle(world_angle - math.pi))
                    else:  # idler
                        off = abs(_norm_angle(world_angle))
                    if off > ARC_HALF:
                        del _engaged[key]
                        data.eq_active[eq_idx] = 0
                elif on_arc and _count_engaged(side, spr_name) < MAX_ENG_PER_SPROCKET:
                    # New engagement — set anchor ONCE (fixed), only if under cap
                    world_angle = math.atan2(dz, dx)
                    local_angle = _norm_angle(world_angle + spr_angle)
                    model.eq_data[eq_idx, 0] = SPROCKET_R * math.cos(local_angle)
                    model.eq_data[eq_idx, 1] = 0.0
                    model.eq_data[eq_idx, 2] = SPROCKET_R * math.sin(local_angle)
                    model.eq_data[eq_idx, 3:6] = 0.0
                    data.eq_active[eq_idx] = 1
                    _engaged[key] = local_angle
                else:
                    data.eq_active[eq_idx] = 0


def step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids,
             left_vel=TARGET_VEL, right_vel=TARGET_VEL):
    t = data.time
    ramp = min(1.0, t / 1.0)
    for spr_name, _ in SPR_DEFS:
        data.ctrl[act_ids[("left", spr_name)]] = left_vel * ramp
        data.ctrl[act_ids[("right", spr_name)]] = right_vel * ramp
    update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)
    mujoco.mj_step(model, data)


def _init_all(model, data):
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids = init_lookups(model)
    set_initial_shape(model, data, hinge_jids)
    seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)
    return eng_ids, link_bids, spr_bids, jnt_ids, act_ids


def run_gui():
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids = _init_all(model, data)

    import mujoco.viewer as mjv
    import time
    with mjv.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        sim_start = data.time
        frame_count = 0
        last_report = wall_start
        while viewer.is_running():
            t0 = time.perf_counter()

            # Step sim toward wall time, but cap to avoid death spiral
            wall_elapsed = t0 - wall_start
            target = sim_start + wall_elapsed
            MAX_STEPS_PER_FRAME = 10
            n_steps = 0
            while data.time < target and n_steps < MAX_STEPS_PER_FRAME:
                step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)
                n_steps += 1
            # If we can't keep up, accept slower-than-realtime
            if data.time < target:
                wall_start = t0
                sim_start = data.time

            t1 = time.perf_counter()
            viewer.sync()
            t2 = time.perf_counter()

            frame_count += 1
            if t2 - last_report > 2.0:
                fps = frame_count / (t2 - last_report)
                sim_ms = (t1 - t0) * 1000
                sync_ms = (t2 - t1) * 1000
                print(f"FPS={fps:.1f}  sim={sim_ms:.1f}ms({n_steps}steps)  sync={sync_ms:.1f}ms  "
                      f"sim_t={data.time:.1f}s")
                frame_count = 0
                last_report = t2


def run_debug():
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids = _init_all(model, data)
    hull_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hull")

    # Obstacle positions for progress tracking
    obstacles = [("ramp", 4.0, 0.03), ("bump", 7.2, 0.05), ("step", 10.7, 0.08)]
    last_obstacle = ""

    # Run longer to reach obstacles
    n_steps = 10000
    history = []

    for step in range(n_steps):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        t = data.time
        hp = data.xpos[hull_bid]
        max_v = np.max(np.abs(data.qvel))
        max_ev = np.max(np.abs(data.efc_pos[:data.nefc])) if data.nefc > 0 else 0
        n_eng = sum(1 for idx in eng_ids.values() if data.eq_active[idx])

        lv = data.qvel[model.jnt_dofadr[jnt_ids[("left", "drive")]]]
        rv = data.qvel[model.jnt_dofadr[jnt_ids[("right", "drive")]]]
        hull_vx = data.cvel[hull_bid][3]  # linear velocity X

        # Ground contact: total normal force and friction force on treads
        total_normal = 0.0
        total_friction = 0.0
        n_ground_contacts = 0
        for c in range(data.ncon):
            con = data.contact[c]
            g1n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            ground_names = ("floor", "ramp", "bump_up", "bump_dn", "step_up", "step_top", "step_dn")
            if g1n in ground_names or g2n in ground_names:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f)
                total_normal += abs(f[0])  # normal
                total_friction += np.linalg.norm(f[1:3])  # tangential
                n_ground_contacts += 1

        # Check obstacle progress
        current_obstacle = ""
        for oname, ox, oh in obstacles:
            if hp[0] > ox - 1.0:
                current_obstacle = oname

        rec = {'t': t, 'hx': hp[0], 'hz': hp[2], 'hvx': hull_vx,
               'lv': lv, 'rv': rv, 'eng': n_eng, 'eq_v': max_ev, 'max_v': max_v,
               'ncon': n_ground_contacts, 'fn': total_normal, 'ff': total_friction,
               'obs': current_obstacle}
        history.append(rec)

        do_print = step < 5 or step % 200 == 0 or max_v > 200
        if current_obstacle != last_obstacle:
            do_print = True
            last_obstacle = current_obstacle
        if np.any(np.isnan(data.qpos)):
            do_print = True

        if do_print:
            obs_str = f" [{current_obstacle}]" if current_obstacle else ""
            print(f"t={t:5.1f}  hx={hp[0]:5.2f} hz={hp[2]:4.2f}  "
                  f"hvx={hull_vx:5.2f}m/s  "
                  f"drv=({lv:4.1f},{rv:4.1f})r/s  eng={n_eng:2d}  "
                  f"eq_v={max_ev:.3f}  v={max_v:5.1f}  "
                  f"gnd={n_ground_contacts:2d} fn={total_normal:6.0f} ff={total_friction:5.0f}"
                  f"{obs_str}")
            if np.any(np.isnan(data.qpos)):
                print(">>> NaN")
                break

    # Summary with obstacle progress
    print()
    print("=== SUMMARY ===")
    print(f"  t={t:.1f}s  hull=({hp[0]:.2f}, {hp[2]:.2f})")
    print(f"  Obstacles:")
    for oname, ox, oh in obstacles:
        reached = any(r['hx'] > ox + 1.0 for r in history)
        max_hx_near = max((r['hx'] for r in history if abs(r['hx'] - ox) < 2.0), default=0)
        if reached:
            print(f"    {oname} (x={ox}, h={oh}m): CLEARED")
        elif max_hx_near > ox - 1.5:
            print(f"    {oname} (x={ox}, h={oh}m): STUCK (got to x={max_hx_near:.2f})")
        else:
            print(f"    {oname} (x={ox}, h={oh}m): NOT REACHED")
    # Hull velocity over time
    if len(history) > 100:
        early = [r['hvx'] for r in history[50:150]]
        late = [r['hvx'] for r in history[-200:]]
        print(f"  Avg hull vx: early={np.mean(early):.3f} late={np.mean(late):.3f} m/s")
        print(f"  Avg ground contacts: {np.mean([r['ncon'] for r in history[-200:]]):.0f}")
        print(f"  Avg normal force: {np.mean([r['fn'] for r in history[-200:]]):.0f}N")
        print(f"  Avg friction force: {np.mean([r['ff'] for r in history[-200:]]):.0f}N")


def run_record():
    """Record mp4 from the tracking camera."""
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids = _init_all(model, data)

    W, H = 1280, 720
    FPS = 24
    SIM_DURATION = 30.0  # seconds of sim time
    frames_per_step = max(1, round(1.0 / (FPS * TIMESTEP)))  # steps between frames

    renderer = mujoco.Renderer(model, width=W, height=H)

    n_steps = int(SIM_DURATION / TIMESTEP)
    n_frames = n_steps // frames_per_step

    import imageio
    out = "/home/sman/Work/CMU/Research/track_synthesis/tank.mp4"
    writer = imageio.get_writer(out, fps=FPS, codec='libx264', quality=8)

    from tqdm import tqdm
    pbar = tqdm(total=n_frames, desc="Recording", unit="frame",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for step in range(n_steps):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        if step % frames_per_step == 0:
            renderer.update_scene(data, camera="front")
            writer.append_data(renderer.render())
            pbar.update(1)

    pbar.close()
    writer.close()
    print(f"Saved {out} ({n_frames} frames, {n_frames/FPS:.1f}s)")


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
