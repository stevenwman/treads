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
TIMESTEP = 0.004
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
SPROCKET_Z = SPROCKET_R + 0.10  # sprocket center height above ground
HULL_Z = SPROCKET_Z + 0.05      # hull center height

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
      f' iterations="300" solver="Newton" tolerance="1e-10" noslip_iterations="10"/>')
    a('  <size nconmax="2000" njmax="6000"/>')
    a('  <visual><global offwidth="1200" offheight="800"/></visual>')
    a('  <default>')
    a('    <geom friction="0.8 0.01 0.01" condim="4" margin="0.005"/>')
    a('    <equality solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    a('  </default>')
    a('  <worldbody>')
    a('    <camera name="overview" pos="0 -5 3" xyaxes="1 0 0 0 0.3 1" fovy="50"/>')
    a('    <camera name="top" pos="0 0 6" xyaxes="1 0 0 0 1 0" fovy="60"/>')
    a('    <camera name="side" pos="-4 0 1.5" xyaxes="0 -1 0 0 0 1" fovy="50"/>')
    a('    <light pos="0 -3 5" dir="0 0.5 -0.5" diffuse="0.8 0.8 0.8"/>')
    a('    <light pos="3 3 5" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>')
    # Ground
    a('    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.4 0.5 0.4 1"'
      '     contype="1" conaffinity="2"/>')  # collides with links (contype=2)
    # RGB axes
    a('    <geom type="capsule" fromto="0 0 0.001 0.5 0 0.001" size="0.008" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0.001 0 0.5 0.001" size="0.008" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0.001 0 0 0.5" size="0.008" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>')

    # ── Hull ──
    a(f'    <body name="hull" pos="0 0 {HULL_Z}">')
    a(f'      <freejoint name="hull_jnt"/>')
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
            a(f'        <joint name="{full}_hinge" type="hinge" axis="0 1 0" damping="0.2"/>')
            col = {"drive": "0.7 0.2 0.2 1", "idler": "0.2 0.2 0.7 1", "mid": "0.2 0.6 0.2 1"}[spr_name]
            a(f'        <geom type="cylinder" size="{HUB_R} {HUB_HALF_Y}" euler="90 0 0"'
              f' rgba="{col}" contype="0" conaffinity="0"/>')
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
              f' pos="{-LINK_PITCH/2:.6f} 0 0" damping="0.05"/>')
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
              f' kv="200" ctrllimited="true" ctrlrange="-5 5"/>')
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


def seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    top_len = 2 * HALF_SPAN
    arc_len = math.pi * SPROCKET_R

    for side, _ in SIDES:
        for i in range(N_LINKS):
            s = (i * LINK_PITCH) % PERIMETER
            spr_name = None
            if top_len < s < top_len + arc_len:
                spr_name = "drive"
            elif 2 * top_len + arc_len < s:
                spr_name = "idler"
            if spr_name is None:
                continue

            key = (side, i, spr_name)
            eq_idx = eng_ids.get(key)
            if eq_idx is None:
                continue

            bid = link_bids[side][i]
            sbid = spr_bids[(side, spr_name)]
            dx = data.xpos[bid][0] - data.xpos[sbid][0]
            dz = data.xpos[bid][2] - data.xpos[sbid][2]
            dist = math.sqrt(dx * dx + dz * dz)
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

                on_arc = False
                if spr_name == "drive":
                    on_arc = lx < sx + LINK_PITCH * 0.3
                elif spr_name == "idler":
                    on_arc = lx > sx - LINK_PITCH * 0.3
                on_arc = on_arc and abs(dist - SPROCKET_R) < 0.10

                if on_arc:
                    world_angle = math.atan2(dz, dx)
                    local_angle = _norm_angle(world_angle + spr_angle)
                    model.eq_data[eq_idx, 0] = SPROCKET_R * math.cos(local_angle)
                    model.eq_data[eq_idx, 1] = 0.0
                    model.eq_data[eq_idx, 2] = SPROCKET_R * math.sin(local_angle)
                    model.eq_data[eq_idx, 3:6] = 0.0
                    data.eq_active[eq_idx] = 1
                    _engaged[key] = local_angle
                else:
                    if key in _engaged:
                        del _engaged[key]
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

    for step in range(5000):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        max_v = np.max(np.abs(data.qvel))
        max_ev = np.max(np.abs(data.efc_pos[:data.nefc])) if data.nefc > 0 else 0
        n_eng = sum(1 for idx in eng_ids.values() if data.eq_active[idx])
        t = data.time
        hp = data.xpos[hull_bid]

        # Per-side drive angles
        ld = math.degrees(data.qpos[model.jnt_qposadr[jnt_ids[("left", "drive")]]])
        rd = math.degrees(data.qpos[model.jnt_qposadr[jnt_ids[("right", "drive")]]])
        lv = data.qvel[model.jnt_dofadr[jnt_ids[("left", "drive")]]]
        rv = data.qvel[model.jnt_dofadr[jnt_ids[("right", "drive")]]]

        if step < 10 or step % 200 == 0 or max_v > 200 or np.any(np.isnan(data.qpos)):
            print(f"step {step:4d}  t={t:.2f}  "
                  f"v={max_v:6.1f}  eq_v={max_ev:.4f}  eng={n_eng:2d}  "
                  f"Ldrv={ld:6.1f}d({lv:5.2f}) Rdrv={rd:6.1f}d({rv:5.2f})  "
                  f"hull=({hp[0]:.2f},{hp[1]:.2f},{hp[2]:.2f})")
            if np.any(np.isnan(data.qpos)):
                print(">>> NaN")
                break

    print()
    print("=== SUMMARY ===")
    print(f"  t={data.time:.2f}  hull=({hp[0]:.2f},{hp[1]:.2f},{hp[2]:.2f})")
    print(f"  Ldrv={ld:.1f}deg  Rdrv={rd:.1f}deg")
    print(f"  engaged={n_eng}  eq_v={max_ev:.4f}  max_v={max_v:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        run_debug()
    else:
        run_gui()
