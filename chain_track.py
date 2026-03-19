"""
Single track: kinematic-tree chain with dynamic sprocket engagement.

Chain links connected by hinge joints in a kinematic tree.
One equality constraint closes the loop.
Sprocket engagement via single connect constraint per link (center of link to drum surface).

Usage:
    uv run python chain_track.py          # GUI viewer
    uv run python chain_track.py --debug  # headless diagnostics
"""

import argparse
import math
import numpy as np
import mujoco

# ── Parameters ──────────────────────────────────────────────────────────────

N_LINKS = 30
SPROCKET_R = 0.35
HALF_SPAN = 1.4
LINK_THICK = 0.02         # half-height (Y) of link box
LINK_WIDTH = 0.10         # half-depth (Z) of link box
TIMESTEP = 0.002
TARGET_VEL = 1.0          # rad/s
TENSION_K = 80.0

HUB_R = SPROCKET_R - 0.04
HUB_Z = LINK_WIDTH + 0.02

PERIMETER = 2 * math.pi * SPROCKET_R + 2 * (2 * HALF_SPAN)
LINK_PITCH = PERIMETER / N_LINKS

# Sprockets: (name, x, y)
SPROCKETS = [
    ("drive", -HALF_SPAN, 0.0),
    ("idler",  HALF_SPAN, 0.0),
    ("mid",    0.0,        0.0),
]

ARC_HALF = math.pi * 0.40  # disengage earlier to allow chain handoff


def _norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def stadium_point(s):
    """(x, y, angle) for arc-length s. XY plane (vertical: Y=up), CCW stadium."""
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

    a('<mujoco model="chain_track">')
    a(f'  <option timestep="{TIMESTEP}" gravity="0 -9.81 0"'
      f' iterations="300" solver="Newton" tolerance="1e-10" noslip_iterations="10"/>')
    a('  <size nconmax="500" njmax="3000"/>')
    a('  <visual><global offwidth="1200" offheight="800"/></visual>')
    a('  <default>')
    a('    <geom friction="0.8 0.01 0.01" condim="4" margin="0.005"/>')
    a('    <equality solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    a('  </default>')
    a('  <worldbody>')
    a('    <camera name="overview" pos="0 0 5" xyaxes="1 0 0 0 1 0" fovy="50" mode="fixed"/>')
    a('    <light pos="0 2 4" dir="0 -0.5 -1" diffuse="0.8 0.8 0.8"/>')
    a('    <light pos="2 -2 3" dir="-0.3 0.3 -1" diffuse="0.4 0.4 0.4"/>')
    # Floor (below the track)
    a('    <geom name="floor" type="plane" size="3 3 0.01" pos="0 -0.6 0"'
      '     euler="90 0 0" rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0"/>')
    # RGB axes at origin: X=red, Y=green, Z=blue
    a('    <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.008" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0 0 0.5 0" size="0.008" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.008" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>')

    # Sprockets (hinge about Z axis — track is in XY plane, Y=up)
    for name, cx, cy in SPROCKETS:
        a(f'    <body name="{name}_spr" pos="{cx} {cy} 0">')
        if name == "idler":
            a(f'      <joint name="{name}_slide" type="slide" axis="1 0 0"'
              f' stiffness="{TENSION_K}" damping="80" range="-0.05 0.3"/>')
        a(f'      <joint name="{name}_hinge" type="hinge" axis="0 0 1" damping="0.2"/>')
        col = {"drive": "0.7 0.2 0.2 1", "idler": "0.2 0.2 0.7 1", "mid": "0.2 0.6 0.2 1"}[name]
        a(f'      <geom type="cylinder" size="{HUB_R} {HUB_Z}" rgba="{col}"'
          f' contype="0" conaffinity="0"/>')
        if name != "mid":
            for mi in range(12):
                th = 2 * math.pi * mi / 12
                a(f'      <geom type="sphere" size="0.012"'
                  f' pos="{SPROCKET_R * math.cos(th):.4f} {SPROCKET_R * math.sin(th):.4f} 0"'
                  f' rgba="1 1 0 0.5" contype="0" conaffinity="0"/>')
        a(f'    </body>')

    # ── Chain: kinematic tree ──
    # Link 0: freejoint root at stadium position
    half_len = LINK_PITCH / 2 - 0.005
    x0, y0, a0 = stadium_point(0)
    qw0, qz0 = math.cos(a0 / 2), math.sin(a0 / 2)

    a(f'    <body name="link_0" pos="{x0:.6f} {y0:.6f} 0" quat="{qw0:.6f} 0 0 {qz0:.6f}">')
    a(f'      <freejoint name="link_0_jnt"/>')
    a(f'      <geom type="box" size="{half_len:.4f} {LINK_THICK} {LINK_WIDTH}"'
      f' rgba="1 0.2 0.2 1" mass="0.05" contype="0" conaffinity="0"/>')
    # Markers: green=fwd, blue=bwd, magenta=center
    a(f'      <geom type="sphere" size="0.015" pos="{LINK_PITCH/2:.4f} 0 0"'
      f' rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
    a(f'      <geom type="sphere" size="0.015" pos="{-LINK_PITCH/2:.4f} 0 0"'
      f' rgba="0 0 1 0.5" contype="0" conaffinity="0"/>')
    a(f'      <geom type="sphere" size="0.02" pos="0 0 0"'
      f' rgba="1 0 1 0.5" contype="0" conaffinity="0"/>')

    # Links 1..N-1: nested children with hinge joints (Z axis = out of track plane)
    for i in range(1, N_LINKS):
        col = "0.9 0.6 0.1 1"
        a(f'      <body name="link_{i}" pos="{LINK_PITCH:.6f} 0 0">')
        a(f'        <joint name="hinge_{i}" type="hinge" axis="0 0 1"'
          f' pos="{-LINK_PITCH/2:.6f} 0 0" damping="0.05"/>')
        a(f'        <geom type="box" size="{half_len:.4f} {LINK_THICK} {LINK_WIDTH}"'
          f' rgba="{col}" mass="0.05" contype="0" conaffinity="0"/>')
        a(f'        <geom type="sphere" size="0.015" pos="{LINK_PITCH/2:.4f} 0 0"'
          f' rgba="0 1 0 0.5" contype="0" conaffinity="0"/>')
        a(f'        <geom type="sphere" size="0.015" pos="{-LINK_PITCH/2:.4f} 0 0"'
          f' rgba="0 0 1 0.5" contype="0" conaffinity="0"/>')
        a(f'        <geom type="sphere" size="0.02" pos="0 0 0"'
          f' rgba="1 0 1 0.5" contype="0" conaffinity="0"/>')

    # Close nested bodies
    for _ in range(N_LINKS):
        a('    </body>')

    a('  </worldbody>')

    # Actuators
    a('  <actuator>')
    for name, _, _ in SPROCKETS:
        a(f'    <velocity name="{name}_motor" joint="{name}_hinge"'
          f' kv="50" ctrllimited="true" ctrlrange="-5 5"/>')
    a('  </actuator>')

    # Equality constraints
    a('  <equality>')
    # Loop closure
    a(f'    <connect name="loop_close" body1="link_{N_LINKS-1}" body2="link_0"'
      f' anchor="{LINK_PITCH/2:.6f} 0 0"/>')
    # Sprocket engagement (one per link per sprocket, all disabled)
    for i in range(N_LINKS):
        for name, _, _ in SPROCKETS:
            if name == "mid":
                continue
            a(f'    <connect name="eng_{i}_{name}" body1="{name}_spr" body2="link_{i}"'
              f' anchor="0 0 0" active="false"/>')
    a('  </equality>')

    a('</mujoco>')
    return '\n'.join(L)


def init_lookups(model):
    eng_ids = {}
    for idx in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if nm and nm.startswith("eng_"):
            parts = nm.split("_")
            link_i = int(parts[1])
            spr = "_".join(parts[2:])
            eng_ids[(link_i, spr)] = idx

    link_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
                 for i in range(N_LINKS)]
    spr_bids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{n}_spr")
                for n, _, _ in SPROCKETS}
    jnt_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_hinge")
               for n, _, _ in SPROCKETS}
    act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
               for n, _, _ in SPROCKETS}

    # Chain hinge joint IDs
    hinge_jids = []
    for i in range(1, N_LINKS):
        hinge_jids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hinge_{i}"))

    return eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids


def set_initial_shape(model, data, hinge_jids):
    """Set hinge qpos so chain forms the stadium, then fix loop closure anchor."""
    for idx, jid in enumerate(hinge_jids):
        i = idx + 1
        _, _, ang_i = stadium_point(i * LINK_PITCH)
        _, _, ang_prev = stadium_point((i - 1) * LINK_PITCH)
        delta = _norm_angle(ang_i - ang_prev)
        # Y-axis hinge: positive rotates X toward -Z (opposite to Z-axis convention)
        data.qpos[model.jnt_qposadr[jid]] = -delta
    mujoco.mj_forward(model, data)

    # Fix loop closure anchor2: MuJoCo auto-computed it from the straight-chain
    # initial pose, but we've bent the chain into a stadium. Recompute anchor2
    # so it matches the current (correct) configuration.
    for idx in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if nm == "loop_close":
            bid1 = model.eq_obj1id[idx]  # link_29
            bid2 = model.eq_obj2id[idx]  # link_0
            a1 = model.eq_data[idx, 0:3]
            # World position of anchor1
            w1 = data.xpos[bid1] + data.xmat[bid1].reshape(3, 3) @ a1
            # Compute anchor2 in body2's (link_0) local frame
            a2 = data.xmat[bid2].reshape(3, 3).T @ (w1 - data.xpos[bid2])
            model.eq_data[idx, 3:6] = a2
            break
    mujoco.mj_forward(model, data)


# Engagement state: {(link_i, spr_name): local_angle}
_engaged = {}


def seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    """Pre-seed engagement constraints for links that start on arcs.

    Uses the known stadium geometry to compute exact anchors that match
    the current chain pose — zero initial constraint error.
    """
    top_len = 2 * HALF_SPAN
    arc_len = math.pi * SPROCKET_R

    for i in range(N_LINKS):
        s = (i * LINK_PITCH) % PERIMETER

        # Determine which sprocket this link is on
        spr_name = None
        if top_len < s < top_len + arc_len:
            spr_name = "drive"  # left arc
        elif 2 * top_len + arc_len < s:
            spr_name = "idler"  # right arc

        if spr_name is None:
            continue

        key = (i, spr_name)
        eq_idx = eng_ids.get(key)
        if eq_idx is None:
            continue

        bid = link_bids[i]
        sbid = spr_bids[spr_name]
        lx, lz = data.xpos[bid][0], data.xpos[bid][2]
        sx, sz = data.xpos[sbid][0], data.xpos[sbid][2]
        dx, dz = lx - sx, lz - sz
        dist = math.sqrt(dx * dx + dz * dz)
        world_angle = math.atan2(dz, dx)
        spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[spr_name]]]
        local_angle = _norm_angle(world_angle - spr_angle)

        # Anchor in sprocket local frame (XZ plane, Y=0)
        model.eq_data[eq_idx, 0] = dist * math.cos(local_angle)
        model.eq_data[eq_idx, 1] = 0.0
        model.eq_data[eq_idx, 2] = dist * math.sin(local_angle)
        model.eq_data[eq_idx, 3:6] = 0.0
        data.eq_active[eq_idx] = 1
        _engaged[key] = local_angle

    mujoco.mj_forward(model, data)


def update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    for i in range(N_LINKS):
        lx = data.xpos[link_bids[i]][0]
        lz = data.xpos[link_bids[i]][2]

        for name, _, _ in SPROCKETS:
            if name == "mid":
                continue
            key = (i, name)
            eq_idx = eng_ids.get(key)
            if eq_idx is None:
                continue

            sx = data.xpos[spr_bids[name]][0]
            sz = data.xpos[spr_bids[name]][2]
            spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[name]]]
            dx, dz = lx - sx, lz - sz
            dist = math.sqrt(dx * dx + dz * dz)

            # Determine if link should be on this sprocket's arc
            on_arc = False
            if name == "drive":
                on_arc = lx < sx + LINK_PITCH * 0.3
            elif name == "idler":
                on_arc = lx > sx - LINK_PITCH * 0.3
            on_arc = on_arc and abs(dist - SPROCKET_R) < 0.10

            if on_arc:
                world_angle = math.atan2(dz, dx)
                local_angle = _norm_angle(world_angle - spr_angle)

                # Anchor in sprocket local frame (XZ plane)
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


def step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids):
    t = data.time
    vel = TARGET_VEL * min(1.0, t / 1.0)
    for name, _, _ in SPROCKETS:
        data.ctrl[act_ids[name]] = vel

    update_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)

    mujoco.mj_step(model, data)


def run_gui():
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids = init_lookups(model)
    set_initial_shape(model, data, hinge_jids)
    seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)

    import mujoco.viewer as mjv
    import time
    with mjv.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        sim_start = data.time
        while viewer.is_running():
            # Step until sim time catches up to wall time
            wall_elapsed = time.perf_counter() - wall_start
            target_sim_time = sim_start + wall_elapsed
            while data.time < target_sim_time:
                step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)
            viewer.sync()


def run_debug():
    xml = build_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids = init_lookups(model)
    set_initial_shape(model, data, hinge_jids)
    seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids)

    # Verify initial shape
    max_err = 0
    for i in range(N_LINKS):
        bid = link_bids[i]
        sx, sy, _ = stadium_point(i * LINK_PITCH)
        err = math.sqrt((data.xpos[bid][0] - sx) ** 2 + (data.xpos[bid][1] - sy) ** 2)
        max_err = max(max_err, err)
    print(f"Initial shape error: {max_err:.4f}")

    # Loop closure gap
    bid0, bid29 = link_bids[0], link_bids[N_LINKS - 1]
    mat0, mat29 = data.xmat[bid0].reshape(3, 3), data.xmat[bid29].reshape(3, 3)
    fwd29 = data.xpos[bid29] + mat29 @ np.array([LINK_PITCH / 2, 0, 0])
    bwd0 = data.xpos[bid0] + mat0 @ np.array([-LINK_PITCH / 2, 0, 0])
    print(f"Loop closure gap: {np.linalg.norm(fwd29 - bwd0):.6f}")

    # State history for diagnostics
    history = []

    for step in range(5000):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        max_v = np.max(np.abs(data.qvel))
        n_eng = sum(1 for idx in eng_ids.values() if data.eq_active[idx])
        max_ev = np.max(np.abs(data.efc_pos[:data.nefc])) if data.nefc > 0 else 0
        t = data.time

        # Sprocket angles
        spr_angles = {}
        for name, _, _ in SPROCKETS:
            jid = jnt_ids[name]
            spr_angles[name] = data.qpos[model.jnt_qposadr[jid]]

        # Track error: distance of each link center from ideal stadium path
        track_errs = []
        for i in range(N_LINKS):
            bid = link_bids[i]
            px, pz = data.xpos[bid][0], data.xpos[bid][2]
            # Nearest point on stadium (XZ plane)
            best = float('inf')
            for name, _, _ in SPROCKETS:
                if name == 'mid':
                    continue
                sbid = spr_bids[name]
                sx, sz = data.xpos[sbid][0], data.xpos[sbid][2]
                d = abs(math.sqrt((px-sx)**2 + (pz-sz)**2) - SPROCKET_R)
                best = min(best, d)
            # Top/bottom straights
            if -HALF_SPAN - 0.2 <= px <= HALF_SPAN + 0.2:
                best = min(best, abs(pz - SPROCKET_R), abs(pz + SPROCKET_R))
            track_errs.append(best)

        # Link_0 arc-length position (how far it has traveled)
        bid0 = link_bids[0]
        link0_pos = (data.xpos[bid0][0], data.xpos[bid0][2])

        # Per-sprocket velocities
        spr_vels = {}
        for name, _, _ in SPROCKETS:
            jid = jnt_ids[name]
            spr_vels[name] = data.qvel[model.jnt_dofadr[jid]]
        chain_speed = spr_vels['drive'] * SPROCKET_R

        # Which links are engaged to which sprocket
        eng_drive = sum(1 for k in _engaged if k[1] == 'drive')
        eng_idler = sum(1 for k in _engaged if k[1] == 'idler')

        rec = {
            't': t, 'step': step, 'max_v': max_v, 'n_eng': n_eng,
            'eq_v': max_ev, 'ncon': data.ncon,
            'drive_angle': math.degrees(spr_angles['drive']),
            'idler_angle': math.degrees(spr_angles['idler']),
            'drive_vel': spr_vels['drive'],
            'idler_vel': spr_vels['idler'],
            'chain_speed': chain_speed,
            'trk_max': max(track_errs), 'trk_mean': np.mean(track_errs),
            'link0': link0_pos,
            'eng_drive': eng_drive, 'eng_idler': eng_idler,
        }
        history.append(rec)

        if step < 10 or step % 200 == 0 or max_v > 100 or np.any(np.isnan(data.qpos)):
            print(f"step {step:4d}  t={t:.2f}  "
                  f"v={max_v:6.1f}  eq_v={max_ev:.4f}  "
                  f"drv={rec['drive_angle']:6.1f}d({spr_vels['drive']:5.2f}r/s) "
                  f"idl={rec['idler_angle']:6.1f}d({spr_vels['idler']:5.2f}r/s) "
                  f"eng=D{eng_drive}+I{eng_idler}  "
                  f"spd={chain_speed:5.2f}  "
                  f"trk={rec['trk_max']:.3f}/{rec['trk_mean']:.3f}  "
                  f"l0=({link0_pos[0]:.2f},{link0_pos[1]:.2f})")
            if np.any(np.isnan(data.qpos)):
                print(">>> NaN")
                break

    # Summary
    print()
    print("=== SUMMARY ===")
    if len(history) > 100:
        late = history[-200:]
        print(f"Last 200 steps (t={late[0]['t']:.1f}-{late[-1]['t']:.1f}):")
        print(f"  drive angle: {late[0]['drive_angle']:.1f} -> {late[-1]['drive_angle']:.1f} deg")
        print(f"  chain speed: mean={np.mean([r['chain_speed'] for r in late]):.3f} m/s")
        print(f"  track error: mean={np.mean([r['trk_mean'] for r in late]):.4f}, max={max(r['trk_max'] for r in late):.4f}")
        print(f"  eq violation: mean={np.mean([r['eq_v'] for r in late]):.4f}, max={max(r['eq_v'] for r in late):.4f}")
        print(f"  engaged: mean={np.mean([r['n_eng'] for r in late]):.1f}")
        # Did link_0 move?
        l0_start = history[0]['link0']
        l0_end = history[-1]['link0']
        l0_dist = math.sqrt((l0_end[0]-l0_start[0])**2 + (l0_end[1]-l0_start[1])**2)
        print(f"  link_0 displacement: {l0_dist:.4f} m")
        print(f"  link_0: ({l0_start[0]:.2f},{l0_start[1]:.2f}) -> ({l0_end[0]:.2f},{l0_end[1]:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        run_debug()
    else:
        run_gui()
