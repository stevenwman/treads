"""
Single track: kinematic-tree chain with dynamic sprocket engagement.

Track in XZ plane (X=forward, Z=up), gravity -Z, hinge about Y.
Floor is the XY plane at z=0.

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
LINK_THICK = 0.02         # half-size in Z (radial, in track plane)
LINK_WIDTH = 0.10         # half-size in Y (lateral, out of track plane)
TIMESTEP = 0.002
TARGET_VEL = 1.0
TENSION_K = 80.0

HUB_R = SPROCKET_R - 0.04
HUB_HALF_Y = LINK_WIDTH + 0.02  # hub half-length along Y

PERIMETER = 2 * math.pi * SPROCKET_R + 2 * (2 * HALF_SPAN)
LINK_PITCH = PERIMETER / N_LINKS

# Sprockets: (name, x_pos, z_pos)
SPROCKET_Z = SPROCKET_R + 0.05  # height of sprocket centers above ground
SPROCKETS = [
    ("drive", -HALF_SPAN, SPROCKET_Z),
    ("idler",  HALF_SPAN, SPROCKET_Z),
    ("mid",    0.0,        SPROCKET_Z),
]

ARC_HALF = math.pi * 0.40


def _norm_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def stadium_point(s):
    """(x_local, z_local, angle) for arc-length s along stadium.

    Returns position relative to sprocket center height (add SPROCKET_Z for world Z).
    The angle is the tangent direction in the XZ plane.
    Top straight: z_local = +R, going -X.  Bottom: z_local = -R, going +X.
    """
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
    a(f'  <option timestep="{TIMESTEP}" gravity="0 0 -9.81"'
      f' iterations="300" solver="Newton" tolerance="1e-10" noslip_iterations="10"/>')
    a('  <size nconmax="500" njmax="3000"/>')
    a('  <visual><global offwidth="1200" offheight="800"/></visual>')
    a('  <default>')
    a('    <geom friction="0.8 0.01 0.01" condim="4" margin="0.005"/>')
    a('    <equality solref="0.005 1" solimp="0.95 0.99 0.001"/>')
    a('  </default>')
    a('  <worldbody>')
    a('    <camera name="overview" pos="0 -4 1.5" xyaxes="1 0 0 0 0 1" fovy="50" mode="fixed"/>')
    a('    <camera name="top" pos="0 0 5" xyaxes="1 0 0 0 1 0" fovy="50" mode="fixed"/>')
    a('    <light pos="0 -3 4" dir="0 0.5 -0.5" diffuse="0.8 0.8 0.8"/>')
    a('    <light pos="2 2 4" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>')
    # Floor (XY plane at z=0)
    a('    <geom name="floor" type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1"'
      '     contype="0" conaffinity="0"/>')
    # RGB axes
    a('    <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.008" rgba="1 0 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0 0 0.5 0" size="0.008" rgba="0 1 0 0.8" contype="0" conaffinity="0"/>')
    a('    <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.008" rgba="0 0 1 0.8" contype="0" conaffinity="0"/>')

    # Sprockets — hinge about Y, in XZ plane
    for name, cx, cz in SPROCKETS:
        a(f'    <body name="{name}_spr" pos="{cx} 0 {cz}">')
        if name == "idler":
            a(f'      <joint name="{name}_slide" type="slide" axis="1 0 0"'
              f' stiffness="{TENSION_K}" damping="80" range="-0.05 0.3"/>')
        a(f'      <joint name="{name}_hinge" type="hinge" axis="0 1 0" damping="0.2"/>')
        col = {"drive": "0.7 0.2 0.2 1", "idler": "0.2 0.2 0.7 1", "mid": "0.2 0.6 0.2 1"}[name]
        # Cylinder default axis is Z; rotate 90 about X to align with Y
        a(f'      <geom type="cylinder" size="{HUB_R} {HUB_HALF_Y}" euler="90 0 0" rgba="{col}"'
          f' contype="1" conaffinity="2"/>')
        a(f'    </body>')

    # ── Chain: kinematic tree ──
    half_len = LINK_PITCH / 2 - 0.005

    # Link 0: freejoint root
    xl, zl, ang = stadium_point(0)
    wz = SPROCKET_Z + zl
    # Y-axis rotation: negate angle (Y hinge positive = X toward -Z, we want CCW in XZ)
    qw = math.cos(-ang / 2)
    qy = math.sin(-ang / 2)

    a(f'    <body name="link_0" pos="{xl:.6f} 0 {wz:.6f}" quat="{qw:.6f} 0 {qy:.6f} 0">')
    a(f'      <freejoint name="link_0_jnt"/>')
    a(f'      <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
      f' rgba="1 0.2 0.2 1" mass="0.05" contype="2" conaffinity="1"/>')

    # Links 1..N-1: nested children with Y-axis hinge
    for i in range(1, N_LINKS):
        col = "0.9 0.6 0.1 1"
        a(f'      <body name="link_{i}" pos="{LINK_PITCH:.6f} 0 0">')
        a(f'        <joint name="hinge_{i}" type="hinge" axis="0 1 0"'
          f' pos="{-LINK_PITCH/2:.6f} 0 0" damping="0.05"/>')
        a(f'        <geom type="box" size="{half_len:.4f} {LINK_WIDTH} {LINK_THICK}"'
          f' rgba="{col}" mass="0.05" contype="2" conaffinity="1"/>')

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
    a(f'    <connect name="loop_close" body1="link_{N_LINKS-1}" body2="link_0"'
      f' anchor="{LINK_PITCH/2:.6f} 0 0"/>')
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
            eng_ids[(int(parts[1]), "_".join(parts[2:]))] = idx

    link_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
                 for i in range(N_LINKS)]
    spr_bids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{n}_spr")
                for n, _, _ in SPROCKETS}
    jnt_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_hinge")
               for n, _, _ in SPROCKETS}
    act_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
               for n, _, _ in SPROCKETS}
    hinge_jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"hinge_{i}")
                  for i in range(1, N_LINKS)]
    return eng_ids, link_bids, spr_bids, jnt_ids, act_ids, hinge_jids


def set_initial_shape(model, data, hinge_jids):
    """Set hinge qpos for stadium shape, then fix loop closure anchor."""
    for idx, jid in enumerate(hinge_jids):
        i = idx + 1
        _, _, ang_i = stadium_point(i * LINK_PITCH)
        _, _, ang_prev = stadium_point((i - 1) * LINK_PITCH)
        delta = _norm_angle(ang_i - ang_prev)
        # Y-axis hinge: positive rotates X toward -Z. Stadium angles are CCW (X toward +Z).
        # So negate.
        data.qpos[model.jnt_qposadr[jid]] = -delta
    mujoco.mj_forward(model, data)

    # Fix loop closure anchor2 (auto-computed from straight-chain pose)
    for idx in range(model.neq):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx) == "loop_close":
            bid1, bid2 = model.eq_obj1id[idx], model.eq_obj2id[idx]
            w1 = data.xpos[bid1] + data.xmat[bid1].reshape(3, 3) @ model.eq_data[idx, 0:3]
            model.eq_data[idx, 3:6] = data.xmat[bid2].reshape(3, 3).T @ (w1 - data.xpos[bid2])
            break
    mujoco.mj_forward(model, data)


_engaged = {}


def seed_initial_engagement(model, data, link_bids, spr_bids, eng_ids, jnt_ids):
    """Pre-seed engagement for links starting on arcs."""
    top_len = 2 * HALF_SPAN
    arc_len = math.pi * SPROCKET_R

    for i in range(N_LINKS):
        s = (i * LINK_PITCH) % PERIMETER
        spr_name = None
        if top_len < s < top_len + arc_len:
            spr_name = "drive"
        elif 2 * top_len + arc_len < s:
            spr_name = "idler"
        if spr_name is None:
            continue

        key = (i, spr_name)
        eq_idx = eng_ids.get(key)
        if eq_idx is None:
            continue

        bid, sbid = link_bids[i], spr_bids[spr_name]
        dx = data.xpos[bid][0] - data.xpos[sbid][0]
        dz = data.xpos[bid][2] - data.xpos[sbid][2]
        dist = math.sqrt(dx * dx + dz * dz)
        world_angle = math.atan2(dz, dx)
        spr_angle = data.qpos[model.jnt_qposadr[jnt_ids[spr_name]]]
        # Y-axis: local = world + spr_angle
        local_angle = _norm_angle(world_angle + spr_angle)

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

            # Apex-only engagement: only near deepest wrap point
            on_arc = False
            if name == "drive":
                on_arc = lx < sx - SPROCKET_R * 0.5
            elif name == "idler":
                on_arc = lx > sx + SPROCKET_R * 0.5
            on_arc = on_arc and abs(dist - SPROCKET_R) < 0.10

            if key in _engaged:
                # Already engaged — angle-based disengagement
                local_angle = _engaged[key]
                world_angle = _norm_angle(local_angle - spr_angle)
                if name == "drive":
                    off = abs(_norm_angle(world_angle - math.pi))
                else:
                    off = abs(_norm_angle(world_angle))
                if off > ARC_HALF:
                    del _engaged[key]
                    data.eq_active[eq_idx] = 0
            elif on_arc:
                # New engagement — fixed anchor
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
        sx, sz_local, _ = stadium_point(i * LINK_PITCH)
        sz = SPROCKET_Z + sz_local
        err = math.sqrt((data.xpos[bid][0] - sx) ** 2 + (data.xpos[bid][2] - sz) ** 2)
        max_err = max(max_err, err)
    print(f"Initial shape error: {max_err:.4f}")

    bid0, bidN = link_bids[0], link_bids[N_LINKS - 1]
    mat0, matN = data.xmat[bid0].reshape(3, 3), data.xmat[bidN].reshape(3, 3)
    fwd = data.xpos[bidN] + matN @ np.array([LINK_PITCH / 2, 0, 0])
    bwd = data.xpos[bid0] + mat0 @ np.array([-LINK_PITCH / 2, 0, 0])
    print(f"Loop closure gap: {np.linalg.norm(fwd - bwd):.6f}")

    history = []
    for step in range(5000):
        step_sim(model, data, eng_ids, link_bids, spr_bids, jnt_ids, act_ids)

        max_v = np.max(np.abs(data.qvel))
        n_eng = sum(1 for idx in eng_ids.values() if data.eq_active[idx])
        max_ev = np.max(np.abs(data.efc_pos[:data.nefc])) if data.nefc > 0 else 0
        t = data.time

        spr_angles = {n: data.qpos[model.jnt_qposadr[jnt_ids[n]]] for n, _, _ in SPROCKETS}
        spr_vels = {n: data.qvel[model.jnt_dofadr[jnt_ids[n]]] for n, _, _ in SPROCKETS}

        track_errs = []
        for i in range(N_LINKS):
            bid = link_bids[i]
            px, pz = data.xpos[bid][0], data.xpos[bid][2]
            best = float('inf')
            for name, _, _ in SPROCKETS:
                if name == 'mid':
                    continue
                sbid = spr_bids[name]
                sx, sz = data.xpos[sbid][0], data.xpos[sbid][2]
                d = abs(math.sqrt((px - sx) ** 2 + (pz - sz) ** 2) - SPROCKET_R)
                best = min(best, d)
            if -HALF_SPAN - 0.2 <= px <= HALF_SPAN + 0.2:
                best = min(best, abs(pz - (SPROCKET_Z + SPROCKET_R)),
                           abs(pz - (SPROCKET_Z - SPROCKET_R)))
            track_errs.append(best)

        link0_pos = (data.xpos[link_bids[0]][0], data.xpos[link_bids[0]][2])
        chain_speed = spr_vels['drive'] * SPROCKET_R
        eng_d = sum(1 for k in _engaged if k[1] == 'drive')
        eng_i = sum(1 for k in _engaged if k[1] == 'idler')

        rec = {'t': t, 'max_v': max_v, 'eq_v': max_ev, 'n_eng': n_eng,
               'drive_angle': math.degrees(spr_angles['drive']),
               'idler_angle': math.degrees(spr_angles['idler']),
               'drive_vel': spr_vels['drive'], 'chain_speed': chain_speed,
               'trk_max': max(track_errs), 'trk_mean': np.mean(track_errs),
               'link0': link0_pos, 'eng_d': eng_d, 'eng_i': eng_i}
        history.append(rec)

        if step < 10 or step % 200 == 0 or max_v > 100 or np.any(np.isnan(data.qpos)):
            print(f"step {step:4d}  t={t:.2f}  "
                  f"v={max_v:6.1f}  eq_v={max_ev:.4f}  "
                  f"drv={rec['drive_angle']:6.1f}d({spr_vels['drive']:5.2f}r/s) "
                  f"idl={rec['idler_angle']:6.1f}d({spr_vels['idler']:5.2f}r/s) "
                  f"eng=D{eng_d}+I{eng_i}  "
                  f"spd={chain_speed:5.2f}  "
                  f"trk={rec['trk_max']:.3f}/{rec['trk_mean']:.3f}  "
                  f"l0=({link0_pos[0]:.2f},{link0_pos[1]:.2f})")
            if np.any(np.isnan(data.qpos)):
                print(">>> NaN")
                break

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
        l0s, l0e = history[0]['link0'], history[-1]['link0']
        print(f"  link_0: ({l0s[0]:.2f},{l0s[1]:.2f}) -> ({l0e[0]:.2f},{l0e[1]:.2f})"
              f"  disp={math.sqrt((l0e[0]-l0s[0])**2+(l0e[1]-l0s[1])**2):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        run_debug()
    else:
        run_gui()
