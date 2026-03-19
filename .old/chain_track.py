"""
Dual-chain track with cross-links and cylindrical-pin sprockets.
Sprocket pins (capsules along Z) create concave valleys that cradle chain links.
Two parallel side chains connected by tread plates.
"""
import numpy as np
import mujoco
import mujoco.viewer
import imageio

# ── Layout ──────────────────────────────────────────────────────────────
HALF_SPAN   = 1.4        # half distance between sprocket centers
SPROCKET_R  = 0.35       # pitch radius (chain rides at this radius)

# Chain
N_LINKS     = 30          # more links = finer chain = better wrapping
CHAIN_Z     = 0.10        # Z offset of each side chain from center
SIDE_THICK  = 0.010       # half-height of side chain link (radial)
SIDE_DEPTH  = 0.012       # half-depth of side chain link (Z)
GAP_FRAC    = 0.25        # gap fraction between link geoms

# Cross-links (tread plates)
CROSS_THICK = 0.008       # half-height (radial)
CROSS_DEPTH = 0.006       # half-width along track direction

# Sprocket pins — capsules arranged in a circle
PIN_RADIUS  = 0.018       # radius of each sprocket pin
ROLLER_RADIUS = 0.014     # radius of chain rollers at each joint

# Physics
DRIVE_TORQUE  = 3.0
GRAVITY       = -3.0
TIMESTEP      = 0.001
SIM_TIME      = 5.0
N_SOLVER_ITER = 200
TENSION_FORCE = 15.0      # force pushing idler outward to tension chain (N)
TENSION_RANGE = 0.3       # max idler slide travel (m)

# ── Derived ─────────────────────────────────────────────────────────────
chain_r    = SPROCKET_R + PIN_RADIUS + SIDE_THICK  # actual chain path radius
perimeter  = 2 * np.pi * chain_r + 2 * (2 * HALF_SPAN)
link_pitch = perimeter / N_LINKS
side_half  = link_pitch * (1 - GAP_FRAC) / 2
gap_size   = link_pitch * GAP_FRAC

# Pins: one per link pitch along sprocket circumference
N_PINS = max(4, round(2 * np.pi * SPROCKET_R / link_pitch))
pin_angular_pitch = 2 * np.pi / N_PINS


def stadium_point(s, half_span, radius):
    """Arc-length parameterized stadium. Returns (x,y), tangent_angle."""
    L_str = 2 * half_span
    L_arc = np.pi * radius
    perim = 2 * L_str + 2 * L_arc
    s = s % perim

    if s < L_str:
        return np.array([half_span - s, radius]), np.pi
    s -= L_str

    if s < L_arc:
        theta = np.pi / 2 + s / radius
        x = -half_span + radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.array([x, y]), theta + np.pi / 2
    s -= L_arc

    if s < L_str:
        return np.array([-half_span + s, -radius]), 0.0
    s -= L_str

    theta = -np.pi / 2 + s / radius
    x = half_span + radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.array([x, y]), theta + np.pi / 2


def build_xml():
    lines = []
    a = lines.append

    a('<mujoco model="dual_chain_track">')
    a(f'  <option timestep="{TIMESTEP}" gravity="0 {GRAVITY} 0" integrator="implicit"')
    a(f'          noslip_iterations="{N_SOLVER_ITER}" viscosity="0.005">')
    a(f'    <flag warmstart="enable"/>')
    a(f'  </option>')
    a(f'  <compiler angle="radian"/>')
    a(f'  <size nconmax="6000" njmax="12000"/>')
    a('')
    a('  <default>')
    a('    <geom condim="4" friction="1.5 0.005 0.001" margin="0.0005"')
    a('          solimp="0.97 0.999 0.0001 0.5 2" solref="0.004 1"/>')
    a('    <equality solimp="0.97 0.999 0.0001 0.5 2" solref="0.002 1"/>')
    a('  </default>')
    a('')
    a('  <visual>')
    a('    <global offwidth="800" offheight="450"/>')
    a('  </visual>')
    a('')
    a('  <worldbody>')
    a('    <camera name="top" pos="0 0 4" zaxis="0 0 1" mode="fixed"/>')
    a('    <camera name="persp" pos="0 -1.0 2.5" xyaxes="1 0 0 0 0.93 0.37" mode="fixed"/>')
    a('    <light pos="0 2 4" dir="0 -0.3 -1" diffuse="1 1 1"/>')
    a('    <light pos="0 -2 4" dir="0 0.3 -1" diffuse="0.5 0.5 0.5"/>')

    # ── Sprockets with cylindrical pins ──
    hub_z_half = CHAIN_Z + SIDE_DEPTH + 0.01   # full track width
    pin_z_half = CHAIN_Z * 0.3                  # short — just poke between cross-links
    hub_r = SPROCKET_R - PIN_RADIUS * 2

    # Drive and mid sprockets — fixed position, hinge only
    for name, xpos, rgba_hub, rgba_pin in [
        ("drive", -HALF_SPAN, "0.55 0.15 0.15 1", "0.8 0.3 0.3 1"),
        ("mid",    0.0,       "0.15 0.45 0.15 1", "0.3 0.7 0.3 1"),
    ]:
        a(f'    <body name="{name}_sprocket" pos="{xpos} 0 0">')
        a(f'      <joint name="{name}_joint" type="hinge" axis="0 0 1" damping="0.1"/>')
        a(f'      <geom name="{name}_hub" type="cylinder" size="{hub_r:.4f} {hub_z_half:.4f}"'
          f' rgba="{rgba_hub}"/>')
        for i in range(N_PINS):
            ang = 2 * np.pi * i / N_PINS
            px = SPROCKET_R * np.cos(ang)
            py = SPROCKET_R * np.sin(ang)
            a(f'      <geom name="{name}_pin_{i}" type="capsule"'
              f' size="{PIN_RADIUS} {pin_z_half:.4f}"'
              f' pos="{px:.5f} {py:.5f} 0"'
              f' rgba="{rgba_pin}" mass="0.002"/>')
        a(f'    </body>')

    # Idler sprocket — on slider joint along X for tensioning
    a(f'    <body name="idler_sprocket" pos="{HALF_SPAN} 0 0">')
    a(f'      <joint name="idler_slide" type="slide" axis="1 0 0"'
      f' damping="2.0" limited="true" range="0 {TENSION_RANGE}"/>')
    a(f'      <joint name="idler_joint" type="hinge" axis="0 0 1" damping="0.1"/>')
    a(f'      <geom name="idler_hub" type="cylinder" size="{hub_r:.4f} {hub_z_half:.4f}"'
      f' rgba="0.15 0.15 0.55 1"/>')
    for i in range(N_PINS):
        ang = 2 * np.pi * i / N_PINS
        px = SPROCKET_R * np.cos(ang)
        py = SPROCKET_R * np.sin(ang)
        a(f'      <geom name="idler_pin_{i}" type="capsule"'
          f' size="{PIN_RADIUS} {pin_z_half:.4f}"'
          f' pos="{px:.5f} {py:.5f} 0"'
          f' rgba="0.3 0.3 0.8 1" mass="0.002"/>')
    a(f'    </body>')

    # ── Track links ──
    # Chain rests ON the pins: offset outward by pin radius + link half-thickness
    chain_r = SPROCKET_R + PIN_RADIUS + SIDE_THICK
    for i in range(N_LINKS):
        s = i * link_pitch
        pos, ang = stadium_point(s, HALF_SPAN, chain_r)
        qw = np.cos(ang / 2)
        qz = np.sin(ang / 2)

        a(f'    <body name="link_{i}" pos="{pos[0]:.6f} {pos[1]:.6f} 0"'
          f' quat="{qw:.6f} 0 0 {qz:.6f}">')
        a(f'      <freejoint name="free_{i}"/>')

        # Left side chain
        a(f'      <geom name="side_L_{i}" type="box"'
          f' size="{side_half:.5f} {SIDE_THICK} {SIDE_DEPTH}"'
          f' pos="0 0 {-CHAIN_Z}" rgba="1 0.6 0 1" mass="0.008"/>')
        # Right side chain
        a(f'      <geom name="side_R_{i}" type="box"'
          f' size="{side_half:.5f} {SIDE_THICK} {SIDE_DEPTH}"'
          f' pos="0 0 {CHAIN_Z}" rgba="1 0.6 0 1" mass="0.008"/>')
        # Cross-link (tread plate)
        cross_half_z = CHAIN_Z + SIDE_DEPTH
        a(f'      <geom name="cross_{i}" type="box"'
          f' size="{CROSS_DEPTH} {CROSS_THICK} {cross_half_z:.5f}"'
          f' pos="0 {SIDE_THICK + CROSS_THICK:.5f} 0" rgba="0.8 0.8 0.2 1" mass="0.003"/>')

        # Roller at forward joint — capsule along Z, sits between sprocket pins
        roller_z_half = CHAIN_Z - SIDE_DEPTH - 0.003
        a(f'      <geom name="roller_{i}" type="capsule"'
          f' size="{ROLLER_RADIUS} {roller_z_half:.4f}"'
          f' pos="{link_pitch/2:.5f} 0 0" rgba="0.9 0.5 0.1 1" mass="0.002"/>')

        # Pin sites at pitch ends
        a(f'      <site name="pin_{i}_fwd" pos="{link_pitch/2:.5f} 0 0" size="0.004" rgba="0 1 0 1"/>')
        a(f'      <site name="pin_{i}_bwd" pos="{-link_pitch/2:.5f} 0 0" size="0.004" rgba="1 0 1 1"/>')
        a(f'    </body>')

    a('  </worldbody>')

    # ── Actuators ──
    a('  <actuator>')
    for name in ["drive", "idler", "mid"]:
        a(f'    <motor name="{name}_motor" joint="{name}_joint" gear="1"'
          f' ctrllimited="true" ctrlrange="-{DRIVE_TORQUE} {DRIVE_TORQUE}"/>')
    # Tensioner: constant force pushing idler outward
    a(f'    <motor name="tensioner" joint="idler_slide" gear="1"'
      f' ctrllimited="true" ctrlrange="0 {TENSION_FORCE}"/>')
    a('  </actuator>')

    # ── Equality: connect consecutive links ──
    a('  <equality>')
    for i in range(N_LINKS):
        j = (i + 1) % N_LINKS
        a(f'    <connect body1="link_{i}" body2="link_{j}"'
          f' anchor="{link_pitch/2:.5f} 0 0"/>')
    a('  </equality>')

    # ── Contact exclusions: adjacent links ──
    a('  <contact>')
    for i in range(N_LINKS):
        j = (i + 1) % N_LINKS
        a(f'    <exclude body1="link_{i}" body2="link_{j}"/>')
    a('  </contact>')

    a('</mujoco>')
    return '\n'.join(lines)


def apply_link_damping(model, data, link_bids):
    """Apply velocity-proportional damping to each link body."""
    for bid in link_bids:
        # Get body velocity (6D: 3 angular + 3 linear in world frame)
        vel = data.cvel[bid]  # (6,) — [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
        # Apply damping force: F = -b * v
        data.xfrc_applied[bid, 3:6] = -LINK_DAMPING * vel[3:6]  # linear damping
        data.xfrc_applied[bid, 0:3] = -LINK_DAMPING * 0.1 * vel[0:3]  # light angular damping


def run_gui():
    """Launch interactive MuJoCo viewer."""
    xml = build_xml()
    with open('/home/sman/Work/CMU/Research/track_synthesis/chain_track.xml', 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    link_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
                 for i in range(N_LINKS)]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            t = step * TIMESTEP
            # Tension always on
            data.ctrl[3] = TENSION_FORCE
            # Drive after warmup
            if t < 1.0:
                torque = 0.0  # let chain tension first
            elif t < 2.0:
                torque = DRIVE_TORQUE * (t - 1.0)
            else:
                torque = DRIVE_TORQUE
            data.ctrl[0] = torque
            data.ctrl[1] = torque
            data.ctrl[2] = torque
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1


def run_record():
    """Record to GIF with diagnostics."""
    xml = build_xml()
    with open('/home/sman/Work/CMU/Research/track_synthesis/chain_track.xml', 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, width=800, height=450)
    frames = []
    n_steps = int(SIM_TIME / TIMESTEP)
    fps = 60
    frame_skip = max(1, int(1 / (fps * TIMESTEP)))

    max_eq_violation = []
    max_contact_force = []
    n_active_contacts = []

    for step in range(n_steps):
        data.ctrl[0] = DRIVE_TORQUE
        data.ctrl[1] = DRIVE_TORQUE
        mujoco.mj_step(model, data)

        if step % frame_skip == 0:
            renderer.update_scene(data, camera="persp")
            frames.append(renderer.render().copy())

            if np.any(np.isnan(data.qpos)):
                print(f"NaN at step {step}, t={step*TIMESTEP:.3f}s")
                break

            if data.nefc > 0:
                eq_viol = np.abs(data.efc_pos[:data.nefc])
                max_eq_violation.append(np.max(eq_viol))
            else:
                max_eq_violation.append(0.0)

            nc = data.ncon
            n_active_contacts.append(nc)
            if nc > 0:
                forces = []
                for c in range(nc):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(model, data, c, f)
                    forces.append(np.linalg.norm(f[:3]))
                max_contact_force.append(max(forces))
            else:
                max_contact_force.append(0.0)

    out_path = '/home/sman/Work/CMU/Research/track_synthesis/chain_track.gif'
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved {out_path} ({len(frames)} frames)")

    max_eq_violation = np.array(max_eq_violation)
    max_contact_force = np.array(max_contact_force)
    n_active_contacts = np.array(n_active_contacts)
    print(f"\n--- Constraint Violation ---")
    print(f"  Max equality violation:  {max_eq_violation.max():.6f} m")
    print(f"  Mean equality violation: {max_eq_violation.mean():.6f} m")
    print(f"  Final equality violation:{max_eq_violation[-1]:.6f} m")
    print(f"\n--- Contact ---")
    print(f"  Max contact force:  {max_contact_force.max():.3f} N")
    print(f"  Mean contact force: {max_contact_force.mean():.3f} N")
    print(f"  Max active contacts: {n_active_contacts.max()}")
    print(f"  Mean active contacts: {n_active_contacts.mean():.1f}")


def nearest_stadium_dist(px, py, half_span, radius):
    """Distance from point (px,py) to the nearest point on the stadium curve."""
    best = float('inf')
    # Arc sections
    for cx in [-half_span, half_span]:
        dx, dy = px - cx, py
        d = abs(np.sqrt(dx**2 + dy**2) - radius)
        if (cx == -half_span and px <= -half_span) or (cx == half_span and px >= half_span):
            best = min(best, d)
    # Straight sections
    if -half_span <= px <= half_span:
        best = min(best, abs(py - radius), abs(py + radius))
    return best


def run_debug():
    """Run headless, print diagnostics + stadium tracking error."""
    xml = build_xml()
    with open('/home/sman/Work/CMU/Research/track_synthesis/chain_track.xml', 'w') as f:
        f.write(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    link_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link_{i}")
                 for i in range(N_LINKS)]
    chain_r = SPROCKET_R + PIN_RADIUS + SIDE_THICK

    for step in range(5000):
        t = step * TIMESTEP
        data.ctrl[3] = TENSION_FORCE
        if t < 1.0:
            torque = 0.0
        elif t < 2.0:
            torque = DRIVE_TORQUE * (t - 1.0)
        else:
            torque = DRIVE_TORQUE
        data.ctrl[0] = torque
        data.ctrl[1] = torque
        data.ctrl[2] = torque
        mujoco.mj_step(model, data)

        max_cf = 0.0
        if data.ncon > 0:
            for c in range(data.ncon):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f)
                max_cf = max(max_cf, np.linalg.norm(f))

        max_ev = 0.0
        if data.nefc > 0:
            max_ev = np.max(np.abs(data.efc_pos[:data.nefc]))

        max_v = np.max(np.abs(data.qvel))

        # Stadium tracking error
        dists = [nearest_stadium_dist(data.xpos[bid][0], data.xpos[bid][1],
                                       HALF_SPAN, chain_r) for bid in link_bids]
        max_trk = max(dists)
        mean_trk = np.mean(dists)

        # Per-link pin gap (distance between consecutive pin sites)
        pin_gaps = []
        for i in range(N_LINKS):
            j = (i + 1) % N_LINKS
            fwd = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pin_{i}_fwd")]
            bwd = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"pin_{j}_bwd")]
            gap = np.linalg.norm(fwd - bwd)
            pin_gaps.append((i, j, gap))

        worst_gaps = sorted(pin_gaps, key=lambda x: x[2], reverse=True)[:3]

        if step < 10 or step % 100 == 0 or max_cf > 500 or max_v > 50 or np.any(np.isnan(data.qpos)):
            worst_str = "  ".join([f"{i}-{j}:{g:.4f}" for i, j, g in worst_gaps])
            print(f"step {step:4d}  t={t:.3f}  "
                  f"ncon={data.ncon:3d}  cf={max_cf:8.1f}  "
                  f"eq_v={max_ev:.4f}  qvel={max_v:6.2f}  "
                  f"trk={max_trk:.3f}  worst_pins: {worst_str}")

        if np.any(np.isnan(data.qpos)):
            print(">>> NaN detected, stopping")
            break


if __name__ == '__main__':
    import sys
    if '--record' in sys.argv:
        run_record()
    elif '--debug' in sys.argv:
        run_debug()
    else:
        run_gui()
