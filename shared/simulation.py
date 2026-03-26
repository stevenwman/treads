"""MuJoCo simulation setup, stepping, and run modes.

This module ties everything together:
    1. Build the XML and create a MuJoCo model
    2. Look up all the body/joint/actuator IDs we need
    3. Set the chain into its initial stadium shape
    4. Seed the sprocket engagement
    5. Run the simulation (GUI viewer, headless debug, or MP4 recording)
"""
import argparse
import math
import time
from dataclasses import dataclass

import numpy as np
import mujoco

from .config import SIDES
from .geometry import stadium_point, normalize_angle
from .xml_builder import build_tank_xml
from .engagement import EngagementManager


# ── MuJoCo ID lookups ────────────────────────────────────────────────────────

@dataclass
class SimLookups:
    """Pre-computed MuJoCo ID mappings for fast access during simulation.

    Instead of calling mj_name2id every step, we look up all IDs once at
    startup and store them here.
    """
    # (side, link_index, sprocket_name) -> equality constraint index
    engagement_eq_ids: dict
    # side -> [body_id for each chain link]
    link_body_ids: dict
    # (side, sprocket_name) -> sprocket body ID
    sprocket_body_ids: dict
    # (side, sprocket_name) -> sprocket hinge joint ID
    sprocket_joint_ids: dict
    # (side, sprocket_name) -> motor actuator ID
    motor_actuator_ids: dict
    # side -> [joint_id for inter-link hinges 1..N-1]
    chain_hinge_ids: dict

    @classmethod
    def from_model(cls, model, config):
        """Build all lookup tables from a MuJoCo model."""
        def body_id(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        def joint_id(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        def actuator_id(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Engagement equality constraints
        engagement_eq_ids = {}
        for idx in range(model.neq):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
            if name is None:
                continue
            for side, _ in SIDES:
                prefix = f"{side}_eng_"
                if name.startswith(prefix):
                    rest = name[len(prefix):]
                    parts = rest.split("_")
                    link_i = int(parts[0])
                    spr_name = "_".join(parts[1:])
                    engagement_eq_ids[(side, link_i, spr_name)] = idx

        # Chain link body IDs
        link_body_ids = {}
        for side, _ in SIDES:
            link_body_ids[side] = [
                body_id(f"{side}_link_{i}")
                for i in range(config.n_links)
            ]

        # Sprocket body, joint, and actuator IDs
        sprocket_body_ids = {}
        sprocket_joint_ids = {}
        motor_actuator_ids = {}
        for side, _ in SIDES:
            for spr in config.sprockets:
                key = (side, spr.name)
                full = f"{side}_{spr.name}"
                sprocket_body_ids[key] = body_id(f"{full}_spr")
                sprocket_joint_ids[key] = joint_id(f"{full}_hinge")
                motor_actuator_ids[key] = actuator_id(f"{full}_motor")

        # Inter-link hinge joint IDs (links 1 through N-1)
        chain_hinge_ids = {}
        for side, _ in SIDES:
            chain_hinge_ids[side] = [
                joint_id(f"{side}_hinge_{i}")
                for i in range(1, config.n_links)
            ]

        return cls(
            engagement_eq_ids=engagement_eq_ids,
            link_body_ids=link_body_ids,
            sprocket_body_ids=sprocket_body_ids,
            sprocket_joint_ids=sprocket_joint_ids,
            motor_actuator_ids=motor_actuator_ids,
            chain_hinge_ids=chain_hinge_ids,
        )


# ── Initialization ────────────────────────────────────────────────────────────

def _set_initial_chain_shape(model, data, lookups, config):
    """Bend the chain links into the stadium shape.

    Each inter-link hinge gets a qpos value equal to the angle difference
    between consecutive links along the stadium path. Then we fix the
    loop-closure constraint anchors to match the actual positions.
    """
    for side, _ in SIDES:
        for idx, joint_id in enumerate(lookups.chain_hinge_ids[side]):
            i = idx + 1  # hinge i connects link i-1 to link i
            _, _, angle_curr = stadium_point(i * config.link_pitch, config)
            _, _, angle_prev = stadium_point((i - 1) * config.link_pitch, config)
            delta = normalize_angle(angle_curr - angle_prev)
            # Y-axis hinge convention: negate the delta
            data.qpos[model.jnt_qposadr[joint_id]] = -delta

    mujoco.mj_forward(model, data)

    # Fix loop-closure anchor positions to match the bent chain
    for idx in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, idx)
        if name and name.endswith("_loop"):
            bid1 = model.eq_obj1id[idx]
            bid2 = model.eq_obj2id[idx]
            # Compute where anchor1 is in world coordinates
            rot1 = data.xmat[bid1].reshape(3, 3)
            world_anchor = data.xpos[bid1] + rot1 @ model.eq_data[idx, 0:3]
            # Express that point in body2's local frame
            rot2 = data.xmat[bid2].reshape(3, 3)
            model.eq_data[idx, 3:6] = rot2.T @ (world_anchor - data.xpos[bid2])

    mujoco.mj_forward(model, data)


def create_simulation(config):
    """Build the MuJoCo model and initialize everything.

    Returns:
        (model, data, lookups, engagement) — everything needed to run.
    """
    xml = build_tank_xml(config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lookups = SimLookups.from_model(model, config)
    _set_initial_chain_shape(model, data, lookups, config)

    engagement = EngagementManager(config)
    engagement.seed(model, data, lookups)

    return model, data, lookups, engagement


# ── Simulation stepping ──────────────────────────────────────────────────────

def step(model, data, lookups, engagement, config,
         left_vel=None, right_vel=None):
    """Advance the simulation by one timestep.

    Args:
        left_vel:  Target angular velocity for left-side sprockets.
        right_vel: Target angular velocity for right-side sprockets.
                   Both default to config.target_velocity.
    """
    if left_vel is None:
        left_vel = config.target_velocity
    if right_vel is None:
        right_vel = config.target_velocity

    # Ramp up velocity over the first second to avoid sudden jolts
    ramp = min(1.0, data.time / 1.0)
    velocities = {"left": left_vel * ramp, "right": right_vel * ramp}

    for side in ("left", "right"):
        for spr in config.sprockets:
            act_id = lookups.motor_actuator_ids[(side, spr.name)]
            data.ctrl[act_id] = velocities[side]

    engagement.update(model, data, lookups)
    mujoco.mj_step(model, data)


# ── Run modes ─────────────────────────────────────────────────────────────────

def run_gui(config):
    """Launch the interactive MuJoCo viewer."""
    model, data, lookups, engagement = create_simulation(config)

    import mujoco.viewer as mjv
    with mjv.launch_passive(model, data) as viewer:
        wall_start = time.perf_counter()
        sim_start = data.time
        frame_count = 0
        last_report = wall_start

        while viewer.is_running():
            t0 = time.perf_counter()

            # Step simulation to match wall-clock time
            wall_elapsed = t0 - wall_start
            target_time = sim_start + wall_elapsed
            MAX_STEPS_PER_FRAME = 10
            n_steps = 0
            while data.time < target_time and n_steps < MAX_STEPS_PER_FRAME:
                step(model, data, lookups, engagement, config)
                n_steps += 1

            # If simulation can't keep up, reset the clock
            if data.time < target_time:
                wall_start = t0
                sim_start = data.time

            t1 = time.perf_counter()
            viewer.sync()
            t2 = time.perf_counter()

            # Performance report every 2 seconds
            frame_count += 1
            if t2 - last_report > 2.0:
                fps = frame_count / (t2 - last_report)
                sim_ms = (t1 - t0) * 1000
                sync_ms = (t2 - t1) * 1000
                print(f"FPS={fps:.1f}  sim={sim_ms:.1f}ms({n_steps}steps)"
                      f"  sync={sync_ms:.1f}ms  sim_t={data.time:.1f}s")
                frame_count = 0
                last_report = t2


def run_debug(config):
    """Run headless simulation with detailed diagnostics."""
    model, data, lookups, engagement = create_simulation(config)
    hull_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hull")

    obstacles = [("ramp", 4.0, 0.03), ("bump", 7.2, 0.05), ("step", 10.7, 0.08)]
    ground_geom_names = ("floor", "ramp", "bump_up", "bump_dn",
                         "step_up", "step_top", "step_dn")
    last_obstacle = ""
    n_steps = 10000
    history = []

    for s in range(n_steps):
        step(model, data, lookups, engagement, config)

        t = data.time
        hull_pos = data.xpos[hull_bid]
        max_vel = np.max(np.abs(data.qvel))
        max_eq_violation = (np.max(np.abs(data.efc_pos[:data.nefc]))
                            if data.nefc > 0 else 0)
        n_engaged = sum(1 for idx in lookups.engagement_eq_ids.values()
                        if data.eq_active[idx])

        drive_jid = lookups.sprocket_joint_ids[("left", "drive")]
        left_drive_vel = data.qvel[model.jnt_dofadr[drive_jid]]
        drive_jid_r = lookups.sprocket_joint_ids[("right", "drive")]
        right_drive_vel = data.qvel[model.jnt_dofadr[drive_jid_r]]
        hull_vx = data.cvel[hull_bid][3]

        # Ground contact forces
        total_normal = 0.0
        total_friction = 0.0
        n_ground_contacts = 0
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
            if g1 in ground_geom_names or g2 in ground_geom_names:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, f)
                total_normal += abs(f[0])
                total_friction += np.linalg.norm(f[1:3])
                n_ground_contacts += 1

        # Track which obstacle we're near
        current_obstacle = ""
        for oname, ox, oh in obstacles:
            if hull_pos[0] > ox - 1.0:
                current_obstacle = oname

        record = {
            't': t, 'hx': hull_pos[0], 'hz': hull_pos[2], 'hvx': hull_vx,
            'lv': left_drive_vel, 'rv': right_drive_vel, 'eng': n_engaged,
            'eq_v': max_eq_violation, 'max_v': max_vel,
            'ncon': n_ground_contacts, 'fn': total_normal, 'ff': total_friction,
            'obs': current_obstacle,
        }
        history.append(record)

        # Print at key moments
        should_print = s < 5 or s % 200 == 0 or max_vel > 200
        if current_obstacle != last_obstacle:
            should_print = True
            last_obstacle = current_obstacle
        if np.any(np.isnan(data.qpos)):
            should_print = True

        if should_print:
            obs_tag = f" [{current_obstacle}]" if current_obstacle else ""
            print(f"t={t:5.1f}  hx={hull_pos[0]:5.2f} hz={hull_pos[2]:4.2f}  "
                  f"hvx={hull_vx:5.2f}m/s  "
                  f"drv=({left_drive_vel:4.1f},{right_drive_vel:4.1f})r/s  "
                  f"eng={n_engaged:2d}  "
                  f"eq_v={max_eq_violation:.3f}  v={max_vel:5.1f}  "
                  f"gnd={n_ground_contacts:2d} fn={total_normal:6.0f}"
                  f" ff={total_friction:5.0f}{obs_tag}")
            if np.any(np.isnan(data.qpos)):
                print(">>> NaN detected — aborting")
                break

    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"  t={t:.1f}s  hull=({hull_pos[0]:.2f}, {hull_pos[2]:.2f})")
    print(f"  Obstacles:")
    for oname, ox, oh in obstacles:
        reached = any(r['hx'] > ox + 1.0 for r in history)
        nearby = [r['hx'] for r in history if abs(r['hx'] - ox) < 2.0]
        max_hx = max(nearby) if nearby else 0
        if reached:
            print(f"    {oname} (x={ox}, h={oh}m): CLEARED")
        elif max_hx > ox - 1.5:
            print(f"    {oname} (x={ox}, h={oh}m): STUCK (got to x={max_hx:.2f})")
        else:
            print(f"    {oname} (x={ox}, h={oh}m): NOT REACHED")

    if len(history) > 100:
        early = [r['hvx'] for r in history[50:150]]
        late = [r['hvx'] for r in history[-200:]]
        print(f"  Avg hull vx: early={np.mean(early):.3f}"
              f" late={np.mean(late):.3f} m/s")
        print(f"  Avg ground contacts:"
              f" {np.mean([r['ncon'] for r in history[-200:]]):.0f}")
        print(f"  Avg normal force:"
              f" {np.mean([r['fn'] for r in history[-200:]]):.0f}N")
        print(f"  Avg friction force:"
              f" {np.mean([r['ff'] for r in history[-200:]]):.0f}N")


def run_record(config):
    """Record an MP4 video from the tracking camera."""
    model, data, lookups, engagement = create_simulation(config)

    W, H = 1280, 720
    FPS = 24
    SIM_DURATION = 30.0
    frames_per_step = max(1, round(1.0 / (FPS * config.timestep)))

    renderer = mujoco.Renderer(model, width=W, height=H)
    n_steps = int(SIM_DURATION / config.timestep)
    n_frames = n_steps // frames_per_step

    import imageio
    output_path = f"{config.name}.mp4"
    writer = imageio.get_writer(output_path, fps=FPS,
                                codec='libx264', quality=8)

    from tqdm import tqdm
    pbar = tqdm(total=n_frames, desc="Recording", unit="frame",
                bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt}"
                            " [{elapsed}<{remaining}, {rate_fmt}]"))

    for s in range(n_steps):
        step(model, data, lookups, engagement, config)
        if s % frames_per_step == 0:
            renderer.update_scene(data, camera="front")
            writer.append_data(renderer.render())
            pbar.update(1)

    pbar.close()
    writer.close()
    print(f"Saved {output_path} ({n_frames} frames, {n_frames / FPS:.1f}s)")


# ── CLI entry point ───────────────────────────────────────────────────────────

def run(config):
    """Parse command-line args and launch the appropriate run mode.

    Usage:
        python -m tank_3spr              # interactive GUI
        python -m tank_3spr --debug      # headless diagnostics
        python -m tank_3spr --record     # save MP4 video
    """
    parser = argparse.ArgumentParser(description=f"Run {config.name} simulation")
    parser.add_argument("--debug", action="store_true",
                        help="Run headless with diagnostics")
    parser.add_argument("--record", action="store_true",
                        help="Record MP4 video")
    args = parser.parse_args()

    if args.debug:
        run_debug(config)
    elif args.record:
        run_record(config)
    else:
        run_gui(config)
