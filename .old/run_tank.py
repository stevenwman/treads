#!/usr/bin/env python3
"""Run the tank simulation with ghost-driven tracks and tread-force feedback."""

import mujoco
import mujoco.viewer
import numpy as np
import time

from stadium import (
    stadium_parametric,
    stadium_perimeter,
    angle_to_quat_y,
    quat_multiply,
    mat_to_quat,
)

# === Parameters (must match generate_tank.py) ===
HALF_LENGTH = 0.28
RADIUS = 0.08
Y_OFFSET = 0.22
Z_OFFSET = 0.0
N_LINKS = 5
CHASSIS_MASS = 20.0

BASE_SPEED = 1.0
TURN_DIFF = 0.5

# Force gains
TREAD_FORCE_SCALE = 0.02   # scale down raw friction (weld forces are very stiff)
DRAG_COEFF = 50.0           # linear drag for deceleration
TURN_TORQUE_GAIN = 15.0     # Nm per m/s of speed difference

# Attitude stabilization
ATTITUDE_KP = 800.0
ATTITUDE_KD = 160.0

# === Load model ===
model = mujoco.MjModel.from_xml_path("tank.xml")
data = mujoco.MjData(model)

# === Cache IDs ===
chassis_id = model.body("chassis").id

ghost_mocap_ids = {}
link_geom_ids = set()
link_geom_to_body = {}

for side in ["L", "R"]:
    for i in range(N_LINKS):
        ghost_name = f"ghost_{side}_{i}"
        link_name = f"link_{side}_{i}"
        ghost_mocap_ids[ghost_name] = model.body_mocapid[model.body(ghost_name).id]

        link_body_id = model.body(link_name).id
        for g in range(model.body_geomnum[link_body_id]):
            gid = model.body_geomadr[link_body_id] + g
            link_geom_ids.add(gid)
            link_geom_to_body[gid] = link_body_id

# === Track state (auto-drive forward on launch) ===
left_speed = BASE_SPEED
right_speed = BASE_SPEED
left_arc = 0.0
right_arc = 0.0

perimeter = stadium_perimeter(HALF_LENGTH, RADIUS)
spacing = perimeter / N_LINKS

keys_held = set()


def update_ghosts():
    """Recompute all ghost positions from chassis pose + stadium offset."""
    global left_arc, right_arc

    left_arc += left_speed * model.opt.timestep
    right_arc += right_speed * model.opt.timestep

    chassis_pos = data.xpos[chassis_id].copy()
    chassis_mat = data.xmat[chassis_id].reshape(3, 3)
    chassis_quat = mat_to_quat(chassis_mat)

    for side in ["L", "R"]:
        y_sign = 1.0 if side == "L" else -1.0
        arc = left_arc if side == "L" else right_arc

        for i in range(N_LINKS):
            s = (arc + i * spacing) % perimeter
            local_x, local_z, angle = stadium_parametric(s, HALF_LENGTH, RADIUS)

            local_pos = np.array([local_x, y_sign * Y_OFFSET, local_z + Z_OFFSET])
            world_pos = chassis_pos + chassis_mat @ local_pos

            mid = ghost_mocap_ids[f"ghost_{side}_{i}"]
            data.mocap_pos[mid] = world_pos

            local_quat = angle_to_quat_y(angle)
            world_quat = quat_multiply(chassis_quat, local_quat)
            data.mocap_quat[mid] = world_quat


def compute_tread_forces():
    """Sum contact forces on track links, reflecting friction for propulsion.

    In a real tank, ground friction on the bottom tread (backward) gets
    converted to forward chassis propulsion through the sprocket mechanism.
    We simulate this by reflecting each contact force's tangential component:
      - Normal component (support): passed through as-is
      - Tangential component (friction): flipped direction

    This means: treads on the ground = propulsion. Treads in the air = nothing.
    Treads on a slope = drive follows the terrain surface.
    """
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    chassis_pos = data.xpos[chassis_id]
    force_buf = np.zeros(6)

    for i in range(data.ncon):
        c = data.contact[i]

        # Identify which geom is the link
        # mj_contactForce returns force on geom2's body (empirically verified)
        if c.geom2 in link_geom_ids:
            link_geom = c.geom2
            sign = 1.0   # raw output is already force on link
        elif c.geom1 in link_geom_ids:
            link_geom = c.geom1
            sign = -1.0  # negate: raw is on the other body
        else:
            continue

        # Get contact force in contact frame [normal, tan1, tan2, ...]
        mujoco.mj_contactForce(model, data, i, force_buf)

        # Contact frame axes
        normal = c.frame[:3]    # points from geom2 toward geom1
        tan1 = c.frame[3:6]
        tan2 = c.frame[6:9]

        # Force on the link body in world frame
        f_on_link = sign * (
            force_buf[0] * normal +
            force_buf[1] * tan1 +
            force_buf[2] * tan2
        )

        # Decompose into normal + tangential
        f_normal = np.dot(f_on_link, normal) * normal
        f_tangential = f_on_link - f_normal

        # Sprocket mechanism: flip friction direction for propulsion.
        # Normal support already comes from the frictionless chassis-ground contact,
        # so we only apply the reflected tangential (friction → propulsion) component.
        f_chassis = -f_tangential

        total_force += f_chassis * TREAD_FORCE_SCALE

        # Torque about chassis COM from force at link position
        link_body = link_geom_to_body[link_geom]
        r = data.xpos[link_body] - chassis_pos
        total_torque += np.cross(r, f_chassis * TREAD_FORCE_SCALE)

    return total_force, total_torque


def apply_forces():
    """Apply tread-derived propulsion, drag, and attitude stabilization."""
    chassis_mat = data.xmat[chassis_id].reshape(3, 3)

    dof = model.body_dofadr[chassis_id]
    lin_vel = data.qvel[dof:dof + 3]
    ang_vel = data.qvel[dof + 3:dof + 6]

    # --- Tread propulsion (from real contact forces) ---
    tread_force, tread_torque = compute_tread_forces()

    force = tread_force.copy()
    torque = tread_torque.copy()

    # --- Drag ---
    ground_vel = lin_vel.copy()
    ground_vel[2] = 0
    force -= DRAG_COEFF * ground_vel

    # --- Turning torque + yaw damping ---
    speed_diff = right_speed - left_speed
    torque[2] += speed_diff * TURN_TORQUE_GAIN
    torque[2] -= 10.0 * ang_vel[2]

    # --- Attitude stabilization (keep level) ---
    roll = np.arctan2(chassis_mat[2, 1], chassis_mat[2, 2])
    pitch = -np.arcsin(np.clip(chassis_mat[2, 0], -1, 1))
    torque[0] += -ATTITUDE_KP * roll - ATTITUDE_KD * ang_vel[0]
    torque[1] += -ATTITUDE_KP * pitch - ATTITUDE_KD * ang_vel[1]

    # xfrc_applied layout: [force(3), torque(3)]
    data.xfrc_applied[chassis_id, :3] = force
    data.xfrc_applied[chassis_id, 3:] = torque


def update_speed_from_keys():
    """Set track speeds based on currently held keys."""
    global left_speed, right_speed

    forward = 265 in keys_held
    backward = 264 in keys_held
    turn_left = 263 in keys_held
    turn_right = 262 in keys_held

    if forward and not backward:
        base = BASE_SPEED
    elif backward and not forward:
        base = -BASE_SPEED
    else:
        base = 0.0

    if turn_left and not turn_right:
        left_speed = base - TURN_DIFF
        right_speed = base + TURN_DIFF
    elif turn_right and not turn_left:
        left_speed = base + TURN_DIFF
        right_speed = base - TURN_DIFF
    else:
        left_speed = base
        right_speed = base


def key_callback(keycode):
    """Handle key press events from MuJoCo viewer (GLFW keycodes)."""
    global left_speed, right_speed

    if keycode == 32:  # Space = stop
        keys_held.clear()
        left_speed = 0.0
        right_speed = 0.0
        return

    if keycode in keys_held:
        keys_held.discard(keycode)
    else:
        keys_held.add(keycode)

    update_speed_from_keys()


# === Main loop ===
print("Tank simulation ready!")
print("Controls: Arrow keys = drive (Up/Down/Left/Right), Space = stop")
print("Treads now provide propulsion through contact force feedback.")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.distance = 2.0
    viewer.cam.elevation = -25.0
    viewer.cam.azimuth = 135.0
    viewer.cam.lookat[:] = [0, 0, 0.15]

    while viewer.is_running():
        step_start = time.time()

        update_ghosts()
        apply_forces()
        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - step_start
        dt = model.opt.timestep
        if elapsed < dt:
            time.sleep(dt - elapsed)
