"""Sprocket-chain engagement manager.

In a real tank, the sprocket teeth mesh with the chain links as they wrap
around. We simulate this with MuJoCo equality constraints: when a link
enters the arc zone around a sprocket, we activate a "connect" constraint
that pins the link to the sprocket at a fixed angle. When the link rotates
past the arc boundary, we deactivate the constraint and the link is free.

Key concepts:
    - "engaging sprockets": drive and idler — the ones that grab the chain.
      Mid-support wheels just provide physical contact, no constraints.
    - "on arc": a link is close enough to the sprocket AND past the apex
      (directly behind drive, or directly in front of idler).
    - "local angle": the angle of the link relative to the sprocket's own
      rotation. This stays fixed while engaged, so the link rides along.
"""
import math
import mujoco

from .config import SIDES
from .geometry import normalize_angle


class EngagementManager:
    """Tracks which chain links are engaged with which sprockets.

    Usage:
        engagement = EngagementManager(config)
        engagement.seed(model, data, lookups)    # call once at startup
        engagement.update(model, data, lookups)  # call every sim step
    """

    def __init__(self, config):
        self.config = config
        # Currently engaged links: {(side, link_index, sprocket_name): local_angle}
        self._engaged = {}

    def _count_engaged(self, side, sprocket_name):
        """How many links are currently engaged to one sprocket."""
        return sum(1 for (s, _, sp) in self._engaged
                   if s == side and sp == sprocket_name)

    def _is_on_arc(self, link_x, sprocket_x, sprocket_name, distance):
        """Check if a link is in the engagement zone of a sprocket.

        A link is "on arc" when it's wrapped around the far side of the
        sprocket (past the apex) AND close to the sprocket radius.
        """
        c = self.config
        R = c.sprocket_radius

        # Position check: link must be past the sprocket apex
        #   Drive sprocket: link is to the LEFT  (lx < sx - R * threshold)
        #   Idler sprocket: link is to the RIGHT (lx > sx + R * threshold)
        past_apex = False
        if sprocket_name == "drive":
            past_apex = link_x < sprocket_x - R * c.engagement_arc_threshold
        elif sprocket_name == "idler":
            past_apex = link_x > sprocket_x + R * c.engagement_arc_threshold

        # Distance check: link must be close to the sprocket radius
        close_enough = abs(distance - R) < c.engagement_distance_tol

        return past_apex and close_enough

    def seed(self, model, data, lookups):
        """Set up initial engagements based on where links start.

        Called once after the chain is placed in its initial stadium shape.
        Finds links that are already on sprocket arcs and engages them.
        """
        c = self.config
        for side, _ in SIDES:
            for spr in c.engaging_sprockets:
                sprocket_bid = lookups.sprocket_body_ids[(side, spr.name)]
                sx = data.xpos[sprocket_bid][0]
                sz = data.xpos[sprocket_bid][2]

                # Score each link by closeness to the sprocket's horizontal
                # center line — we want the links nearest the apex
                candidates = []
                for i in range(c.n_links):
                    link_bid = lookups.link_body_ids[side][i]
                    lx, lz = data.xpos[link_bid][0], data.xpos[link_bid][2]
                    dx, dz = lx - sx, lz - sz
                    dist = math.sqrt(dx * dx + dz * dz)

                    if self._is_on_arc(lx, sx, spr.name, dist):
                        candidates.append((abs(dz), i, dx, dz, dist))

                # Engage the closest ones (up to the cap)
                candidates.sort()
                for _, i, dx, dz, dist in candidates[:c.max_engaged_per_sprocket]:
                    self._engage_link(model, data, lookups,
                                      side, i, spr.name, dx, dz, dist)

        mujoco.mj_forward(model, data)

    def update(self, model, data, lookups):
        """Check every link and engage/disengage as needed.

        Called once per simulation step. For each link and each engaging
        sprocket, we either:
            - Keep it engaged (still on arc)
            - Disengage it (rotated past the arc boundary)
            - Newly engage it (just entered the arc zone)
        """
        c = self.config
        for side, _ in SIDES:
            for i in range(c.n_links):
                link_bid = lookups.link_body_ids[side][i]
                lx = data.xpos[link_bid][0]
                lz = data.xpos[link_bid][2]

                for spr in c.engaging_sprockets:
                    key = (side, i, spr.name)
                    eq_idx = lookups.engagement_eq_ids.get(key)
                    if eq_idx is None:
                        continue

                    sprocket_bid = lookups.sprocket_body_ids[(side, spr.name)]
                    sx = data.xpos[sprocket_bid][0]
                    sz = data.xpos[sprocket_bid][2]
                    joint_id = lookups.sprocket_joint_ids[(side, spr.name)]
                    spr_angle = data.qpos[model.jnt_qposadr[joint_id]]
                    dx, dz = lx - sx, lz - sz
                    dist = math.sqrt(dx * dx + dz * dz)

                    if key in self._engaged:
                        # --- Already engaged: check if it should disengage ---
                        self._maybe_disengage(data, key, eq_idx,
                                              spr.name, spr_angle)
                    elif (self._is_on_arc(lx, sx, spr.name, dist)
                          and self._count_engaged(side, spr.name)
                              < c.max_engaged_per_sprocket):
                        # --- New engagement ---
                        self._engage_link(model, data, lookups,
                                          side, i, spr.name, dx, dz, dist)
                    else:
                        data.eq_active[eq_idx] = 0

    def _engage_link(self, model, data, lookups,
                     side, link_index, spr_name, dx, dz, dist):
        """Activate the constraint that pins a link to a sprocket."""
        key = (side, link_index, spr_name)
        eq_idx = lookups.engagement_eq_ids.get(key)
        if eq_idx is None:
            return

        R = self.config.sprocket_radius
        joint_id = lookups.sprocket_joint_ids[(side, spr_name)]
        spr_angle = data.qpos[model.jnt_qposadr[joint_id]]

        # Compute where the link attaches in the sprocket's local frame
        world_angle = math.atan2(dz, dx)
        local_angle = normalize_angle(world_angle + spr_angle)

        # Set the anchor point on the sprocket (body1) at radius R
        model.eq_data[eq_idx, 0] = R * math.cos(local_angle)
        model.eq_data[eq_idx, 1] = 0.0
        model.eq_data[eq_idx, 2] = R * math.sin(local_angle)
        # Anchor on the link (body2) at its center
        model.eq_data[eq_idx, 3:6] = 0.0

        data.eq_active[eq_idx] = 1
        self._engaged[key] = local_angle

    def _maybe_disengage(self, data, key, eq_idx, spr_name, spr_angle):
        """Deactivate a constraint if the link has rotated past the arc."""
        local_angle = self._engaged[key]
        # Convert back to world angle
        world_angle = normalize_angle(local_angle - spr_angle)

        # Check how far the link has rotated from the sprocket's apex
        if spr_name == "drive":
            offset = abs(normalize_angle(world_angle - math.pi))
        else:  # idler
            offset = abs(normalize_angle(world_angle))

        if offset > self.config.arc_half:
            del self._engaged[key]
            data.eq_active[eq_idx] = 0
