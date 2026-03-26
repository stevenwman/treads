"""Configuration dataclasses for tank track simulations.

Every tunable parameter lives here. The two tank variants (3-sprocket and
4-sprocket) each create a TankConfig with different values, but the simulation
code is identical.
"""
from dataclasses import dataclass
import math


# The tank has a left and right track, mirrored in Y.
# y_sign: +1 for left, -1 for right.
SIDES = [("left", 1), ("right", -1)]


@dataclass
class Sprocket:
    """One sprocket wheel on a track.

    Attributes:
        name:           Unique name (e.g. "drive", "idler", "mid").
        x_offset:       X position relative to hull center.
        engages_chain:  If True, chain links latch onto this sprocket via
                        equality constraints. Drive and idler engage; mid
                        wheels only provide physical support.
        has_tensioner:  If True, a spring-loaded slide joint keeps the chain
                        taut (typically only the idler).
        color:          RGBA string for the hub geometry.
    """
    name: str
    x_offset: float
    engages_chain: bool = True
    has_tensioner: bool = False
    color: str = "0.2 0.6 0.2 1"   # default green (mid sprockets)


@dataclass
class TankConfig:
    """All parameters that define a tank simulation.

    Derived quantities (perimeter, link_pitch, hull dimensions, etc.) are
    computed as properties so the config stays consistent if you tweak a
    parameter.
    """
    name: str                       # e.g. "tank_3spr"

    # -- Track shape --
    sprocket_radius: float          # radius of drive/idler sprocket wheels
    n_links: int                    # number of chain links per track
    half_span: float                # half the distance between outermost sprockets
    sprockets: list                 # list of Sprocket objects

    # -- Link geometry --
    link_thickness: float = 0.02    # half-height of each link box (radial)
    link_width: float = 0.10        # half-width of each link box (lateral)

    # -- Physics --
    timestep: float = 0.002
    target_velocity: float = 1.0    # sprocket angular velocity target (rad/s)
    tension_stiffness: float = 80.0 # idler tensioner spring stiffness
    hinge_damping: float = 0.2      # damping on inter-link hinges
    hinge_range: tuple = (-70, 70)  # angular limits on inter-link hinges (degrees)
    sprocket_hinge_damping: float = 2.0

    # -- Engagement --
    arc_half: float = None          # half-angle of the engagement arc (default pi*0.4)
    max_engaged_per_sprocket: int = 2
    engagement_distance_tol: float = 0.10   # how close to sprocket_radius to engage
    engagement_arc_threshold: float = 0.85  # fraction of R beyond which link is "on arc"

    # -- Hull --
    track_gauge: float = 1.2        # Y distance between left and right track centers
    hull_half_x_margin: float = 0.25  # hull extends this far beyond the track span
    hull_half_z: float = 0.12
    hull_mass: float = 20.0

    def __post_init__(self):
        if self.arc_half is None:
            self.arc_half = math.pi * 0.40

    # -- Derived geometry (computed from the parameters above) --

    @property
    def hub_radius(self):
        """Hub is slightly smaller than the sprocket so links wrap outside."""
        return self.sprocket_radius - 0.04

    @property
    def hub_half_width(self):
        """Hub cylinder extends slightly wider than the links."""
        return self.link_width + 0.02

    @property
    def perimeter(self):
        """Total path length of the stadium-shaped track."""
        return 2 * math.pi * self.sprocket_radius + 4 * self.half_span

    @property
    def link_pitch(self):
        """Center-to-center distance between adjacent links."""
        return self.perimeter / self.n_links

    @property
    def hull_half_x(self):
        return self.half_span + self.hull_half_x_margin

    @property
    def hull_half_y(self):
        return self.track_gauge / 2 - 0.05

    @property
    def sprocket_z(self):
        """Height of sprocket centers above the ground plane."""
        return self.sprocket_radius + 0.02

    @property
    def hull_z(self):
        """Height of hull center above the ground plane."""
        return self.sprocket_z + 0.15

    @property
    def engaging_sprockets(self):
        """Only sprockets that grab chain links (drive and idler)."""
        return [s for s in self.sprockets if s.engages_chain]
