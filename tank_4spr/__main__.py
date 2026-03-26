"""4-sprocket tank simulation.

Layout: drive (rear), two mid supports, idler (front).
Smaller wheels, more evenly distributed weight.

    ╭──────────────────────────────────╮
    │ drive   mid1    mid2     idler   │
    │ (rear)                  (front)  │
    ╰──────────────────────────────────╯

Run:
    python -m tank_4spr              # interactive viewer
    python -m tank_4spr --debug      # headless diagnostics
    python -m tank_4spr --record     # save MP4 video
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import TankConfig, Sprocket
from shared.simulation import run


SPAN_BETWEEN = 0.7       # distance between adjacent sprocket centers
N_SPROCKETS = 4
HALF_SPAN = SPAN_BETWEEN * (N_SPROCKETS - 1) / 2  # = 1.05

CONFIG = TankConfig(
    name="tank_4spr",
    sprocket_radius=0.25,
    n_links=30,
    half_span=HALF_SPAN,
    sprockets=[
        Sprocket("drive", x_offset=-HALF_SPAN,
                 engages_chain=True,
                 color="0.7 0.2 0.2 1"),       # red
        Sprocket("mid1",  x_offset=-HALF_SPAN + SPAN_BETWEEN,
                 engages_chain=False,
                 color="0.2 0.6 0.2 1"),       # green
        Sprocket("mid2",  x_offset=-HALF_SPAN + 2 * SPAN_BETWEEN,
                 engages_chain=False,
                 color="0.2 0.6 0.2 1"),       # green
        Sprocket("idler", x_offset=HALF_SPAN,
                 engages_chain=True,
                 has_tensioner=True,
                 color="0.2 0.2 0.7 1"),       # blue
    ],
    hull_half_x_margin=0.2,
    hinge_range=(-80, 80),
)

if __name__ == "__main__":
    run(CONFIG)
