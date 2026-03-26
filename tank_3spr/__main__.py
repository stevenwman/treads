"""3-sprocket tank simulation.

Layout: one drive sprocket (rear), one idler (front), one mid support.

    ╭────────────────────────────────────╮
    │ drive        mid           idler   │
    │ (rear)     (center)       (front)  │
    ╰────────────────────────────────────╯

Run:
    python -m tank_3spr              # interactive viewer
    python -m tank_3spr --debug      # headless diagnostics
    python -m tank_3spr --record     # save MP4 video
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import TankConfig, Sprocket
from shared.simulation import run


HALF_SPAN = 1.3394  # tuned for zero-slack at 28 links

CONFIG = TankConfig(
    name="tank_3spr",
    sprocket_radius=0.35,
    n_links=28,
    half_span=HALF_SPAN,
    sprockets=[
        Sprocket("drive", x_offset=-HALF_SPAN,
                 engages_chain=True,
                 color="0.7 0.2 0.2 1"),       # red
        Sprocket("idler", x_offset=HALF_SPAN,
                 engages_chain=True,
                 has_tensioner=True,
                 color="0.2 0.2 0.7 1"),       # blue
        Sprocket("mid",   x_offset=0.0,
                 engages_chain=False,
                 color="0.2 0.6 0.2 1"),       # green
    ],
    hull_half_x_margin=0.25,
    hinge_range=(-70, 70),
)

if __name__ == "__main__":
    run(CONFIG)
