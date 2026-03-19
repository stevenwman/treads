---
name: no_gif_generation
description: User prefers to run simulations themselves and only generate GIFs when explicitly needed for Claude to visualize
type: feedback
---

Don't generate GIFs automatically. Tell the user to run the script themselves. Only ask for a GIF if Claude needs to visually inspect the result.

**Why:** User wants to see the simulation live themselves rather than waiting for GIF rendering.
**How to apply:** After making code changes, tell the user to run the script. Print diagnostics to stdout instead of relying on visual inspection.
