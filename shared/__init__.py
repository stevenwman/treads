"""Shared modules for tank track simulations.

This package contains the reusable building blocks:
    config      - TankConfig / Sprocket dataclasses (all tunable parameters)
    geometry    - Stadium-shaped track path math
    xml_builder - Generates MuJoCo MJCF XML for the tank scene
    engagement  - Sprocket-chain engagement manager
    simulation  - MuJoCo setup, stepping, and run modes (GUI / debug / record)
"""
from .config import TankConfig, Sprocket
from .simulation import run
