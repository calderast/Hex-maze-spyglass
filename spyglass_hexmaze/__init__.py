"""Spyglass Hex Maze extension package.

This package provides DataJoint tables and analysis tools for hex maze experiments
using the Spyglass neurophysiology data analysis framework.
"""

__version__ = "0.1.0"

from spyglass_hexmaze import (
    berke_fiber_photometry,
    hex_maze_behavior,
    hex_maze_decoding,
)

__all__ = [
    "hex_maze_behavior",
    "hex_maze_decoding",
    "berke_fiber_photometry",
]
