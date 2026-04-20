"""Spyglass Hex Maze extension package.

This package provides DataJoint tables and analysis tools for hex maze experiments
using the Spyglass neurophysiology data analysis framework.
"""

__version__ = "0.1.0"

# PyNWB 3.x introduced DeviceModel, which crashes when reading NWB files whose
# device names contain '/' or ':' (e.g. 'MFC_200/250-0.66_40mm_MF2.5_FLT').
# Patch DeviceMapper to silently skip invalid device model names on read.
try:
    from pynwb.io.device import DeviceMapper

    _orig_model_fn = DeviceMapper.constructor_args.get("model")

    if _orig_model_fn:
        def _safe_model_fn(self, builder, manager):
            try:
                return _orig_model_fn(self, builder, manager)
            except ValueError:
                return None
        DeviceMapper.constructor_args["model"] = _safe_model_fn
except Exception:
    pass  # older pynwb without DeviceMapper — no patch needed

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
