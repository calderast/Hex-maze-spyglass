import re

import datajoint as dj
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import non_local_detector.analysis as analysis
from non_local_detector.model_checking import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
import spyglass.common as sgc
from spyglass.common import TaskEpoch, IntervalList, AnalysisNwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile as custom_AnalysisNwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.utils import SpyglassMixin

from spyglass_hexmaze.hex_maze_behavior import HexCentroids, HexMazeBlock

from hexmaze import (
    are_points_in_maze,
    classify_maze_hexes,
    divide_into_thirds,
    get_all_choice_points,
    get_critical_choice_points,
    get_junction_left_right_map,
    get_hexes_before_divergence,
    get_hexes_from_port,
    get_optimal_path_hexes_after_divergence,
    get_path_divergence_point,
    get_unreachable_hexes,
    maze_to_barrier_set,
    maze_to_graph,
    get_hex_distance,
    plot_hex_maze,
)


schema = dj.schema("hex_maze_decoding")

# The theta tables moved to hex_maze_theta.py (they are LFP signal processing, not decoding)
# Re-exported here so existing
#     from spyglass_hexmaze.hex_maze_decoding import HexMazeThetaV1
# imports keep working.
from spyglass_hexmaze.hex_maze_theta import (  # noqa: E402,F401
    HexMazeThetaV1,
    HexMazeThetaReference,
)


def _decode_interval_epoch(nwb_file_name, decoding_interval, run_intervals):
    """Resolve which epoch a decode belongs to, from its decoding_interval name.

    Decodes name their interval a few different ways:
      - the run interval itself: "00_r1" (Berke) or "07_r4" (Frank)
      - a derived run interval: "01_r1_noPreTrialTimes" (run interval, pre-trial times removed)
      - an interval named for its epoch: "epoch3_block2", "epoch7_nonLocal_ALL"

    run_intervals maps {nwb_file_name: {interval_list_name: epoch}} (from TaskEpoch).
    Returns the epoch number, or None if the interval can't be placed in an epoch.
    """
    name = str(decoding_interval)
    # Run interval, exact or with a suffix like "_noPreTrialTimes"
    for interval, epoch in run_intervals.get(nwb_file_name, {}).items():
        if name.startswith(interval):
            return epoch
    # Interval that names its epoch up front (block / barrier shift / nonlocal decodes)
    match = re.match(r"epoch(\d+)_", name)
    return int(match.group(1)) if match else None


def valid_decoded_position_keys():
    """Every real (decoding_merge_id, nwb_file_name, epoch) key for HexMazeDecodedPosition.

    Pairs each DecodingOutput entry that belongs to a hex maze session with the single
    epoch its decoding_interval belongs to. Used as the key_source (see below) so that a
    blank .populate() only considers real (decode, epoch) pairs instead of the full
    DecodingOutput x TaskEpoch cross product.
    """
    hex_sessions = set(HexMazeBlock.fetch("nwb_file_name"))

    # Build {nwb_file_name: {run interval name: epoch}} from TaskEpoch
    run_intervals = {}
    for nwb, epoch, interval in zip(
        *(TaskEpoch & [{"nwb_file_name": s} for s in hex_sessions]).fetch(
            "nwb_file_name", "epoch", "interval_list_name"
        )
    ):
        run_intervals.setdefault(nwb, {})[interval] = epoch

    # Walk the DecodingOutput parts for each decode's nwb_file_name + decoding_interval.
    # (We can't use merge_fetch here because parts key on different attributes.)
    keys = []
    for part in DecodingOutput().parts(as_objects=True):
        if "nwb_file_name" not in part.heading.names:
            continue
        for merge_id, nwb, interval in zip(
            *part.fetch("merge_id", "nwb_file_name", "decoding_interval")
        ):
            if nwb not in hex_sessions:
                continue
            epoch = _decode_interval_epoch(nwb, interval, run_intervals)
            if epoch is not None:
                keys.append(
                    {"decoding_merge_id": merge_id, "nwb_file_name": nwb, "epoch": epoch}
                )
    return keys


@schema
class HexMazeDecodedPosition(SpyglassMixin, dj.Computed):
    """
    Calculates most likely decoded position at each time point.
    Stores combined dataframe of decoded and actual position, including
    decode confidence metrics (hpd threshold, spatial coverage of 95% confidence region)
    and distance between decoded and actual position
    """

    definition = """
    -> DecodingOutput.proj(decoding_merge_id = "merge_id")
    -> TaskEpoch
    ---
    -> custom_AnalysisNwbfile
    decoded_position_object_id: varchar(128)
    """

    @property
    def key_source(self):
        # DecodingOutput and TaskEpoch share no attribute, so DataJoint's default
        # key_source is their full cross product (every decode x every epoch). Restrict it
        # to the real (decode, epoch) pairs (each decode paired with the epoch its
        # decoding_interval actually belongs to) so a blank .populate() works instead
        # of trying to populate every single entry in DecodingOutput
        cross = DecodingOutput.proj(decoding_merge_id="merge_id") * TaskEpoch
        # .proj() drops TaskEpoch's secondary columns so key_source is just the primary key
        return (cross & valid_decoded_position_keys()).proj()

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get decode results
        decode_key = {"merge_id": key["decoding_merge_id"]}
        results = DecodingOutput.fetch_results(decode_key)
        
        # Get the posterior (probability of decode at each x,y location at each time point)
        # posterior has shape (n_time, n_x_bins, n_y_bins)
        posterior = results.acausal_posterior.squeeze().unstack("state_bins").sum("state")
        
        # Get timestamps
        # timestamps have shape (n_time,)
        timestamps = posterior.time.values
        
        # Get the max likelihood x,y coordinate at each time point
        # max_likelihood_position has shape (n_time, 2)
        max_likelihood_position = analysis.maximum_a_posteriori_estimate(posterior)
        
        # Get the threshold to plug into get_HPD_spatial_coverage
        # hpd_thresh has shape (n_time,)
        hpd_thresh = get_highest_posterior_threshold(posterior, coverage=0.95).squeeze()
        
        # posterior_stacked has shape (n_time, n_x_bins times n_y_bins)
        posterior_stacked = posterior.stack(position=["x_position", "y_position"])
        posterior_stacked = posterior_stacked.assign_coords(position=np.arange(posterior_stacked.position.size))

        # spatial_cov has shape (n_time,)
        spatial_cov = get_HPD_spatial_coverage(posterior_stacked, hpd_thresh)

        # Make dataframe of decoded position info
        decoded_position_df = pd.DataFrame(
            {
                "time": timestamps,
                "hpd_thresh": hpd_thresh,
                "spatial_cov": spatial_cov,
                "decode_position_x": max_likelihood_position[:, 0],
                "decode_position_y": max_likelihood_position[:, 1],
            }
        ).set_index("time")

        # Get source table (either ClusterlessDecodingV1 or SortedSpikesDecodingV1)
        source_table = DecodingOutput().merge_restrict_class(decode_key)
        classifier = source_table.fetch_model()

        # Get actual position and orientation data from source table
        # We expect position_df cols: position_x, position_y, orientation, velocity_x, velocity_y, speed
        # We expect position_variable_names: ['position_x', 'position_y']; orientation_name: 'orientation'
        position_df, position_variable_names = source_table.fetch_position_info(source_table.fetch1("KEY"))
        orientation_name = source_table.get_orientation_col(position_df)

        # Enforce that position columns are 'position_x', 'position_y' for consistency (just in case)
        position_df = position_df.rename(columns={
            position_variable_names[0]: "position_x",
            position_variable_names[1]: "position_y",
        })

        # Create combined df of actual and decode position
        combined_df = pd.merge(position_df, decoded_position_df, left_index=True, right_index=True)

        # Add distance between actual and decode position to the df
        ahead_behind_distance = analysis.get_ahead_behind_distance2D(
            combined_df[["position_x", "position_y"]].to_numpy(),
            combined_df[orientation_name].to_numpy(),
            combined_df[["decode_position_x", "decode_position_y"]].to_numpy(),
            classifier.environments[0].track_graph,
            classifier.environments[0].edges_,
        )
        combined_df["decode_distance"] = ahead_behind_distance

        # Rearrange columns: actual position first, then decoded
        actual_cols = ["position_x", "position_y", "orientation", "velocity_x", "velocity_y", "speed"]
        decode_cols = ["decode_position_x", "decode_position_y", "decode_distance", "hpd_thresh", "spatial_cov"]
        combined_df = combined_df[[c for c in actual_cols + decode_cols if c in combined_df.columns]]

        # Save time as a column instead of index (NWB requires integer index)
        # reset_index() puts time as the first column automatically
        combined_df = combined_df.reset_index()
        
        # Create an AnalysisNwbfile with a link to the original nwb and add the df
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["decoded_position_object_id"] = builder.add_nwb_object(
                combined_df, "decoded_position"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["decoded_position"].set_index("time")


def compute_aheadness(
    df,
    orientation_col="orientation",
    position_cols=("position_x", "position_y"),
    decode_cols=("decode_position_x", "decode_position_y"),
):
    """Raw "aheadness" = cos of the angle between the rat's heading and the
    straight-line direction from the rat to the decoded position.

        +1  -> decode is straight ahead of the rat
         0  -> decode is 90 degrees off to the side
        -1  -> decode is straight behind the rat

    This is the raw quantity whose *sign* HexMazeDecodedPosition bakes into
    `decode_distance` (decode_distance = sign(aheadness) * graph distance).
    Returning the raw cosine lets us recompute the ahead/behind sign later with
    any threshold or deadband we want, without touching the distance magnitude.

    Note: this uses the straight-line (Euclidean) direction to the decode, so it
    needs only the columns already stored in HexMazeDecodedPosition (no track
    graph). It is topology-blind (near a corner or junction the straight line 
    can cut through a barrier).

    Parameters:
        df (pd.DataFrame): Typically HexMazeDecodedPosition.fetch1_dataframe(). Must
            contain the orientation, actual-position, and decoded-position columns below.
        orientation_col (str): Column holding the rat's head direction, in radians
        position_cols (tuple[str, str]): (x, y) column names for the rat's actual position
        decode_cols (tuple[str, str]): (x, y) column names for the decoded position

    Returns:
        pd.Series: cos(Δθ) per time point, aligned to df's index
    """
    # Straight-line direction (radians) from the rat to the decoded position
    direction_to_decode = np.arctan2(
        df[decode_cols[1]] - df[position_cols[1]],
        df[decode_cols[0]] - df[position_cols[0]],
    )
    # cos of the difference: how aligned the decode is with where the rat faces
    return np.cos(df[orientation_col] - direction_to_decode)


def get_open_hexes(maze):
    """All reachable open hexes in the maze (1-49 minus barriers and unreachable hexes)."""
    return set(range(1, 50)) - maze_to_barrier_set(maze) - get_unreachable_hexes(maze)


def get_epoch_blocks_in_order(nwb_file_name, epoch):
    """
    Every block in this epoch in order with its time bounds and open hexes.
    Helpful so we can get hexes open in this block or previous blocks for assigning decode to hex.

    Note that "open" excludes unreachable hexes as well as barriers (see get_open_hexes), so
    a hex walled off into an unreachable island counts as blocked for as long as it is
    stranded, exactly like a hex with a barrier on it.

    Parameters:
        nwb_file_name (str): The session the epoch belongs to
        epoch (int): The epoch to get blocks for

    Returns:
        list[dict]: One dict per block, ordered by block number, with keys
            "block", "start_time", "end_time", and "open_hexes" (set[int])
    """
    blocks = []
    for block in HexMazeBlock & {"nwb_file_name": nwb_file_name, "epoch": epoch}:
        # Get the block start and end times
        start_time, end_time = (
            sgc.IntervalList
            & {
                "nwb_file_name": nwb_file_name,
                "interval_list_name": block["interval_list_name"],
            }
        ).fetch1("valid_times")[0]

        blocks.append(
            {
                "block": block["block"],
                "start_time": start_time,
                "end_time": end_time,
                "open_hexes": get_open_hexes(block["config_id"]),
            }
        )

    return sorted(blocks, key=lambda b: b["block"])


def blocks_ago_label(blocks_ago):
    """
    Human-readable label for how recently a hex was open.

    Parameters:
        blocks_ago (int or None): How many blocks back the hex was last open
            (0 = open right now, None = it has not been open yet this epoch)

    Returns:
        str: "open", "open_1_block_ago", "open_2_blocks_ago", ..., or "never_open"
    """
    if blocks_ago is None:
        return "never_open"
    if blocks_ago == 0:
        return "open"
    # "1 block ago" but "2 blocks ago"
    return f"open_{blocks_ago}_block_ago" if blocks_ago == 1 else f"open_{blocks_ago}_blocks_ago"


def core_hex(hex_id):
    """
    The core hex (1-49) behind a hex id: "17" -> 17, and "4_left" or "4_right" -> 4.

    The 6 side half-hexes next to the reward ports are keyed by the hex they hang off plus
    a side, so this is how we fold them back into a real hex.

    Parameters:
        hex_id (str or int): a hex id as keyed in a hex_centroids dict, e.g. "17" or "4_left"

    Returns:
        int: the core hex (1-49)
    """
    return int(re.match(r"\d+", str(hex_id)).group())


def centroids_for_hexes(hex_centroids, allowed_hexes):
    """
    Narrow a hex_centroids dict to just the given hexes, so assign_position_to_hex can only
    assign to those - most often the hexes open in this block's maze, via get_open_hexes(maze),
    so that a position never lands on a hex the rat could not have been in. 
    
    A side half-hex is kept whenever its core hex is: e.g. "4_left" kept when 4 in allowed_hexes.

    Parameters:
        hex_centroids (dict): hex id to (x, y) centroid, including the side half-hexes
        allowed_hexes (iterable of int): core hex ids (1-49) to keep

    Returns:
        dict: the subset of hex_centroids belonging to allowed_hexes, in the same order
    """
    allowed = set(allowed_hexes)
    return {h: xy for h, xy in hex_centroids.items() if core_hex(h) in allowed}


def assign_position_to_hex(positions_xy, hex_centroids):
    """
    Assign each (x, y) position to the nearest hex centroid.
    
    Also return the distance from a point to its assigned centroid. 
    This is useful for downstream filtering: if a point is more than a hex radius from 
    its assigned centroid, it is probably not actually in that hex. (This happens e.g. 
    when the rat's head is out of the maze, or if a decoded position is in a barrier
    hex that is not included in the centroids list).

    Parameters:
        positions_xy (np.ndarray): shape (n_positions, 2), the (x, y) positions to assign
        hex_centroids (dict): hex id to (x, y) centroid to assign to

    Returns:
        assigned_hex (list): id of the nearest centroid for each position, keyed as in
            hex_centroids (e.g. "17" or "4_left")
        distance_from_centroid (np.ndarray): distance from each position to that centroid
    """
    hex_ids = list(hex_centroids.keys())
    hex_coords = np.array(list(hex_centroids.values()))  # shape (n_hexes, 2)

    # Find the closest centroid for each position and how far away it is. 
    # Work in chunks of positions: 'diffs' below is (n_positions, n_hexes, 2) floats, 
    # so a whole session at once would need > 1 GB. Chunking caps it at ~44 MB 
    chunk_size = 50_000
    closest_idx = np.empty(len(positions_xy), dtype=int)
    distance_from_centroid = np.empty(len(positions_xy))
    for start in range(0, len(positions_xy), chunk_size):
        chunk = positions_xy[start : start + chunk_size]
        stop = start + len(chunk)
        # Distance from every position in the chunk to every centroid, (n_chunk, n_hexes)
        diffs = chunk[:, np.newaxis, :] - hex_coords[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        closest_idx[start:stop] = np.argmin(distances, axis=1)
        distance_from_centroid[start:stop] = np.min(distances, axis=1)

    # Return a list (same length as positions_xy) of the closest hex for each position
    return [hex_ids[i] for i in closest_idx], distance_from_centroid


@schema
class HexMazeDecodedPositionHex(SpyglassMixin, dj.Computed):
    """
    Assigns actual and decoded position from HexMazeDecodedPosition to the nearest maze hex.
    Stores combined dataframe of decoded and actual position and their assigned hexes, 
    including hex distance between actual and decoded hex, assigned side hexes (half-hexes next to reward ports), 
    and distance from assigned centroid
    """

    definition = """
    -> HexMazeDecodedPosition
    -> HexCentroids
    ---
    -> custom_AnalysisNwbfile
    hex_assignment_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get a dict of hex: (x, y) centroid in cm for this nwbfile
        hex_centroids = HexCentroids.get_hex_centroids_dict_cm(key)

        # Fetch the combined actual/decoded position dataframe from HexMazeDecodedPosition
        position_df = (HexMazeDecodedPosition & key).fetch1_dataframe()

        # Set up a new df to store assigned hex info for each index in position_df
        # (We use -100 and "None" instead of nan to avoid HDF5 datatype issues)
        n = len(position_df)
        hex_df = pd.DataFrame(
            {
                "hex": np.full(n, -100),
                "hex_including_sides": ["None"] * n,
                "distance_from_centroid": np.full(n, -100.0),
                "decode_hex": np.full(n, -100),
                "decode_hex_including_sides": ["None"] * n,
                "decode_distance_from_centroid": np.full(n, -100.0),
                "decode_hex_distance": np.full(n, -100),
            },
            index=position_df.index,
        )

        # Loop through all blocks in this epoch
        for block in (HexMazeBlock & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}):
            # Get maze config for this block
            maze = block.get("config_id")
            
            # Get the block start and end times
            block_start, block_end = (
                sgc.IntervalList
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": block["interval_list_name"],
                }
            ).fetch1("valid_times")[0]

            # Filter position_df to only include times for this block
            block_pos = position_df.loc[block_start:block_end]

            # Only assign positions to hexes that are open in this block's maze, so a
            # position never lands on a hex the rat could not have been in
            open_centroids = centroids_for_hexes(hex_centroids, get_open_hexes(maze))

            # Assign actual position to hex
            actual_xy = block_pos[["position_x", "position_y"]].to_numpy()
            hex_incl_sides, dist_from_centroid = assign_position_to_hex(actual_xy, open_centroids)
            actual_core_hex = [core_hex(h) for h in hex_incl_sides]
            hex_df.loc[block_pos.index, "hex"] = actual_core_hex
            hex_df.loc[block_pos.index, "hex_including_sides"] = hex_incl_sides
            hex_df.loc[block_pos.index, "distance_from_centroid"] = dist_from_centroid

            # Assign decoded position to hex
            decode_xy = block_pos[["decode_position_x", "decode_position_y"]].to_numpy()
            hex_incl_sides, dist_from_centroid = assign_position_to_hex(decode_xy, open_centroids)
            decode_core_hex = [core_hex(h) for h in hex_incl_sides]
            hex_df.loc[block_pos.index, "decode_hex"] = decode_core_hex
            hex_df.loc[block_pos.index, "decode_hex_including_sides"] = hex_incl_sides
            hex_df.loc[block_pos.index, "decode_distance_from_centroid"] = dist_from_centroid

            # Calculate hex distance between actual and decoded hex for each time point
            hex_df.loc[block_pos.index, "decode_hex_distance"] = [
                get_hex_distance(maze=maze, start_hex=actual, target_hex=decode)
                for actual, decode in zip(actual_core_hex, decode_core_hex)
            ]

        # Combine position data with hex assignments
        combined_df = position_df.join(hex_df)

        # Rearrange columns: actual position/hex first, then decoded position/hex
        actual_cols = ["position_x", "position_y", "orientation", "velocity_x", "velocity_y", "speed",
                       "hex", "hex_including_sides", "distance_from_centroid"]
        decode_cols = ["decode_position_x", "decode_position_y",
                       "decode_hex", "decode_hex_including_sides", "decode_distance_from_centroid",
                       "decode_distance", "decode_hex_distance", "hpd_thresh", "spatial_cov"]
        combined_df = combined_df[[c for c in actual_cols + decode_cols if c in combined_df.columns]]

        # Save time as a column instead of index (NWB requires integer index)
        # reset_index() puts time as the first column automatically
        combined_df = combined_df.reset_index()

        # Create an AnalysisNwbfile with a link to the original nwb and add the df
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["hex_assignment_object_id"] = builder.add_nwb_object(combined_df, "hex_assignment")
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    _drop_cols = [
        "hex_including_sides", "distance_from_centroid",
        "decode_hex_including_sides", "decode_distance_from_centroid",
    ]

    def fetch1_dataframe(self):
        # Return the clean dataframe (drop hex including sides, distance from centroid)
        return self.fetch1_dataframe_full().drop(columns=self._drop_cols)

    def fetch1_dataframe_full(self):
        # Return the full dataframe if we need more precise hex assignment info
        return self.fetch_nwb()[0]["hex_assignment"].set_index("time")


@schema
class HexMazeDecodedPositionHexV2(SpyglassMixin, dj.Computed):
    """
    Extension of HexMazeDecodedPositionHex with alternative hex assignments.

    HexMazeDecodedPositionHex assigns each decoded position to the nearest hex that is open in the
    current block, which means a decode that lands on a barrier hex gets snapped to whatever
    open hex happens to be closest. This hides the cases we sometimes care about: decodes of hexes 
    that were open in previous blocks (or even decodes of hexes that were never open in this epoch 
    but the rat might be thinking about anyway based on previous knowledge of the maze).

    History here is scoped to the epoch, and looks only at the past: "open at some point"
    always means open in the current block or an earlier block of the same epoch, never a
    later one and never a different epoch. A hex that is blocked in block 3 and only opens up
    in block 4 counts as never having been open during block 3.

    This table keeps every column from HexMazeDecodedPositionHex and adds:

    Assignment allowing any hex open at any point in the epoch so far
    (the union of open hexes over every block up to and including the current one):
        decode_hex_epoch_open, decode_hex_epoch_open_including_sides,
        decode_epoch_open_distance_from_centroid

    Assignment allowing all hexes, regardless of whether they were ever open:
        decode_hex_any, decode_hex_any_including_sides, decode_any_distance_from_centroid

    Whether the position falls inside the maze's physical footprint (1 = yes, 0 = no):
        in_maze, decode_in_maze

    How recently the decode's true nearest hex (decode_hex_any) was last open, as of the
    block the decode happened in:
        decode_hex_blocks_since_open: 0 if the hex is open in the current block, N if the
            most recent block it was open in was N blocks ago, -1 if it has not been open
            yet this epoch, and -100 for timepoints outside any block.
        decode_hex_open_status: the same thing as a label: "open", "open_1_block_ago",
            "open_2_blocks_ago", ..., "never_open", or "None" outside any block.

    Note that decode_hex_any, in_maze, and decode_in_maze do not depend on the maze
    config, so they are filled for every timepoint in the decode. The epoch-open and
    open-status columns do depend on which block we are in, so (like the parent table's
    columns) they are only filled during blocks. (We should always be within block time 
    bounds, so the outside-block defaults are just a safety net).
    """

    definition = """
    -> HexMazeDecodedPositionHex
    ---
    -> custom_AnalysisNwbfile
    hex_assignment_v2_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return

        # Get the full hex assignment dataframe from the parent table (we keep all of its
        # columns, including the "including sides" / "distance from centroid" detail columns)
        combined_df = (HexMazeDecodedPositionHex & key).fetch1_dataframe_full()

        # Get a dict of hex: (x, y) centroid in cm for this nwbfile.
        # hex_centroids includes the 6 side half-hexes next to the reward ports (e.g. "4_left");
        # core_hex_centroids has just the 49 real hexes, keyed by int, which is what the
        # maze bounding box needs (it infers hex size from centroid spacing).
        hex_centroids = HexCentroids.get_hex_centroids_dict_cm(key)
        core_hex_centroids = HexCentroids.get_core_hex_centroids_dict_cm(key)

        actual_xy = combined_df[["position_x", "position_y"]].to_numpy()
        decode_xy = combined_df[["decode_position_x", "decode_position_y"]].to_numpy()

        # Assign decoded position to the nearest hex out of all hexes, whether or
        # not that hex was ever open (we pass the full centroid dict). This
        # doesn't depend on the maze config, so unlike the parent table we can do it in one
        # pass over the whole session instead of block by block.
        decode_any_incl_sides, decode_any_dist = assign_position_to_hex(decode_xy, hex_centroids)
        combined_df["decode_hex_any"] = [core_hex(h) for h in decode_any_incl_sides]
        combined_df["decode_hex_any_including_sides"] = decode_any_incl_sides
        combined_df["decode_any_distance_from_centroid"] = decode_any_dist

        # Flag positions that fall outside the physical footprint of the maze (with tolerance 1/20 of hex)
        combined_df["in_maze"] = are_points_in_maze(actual_xy, core_hex_centroids).astype(int)
        combined_df["decode_in_maze"] = are_points_in_maze(decode_xy, core_hex_centroids).astype(int)

        # The remaining columns depend on which blocks of this epoch have already happened,
        # so we walk the epoch's blocks in chronological order
        # (Use -100 and "None", as defaults to avoid nan/HDF5 datatype issues)
        combined_df["decode_hex_epoch_open"] = -100
        combined_df["decode_hex_epoch_open_including_sides"] = "None"
        combined_df["decode_epoch_open_distance_from_centroid"] = -100.0
        combined_df["decode_hex_blocks_since_open"] = -100
        combined_df["decode_hex_open_status"] = "None"

        # Every block in this epoch, oldest first. History does not carry across epochs:
        # each epoch starts over with nothing having been open yet.
        epoch_blocks = get_epoch_blocks_in_order(key["nwb_file_name"], key["epoch"])

        # As we walk forward through blocks, remember the most recent block each hex was open in
        # We only add hexes to this set once they are reachable in a block 
        # (so it exludes hexes that were never open in this epoch by default)
        last_open_block_idx = {}  # hex -> index of the most recent block it was open in

        for block_idx, block in enumerate(epoch_blocks):
            # Record this block's open hexes first, so a hex open right now is "0 blocks ago"
            for open_hex in block["open_hexes"]:
                last_open_block_idx[open_hex] = block_idx

            # Filter dataframe to only include times for this block
            block_pos = combined_df.loc[block["start_time"]:block["end_time"]]
            if block_pos.empty:
                continue

            # Every hex open in this block or any earlier block of this epoch
            open_so_far = set(last_open_block_idx)

            # How many blocks back each hex was last open (0 = open in this block)
            blocks_since_open = {h: block_idx - idx for h, idx in last_open_block_idx.items()}

            # Assign decoded position to the nearest hex that has been open at some point so far
            decode_xy_block = block_pos[["decode_position_x", "decode_position_y"]].to_numpy()
            epoch_open_centroids = centroids_for_hexes(hex_centroids, open_so_far)
            epoch_incl_sides, epoch_dist = assign_position_to_hex(
                decode_xy_block, epoch_open_centroids
            )
            combined_df.loc[block_pos.index, "decode_hex_epoch_open"] = [core_hex(h) for h in epoch_incl_sides]
            combined_df.loc[block_pos.index, "decode_hex_epoch_open_including_sides"] = epoch_incl_sides
            combined_df.loc[block_pos.index, "decode_epoch_open_distance_from_centroid"] = epoch_dist

            # Record how recently decode_hex_any (the true nearest hex) was last open, 
            # as a number of  blocks (-1 if it has not been open yet this epoch) and as a readable label
            combined_df.loc[block_pos.index, "decode_hex_blocks_since_open"] = [
                blocks_since_open.get(h, -1) for h in block_pos["decode_hex_any"]
            ]
            combined_df.loc[block_pos.index, "decode_hex_open_status"] = [
                blocks_ago_label(blocks_since_open.get(h)) for h in block_pos["decode_hex_any"]
            ]

        # Rearrange columns
        # Actual position (same as hexMazeDecodedPositionHex) - only assigned to currently open hexes
        actual_cols = ["position_x", "position_y", "orientation", "velocity_x", "velocity_y", "speed",
                       "hex", "hex_including_sides", "distance_from_centroid", "in_maze"]
        # Decoded position - assigned in a variety of ways 
        decode_cols = ["decode_position_x", "decode_position_y",
                       # hexes currently open (same as hexMazeDecodedPositionHex)
                       "decode_hex", "decode_hex_including_sides", "decode_distance_from_centroid",
                       # hexes open at any point in this epoch (up until this point)
                       "decode_hex_epoch_open", "decode_hex_epoch_open_including_sides",
                       "decode_epoch_open_distance_from_centroid",
                       # any hex 1-49 even if never open/not open yet in this epoch
                       "decode_hex_any", "decode_hex_any_including_sides",
                       "decode_any_distance_from_centroid",
                       # for decode assigned to any hex, note if that hex is currently open/when it was last open
                       "decode_hex_blocks_since_open", "decode_hex_open_status", 
                       # distance measures (same as hexMazeDecodedPositionHex) + is decode in maze
                       "decode_in_maze", "decode_distance", "decode_hex_distance", 
                       # decode quality metrics
                       "hpd_thresh", "spatial_cov"]
        combined_df = combined_df[[c for c in actual_cols + decode_cols if c in combined_df.columns]]

        # Save time as a column instead of index (NWB requires integer index)
        # reset_index() puts time as the first column automatically
        combined_df = combined_df.reset_index()

        # Create an AnalysisNwbfile with a link to the original nwb and add the df
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["hex_assignment_v2_object_id"] = builder.add_nwb_object(
                combined_df, "hex_assignment_v2"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    # The "including sides" and "distance from centroid" columns are extra precision that
    # most analyses don't need, so fetch1_dataframe drops them (same as the parent table)
    _drop_cols = [
        "hex_including_sides", "distance_from_centroid",
        "decode_hex_including_sides", "decode_distance_from_centroid",
        "decode_hex_epoch_open_including_sides", "decode_epoch_open_distance_from_centroid",
        "decode_hex_any_including_sides", "decode_any_distance_from_centroid",
    ]

    def fetch1_dataframe(self):
        # Return the clean dataframe (drop hex including sides, distance from centroid)
        return self.fetch1_dataframe_full().drop(columns=self._drop_cols)

    def fetch1_dataframe_full(self):
        # Return the full dataframe if we need more precise hex assignment info
        return self.fetch_nwb()[0]["hex_assignment_v2"].set_index("time")


@schema
class HexMazeDecodedHexPath(SpyglassMixin, dj.Computed):
    """
    Stores each hex transition within a trial, including entry/exit times,
    maze component, and distance to/from ports.
    Built from HexMazeDecodedPositionHex.
    """

    definition = """
    -> HexMazeDecodedPositionHex
    ---
    -> custom_AnalysisNwbfile
    hex_path_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get hex position dataframe for this nwb+epoch
        hex_position_df = (HexMazeDecodedPositionHex & key).fetch1_dataframe()
        nwb_file = key["nwb_file_name"]
        epoch = key["epoch"]

        # Get trials for this nwb+epoch
        trials = HexMazeBlock().Trial() & {"nwb_file_name": nwb_file, "epoch": epoch}

        # Accumulate per-trial dataframes
        all_hex_paths = []

        for trial in trials:
            # Get trial time bounds
            trial_start, trial_end = (
                sgc.IntervalList
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "interval_list_name": trial["interval_list_name"],
                }
            ).fetch1("valid_times")[0]

            # Get maze configuration and attributes
            maze = (
                HexMazeBlock()
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "block": trial["block"],
                    "epoch": trial["epoch"],
                }
            ).fetch1("config_id")

            # Filter decoded position data to this trial
            trial_df = hex_position_df.loc[trial_start:trial_end].copy()

            # Identify contiguous segments: new segment whenever hex OR decode_hex changes
            hex_changed = trial_df["hex"] != trial_df["hex"].shift()
            decode_hex_changed = trial_df["decode_hex"] != trial_df["decode_hex"].shift()
            trial_df["segment"] = (hex_changed | decode_hex_changed).cumsum()

            # Set up dataframe of hex entries for this trial
            hex_path = (
                trial_df.groupby("segment")
                .agg(
                    hex=("hex", "first"),
                    decode_hex=("decode_hex", "first"),
                    entry_time=("hex", lambda x: x.index[0]),
                    exit_time=("hex", lambda x: x.index[-1]),
                )
                .reset_index(drop=True)
            )

            # Time spent in each segment
            hex_path["duration"] = hex_path["exit_time"] - hex_path["entry_time"]

            # What number segment in the trial this is
            hex_path["hex_in_trial"] = range(1, len(hex_path) + 1)

            # Count the number of times the rat has entered this specific hex in this trial
            hex_path["hex_entry_num"] = hex_path.groupby("hex").cumcount() + 1

            # Count the number of times decode has entered this specific hex in this trial
            hex_path["decode_hex_entry_num"] = hex_path.groupby("decode_hex").cumcount() + 1

            # For each hex, compute distances to start and end port (actual and decoded)
            start_port, end_port = trial["start_port"], trial["end_port"]
            if start_port == "None":
                # First trial does not have a start port, so we just fill with -100
                hex_path["hexes_from_start"] = -100
                hex_path["decode_hexes_from_start"] = -100
            else:
                hex_path["hexes_from_start"] = [
                    get_hexes_from_port(maze, start_hex=h, reward_port=start_port)
                    for h in hex_path["hex"]
                ]
                hex_path["decode_hexes_from_start"] = [
                    get_hexes_from_port(maze, start_hex=h, reward_port=start_port)
                    for h in hex_path["decode_hex"]
                ]
            hex_path["hexes_from_end"] = [
                get_hexes_from_port(maze, start_hex=h, reward_port=end_port)
                for h in hex_path["hex"]
            ]
            hex_path["decode_hexes_from_end"] = [
                get_hexes_from_port(maze, start_hex=h, reward_port=end_port)
                for h in hex_path["decode_hex"]
            ]

            # Hex distance between actual and decoded hex for each segment
            hex_path["decode_hex_distance"] = [
                get_hex_distance(maze=maze, start_hex=a, target_hex=d)
                for a, d in zip(hex_path["hex"], hex_path["decode_hex"])
            ]

            # Classify each hex as optimal, non-optimal, or dead-end
            hex_to_type = {
                h: name.replace("_hexes", "")
                for name, hexes in classify_maze_hexes(maze).items()
                if name in {"optimal_hexes", "non_optimal_hexes", "dead_end_hexes"}
                for h in hexes
            }
            hex_path["hex_type"] = hex_path["hex"].map(hex_to_type)
            hex_path["decode_hex_type"] = hex_path["decode_hex"].map(hex_to_type)

            # Map each hex to the section of the maze it's in (1, 2, or 3 for near port A, B, or C)
            hex_to_maze_third = {
                h: third_num
                for third_num, hexes in enumerate(divide_into_thirds(maze), start=1)
                for h in hexes
            }
            # Map choice points to section 0
            hex_to_maze_third.update({h: 0 for h in get_critical_choice_points(maze, start_port if start_port != "None" else None)})

            # Identify the maze sections as 'start', 'chosen', or 'unchosen'
            # Note that for the first trial, start_port is None so start_section and unchosen_section will both be None
            port_map = {"A": 1, "B": 2, "C": 3}
            start_section = port_map.get(start_port)
            chosen_section = port_map.get(end_port)
            unchosen_section = {1, 2, 3} - {chosen_section} - {start_section}
            unchosen_section = unchosen_section.pop() if len(unchosen_section) == 1 else None

            # Map maze section number to its label
            label = {
                start_section: "start",
                chosen_section: "chosen",
                unchosen_section: "unchosen",
                0: "choice_point",
            }

            # Assign maze section label for each hex (if no label, e.g. first section of first trial, it will be "None")
            hex_to_label = lambda h: str(label.get(hex_to_maze_third.get(h)))
            hex_path["maze_portion"] = hex_path["hex"].map(hex_to_label)
            hex_path["decode_maze_portion"] = hex_path["decode_hex"].map(hex_to_label)

            # Compute distance from each hex to the critical choice point
            # Pass start_port (or None for first trial) to get the relevant choice point(s)
            choice_points = get_critical_choice_points(maze, start_port if start_port != "None" else None)
            # Distance from actual hex to nearest choice point
            # Distance is negative for hexes in the 'start' section (pre choice point), positive after
            hex_path["hexes_from_choice"] = [
                min(get_hex_distance(maze=maze, start_hex=h, target_hex=cp) for cp in choice_points)
                * (-1 if hex_to_maze_third.get(h) == start_section else 1)
                for h in hex_path["hex"]
            ]
            # Distance from decoded hex to nearest choice point
            hex_path["decode_hexes_from_choice"] = [
                min(get_hex_distance(maze=maze, start_hex=h, target_hex=cp) for cp in choice_points)
                * (-1 if hex_to_maze_third.get(h) == start_section else 1)
                for h in hex_path["decode_hex"]
            ]

            # Add block/trial key columns and put them on the left
            key_cols = ["nwb_file_name", "epoch", "block", "block_trial_num", "epoch_trial_num"]
            for col in key_cols:
                hex_path[col] = trial[col]
            hex_path = hex_path[key_cols + [c for c in hex_path.columns if c not in key_cols]]

            # Add the hex path for this trial
            all_hex_paths.append(hex_path)

        # Concatenate per-trial dataframes into one big dataframe
        hex_path_all_trials = pd.concat(all_hex_paths, ignore_index=True)

        # Create an empty AnalysisNwbfile with a link to the original nwb
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            # Add the hex path dataframe to the AnalysisNwbfile
            key["hex_path_object_id"] = builder.add_nwb_object(hex_path_all_trials, "hex_path")

            # File automatically registered on exit!
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["hex_path"]

    def fetch_block(self, block):
        """Return hex_path rows for a specific block."""
        df = self.fetch1_dataframe()
        df_block = df[df["block"] == block]
        return df_block.reset_index(drop=True)

    def fetch_trial(self, block, block_trial_num):
        """Return hex_path rows for a specific trial within a block."""
        df = self.fetch1_dataframe()
        df_trial = df[
            (df["block"] == block) & (df["block_trial_num"] == block_trial_num)
        ]
        return df_trial.reset_index(drop=True)

    def fetch_trials(self, block=None, block_trial_num=None):
        """Return hex_path rows optionally filtered to specific blocks or trials"""
        df = self.fetch1_dataframe()

        if block is not None:
            if isinstance(block, (list, tuple, set)):
                df = df[df["block"].isin(block)]
            else:
                df = df[df["block"] == block]

        if block_trial_num is not None:
            if isinstance(block_trial_num, (list, tuple, set)):
                df = df[df["block_trial_num"].isin(block_trial_num)]
            else:
                df = df[df["block_trial_num"] == block_trial_num]

        return df.reset_index(drop=True)

    def plot_trial(self, block, block_trial_num, ax=None, show_stats=True):
        """Plot a single trial's trajectory on the hex maze."""

        # Fetch the hex path for this trial
        df = self.fetch_trial(block, block_trial_num)
        if df.empty:
            raise ValueError(
                f"No hex path found for block {block}, trial {block_trial_num}"
            )
        hex_path = df["hex"].tolist()

        # Fetch the key for this HexPath entry
        key = self.fetch1("KEY")  # contains nwb_file_name + epoch

        # Fetch maze config for the given block in this epoch
        block_entry = HexMazeBlock() & {
            "nwb_file_name": key["nwb_file_name"],
            "epoch": key["epoch"],
            "block": block,
        }
        maze_config = block_entry.fetch1("config_id")

        if show_stats:
            reward_probs = [int(block_entry.fetch1(f"p_{x}")) for x in ["a", "b", "c"]]
        else:
            reward_probs = None

        # Create figure if no axis provided
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            created_fig = True

        # Plot the maze with the hex path
        plot_hex_maze(
            barriers=maze_config,
            ax=ax,
            hex_path=hex_path,
            show_barriers=False,
            show_choice_points=False,
            show_hex_labels=False,
            show_stats=show_stats,
            reward_probabilities=reward_probs,
        )
        ax.set_title(f"Block {block}, Trial {block_trial_num}")

        if created_fig:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_block(self, block, trials=None, show_stats=True):
        """Plot trial trajectories for all trials in a block on the hex maze."""

        # Fetch all trial paths for the block at once
        df_block = self.fetch_block(block)
        if df_block.empty:
            raise ValueError(f"No hex path found for block {block}")

        if trials is None:
            trials = sorted(df_block["block_trial_num"].unique())

        num_trials = len(trials)

        # Fetch block info
        key = self.fetch1("KEY")  # contains nwb_file_name + epoch
        nwb_file, epoch = key["nwb_file_name"], key["epoch"]

        # Fetch maze config and reward probabilities for this block
        block_entry = HexMazeBlock() & {
            "nwb_file_name": nwb_file,
            "epoch": epoch,
            "block": block,
        }
        maze_config = block_entry.fetch1("config_id")
        if show_stats:
            reward_probs = [int(block_entry.fetch1(f"p_{x}")) for x in ["a", "b", "c"]]
        else:
            reward_probs = None

        # Determine square-ish grid
        ncols = int(np.ceil(np.sqrt(num_trials)))
        nrows = int(np.ceil(num_trials / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

        # Make sure axes is 1D so flatten doesn't break
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()

        # Big title
        fig.suptitle(f"{nwb_file} epoch {epoch}, block {block}", fontsize=20, y=1.02)

        # Loop over trials and plot hex path for each one
        for i, tri_num in enumerate(trials):
            df_trial = df_block[df_block["block_trial_num"] == tri_num]
            if df_trial.empty:
                raise ValueError(
                    f"No hex path found for block {block}, trial {tri_num}"
                )
            hex_path = df_trial["hex"].tolist()

            plot_hex_maze(
                barriers=maze_config,
                ax=axes[i],
                hex_path=hex_path,
                show_barriers=False,
                show_choice_points=False,
                show_hex_labels=False,
                show_stats=show_stats,
                reward_probabilities=reward_probs,
            )
            axes[i].set_title(f"Trial {tri_num}")

        # Hide unused axes
        for j in range(num_trials, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

        return axes



#### These are in dev and Steph might mess with them at any time, fyi

@schema
class HexMazeDecodedPositionHexAnnotated(SpyglassMixin, dj.Computed):
    """
    Adds per-timepoint maze annotations to HexMazeDecodedPositionHex.
    Same columns as HexMazeDecodedHexPath (hex_type, maze_portion, hexes_from_start/end/choice, etc.)
    but without aggregation — one row per timepoint, not per hex segment.
    This is nice so we can filter by speed, spatial coverage, etc. at each time point
    """

    definition = """
    -> HexMazeDecodedPositionHex
    ---
    -> custom_AnalysisNwbfile
    annotated_hex_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get hex position dataframe for this nwb+epoch
        hex_position_df = (HexMazeDecodedPositionHex & key).fetch1_dataframe()
        nwb_file = key["nwb_file_name"]
        epoch = key["epoch"]

        # Initialize new columns with defaults (-100 for numeric, "None" for string)
        # Rows that fall outside any trial will keep these defaults
        hex_position_df["block"] = -100
        hex_position_df["block_trial_num"] = -100
        hex_position_df["epoch_trial_num"] = -100
        hex_position_df["hex_type"] = "None"
        hex_position_df["decode_hex_type"] = "None"
        hex_position_df["maze_portion"] = "None"
        hex_position_df["decode_maze_portion"] = "None"
        hex_position_df["hexes_from_start"] = -100
        hex_position_df["hexes_from_end"] = -100
        hex_position_df["hexes_from_unchosen"] = -100
        hex_position_df["hexes_from_choice"] = -100
        hex_position_df["decode_hexes_from_start"] = -100
        hex_position_df["decode_hexes_from_end"] = -100
        hex_position_df["decode_hexes_from_unchosen"] = -100
        hex_position_df["decode_hexes_from_choice"] = -100

        # Get trials for this nwb+epoch
        trials = HexMazeBlock().Trial() & {"nwb_file_name": nwb_file, "epoch": epoch}

        port_map = {"A": 1, "B": 2, "C": 3}

        # Caches for per-maze computations (avoid recomputing for trials in the same block, makes it way faster)
        hex_type_cache = {}       # maze -> {hex: type_str}
        thirds_cache = {}         # maze -> {hex: third_num}
        port_dist_cache = {}      # (maze, port) -> {hex: distance}
        choice_dist_cache = {}    # (maze, start_port) -> {hex: signed_distance}

        def _get_hex_type_map(maze):
            """Hex -> classification (optimal, non_optimal, or dead_end)."""
            if maze not in hex_type_cache:
                hex_type_cache[maze] = {
                    h: name.replace("_hexes", "")
                    for name, hexes in classify_maze_hexes(maze).items()
                    if name in {"optimal_hexes", "non_optimal_hexes", "dead_end_hexes"}
                    for h in hexes
                }
            return hex_type_cache[maze]

        def _get_thirds_map(maze):
            """Hex -> maze third (1, 2, or 3 for near port A, B, or C)."""
            if maze not in thirds_cache:
                thirds_cache[maze] = {
                    h: third_num
                    for third_num, hexes in enumerate(divide_into_thirds(maze), start=1)
                    for h in hexes
                }
            return thirds_cache[maze]

        def _get_port_dist_map(maze, port):
            """Hex -> distance from the given reward port."""
            cache_key = (maze, port)
            if cache_key not in port_dist_cache:
                port_dist_cache[cache_key] = {
                    h: get_hexes_from_port(maze, start_hex=h, reward_port=port)
                    for h in get_open_hexes(maze)
                }
            return port_dist_cache[cache_key]

        def _get_choice_dist_map(maze, start_port):
            """Hex -> signed distance from the critical choice point.

            Negative for hexes in the start section (before choice point),
            positive for hexes after the choice point.
            """
            cache_key = (maze, start_port)
            if cache_key not in choice_dist_cache:
                sp = start_port if start_port != "None" else None
                choice_points = get_critical_choice_points(maze, sp)
                start_section = port_map.get(start_port)

                # Build hex -> section map including choice points as section 0
                hex_to_section = dict(_get_thirds_map(maze))
                hex_to_section.update({h: 0 for h in choice_points})

                choice_dist_cache[cache_key] = {
                    h: min(get_hex_distance(maze=maze, start_hex=h, target_hex=cp) for cp in choice_points)
                    * (-1 if hex_to_section.get(h) == start_section else 1)
                    for h in get_open_hexes(maze)
                }
            return choice_dist_cache[cache_key]

        for trial in trials:
            # Get trial time bounds
            trial_start, trial_end = (
                sgc.IntervalList
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "interval_list_name": trial["interval_list_name"],
                }
            ).fetch1("valid_times")[0]

            # Get maze configuration for this block
            maze = (
                HexMazeBlock()
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "block": trial["block"],
                    "epoch": trial["epoch"],
                }
            ).fetch1("config_id")

            # Filter to this trial's timepoints
            trial_df = hex_position_df.loc[trial_start:trial_end]
            idx = trial_df.index
            start_port, end_port = trial["start_port"], trial["end_port"]

            # Trial identifiers
            hex_position_df.loc[idx, "block"] = trial["block"]
            hex_position_df.loc[idx, "block_trial_num"] = trial["block_trial_num"]
            hex_position_df.loc[idx, "epoch_trial_num"] = trial["epoch_trial_num"]

            # Hex classification (optimal, non-optimal, dead-end)
            hex_to_type = _get_hex_type_map(maze)
            hex_position_df.loc[idx, "hex_type"] = trial_df["hex"].map(hex_to_type).fillna("None")
            hex_position_df.loc[idx, "decode_hex_type"] = trial_df["decode_hex"].map(hex_to_type).fillna("None")

            # Maze portion (start, chosen, unchosen, choice_point)
            # Build hex -> section map: thirds (1/2/3) + choice points (0)
            hex_to_section = dict(_get_thirds_map(maze))
            hex_to_section.update({
                h: 0 for h in get_critical_choice_points(maze, start_port if start_port != "None" else None)
            })

            # Map section numbers to trial-relative labels
            start_section = port_map.get(start_port)
            chosen_section = port_map.get(end_port)
            unchosen_section = {1, 2, 3} - {chosen_section} - {start_section}
            unchosen_section = unchosen_section.pop() if len(unchosen_section) == 1 else None

            section_to_label = {
                start_section: "start",
                chosen_section: "chosen",
                unchosen_section: "unchosen",
                0: "choice_point",
            }

            hex_to_label = {h: str(section_to_label.get(section)) for h, section in hex_to_section.items()}
            hex_position_df.loc[idx, "maze_portion"] = trial_df["hex"].map(hex_to_label).fillna("None")
            hex_position_df.loc[idx, "decode_maze_portion"] = trial_df["decode_hex"].map(hex_to_label).fillna("None")

            # Distance from ports
            if start_port != "None":
                start_dist = _get_port_dist_map(maze, start_port)
                hex_position_df.loc[idx, "hexes_from_start"] = trial_df["hex"].map(start_dist).fillna(-100)
                hex_position_df.loc[idx, "decode_hexes_from_start"] = trial_df["decode_hex"].map(start_dist).fillna(-100)

            end_dist = _get_port_dist_map(maze, end_port)
            hex_position_df.loc[idx, "hexes_from_end"] = trial_df["hex"].map(end_dist).fillna(-100)
            hex_position_df.loc[idx, "decode_hexes_from_end"] = trial_df["decode_hex"].map(end_dist).fillna(-100)

            unchosen_port = {"A", "B", "C"} - {start_port, end_port}
            if len(unchosen_port) == 1:
                unchosen_dist = _get_port_dist_map(maze, unchosen_port.pop())
                hex_position_df.loc[idx, "hexes_from_unchosen"] = trial_df["hex"].map(unchosen_dist).fillna(-100)
                hex_position_df.loc[idx, "decode_hexes_from_unchosen"] = trial_df["decode_hex"].map(unchosen_dist).fillna(-100)

            # Signed distance from choice point
            choice_dist = _get_choice_dist_map(maze, start_port)
            hex_position_df.loc[idx, "hexes_from_choice"] = trial_df["hex"].map(choice_dist).fillna(-100)
            hex_position_df.loc[idx, "decode_hexes_from_choice"] = trial_df["decode_hex"].map(choice_dist).fillna(-100)

        # Save time as a column instead of index (NWB requires integer index)
        hex_position_df = hex_position_df.reset_index()

        # Write to NWB analysis file
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["annotated_hex_object_id"] = builder.add_nwb_object(hex_position_df, "annotated_hex")
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["annotated_hex"].set_index("time")


@schema
class HexMazeDecodedHexPathBarrierChange(SpyglassMixin, dj.Computed):
    """
    Extension of HexMazeDecodedHexPath for barrier change sessions.
    Adds columns for hex distance from the path divergence point, and
    classifies each hex relative to the barrier change:
    old_path, new_path, before_divergence, after_convergence, or other.

    Only populates for barrier change sessions.
    """

    definition = """
    -> HexMazeDecodedHexPath
    ---
    -> custom_AnalysisNwbfile
    barrier_change_hex_path_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        nwb_file = key["nwb_file_name"]
        epoch = key["epoch"]

        # Get all blocks for this epoch, ordered by block number
        blocks = (
            HexMazeBlock & {"nwb_file_name": nwb_file, "epoch": epoch}
        ).fetch(as_dict=True, order_by="block")

        # Check that this is a barrier change session
        if not any(b["task_type"] == "barrier change" for b in blocks):
            return

        # Build a map of block number to its maze config and the previous block's config
        # For the first block (or non-barrier-change blocks), old_maze is None
        block_maze_map = {}
        for i, block in enumerate(blocks):
            old_maze = blocks[i - 1]["config_id"] if i > 0 and block["task_type"] == "barrier change" else None
            block_maze_map[block["block"]] = {
                "new_maze": block["config_id"],
                "old_maze": old_maze,
            }

        # Get the hex path dataframe from HexMazeDecodedHexPath
        hex_path_df = (HexMazeDecodedHexPath & key).fetch1_dataframe()

        # Initialize new columns
        hex_path_df["hexes_from_divergence"] = -100
        hex_path_df["decode_hexes_from_divergence"] = -100
        hex_path_df["barrier_change_hex_class"] = "None"
        hex_path_df["decode_barrier_change_hex_class"] = "None"

        # Get trials for this epoch
        trials = HexMazeBlock().Trial() & {"nwb_file_name": nwb_file, "epoch": epoch}

        # Process each trial
        for trial in trials:
            block_num = trial["block"]
            maze_info = block_maze_map[block_num]
            old_maze = maze_info["old_maze"]
            new_maze = maze_info["new_maze"]

            # Skip trials in blocks that are not barrier changes
            if old_maze is None:
                continue

            start_port = trial["start_port"]
            end_port = trial["end_port"]

            # Skip first trial with no start port
            if start_port == "None":
                continue

            # Get rows for this trial
            trial_mask = (
                (hex_path_df["block"] == block_num)
                & (hex_path_df["block_trial_num"] == trial["block_trial_num"])
            )

            # Get the path divergence point hex for this trial's start/end ports
            divergence_hex = get_path_divergence_point(old_maze, new_maze, start_port, end_port)

            # Classify hexes relative to the barrier change:
            # before_divergence: hexes shared by old and new paths before the divergence point
            # old_path: hexes unique to the old (pre-barrier-change) optimal path
            # new_path: hexes unique to the new (post-barrier-change) optimal path
            # after_convergence: hexes shared by old and new paths after convergence
            # other: hexes not on either optimal path (dead ends, non-optimal)
            before_div = get_hexes_before_divergence(old_maze, new_maze, start_port, end_port)

            # Compute hex distance from divergence point for actual and decoded hex
            # Negative for hexes before the divergence point (on the approach from start)
            # If paths are identical (no divergence), fill with -100
            trial_rows = hex_path_df.loc[trial_mask]
            if divergence_hex is None:
                hex_path_df.loc[trial_mask, "hexes_from_divergence"] = -100
                hex_path_df.loc[trial_mask, "decode_hexes_from_divergence"] = -100
            else:
                hex_path_df.loc[trial_mask, "hexes_from_divergence"] = [
                    get_hex_distance(maze=new_maze, start_hex=h, target_hex=divergence_hex)
                    * (-1 if h in before_div else 1)
                    for h in trial_rows["hex"]
                ]
                hex_path_df.loc[trial_mask, "decode_hexes_from_divergence"] = [
                    get_hex_distance(maze=new_maze, start_hex=h, target_hex=divergence_hex)
                    * (-1 if h in before_div else 1)
                    for h in trial_rows["decode_hex"]
                ]
            hexes_on_old_path, hexes_on_new_path = get_optimal_path_hexes_after_divergence(
                old_maze, new_maze, start_port, end_port
            )
            after_conv = get_hexes_before_divergence(old_maze, new_maze, end_port, start_port)

            # Build hex -> classification lookup
            hex_class = {}
            for h in before_div:
                hex_class[h] = "before_divergence"
            for h in hexes_on_old_path:
                hex_class[h] = "old_path"
            for h in hexes_on_new_path:
                hex_class[h] = "new_path"
            for h in after_conv:
                hex_class[h] = "after_convergence"

            # Classify actual and decoded hexes (default to "other" if not in any category)
            hex_path_df.loc[trial_mask, "barrier_change_hex_class"] = [
                hex_class.get(h, "other") for h in trial_rows["hex"]
            ]
            hex_path_df.loc[trial_mask, "decode_barrier_change_hex_class"] = [
                hex_class.get(h, "other") for h in trial_rows["decode_hex"]
            ]

        # Create an AnalysisNwbfile and save the dataframe
        with custom_AnalysisNwbfile().build(nwb_file) as builder:
            key["barrier_change_hex_path_object_id"] = builder.add_nwb_object(
                hex_path_df, "barrier_change_hex_path"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["barrier_change_hex_path"]


@schema
class HexMazeJunctionDecode(SpyglassMixin, dj.Computed):
    """
    Adds per-timepoint junction context to HexMazeDecodedPositionHexAnnotated.

    For every 3-way junction (degree-3 hex) in the maze, this table identifies
    when the rat passes through and classifies:
    - The entry direction (which neighbor the rat came from)
    - Which exit is "left" vs "right" (via cross-product geometry)
    - Which direction the rat actually went
    - Which direction the decode_hex points toward (via graph distance)

    This enables analysis of whether the neural decoder anticipates upcoming
    turns at arbitrary maze junctions, not just the critical choice point.
    """

    definition = """
    -> HexMazeDecodedPositionHexAnnotated
    ---
    -> custom_AnalysisNwbfile
    junction_decode_object_id: varchar(128)
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get per-timepoint annotated dataframe from parent table
        annotated_df = (HexMazeDecodedPositionHexAnnotated & key).fetch1_dataframe()
        nwb_file = key["nwb_file_name"]
        epoch = key["epoch"]

        # Initialize new columns with defaults
        # -100 for numeric columns, "None" for string columns
        annotated_df["nearest_junction"] = -100
        annotated_df["hexes_from_junction"] = -100
        annotated_df["junction_entry_hex"] = -100
        annotated_df["junction_left_exit"] = -100
        annotated_df["junction_right_exit"] = -100
        annotated_df["rat_junction_direction"] = "None"
        annotated_df["decode_junction_direction"] = "None"

        # Get trials for this nwb+epoch
        trials = HexMazeBlock().Trial() & {"nwb_file_name": nwb_file, "epoch": epoch}

        # Caches for per-maze computations (avoid recomputing across trials in same block)
        junction_lr_cache = {}       # maze -> junction left/right map
        all_pairs_dist_cache = {}    # maze -> dict of all-pairs shortest path lengths
        junction_set_cache = {}      # maze -> set of junction hex IDs

        def _get_junction_set(maze):
            """Get the set of 3-way junction hexes for this maze."""
            if maze not in junction_set_cache:
                junction_set_cache[maze] = get_all_choice_points(maze)
            return junction_set_cache[maze]

        def _get_junction_lr(maze):
            """Get the junction left/right map for this maze."""
            if maze not in junction_lr_cache:
                junction_lr_cache[maze] = get_junction_left_right_map(maze)
            return junction_lr_cache[maze]

        def _get_all_pairs_dist(maze):
            """Get all-pairs shortest path lengths for this maze (for fast distance lookups)."""
            if maze not in all_pairs_dist_cache:
                graph = maze_to_graph(maze)
                all_pairs_dist_cache[maze] = dict(nx.all_pairs_shortest_path_length(graph))
            return all_pairs_dist_cache[maze]

        def _classify_decode_direction(decode_hex, left_exit, right_exit, dist_dict):
            """Classify decode_hex as 'left', 'right', or 'ambiguous' based on graph distance."""
            # If decode_hex is not reachable (e.g., -100 sentinel), return "None"
            if decode_hex not in dist_dict:
                return "None"
            dist_left = dist_dict[decode_hex].get(left_exit, float("inf"))
            dist_right = dist_dict[decode_hex].get(right_exit, float("inf"))
            if dist_left < dist_right:
                return "left"
            elif dist_right < dist_left:
                return "right"
            else:
                return "ambiguous"

        for trial in trials:
            # Get trial time bounds
            trial_start, trial_end = (
                sgc.IntervalList
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "interval_list_name": trial["interval_list_name"],
                }
            ).fetch1("valid_times")[0]

            # Get maze configuration for this block
            maze = (
                HexMazeBlock()
                & {
                    "nwb_file_name": trial["nwb_file_name"],
                    "block": trial["block"],
                    "epoch": trial["epoch"],
                }
            ).fetch1("config_id")

            # Get per-maze data structures
            junction_hexes = _get_junction_set(maze)
            junction_lr = _get_junction_lr(maze)
            dist_dict = _get_all_pairs_dist(maze)

            # Filter to this trial's timepoints
            trial_df = annotated_df.loc[trial_start:trial_end]
            if len(trial_df) == 0:
                continue

            # Build the hex sequence for this trial by finding contiguous segments
            # where the rat stays at the same hex
            hex_series = trial_df["hex"]
            hex_changed = hex_series != hex_series.shift()
            segment_ids = hex_changed.cumsum()

            # Build a list of (hex_id, start_time, end_time) for each segment
            segments = []
            for seg_id, seg_df in trial_df.groupby(segment_ids):
                segments.append({
                    "hex": seg_df["hex"].iloc[0],
                    "start_time": seg_df.index[0],
                    "end_time": seg_df.index[-1],
                })

            # Walk through segments to find junction encounters
            # For each junction segment, we need the previous and next hex to determine
            # entry direction and actual exit
            for i, seg in enumerate(segments):
                seg_hex = seg["hex"]
                if seg_hex not in junction_hexes:
                    continue

                # Need a previous segment to determine entry direction
                if i == 0:
                    continue
                # Need a next segment to determine actual exit
                if i == len(segments) - 1:
                    continue

                entry_hex = segments[i - 1]["hex"]
                actual_exit = segments[i + 1]["hex"]

                # Verify entry_hex is a valid neighbor of this junction
                lr_key = (seg_hex, entry_hex)
                if lr_key not in junction_lr:
                    # Entry hex is not a direct neighbor (hex was skipped), skip
                    continue

                lr = junction_lr[lr_key]
                left_exit = lr["left"]
                right_exit = lr["right"]

                # Determine which direction the rat actually went
                if actual_exit == left_exit:
                    rat_direction = "left"
                elif actual_exit == right_exit:
                    rat_direction = "right"
                elif actual_exit == entry_hex:
                    rat_direction = "back"
                else:
                    # Exit is not one of the expected neighbors (hex skipped)
                    rat_direction = "None"

                # Get timepoint indices for this junction segment
                seg_idx = trial_df.loc[seg["start_time"]:seg["end_time"]].index

                # Annotate each timepoint at this junction
                annotated_df.loc[seg_idx, "nearest_junction"] = seg_hex
                annotated_df.loc[seg_idx, "hexes_from_junction"] = 0
                annotated_df.loc[seg_idx, "junction_entry_hex"] = entry_hex
                annotated_df.loc[seg_idx, "junction_left_exit"] = left_exit
                annotated_df.loc[seg_idx, "junction_right_exit"] = right_exit
                annotated_df.loc[seg_idx, "rat_junction_direction"] = rat_direction

                # Classify decode direction for each timepoint at the junction
                annotated_df.loc[seg_idx, "decode_junction_direction"] = [
                    _classify_decode_direction(dh, left_exit, right_exit, dist_dict)
                    for dh in annotated_df.loc[seg_idx, "decode_hex"]
                ]

                # Annotate approach segments BEFORE the junction (walking backwards)
                # Stop when we hit another junction or an already-annotated hex
                for j in range(i - 1, -1, -1):
                    prev_seg = segments[j]
                    prev_hex = prev_seg["hex"]
                    # Stop if we hit another junction — it owns its own timepoints
                    if prev_hex in junction_hexes:
                        break
                    # Compute graph distance from this hex to the junction
                    if prev_hex not in dist_dict or seg_hex not in dist_dict.get(prev_hex, {}):
                        break
                    dist_to_junc = dist_dict[prev_hex][seg_hex]

                    prev_idx = trial_df.loc[prev_seg["start_time"]:prev_seg["end_time"]].index
                    # Only annotate if not already claimed by a closer junction
                    current_vals = annotated_df.loc[prev_idx, "hexes_from_junction"]
                    if (current_vals != -100).any():
                        existing_dist = current_vals[current_vals != -100].abs().min()
                        if dist_to_junc >= existing_dist:
                            continue

                    # Negative distance = approaching the upcoming junction
                    annotated_df.loc[prev_idx, "nearest_junction"] = seg_hex
                    annotated_df.loc[prev_idx, "hexes_from_junction"] = -dist_to_junc
                    annotated_df.loc[prev_idx, "junction_entry_hex"] = entry_hex
                    annotated_df.loc[prev_idx, "junction_left_exit"] = left_exit
                    annotated_df.loc[prev_idx, "junction_right_exit"] = right_exit
                    annotated_df.loc[prev_idx, "rat_junction_direction"] = rat_direction
                    annotated_df.loc[prev_idx, "decode_junction_direction"] = [
                        _classify_decode_direction(dh, left_exit, right_exit, dist_dict)
                        for dh in annotated_df.loc[prev_idx, "decode_hex"]
                    ]

        # Save time as a column instead of index (NWB requires integer index)
        annotated_df = annotated_df.reset_index()

        # Write to NWB analysis file
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["junction_decode_object_id"] = builder.add_nwb_object(
                annotated_df, "junction_decode"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["junction_decode"].set_index("time")
