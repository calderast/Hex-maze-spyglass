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
from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1, LFPBandSelection
from spyglass.utils import SpyglassMixin, logger

from spyglass_hexmaze.hex_maze_behavior import HexCentroids, HexMazeBlock

try:
    from hexmaze import (
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
except ImportError:
    logger.error("required hexmaze functions could not be imported")
    (
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
    ) = (None,) * 14


schema = dj.schema("hex_maze_decoding")


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

    def make(self, key):
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

        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["decoded_position"].set_index("time")



def assign_position_to_hex(positions_xy, hex_centroids, maze):
    """
    Assign each (x, y) position to the nearest hex.
    
    Parameters:
        positions_xy: np.ndarray, shape (n_positions, 2)
        hex_centroids: dict mapping hex_id to (x, y) centroid (including side hexes)
        maze: hex maze config (to exclude hexes that are barriers or unreachable)

    Returns:
        core_hex (list[int]): Closest core hex (1-49) for each hex for each xy position
        hex_including_sides (list[str]): Closest hex (including side hexes) for each xy position
        distance_from_centroid (np.ndarray): Distance between each xy position and centroid of assigned hex
    """
    
    # Exclude hexes that are barriers or unreachable so we don't assign position to them
    exclude_hexes = {str(h) for h in maze_to_barrier_set(maze) | get_unreachable_hexes(maze)}
    centroids_to_use = {hex: coords for hex, coords in hex_centroids.items() if hex not in exclude_hexes}

    # Convert centroids to array for fast computation
    hex_ids = list(centroids_to_use.keys())
    hex_coords = np.array(list(centroids_to_use.values())) # shape (n_hexes, 2)

    # Compute distances from each x, y position to each hex centroid
    diffs = positions_xy[:, np.newaxis, :] - hex_coords[np.newaxis, :, :] # shape (n_positions, n_hexes, 2)
    dists = np.linalg.norm(diffs, axis=2) # shape (n_positions, n_hexes)

    # Find the closest hex centroid for each x, y position
    closest_idx = np.argmin(dists, axis=1)
    closest_hex_including_sides = [hex_ids[i] for i in closest_idx]
    
    # Calculate the distance from the centroid for each closest hex
    distance_from_centroid = np.min(dists, axis=1)
    
    # Closest_hex_including_sides includes ids for the 6 side hexes next to the reward ports (e.g '4_left')
    # Closest_core_hex assigns the side hexes to their "core" hex (e.g. '4_left' and '4_right') become 4
    closest_core_hex = [int(re.match(r"\d+", h).group()) for h in closest_hex_including_sides]

    # Return lists (same length as position_xy) of the closest hex for each position
    return closest_core_hex, closest_hex_including_sides, distance_from_centroid


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
            
            # Assign actual position to hex
            actual_xy = block_pos[["position_x", "position_y"]].to_numpy()
            actual_core_hex, hex_incl_sides, dist_from_centroid = assign_position_to_hex(actual_xy, hex_centroids, maze)
            hex_df.loc[block_pos.index, "hex"] = actual_core_hex
            hex_df.loc[block_pos.index, "hex_including_sides"] = hex_incl_sides
            hex_df.loc[block_pos.index, "distance_from_centroid"] = dist_from_centroid

            # Assign decoded position to hex
            decode_xy = block_pos[["decode_position_x", "decode_position_y"]].to_numpy()
            decode_core_hex, hex_incl_sides, dist_from_centroid = assign_position_to_hex(decode_xy, hex_centroids, maze)
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

        self.insert1(key)

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

        self.insert1(key)

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

        def _get_open_hexes(maze):
            """All reachable open hexes in the maze (1-49 minus barriers and unreachable)."""
            return set(range(1, 50)) - maze_to_barrier_set(maze) - get_unreachable_hexes(maze)

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
                    for h in _get_open_hexes(maze)
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
                    for h in _get_open_hexes(maze)
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

        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["annotated_hex"].set_index("time")


@schema
class HexMazeDecodedPositionAll(SpyglassMixin, dj.Computed):
    """
    Calculates most likely decoded position at each time point.
    Assigns actual and decoded position at each time point to the nearest maze hex.
    Stores combined dataframe of decoded and actual position, including assigned hex,
    decode confidence metrics (hpd threshold, spatial coverage of 95% confidence region),
    and distance between decoded and actual position.
    """

    definition = """
    -> DecodingOutput.proj(decoding_merge_id = "merge_id")
    -> TaskEpoch
    -> HexCentroids
    ---
    -> custom_AnalysisNwbfile
    decoded_position_all_object_id: varchar(128)
    """

    def make(self, key):
        decode_key = {"merge_id": key["decoding_merge_id"]}

        # Get decode results
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
        combined_df["decode_distance"] = analysis.get_ahead_behind_distance2D(
            combined_df[["position_x", "position_y"]].to_numpy(),
            combined_df[orientation_name].to_numpy(),
            combined_df[["decode_position_x", "decode_position_y"]].to_numpy(),
            classifier.environments[0].track_graph,
            classifier.environments[0].edges_,
        )

        ## Hex assignment
        # Get a dict of hex: (x, y) centroid in cm for this nwbfile
        hex_centroids = HexCentroids.get_hex_centroids_dict_cm(key)

        # Set up hex columns (use -100 and "None" instead of nan to avoid HDF5 datatype issues)
        for col in ["hex", "decode_hex"]:
            combined_df[col] = -100
        for col in ["hex_including_sides", "decode_hex_including_sides"]:
            combined_df[col] = "None"
        for col in ["distance_from_centroid", "decode_distance_from_centroid"]:
            combined_df[col] = -100.0
        combined_df["decode_hex_distance"] = -100

        # Loop through all blocks in this epoch
        for block in HexMazeBlock & {
            "nwb_file_name": key["nwb_file_name"],
            "epoch": key["epoch"],
        }:
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

            # Filter to only include times for this block
            block_pos = combined_df.loc[block_start:block_end]

            # Assign actual position to hex
            actual_xy = block_pos[["position_x", "position_y"]].to_numpy()
            actual_core_hex, incl_sides, dist = assign_position_to_hex(actual_xy, hex_centroids, maze)
            combined_df.loc[block_pos.index, "hex"] = actual_core_hex
            combined_df.loc[block_pos.index, "hex_including_sides"] = incl_sides
            combined_df.loc[block_pos.index, "distance_from_centroid"] = dist

            # Assign decoded position to hex
            decode_xy = block_pos[["decode_position_x", "decode_position_y"]].to_numpy()
            decode_core_hex, incl_sides, dist = assign_position_to_hex(decode_xy, hex_centroids, maze)
            combined_df.loc[block_pos.index, "decode_hex"] = decode_core_hex
            combined_df.loc[block_pos.index, "decode_hex_including_sides"] = incl_sides
            combined_df.loc[block_pos.index, "decode_distance_from_centroid"] = dist

            # Calculate hex distance between actual and decoded hex for each time point
            combined_df.loc[block_pos.index, "decode_hex_distance"] = [
                get_hex_distance(maze=maze, start_hex=a, target_hex=d)
                for a, d in zip(actual_core_hex, decode_core_hex)
            ]

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
            key["decoded_position_all_object_id"] = builder.add_nwb_object(
                combined_df, "decoded_position_all"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key)

    _drop_cols = [
        "hex_including_sides", "distance_from_centroid",
        "decode_hex_including_sides", "decode_distance_from_centroid",
    ]

    def fetch1_dataframe(self):
        return self.fetch1_dataframe_full().drop(columns=self._drop_cols)

    def fetch1_dataframe_full(self):
        return self.fetch_nwb()[0]["decoded_position_all"].set_index("time")


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

        self.insert1(key)

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

        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["junction_decode"].set_index("time")


@schema
class HexMazeThetaV1(SpyglassMixin, dj.Computed):
    """
    Computes theta-band analytic signal, phase, and power from LFPBandV1.
    Saves all three to a single analysis NWB file.

    The analytic signal is stored as real + imaginary float columns because
    HDF5/NWB can't store complex dtype directly. To reconstruct:
        z = df['electrode 5_real'] + 1j * df['electrode 5_imag']

    Phase is in radians, shifted to [0, 2π] (matching spyglass convention).
    Power is amplitude squared (|analytic signal|²).

    Dependencies (all must be populated before running make):
        LFPElectrodeGroup  →  LFPSelection  →  LFPV1
        LFPBandSelection  →  LFPBandV1

    Use HexMazeThetaV1.setup_theta_pipeline() to create all those entries,
    then call LFPBandV1().populate(lfp_band_key) before populating this table.

    Typical usage:
        # Step 1: set up the pipeline and populate broadband LFP (~2h locally)
        lfp_band_key = HexMazeThetaV1.setup_theta_pipeline(
            nwb_file_name="IM-1478_20220726_.nwb",
            lfp_electrode_group_name="my_lfp_group",
            electrode_ids=[0, 1, 2, ...],
            interval_list_name="00_r1",
        )
        # Step 2: populate theta-band LFP
        LFPBandV1().populate(lfp_band_key)
        # Step 3: compute and store analytic signal, phase, power
        HexMazeThetaV1().populate(lfp_band_key)
    """

    definition = """
    -> LFPBandV1
    ---
    -> custom_AnalysisNwbfile
    analytic_signal_object_id : varchar(128)   # real+imag parts of Hilbert transform
    theta_phase_object_id     : varchar(128)   # instantaneous phase [0, 2π] in radians
    theta_power_object_id     : varchar(128)   # instantaneous power (amplitude²)
    """

    def make(self, key):
        # Get the list of electrodes selected for this LFP band entry
        electrode_ids = sorted(
            (LFPBandSelection.LFPBandElectrode & key).fetch("electrode_id").tolist()
        )

        lfp_band_entry = LFPBandV1 & key

        # Compute analytic signal (complex), instantaneous phase, and power.
        # Each DataFrame has shape (n_timesteps × n_electrodes), time-indexed.
        # Column names are "electrode {electrode_id}" for each electrode.
        analytic_df = lfp_band_entry.compute_analytic_signal(electrode_list=electrode_ids)
        phase_df = lfp_band_entry.compute_signal_phase(electrode_list=electrode_ids)
        power_df = lfp_band_entry.compute_signal_power(electrode_list=electrode_ids)

        # Split complex analytic signal into real and imaginary float columns
        # so the DataFrame can be stored as HDF5 (NWB doesn't support complex dtype).
        # Use .values.real/.values.imag — pandas DataFrame has no .real/.imag directly.
        # Real columns: "electrode 5_real", imaginary: "electrode 5_imag"
        analytic_stored = pd.concat(
            [
                pd.DataFrame(
                    analytic_df.values.real,
                    index=analytic_df.index,
                    columns=[c + "_real" for c in analytic_df.columns],
                ),
                pd.DataFrame(
                    analytic_df.values.imag,
                    index=analytic_df.index,
                    columns=[c + "_imag" for c in analytic_df.columns],
                ),
            ],
            axis=1,
        ).reset_index()  # move time index to column (NWB requires integer index)

        phase_stored = phase_df.reset_index()
        power_stored = power_df.reset_index()

        # Save all three DataFrames to a single analysis NWB file
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["analytic_signal_object_id"] = builder.add_nwb_object(
                analytic_stored, "analytic_signal"
            )
            key["theta_phase_object_id"] = builder.add_nwb_object(
                phase_stored, "theta_phase"
            )
            key["theta_power_object_id"] = builder.add_nwb_object(
                power_stored, "theta_power"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key)

    def fetch1_analytic_signal(self) -> pd.DataFrame:
        """Return the analytic signal DataFrame, indexed by time.

        Columns are 'electrode N_real' and 'electrode N_imag' for each electrode N.
        To reconstruct complex signal: df['electrode 5_real'] + 1j * df['electrode 5_imag']
        """
        return self.fetch_nwb()[0]["analytic_signal"].set_index("time")

    def fetch1_theta_phase(self) -> pd.DataFrame:
        """Return instantaneous theta phase in radians [0, 2π], indexed by time.

        Columns are 'electrode N' for each electrode N.
        """
        return self.fetch_nwb()[0]["theta_phase"].set_index("time")

    def fetch1_theta_power(self) -> pd.DataFrame:
        """Return instantaneous theta power (amplitude²), indexed by time.

        Columns are 'electrode N' for each electrode N.
        """
        return self.fetch_nwb()[0]["theta_power"].set_index("time")

    def average_analytic_signal(self, electrode_list) -> pd.Series:
        """Average the complex analytic signal across a set of electrodes, indexed by time.

        Averaging the complex signal (not the phases) is the correct way to combine
        channels: in-phase theta adds, noise cancels. electrode_list entries can be
        ints or strings ("57").
        """
        analytic = self.fetch1_analytic_signal()
        z = sum(
            analytic[f"electrode {int(e)}_real"] + 1j * analytic[f"electrode {int(e)}_imag"]
            for e in electrode_list
        ) / len(electrode_list)
        z.name = "analytic_signal"
        return z

    def average_theta_phase(self, electrode_list) -> pd.Series:
        """Averaged theta phase (radians [0, 2π]) across a set of electrodes, indexed by time."""
        z = self.average_analytic_signal(electrode_list)
        return pd.Series(np.mod(np.angle(z), 2 * np.pi), index=z.index, name="theta_phase")

    def average_theta_power(self, electrode_list) -> pd.Series:
        """Averaged theta power across a set of electrodes, indexed by time.

        This is the mean of the per-channel power envelopes (average of powers), which is
        robust to small phase differences between channels. For the *coherent* power of the
        combined waveform instead (power of the average, which sags if channels drift out
        of phase) use np.abs(self.average_analytic_signal(electrode_list)) ** 2.
        """
        power_df = self.fetch1_theta_power()
        columns = [f"electrode {int(e)}" for e in electrode_list]
        return power_df[columns].mean(axis=1).rename("theta_power")

    def get_high_theta_intervals(
        self,
        threshold_percentile: float = 75.0,
        electrode=None,
        min_duration: float = 0.1,
    ) -> np.ndarray:
        """Return time intervals where theta power exceeds a percentile threshold.

        Parameters
        ----------
        threshold_percentile : float
            Percentile of the theta power distribution to use as the threshold.
            Default 75 = top quartile of theta power.
        electrode : str | list, optional
            Which signal to threshold:
            - a single electrode as 3, "3", or "electrode 3": that one electrode.
            - a list/set of electrode ids (ints or "57" strings): average the per-channel
              theta power across them — use this to threshold on a saved reference set
              (e.g. stratum radiatum).
            - None (default): average power across all electrodes.
        min_duration : float
            Minimum interval duration in seconds. Intervals shorter than this
            are discarded. Default 0.1 s.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Array of [start_time, end_time] rows in the Spyglass valid_times format.
            Returns an empty (0, 2) array if no intervals pass the threshold.

        Examples
        --------
        # Average a set of reference electrodes (e.g. a saved radiatum set), top 25%
        high_theta = (HexMazeThetaV1 & key).get_high_theta_intervals(electrode=[57, 58, 65])

        # Use a single electrode
        high_theta = (HexMazeThetaV1 & key).get_high_theta_intervals(electrode="electrode 57")

        # Filter a dataframe to high-theta times
        mask = np.zeros(len(df), dtype=bool)
        for start, end in high_theta:
            mask |= (df.index >= start) & (df.index <= end)
        df_high_theta = df[mask]
        """
        # Select the theta-power signal to threshold
        if isinstance(electrode, (list, tuple, set, np.ndarray)):
            # Average the per-channel theta power across a set of reference electrodes
            # and threshold that (robust to small inter-channel phase differences).
            power_series = self.average_theta_power(electrode)
            power = power_series.to_numpy(dtype=float)
            times = power_series.index.to_numpy()
        else:
            power_df = self.fetch1_theta_power()
            times = power_df.index.to_numpy()
            if electrode is not None:
                # Accept 57, "57", or "electrode 57" -- normalize to the column name
                column = f"electrode {int(str(electrode).split()[-1])}"
                power = power_df[column].to_numpy(dtype=float)
            else:
                # Average across all electrodes (not recommended for sessions with
                # many channels from mixed brain regions)
                power = power_df.mean(axis=1).to_numpy(dtype=float)

        # Threshold at the requested percentile
        threshold = float(np.nanpercentile(power, threshold_percentile))
        above = power >= threshold

        # Find contiguous runs where power is above threshold.
        # Pad with False at both ends so np.diff catches edges correctly.
        padded = np.concatenate([[False], above, [False]])
        diff = np.diff(padded.astype(int))
        # diff==1: False→True transition, index = start of run in `times`
        # diff==-1: True→False transition, index = exclusive end of run in `times`
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        intervals = []
        for s, e in zip(starts, ends):
            t_start = times[s]
            t_end = times[e - 1]  # last sample in the run
            if (t_end - t_start) >= min_duration:
                intervals.append([t_start, t_end])

        if not intervals:
            return np.empty((0, 2))
        return np.array(intervals)

    @classmethod
    def setup_theta_pipeline(
        cls,
        nwb_file_name: str,
        lfp_electrode_group_name: str,
        electrode_ids: list,
        interval_list_name: str,
        target_sampling_rate: int = 1000,
        lfp_band_sampling_rate: int = 1000,
        theta_filter_name: str = "Theta 5-11 Hz",
        theta_band_edges: list = None,
    ) -> dict:
        """Set up the full LFP → theta pipeline for a given NWB file.

        Creates all upstream entries needed before calling LFPBandV1.populate():
          1. LFP lowpass filter matched to the file's actual raw sampling rate
             (fetched from the Raw table — no hardcoding of session-specific rates)
          2. LFP electrode group (the set of channels to filter)
          3. LFPSelection entry (links electrode group + interval + filter)
          4. Populates LFPV1 — broadband LFP downsampled to ~target_sampling_rate Hz
             NOTE: This step takes ~2 hours locally for a full recording.
          5. Theta bandpass filter matched to the actual LFP output sampling rate
          6. LFPBandSelection entry (links LFP output + theta filter + interval)

        Parameters
        ----------
        nwb_file_name : str
            NWB file to process (must already be inserted in the database).
        lfp_electrode_group_name : str
            Name for the LFP electrode group — arbitrary but must be unique per file.
        electrode_ids : list[int]
            Electrode IDs (from the electrode table) to include in LFP processing.
        interval_list_name : str
            Name of the valid time interval to process (e.g. "00_r1" for Berke lab data).
        target_sampling_rate : int
            Desired output rate for the broadband LFP in Hz. Default 1000.
            The actual rate may differ slightly due to integer decimation
            (e.g. raw 29998 Hz with target 1000 → decimation=29 → actual 1034 Hz).
        lfp_band_sampling_rate : int
            Desired output rate for the theta-band LFP in Hz. Default 1000.
        theta_filter_name : str
            Name for the theta bandpass filter entry in FirFilterParameters.
        theta_band_edges : list[float], optional
            FIR band edges in Hz: [lo_pass, lo_stop, hi_stop, hi_pass].
            Defaults to [4, 5, 11, 12] (standard theta band).

        Returns
        -------
        dict
            LFPBandSelection key — pass this to LFPBandV1().populate() to run theta
            filtering, then to HexMazeThetaV1().populate() to compute and store results.
        """
        import spyglass.lfp as lfp

        if theta_band_edges is None:
            theta_band_edges = [4, 5, 11, 12]

        # Get the actual raw sampling rate for this recording from the database.
        # This varies per session (e.g. 29998 Hz for IM-1478 instead of the nominal 30000 Hz),
        # so we always fetch it rather than hardcoding.
        raw_sampling_rate = int(
            np.round((sgc.Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate"))
        )

        # The standard set of filters is designed for 30000 Hz data.
        # We copy the band edges from the standard LFP filter and re-design it at the
        # actual sampling rate, so cutoff frequencies are the same but filter coefficients differ.
        sgc.FirFilterParameters().create_standard_filters()
        standard_filter = sgc.FirFilterParameters() & {
            "filter_name": "LFP 0-400 Hz",
            "filter_sampling_rate": 30000,
        }
        lfp_band_edges = standard_filter.fetch1("filter_band_edges")

        lfp_filter_name = f"LFP 0-400 Hz {raw_sampling_rate}Hz"
        if not (
            sgc.FirFilterParameters
            & {"filter_name": lfp_filter_name, "filter_sampling_rate": raw_sampling_rate}
        ):
            sgc.FirFilterParameters().add_filter(
                filter_name=lfp_filter_name,
                fs=raw_sampling_rate,
                filter_type="lowpass",
                band_edges=lfp_band_edges,
                comments=f"Standard LFP 0-400 Hz filter adapted for {raw_sampling_rate} Hz data",
            )

        # Create the LFP electrode group (which channels to compute LFP for)
        lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
            nwb_file_name=nwb_file_name,
            group_name=lfp_electrode_group_name,
            electrode_list=electrode_ids,
            skip_duplicates=True,
        )

        # Insert LFPSelection: links electrode group + valid time interval + filter into one entry
        lfp_s_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
            "target_interval_list_name": interval_list_name,
            "filter_name": lfp_filter_name,
            "filter_sampling_rate": raw_sampling_rate,
            "target_sampling_rate": target_sampling_rate,
        }
        lfp.v1.LFPSelection.insert1(lfp_s_key, skip_duplicates=True)

        # Populate LFPV1: applies the LFP filter and downsamples to ~target_sampling_rate Hz.
        # NOTE: For full recordings this takes ~2 hours when run locally.
        lfp.v1.LFPV1().populate(lfp_s_key)

        # Get the LFP merge ID so we can reference this result downstream
        lfp_key = {
            "merge_id": (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1("merge_id")
        }

        # The actual LFP sampling rate after integer decimation
        # (may differ from target, e.g. 29998 → 1034 Hz with decimation=29)
        lfp_sampling_rate = int(
            lfp.LFPOutput.merge_get_parent(lfp_key).fetch1("lfp_sampling_rate")
        )

        # Create the theta bandpass filter matched to the actual LFP output sampling rate.
        # We can't reuse a filter entry from a different session if their LFP rates differ.
        if not (
            sgc.FirFilterParameters
            & {"filter_name": theta_filter_name, "filter_sampling_rate": lfp_sampling_rate}
        ):
            sgc.common_filter.FirFilterParameters().add_filter(
                theta_filter_name,
                lfp_sampling_rate,
                "bandpass",
                theta_band_edges,
                f"Theta 5-11 Hz bandpass filter for {lfp_sampling_rate} Hz LFP data",
            )

        # Insert LFPBandSelection: links LFP output + theta filter + valid time interval
        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=nwb_file_name,
            lfp_merge_id=lfp_key["merge_id"],
            electrode_list=electrode_ids,
            filter_name=theta_filter_name,
            interval_list_name=interval_list_name,
            reference_electrode_list=[-1],  # -1 means no reference electrode
            lfp_band_sampling_rate=lfp_band_sampling_rate,
        )

        # set_lfp_band_electrodes stores lfp_band_sampling_rate as
        # lfp_sampling_rate // decimation (integer floor division), not the
        # target we passed in. When the LFP rate isn't a clean multiple of the
        # target (e.g. 1034 Hz target 1000 → decimation=1 → stored rate 1034),
        # querying with the target value returns 0 rows.
        actual_lfp_band_rate = lfp_sampling_rate // (lfp_sampling_rate // lfp_band_sampling_rate)

        # Return the LFPBandSelection key.
        # Pass this to LFPBandV1().populate() then HexMazeThetaV1().populate().
        lfp_band_key = (
            LFPBandSelection
            & {
                "lfp_merge_id": lfp_key["merge_id"],
                "filter_name": theta_filter_name,
                "lfp_band_sampling_rate": actual_lfp_band_rate,
            }
        ).fetch1("KEY")

        return lfp_band_key


@schema
class HexMazeThetaReference(SpyglassMixin, dj.Manual):
    """A named set of reference electrodes for a HexMazeThetaV1 entry.

    Lets you save the electrodes you chose for a layer (e.g. stratum radiatum) so that
    later you can pull theta phase or power as the AVERAGE of those electrodes, without
    re-running the selection. Averaging is done on the complex analytic signal (the
    correct way to combine phases), then phase and power are derived from that average.

    Typical usage:
        # save a selection (electrode_ids can be ints or strings like "57")
        HexMazeThetaReference.add_reference(
            key={"nwb_file_name": "IM-1478_20220727_.nwb"},
            reference_name="radiatum",
            electrode_ids=[57, 58, 65],
        )

        # later, fetch the averaged theta phase / power for that set
        ref = HexMazeThetaReference & {"nwb_file_name": "IM-1478_20220727_.nwb",
                                       "reference_name": "radiatum"}
        phase = ref.fetch1_reference_phase()   # radians [0, 2π], indexed by time
        power = ref.fetch1_reference_power()    # amplitude², indexed by time
    """

    definition = """
    -> HexMazeThetaV1
    reference_name : varchar(64)    # label for this electrode set, e.g. "radiatum"
    ---
    electrode_ids  : blob           # list of electrode ids to average over
    description="" : varchar(255)   # optional note about how the set was chosen
    """

    @classmethod
    def add_reference(cls, key, reference_name, electrode_ids, description="", replace=False):
        """Save a set of reference electrodes for one HexMazeThetaV1 entry.

        Parameters
        ----------
        key : dict
            Restriction that uniquely identifies one HexMazeThetaV1 row
            (e.g. {"nwb_file_name": "IM-1478_20220727_.nwb"}).
        reference_name : str
            Label for this set (e.g. "radiatum"). Part of the primary key, so you can
            store several named sets per session.
        electrode_ids : list
            Electrode ids to average over. Ints or strings ("57") both work.
        description : str
            Optional note (e.g. "k=6 cluster 0, stratum radiatum").
        replace : bool
            If True, overwrite an existing set stored under the same name.
        """
        theta_key = (HexMazeThetaV1 & key).fetch1("KEY")
        cls.insert1(
            {
                **theta_key,
                "reference_name": reference_name,
                "electrode_ids": [int(e) for e in electrode_ids],
                "description": description,
            },
            replace=replace,
        )

    def fetch1_reference_analytic(self) -> pd.Series:
        """Complex analytic signal averaged across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_analytic_signal(self.fetch1("electrode_ids"))

    def fetch1_reference_phase(self) -> pd.Series:
        """Averaged theta phase (radians [0, 2π]) across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_theta_phase(self.fetch1("electrode_ids"))

    def fetch1_reference_power(self) -> pd.Series:
        """Averaged theta power (amplitude²) across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_theta_power(self.fetch1("electrode_ids"))
