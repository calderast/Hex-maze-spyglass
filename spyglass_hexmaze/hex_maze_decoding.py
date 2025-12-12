import re

import datajoint as dj
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from non_local_detector.analysis import maximum_a_posteriori_estimate
from non_local_detector.model_checking import (
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
)
from spyglass.common import Nwbfile, TaskEpoch, IntervalList, Session, AnalysisNwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile as custom_AnalysisNwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.utils import SpyglassMixin, logger

from spyglass_hexmaze.hex_maze_behavior import HexCentroids, HexMazeBlock

try:
    from hexmaze import (
        classify_maze_hexes,
        divide_into_thirds,
        get_critical_choice_points,
        get_hexes_from_port,
        plot_hex_maze,
    )
except ImportError:
    logger.error("required hexmaze functions could not be imported")
    (
        get_critical_choice_points,
        divide_into_thirds,
        classify_maze_hexes,
        get_hexes_from_port,
        plot_hex_maze,
    ) = (None,) * 5


schema = dj.schema("hex_maze_decoding")


@schema
class DecodedPosition(SpyglassMixin, dj.Computed):
    definition = """
    -> DecodingOutput.proj(decoding_merge_id = "merge_id")
    -> Session
    ---
    -> AnalysisNwbfile
    decoded_position_object_id: varchar(128)
    """

    def make(self, key):
        # Get decode results
        decode_key = {
            "merge_id": key["decoding_merge_id"]
        }  # in case the key contains multiple 'merge_id'
        results = DecodingOutput.fetch_results(decode_key)

        # Get the posterior (probability of decode at each x,y location at each time point)
        # posterior has shape (n_time, n_x_bins, n_y_bins)
        posterior = results.acausal_posterior.unstack("state_bins").sum("state")

        # posterior = np.squeeze(posterior, axis=0) # prev
        posterior = posterior.squeeze()

        # Get timestamps
        # timestamps have shape (n_time,)
        timestamps = posterior.time.values

        # Get the max likelihood x,y coordinate at each time point
        # max_likelihood_position has shape (n_time, 2)
        max_likelihood_position = maximum_a_posteriori_estimate(posterior)

        # Get the threshold to plug into get_HPD_spatial_coverage
        # hpd_thresh has shape (n_time,)
        hpd_thresh = get_highest_posterior_threshold(posterior, coverage=0.95)

        # hpd_thresh = np.squeeze(hpd_thresh)
        hpd_thresh = hpd_thresh.squeeze()

        # posterior_stacked has shape (n_time, n_x_bins times n_y_bins)
        posterior_stacked = posterior.stack(position=["x_position", "y_position"])
        posterior_stacked = posterior_stacked.assign_coords(
            position=np.arange(posterior_stacked.position.size)
        )

        # spatial_cov has shape (n_time,)
        spatial_cov = get_HPD_spatial_coverage(posterior_stacked, hpd_thresh)

        # Make combined dataframe of decode info
        decoded_position_df = pd.DataFrame(
            {
                "time": timestamps,
                "hpd_thresh": hpd_thresh,
                "spatial_cov": spatial_cov,
                "pred_x": max_likelihood_position[:, 0],
                "pred_y": max_likelihood_position[:, 1],
            }
        )

        # Create an empty AnalysisNwbfile with a link to the original nwb
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        # Store the name of this newly created AnalysisNwbfile
        key["analysis_file_name"] = analysis_file_name
        # Add the computed decoded position dataframe to the AnalysisNwbfile
        key["decoded_position_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name,
            decoded_position_df,
            "decoded_position_dataframe",
        )
        # Create an entry in the AnalysisNwbfile table (like insert1)
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["decoded_position"].set_index("time")


@schema
class DecodedHexPositionSelection(SpyglassMixin, dj.Manual):
    """
    Note we inherit from TaskEpoch instead of HexMazeBlock because we want
    nwb_file_name and epoch (but not block) as primary keys.
    The session must exist in the HexMazeBlock table (populated via populate_all_hexmaze).
    """

    definition = """
    -> DecodedPosition
    -> TaskEpoch
    -> HexCentroids
    ---
    """

    @classmethod
    def get_all_valid_keys(cls, verbose=True):
        """
        Return a list of valid composite keys (nwb_file_name, epoch, merge_id)
        for sessions that have HexMazeBlock, DecodedPosition, and HexCentroids data.
        These keys can be used to populate the DecodedHexPositionSelection table.

        Use verbose=False to suppress print output.
        """
        all_valid_keys = []

        # Loop through all unique nwbfiles in the HexMazeBlock table
        for nwb_file_name in set(HexMazeBlock.fetch("nwb_file_name")):
            key = {"nwb_file_name": nwb_file_name}

            # Make sure an entry in HexCentroids exists for this nwbfile
            if not len(HexCentroids & key):
                if verbose:
                    print(
                        f"No HexCentroids entry found for nwbfile {nwb_file_name}, skipping."
                    )
                continue

            # Fetch the DecodedPosition merge_ids for this nwb (if it exists in the DecodedPosition table)
            merge_ids = (DecodedPosition & key).fetch("KEY")

            if not merge_ids:
                if verbose:
                    print(
                        f"No DecodedPosition entry found for {nwb_file_name}, skipping."
                    )
                continue

            # Loop through all unique merge_ids
            for merge_id in merge_ids:
                # Loop through all unique epochs
                for epoch in set((HexMazeBlock & key).fetch("epoch")):
                    composite_key = {
                        "nwb_file_name": nwb_file_name,
                        "epoch": epoch,
                        **merge_id,
                    }
                    all_valid_keys.append(composite_key)
        return all_valid_keys


@schema
class DecodedHexPosition(SpyglassMixin, dj.Computed):
    definition = """
    -> DecodedHexPositionSelection
    ---
    -> AnalysisNwbfile
    hex_assignment_object_id: varchar(128)
    """

    def make(self, key):
        # Get a dict of hex: (x, y) centroid in cm for this nwbfile
        hex_centroids = HexCentroids.get_hex_centroids_dict_cm(key)

        # Get the rat's position for this epoch from the DecodedPosition table
        decoded_pos_key = {
            "decoding_merge_id": key[
                "decoding_merge_id"
            ],  # in case the key contains multiple 'merge_id'
            "nwb_file_name": key["nwb_file_name"],
        }
        decoded_position_df = (DecodedPosition & decoded_pos_key).fetch1_dataframe()

        # Set up a new df to store assigned hex info for each index in decoded_position_df
        # (We use -1 and "None" instead of nan to avoid HDF5 datatype issues)
        hex_df = pd.DataFrame(
            {
                "hex": np.full(len(decoded_position_df), -1),
                "hex_including_sides": ["None"] * len(decoded_position_df),
                "distance_from_centroid": np.full(len(decoded_position_df), -1.0),
            },
            index=decoded_position_df.index,
        )

        # Loop through all blocks within this epoch
        for block in HexMazeBlock & {
            "nwb_file_name": key["nwb_file_name"],
            "epoch": key["epoch"],
        }:

            # Get the block start and end times
            block_start, block_end = (
                IntervalList
                & {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": block["interval_list_name"],
                }
            ).fetch1("valid_times")[0]

            # Filter position_df to only include times for this block
            block_mask = (decoded_position_df.index >= block_start) & (
                decoded_position_df.index <= block_end
            )
            block_positions = decoded_position_df.loc[block_mask]

            # Get the hex maze config for this block
            maze_config = block.get("config_id")
            barrier_hexes = maze_config.split(",")

            # Remove the barrier hexes from our centroids dict
            block_hex_centroids = hex_centroids.copy()
            for hex_id in barrier_hexes:
                block_hex_centroids.pop(hex_id, None)

            # Convert hex_centroids to array for fast computation
            hex_ids = list(block_hex_centroids.keys())
            hex_coords = np.array(
                list(block_hex_centroids.values())
            )  # shape (n_hexes, 2)

            # Compute distances from each x, y position to each hex centroid
            positions = block_positions[
                ["pred_x", "pred_y"]
            ].to_numpy()  # shape (n_positions, 2)
            diffs = (
                positions[:, np.newaxis, :] - hex_coords[np.newaxis, :, :]
            )  # shape (n_positions, n_hexes, 2)
            dists = np.linalg.norm(diffs, axis=2)  # shape (n_positions, n_hexes)

            # Find the closest hex centroid for each x, y position
            closest_idx = np.argmin(dists, axis=1)
            closest_hex_incl_sides = [hex_ids[i] for i in closest_idx]

            # Calculate the distance from the centroid for each closest hex
            distance_from_centroid = np.min(dists, axis=1)

            # Closest_hex_incl_sides includes ids for the 6 side hexes next to the reward ports (e.g '4_left')
            # Closest_core_hex assigns the side hexes to their "core" hex (e.g. '4_left' and '4_right') become 4
            closest_core_hex = [
                int(re.match(r"\d+", hex_id).group())
                for hex_id in closest_hex_incl_sides
            ]

            # Add info for this block to hex_df
            hex_df.loc[block_positions.index, "hex"] = closest_core_hex
            hex_df.loc[block_positions.index, "hex_including_sides"] = (
                closest_hex_incl_sides
            )
            hex_df.loc[block_positions.index, "distance_from_centroid"] = (
                distance_from_centroid
            )

        # Save time as a column instead so we don't have float indices
        hex_df["time"] = hex_df.index
        hex_df = hex_df.reset_index(drop=True)

        # Create an empty AnalysisNwbfile with a link to the original nwb
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        # Store the name of this newly created AnalysisNwbfile
        key["analysis_file_name"] = analysis_file_name
        # Add the computed hex dataframe to the AnalysisNwbfile
        key["hex_assignment_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, hex_df, "hex_dataframe"
        )
        # Create an entry in the AnalysisNwbfile table (like insert1)
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["hex_assignment"].set_index("time")

    def fetch_hex_and_position_dataframe(self, key=None):
        """
        Fetch a combined hex and decoded position dataframe filtered to valid times.

        Works whether called as:
            DecodedHexPosition().fetch_hex_and_position_dataframe(key)
        or
            (DecodedHexPosition & key).fetch_hex_and_position_dataframe()

        Returns
        -------
        pd.DataFrame
            Combined decoded position + hex dataframe filtered to valid block times.
        """

        # Allow usage with restricted table or explicit key
        entry = self if key is None else (self & key)
        key = entry.fetch1("KEY")

        # Get all blocks for this epoch so we can filter to only valid times
        blocks = (HexMazeBlock & key).fetch()

        if len(blocks) == 0:
            raise ValueError(f"No HexMazeBlock entries found for key: {key}")

        first_block_start, first_block_end = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": blocks[0]["interval_list_name"],
            }
        ).fetch1("valid_times")[0]

        last_block_start, last_block_end = (
            IntervalList
            & {
                "nwb_file_name": key["nwb_file_name"],
                "interval_list_name": blocks[-1]["interval_list_name"],
            }
        ).fetch1("valid_times")[0]

        # Get decoded xy position from the DecodedPosition table
        xy_position_df = (DecodedPosition & key).fetch1_dataframe()

        # Get hex position from the HexPosition table
        hex_position_df = entry.fetch1_dataframe()

        # Combine x,y position with assigned hex position
        full_position_df = xy_position_df.join(hex_position_df, on="time")

        # Filter position data to only include times between first block start and last block end
        mask = (full_position_df.index >= first_block_start) & (
            full_position_df.index <= last_block_end
        )
        full_position_df = full_position_df.loc[mask]

        return full_position_df


@schema
class DecodedHexPath(SpyglassMixin, dj.Computed):
    """
    Stores each hex transition within a trial, including entry/exit times,
    maze component, and distance to/from ports.
    """

    definition = """
    -> DecodedHexPosition
    ---
    -> custom_AnalysisNwbfile
    hex_path_object_id: varchar(128)
    """

    _nwb_table = custom_AnalysisNwbfile() 

    def make(self, key):
        # Get hex position dataframe for this nwb+epoch
        hex_position_df = (DecodedHexPosition & key).fetch1_dataframe()

        # Get trials for this nwb+epoch
        trials = HexMazeBlock().Trial() & {
            "nwb_file_name": key["nwb_file_name"],
            "epoch": key["epoch"],
        }

        # Accumulate per-trial dataframes
        all_hex_paths = []

        for trial in trials:
            # Get trial time bounds
            trial_start, trial_end = (
                IntervalList
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
            critical_choice_points = get_critical_choice_points(maze)
            hex_type_dict = classify_maze_hexes(maze)

            # Filter decoded position data to this trial
            trial_mask = (hex_position_df.index >= trial_start) & (
                hex_position_df.index <= trial_end
            )
            trial_position_df = hex_position_df.loc[trial_mask].copy()

            # Identify contiguous hex segments
            trial_position_df["segment"] = (
                trial_position_df["hex"] != trial_position_df["hex"].shift()
            ).cumsum()

            # Set up dataframe of hex entries for this trial
            hex_path = (
                trial_position_df.groupby("segment")
                .agg(
                    hex=("hex", "first"),
                    entry_time=("hex", lambda x: x.index[0]),
                    exit_time=("hex", lambda x: x.index[-1]),
                )
                .reset_index(drop=True)
            )

            # Time spent in each hex
            hex_path["duration"] = hex_path["exit_time"] - hex_path["entry_time"]

            # What number hex entry in the trial this is
            hex_path["hex_in_trial"] = range(1, len(hex_path) + 1)

            # Count the number of times the rat has entered this specific hex in this trial
            hex_path["hex_entry_num"] = hex_path.groupby("hex").cumcount() + 1

            # For each hex, compute distances to start and end port
            if (
                trial["start_port"] == "None"
            ):  # first trial does not have a start port, so we just fill with -1
                hex_path["hexes_from_start"] = [-1] * len(hex_path)
            else:
                hex_path["hexes_from_start"] = [
                    get_hexes_from_port(
                        maze, start_hex=h, reward_port=trial["start_port"]
                    )
                    for h in hex_path["hex"]
                ]
            hex_path["hexes_from_end"] = [
                get_hexes_from_port(maze, start_hex=h, reward_port=trial["end_port"])
                for h in hex_path["hex"]
            ]

            # Classify each hex as optimal, non-optimal, or dead-end
            hex_to_type = {
                h: group_name.replace("_hexes", "")
                for group_name, hexes in hex_type_dict.items()
                if group_name
                in {"optimal_hexes", "non_optimal_hexes", "dead_end_hexes"}
                for h in hexes
            }
            hex_path["hex_type"] = hex_path["hex"].map(hex_to_type)

            # Map each hex to the section of the maze it's in (1, 2, or 3 for near port A, B, or C)
            thirds = divide_into_thirds(maze)
            hex_to_maze_third = {
                h: third_num
                for third_num, hexes in enumerate(thirds, start=1)
                for h in hexes
            }
            # Map choice points to section 0
            hex_to_maze_third.update({h: 0 for h in critical_choice_points})

            # Identify the maze sections as 'start', 'chosen', or 'unchosen'
            # Note that for the first trial, start_port is None so start_section and unchosen_section will both be None
            port_map = {"A": 1, "B": 2, "C": 3}
            start_section = port_map.get(trial["start_port"])
            chosen_section = port_map.get(trial["end_port"])
            unchosen_section = {1, 2, 3} - {chosen_section} - {start_section}
            unchosen_section = (
                unchosen_section.pop() if len(unchosen_section) == 1 else None
            )

            # Map maze section number to its label
            label = {
                start_section: "start",
                chosen_section: "chosen",
                unchosen_section: "unchosen",
                0: "choice_point",
            }

            # Assign maze section label for each hex (if no label, e.g. first section of first trial, it will be "None")
            hex_path["maze_portion"] = (
                hex_path["hex"]
                .map(lambda h: label.get(hex_to_maze_third.get(h)))
                .astype("str")
            )

            # Add block/trial key columns for ease of combination later
            hex_path["nwb_file_name"] = trial["nwb_file_name"]
            hex_path["epoch"] = trial["epoch"]
            hex_path["block"] = trial["block"]
            hex_path["block_trial_num"] = trial["block_trial_num"]
            hex_path["epoch_trial_num"] = trial["epoch_trial_num"]
            # Put the key columns on the left
            hex_path = hex_path[
                [
                    "nwb_file_name",
                    "epoch",
                    "block",
                    "block_trial_num",
                    "epoch_trial_num",
                ]
                + [
                    c
                    for c in hex_path.columns
                    if c
                    not in {
                        "nwb_file_name",
                        "epoch",
                        "block",
                        "block_trial_num",
                        "epoch_trial_num",
                    }
                ]
            ]

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
        # If show_stats, add path lengths and reward probabilities
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
