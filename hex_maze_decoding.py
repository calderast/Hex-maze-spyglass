import re
import numpy as np
import pandas as pd
import datajoint as dj
from spyglass.common import Nwbfile, TaskEpoch, IntervalList, Session, AnalysisNwbfile
from spyglass.decoding.decoding_merge import DecodingOutput
from spyglass.utils.dj_mixin import SpyglassMixin

from hex_maze_behavior import HexCentroids, HexMazeBlock

from non_local_detector.analysis import maximum_a_posteriori_estimate
from non_local_detector.model_checking import get_highest_posterior_threshold, get_HPD_spatial_coverage

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
        decode_key = {"merge_id": key["decoding_merge_id"]} # in case the key contains multiple 'merge_id'
        results = DecodingOutput.fetch_results(decode_key)

        # Get the posterior (probability of decode at each x,y location at each time point)
        # posterior has shape (n_time, n_x_bins, n_y_bins)
        posterior = results.acausal_posterior.unstack("state_bins").sum("state")
        
        #posterior = np.squeeze(posterior, axis=0) # prev
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
        
        #hpd_thresh = np.squeeze(hpd_thresh)
        hpd_thresh = hpd_thresh.squeeze()

        # posterior_stacked has shape (n_time, n_x_bins times n_y_bins)
        posterior_stacked = posterior.stack(position=["x_position", "y_position"])
        posterior_stacked = posterior_stacked.assign_coords(
            position=np.arange(posterior_stacked.position.size)
        )

        # spatial_cov has shape (n_time,)
        spatial_cov = get_HPD_spatial_coverage(posterior_stacked, hpd_thresh)

        # Make combined dataframe of decode info
        decoded_position_df = pd.DataFrame({
            "time": timestamps,
            "hpd_thresh": hpd_thresh,
            "spatial_cov": spatial_cov,
            "pred_x": max_likelihood_position[:, 0],
            "pred_y": max_likelihood_position[:, 1]
        })

        # Create an empty AnalysisNwbfile with a link to the original nwb
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        # Store the name of this newly created AnalysisNwbfile 
        key["analysis_file_name"] = analysis_file_name
        # Add the computed decoded position dataframe to the AnalysisNwbfile 
        key["decoded_position_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, decoded_position_df, "decoded_position_dataframe"
        )
        # Create an entry in the AnalysisNwbfile table (like insert1)
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
        self.insert1(key)

    def fetch1_dataframe(self):
        return self.fetch_nwb()[0]["decoded_position"].set_index('time')



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
                    print(f"No HexCentroids entry found for nwbfile {nwb_file_name}, skipping.")
                continue

            # Fetch the DecodedPosition merge_ids for this nwb (if it exists in the DecodedPosition table)
            merge_ids = (DecodedPosition & key).fetch("KEY")

            if not merge_ids:
                if verbose:
                    print(f"No DecodedPosition entry found for {nwb_file_name}, skipping.")
                continue

            # Loop through all unique merge_ids
            for merge_id in merge_ids:
                # Loop through all unique epochs
                for epoch in set((HexMazeBlock & key).fetch("epoch")):
                    composite_key = {
                        "nwb_file_name": nwb_file_name,
                        "epoch": epoch,
                        **merge_id
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
            "decoding_merge_id": key["decoding_merge_id"], # in case the key contains multiple 'merge_id'
            "nwb_file_name": key["nwb_file_name"]
        }
        decoded_position_df = (DecodedPosition & decoded_pos_key).fetch1_dataframe()

        # Set up a new df to store assigned hex info for each index in decoded_position_df
        # (We use -1 and "None" instead of nan to avoid HDF5 datatype issues)
        hex_df = pd.DataFrame({
            "hex": np.full(len(decoded_position_df), -1),
            "hex_including_sides": ["None"] * len(decoded_position_df),
            "distance_from_centroid": np.full(len(decoded_position_df), -1.0)
        }, index=decoded_position_df.index)

        # Loop through all blocks within this epoch
        for block in (HexMazeBlock & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}):

            # Get the block start and end times
            block_start, block_end = (IntervalList & 
                        {'nwb_file_name': key['nwb_file_name'], 
                        'interval_list_name': block['interval_list_name']}
                        ).fetch1('valid_times')[0]

            # Filter position_df to only include times for this block
            block_mask = (decoded_position_df.index >= block_start) & (decoded_position_df.index <= block_end)
            block_positions = decoded_position_df.loc[block_mask]

            # Get the hex maze config for this block
            maze_config = block.get('config_id')
            barrier_hexes = maze_config.split(',')

            # Remove the barrier hexes from our centroids dict
            block_hex_centroids = hex_centroids.copy()
            for hex_id in barrier_hexes:
                block_hex_centroids.pop(hex_id, None)

            # Convert hex_centroids to array for fast computation
            hex_ids = list(block_hex_centroids.keys())
            hex_coords = np.array(list(block_hex_centroids.values()))  # shape (n_hexes, 2)

            # Compute distances from each x, y position to each hex centroid
            positions = block_positions[['pred_x', 'pred_y']].to_numpy()  # shape (n_positions, 2)
            diffs = positions[:, np.newaxis, :] - hex_coords[np.newaxis, :, :]  # shape (n_positions, n_hexes, 2)
            dists = np.linalg.norm(diffs, axis=2)  # shape (n_positions, n_hexes)

            # Find the closest hex centroid for each x, y position
            closest_idx = np.argmin(dists, axis=1)
            closest_hex_incl_sides = [hex_ids[i] for i in closest_idx]

            # Calculate the distance from the centroid for each closest hex
            distance_from_centroid = np.min(dists, axis=1)

            # Closest_hex_incl_sides includes ids for the 6 side hexes next to the reward ports (e.g '4_left')
            # Closest_core_hex assigns the side hexes to their "core" hex (e.g. '4_left' and '4_right') become 4
            closest_core_hex = [int(re.match(r"\d+", hex_id).group()) for hex_id in closest_hex_incl_sides]

            # Add info for this block to hex_df
            hex_df.loc[block_positions.index, "hex"] = closest_core_hex
            hex_df.loc[block_positions.index, "hex_including_sides"] = closest_hex_incl_sides
            hex_df.loc[block_positions.index, "distance_from_centroid"] = distance_from_centroid

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
        return self.fetch_nwb()[0]["hex_assignment"].set_index('time')
