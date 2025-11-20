import re
import numpy as np
import pandas as pd
import datajoint as dj
from pynwb import NWBHDF5IO
import spyglass.common as sgc
from spyglass.common import Nwbfile, TaskEpoch, IntervalList, Session, AnalysisNwbfile
from spyglass.position import PositionOutput
from spyglass.utils.dj_mixin import SpyglassMixin

from hexmaze import (
    get_maze_attributes, 
    get_critical_choice_points, 
    divide_into_thirds, 
    classify_maze_hexes, 
    get_hexes_from_port,
    get_choice_direction,
    get_path_length_difference,
)

schema = dj.schema("hex_maze")

def populate_all_hexmaze(nwb_file_name):
    """
    Populate basic hex maze tables for a given NWB file:
    HexMazeBlock, HexMazeBlock.Trial, HexMazeChoice, HexCentroids, HexMazeConfig
    """

    # Populate the HexMazeBlock table, Trial part table, HexMazeChoice, and HexMazeConfig table
    HexMazeBlock().load_from_nwb(nwb_file_name)
    # Populate the HexCentroids table
    HexCentroids.populate({'nwb_file_name': nwb_file_name})


def populate_all_hex_position():
    """
    Find all valid HexPositionSelection keys, insert them into 
    the HexPositionSelection table, and populate HexPosition.
    """

    # Get all valid keys that can be used to populate the HexPositionSelection table
    all_valid_keys = HexPositionSelection.get_all_valid_keys()

    # Insert each key into HexPositionSelection with renamed key field
    for key in all_valid_keys:
        selection_key = key.copy()
        selection_key["pos_merge_id"] = selection_key.pop("merge_id")

        # Skip inserting the key if it already exists in the table
        if selection_key in HexPositionSelection:
            continue
        try:
            HexPositionSelection.insert1(selection_key, skip_duplicates=True)
            print(f"Inserted new key {selection_key} into HexPositionSelection")
        except Exception as e:
            print(f"Skipping insert for {selection_key}: {e}")

    # Populate HexPosition table
    HexPosition.populate()


def populate_hex_position(nwb_file_name):
    """
    Populate the HexPositionSelection and HexPosition tables for a given nwb_file_name.
    """
    # Get all valid keys for the that HexPositionSelection table for this nwb
    all_valid_keys = HexPositionSelection.get_all_valid_keys(verbose=False)
    nwb_file_keys = [key for key in all_valid_keys if key["nwb_file_name"] == nwb_file_name]

    if not nwb_file_keys:
        print(f"No valid HexPositionSelection keys found for {nwb_file_name}")
        return

    # Insert each key into HexPositionSelection with renamed key field
    for key in nwb_file_keys:
        selection_key = key.copy()
        selection_key["pos_merge_id"] = selection_key.pop("merge_id")

        # Skip inserting the key if it already exists in the table
        if selection_key in HexPositionSelection:
            continue
        try:
            HexPositionSelection.insert1(selection_key, skip_duplicates=True)
            print(f"Inserted new key {selection_key} into HexPositionSelection")
        except Exception as e:
            print(f"Skipping insert for {selection_key}: {e}")

    # Only populate HexPosition with keys for this nwb
    selection_keys = (HexPositionSelection & {"nwb_file_name": nwb_file_name}).fetch("KEY")
    print(f"Populating HexPosition for {len(selection_keys)} entries in {nwb_file_name}")
    HexPosition.populate(selection_keys)


@schema
class HexMazeConfig(SpyglassMixin, dj.Manual):
    """
    Contains data for each hex maze configuration, defined as the hexes where
    movable barriers are placed in the hex maze.
    """

    definition = """
    config_id: varchar(64)  # maze configuration as a string
    ---
    len_ab: int             # number of hexes on optimal path between ports A and B
    len_bc: int             # number of hexes on optimal path between ports B and C
    len_ac: int             # number of hexes on optimal path between ports A and C
    path_length_diff: int   # max path length difference between lenAB, lenBC, lenAC
    num_choice_points: int  # number of critical choice points for this maze config
    num_cycles: int         # number of graph cycles (closed loops) for this maze config
    choice_points: blob     # list of hexes that are choice points (not query-able)
    num_dead_ends: int      # number of dead ends at least 3 hexes long
    optimal_pct: float      # percentage of maze hexes that are on optimal paths
    non_optimal_pct: float  # percentage of maze hexes that are on non-optimal paths
    dead_end_pct: float     # percentage of maze hexes that are on dead-end paths
    """

    def insert_config(self, key):
        """
        Calculate secondary keys (maze attributes) based on the primary key (config_id)
        and add them to the HexMazeConfig table.
        """
        # Get config_id as a string
        config_id = key['config_id']
    
        # Calculate maze attributes for this maze
        maze_attributes = get_maze_attributes(config_id)
        
        # Add maze attributes to key dict
        key.update({
            'len_ab': maze_attributes.get('len12'),
            'len_bc': maze_attributes.get('len23'),
            'len_ac': maze_attributes.get('len13'),
            'path_length_diff': maze_attributes.get('path_length_difference'),
            'num_choice_points': maze_attributes.get('num_choice_points'),
            'num_cycles': maze_attributes.get('num_cycles'),
            'choice_points': list(maze_attributes.get('choice_points')),
            'num_dead_ends': maze_attributes.get('num_dead_ends_min_length_3'),
            'optimal_pct': maze_attributes.get('optimal_pct'),
            'non_optimal_pct': maze_attributes.get('non_optimal_pct'),
            'dead_end_pct': maze_attributes.get('dead_end_pct'),
        })

        self.insert1(key, skip_duplicates=True)


@schema
class HexMazeBlock(SpyglassMixin, dj.Manual):
    """
    Contains data for each block in the Hex Maze task.
    Calling load_from_nwb to populate this table also populates the 
    Trial part table, the HexMazeConfig table, and the HexMazeChoice table.

    HexMazeBlock inherits primary keys nwb_file_name and epoch from TaskEpoch, 
    and inherits secondary keys config_id from HexMazeConfig 
    and interval_list_name from IntervalList
    """

    definition = """
    -> TaskEpoch                    # gives nwb_file_name and epoch
    block: int                      # the block number within the epoch
    ---
    -> HexMazeConfig                # gives config_id
    -> IntervalList                 # [start_time, end_time] defining block bounds
    p_a: float                      # probability of reward at port A
    p_b: float                      # probability of reward at port B
    p_c: float                      # probability of reward at port C
    num_trials: int                 # number of trials in this block
    task_type: varchar(64)          # 'barrier shift' or 'probabilty shift'
    """

    class Trial(SpyglassMixin, dj.Part):
        """
        Contains data for each trial within a block in the Hex Maze task.
        This is a part table based on HexMazeBlock.

        Trial inherits primary keys nwb_file_name and epoch from TaskEpoch, 
        and block from HexMazeBlock
        """

        definition = """
        -> master                       # gives nwb_file_name, epoch, block
        block_trial_num: int            # trial number within the block
        ---
        -> IntervalList                 # [start_time, end_time] defining trial bounds
        epoch_trial_num: int            # trial number within the epoch
        reward: bool                    # if the rat got a reward
        start_port: varchar(5)          # A, B, or C
        end_port: varchar(5)            # A, B, or C
        opto_cond=NULL: varchar(64)     # description of opto condition, if any (delay / no_delay)
        poke_interval: blob             # np.array of [poke_in, poke_out]
        duration: float                 # trial duration in seconds
        """

    def load_from_nwb(self, nwb_file_name):

        nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)

        with NWBHDF5IO(nwb_file_path, 'r') as io:
            nwbfile = io.read()

            # Get trial and block data from the nwb
            block_data = nwbfile.intervals["block"].to_dataframe()
            trial_data = nwbfile.intervals["trials"].to_dataframe()

            for block in block_data.itertuples():
                # Add maze for this block to the HexMazeConfig table
                HexMazeConfig().insert_config({"config_id": block.maze_configuration})

                # Add the block interval to the IntervalList table
                block_interval_list_name = f"epoch{block.epoch}_block{block.block}"
                IntervalList.insert1(
                    {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": block_interval_list_name,
                        "valid_times": np.array([[block.start_time, block.stop_time]]),
                        "pipeline": "hex_maze"
                    }, skip_duplicates=True
                )

                # Add the block to the HexMazeBlock table
                block_key = {
                    'nwb_file_name': nwb_file_name,
                    'epoch': block.epoch,
                    'block': block.block,
                    'config_id': block.maze_configuration,
                    'interval_list_name': block_interval_list_name,
                    'p_a': block.pA,
                    'p_b': block.pB,
                    'p_c': block.pC,
                    'num_trials': block.num_trials,
                    'task_type': block.task_type
                }
                self.insert1(block_key, skip_duplicates=True)

            # After populating the HexMazeBlock table, add each trial to the Trial part table
            trials_to_insert = []
            for trial in trial_data.itertuples():

                # Insert the trial interval into the IntervalList table
                trial_interval_list_name = f"epoch{trial.epoch}_block{trial.block}_trial{trial.trial_within_block}"
                IntervalList.insert1(
                    {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": trial_interval_list_name,
                        "valid_times": np.array([[trial.start_time, trial.stop_time]]),
                        "pipeline": "hex_maze"
                    }, skip_duplicates=True
                )

                # Add each trial to the Trial part table
                trial_key = {
                    'nwb_file_name': nwb_file_name,
                    'epoch': trial.epoch,
                    'block': trial.block,
                    'block_trial_num': trial.trial_within_block,
                    'epoch_trial_num': trial.trial_within_epoch,
                    'interval_list_name': trial_interval_list_name,
                    'reward': trial.reward,
                    'start_port': trial.start_port,
                    'end_port': trial.end_port,
                    'opto_cond': trial.opto_condition,
                    'poke_interval': np.array([trial.poke_in, trial.poke_out]),
                    'duration': trial.duration
                }
                trials_to_insert.append(trial_key)

            HexMazeBlock.Trial.insert(trials_to_insert, skip_duplicates=True)

            # Populate HexMazeChoice for all new trials
            HexMazeChoice.populate()

            # Populate HexMazeTrialHistory for all new trials
            HexMazeTrialHistory.populate()


    def join_with_trial(self):
        """
        Join a restricted HexMazeBlock query (self) with its Trial part table,
        renaming conflicting `interval_list_name` columns to `block_interval_list_name`
        and `trial_interval_list_name`.
        """
        hmb = self # HexMazeBlock
        hmbt = self.Trial # HexMazeBlock.Trial

        # Build projection that renames interval_list_name to block_interval_list_name and keeps everything else
        hmb_attrs_to_keep = [a for a in hmb.heading.names if a != 'interval_list_name']
        hmb_proj_dict = {**{a: a for a in hmb_attrs_to_keep}, 'block_interval_list_name': 'interval_list_name'}
        hmb = hmb.proj(**hmb_proj_dict)

        # Build projection that renames interval_list_name to trial_interval_list_name and keeps everything else
        hmbt_attrs_to_keep = [a for a in hmbt.heading.names if a != 'interval_list_name']
        hmbt_proj_dict = {**{a: a for a in hmbt_attrs_to_keep}, 'trial_interval_list_name': 'interval_list_name'}
        hmbt = hmbt.proj(**hmbt_proj_dict)

        return hmb * hmbt


@schema
class HexMazeChoice(dj.Computed):
    """
    Automatically computes choice direction, reward probability difference, 
    and path length difference for each entry in HexMazeBlock.Trial.
    The first trial of each epoch is excluded because there is no start_port.
    """
    definition = """
    -> HexMazeBlock.Trial
    ---
    choice_direction: varchar(16)   # 'left' or 'right'
    reward_prob_diff: float         # reward probability difference
    path_length_diff: float         # path length difference
    """

    @staticmethod
    def get_reward_prob_difference(trial_row) -> float:
        """
        Helper to compute the reward probability difference for a trial row.

        Parameters:
            trial_row (dict or DataFrame row): must contain keys:
                'p_a', 'p_b', 'p_c', 'start_port', 'end_port'

        Returns:
            float: p(chosen) - p(unchosen)
        """
        pa, pb, pc = trial_row['p_a'], trial_row['p_b'], trial_row['p_c']
        start_port = trial_row['start_port']
        end_port = trial_row['end_port']

        port_probs = {"A": pa, "B": pb, "C": pc}

        chosen_prob = port_probs[end_port]
        unchosen_port = ({'A', 'B', 'C'} - {start_port, end_port}).pop()
        unchosen_prob = port_probs[unchosen_port]

        return chosen_prob - unchosen_prob


    def make(self, key):
        # Fetch the block + trial row
        trial_row = (HexMazeBlock().join_with_trial() & key).fetch1()
        
        # Skip this trial if start_port is None (first trial of the epoch)
        if trial_row['start_port'] == 'None':
            return

        maze = trial_row['config_id']
        start_port = trial_row['start_port']
        end_port = trial_row['end_port']

        # Compute choice features
        choice_direction = get_choice_direction(start_port, end_port)
        reward_prob_diff= self.get_reward_prob_difference(trial_row)
        path_length_diff = get_path_length_difference(maze, start_port, end_port)

        # Insert features
        self.insert1({
            **key,
            'choice_direction': choice_direction,
            'reward_prob_diff': reward_prob_diff,
            'path_length_diff': path_length_diff
        })


class HexMazeTrialContext:
    """
    Context for analyzing a single trial's history within its epoch.
    Initialize with a trial key, automatically loads epoch history.
    """

    def __init__(self, trial_key):
        # Load the current trial
        self.trial = (HexMazeBlock.Trial & trial_key).fetch1()
        self.trial_num = self.trial["epoch_trial_num"]

        # Load all trials from this epoch
        epoch_key = {"nwb_file_name": trial_key["nwb_file_name"], "epoch": trial_key["epoch"]}
        self.epoch_trials = (HexMazeBlock.Trial & epoch_key).fetch(order_by="epoch_trial_num", as_dict=True)

    @property
    def history(self):
        """All trials before the current one in chronological order."""
        return self.epoch_trials[:self.trial_num - 1]

    def was_rewarded_n_trials_ago(self, trials_ago=1):
        """Was the trial N trials ago rewarded?"""
        idx = self.trial_num - 1 - trials_ago
        if 0 <= idx < len(self.history):
            return bool(self.history[idx]["reward"])
        return False

    def previous_visits_to_port(self):
        """All previous visits to the current trial's end port."""
        end_port = self.trial["end_port"]
        return [t for t in self.history if t["end_port"] == end_port]

    def previous_visits_same_path(self):
        """All previous trials with same start and end port as current trial."""
        start_port = self.trial["start_port"]
        end_port = self.trial["end_port"]
        return [t for t in self.history 
                if t["start_port"] == start_port and t["end_port"] == end_port]

    def last_visit_to_port_rewarded(self, visits_ago=1):
        """Was the Nth most recent visit to this port rewarded?"""
        visits = self.previous_visits_to_port()
        if len(visits) >= visits_ago:
            return bool(visits[-visits_ago]["reward"])
        return False

    def last_visit_to_port_same_path(self, visits_ago=1):
        """Was the Nth most recent visit to this port via the same path?"""
        port_visits = self.previous_visits_to_port()
        same_path_visits = self.previous_visits_same_path()
        
        if len(port_visits) < visits_ago:
            return False
        
        # Get the visit to this port N visits ago
        target_visit = port_visits[-visits_ago]
        # Check if this specific port visit is in the same_path list
        return target_visit in same_path_visits

    def reward_count_last_n_trials(self, n=5):
        """How many rewards in the last N trials (all ports)?"""
        recent_trials = self.history[-n:] if len(self.history) >= n else self.history
        return sum(t["reward"] for t in recent_trials)

    def reward_count_last_n_port_visits(self, n=5):
        """How many rewards in the last N visits to this port?"""
        port_visits = self.previous_visits_to_port()
        recent_visits = port_visits[-n:] if len(port_visits) >= n else port_visits
        return sum(t["reward"] for t in recent_visits)


@schema
class HexMazeTrialHistory(dj.Computed):
    """
    Automatically computes choice and reward history.
    """
    definition = """
    -> HexMazeBlock.Trial
    ---
    port_visit_1prev_rwd: bool              # if the previous visit to this port was rewarded
    port_visit_2prev_rwd: bool              # if the second-to-last visit to this port was rewarded
    port_visit_3prev_rwd: bool              # if the third-to-last visit to this port was rewarded
    port_visit_1prev_same_path: bool        # if the previous visit to this port was via the same path
    port_visit_2prev_same_path: bool        # if the second-to-last visit to this port was via the same path
    port_visit_3prev_same_path: bool        # if the third-to-last visit to this port was via the same path
    trial_1prev_rwd: bool                   # if the rat got a reward on the previous trial
    trial_2prev_rwd: bool                   # if the rat got a reward 2 trials ago
    trial_3prev_rwd: bool                   # if the rat got a reward 3 trials ago
    trial_4prev_rwd: bool                   # if the rat got a reward 4 trials ago
    trial_5prev_rwd: bool                   # if the rat got a reward 5 trials ago
    """

    def make(self, key):
        # Create HexMazeTrialContext from trial key
        trial_context = HexMazeTrialContext(key)

        # Create the HexMazeTrialHistory key for this trial
        self.insert1({
            **key,
            "port_visit_1prev_rwd": trial_context.last_visit_to_port_rewarded(visits_ago=1),
            "port_visit_2prev_rwd": trial_context.last_visit_to_port_rewarded(visits_ago=2),
            "port_visit_3prev_rwd": trial_context.last_visit_to_port_rewarded(visits_ago=3),
            "port_visit_1prev_same_path": trial_context.last_visit_to_port_same_path(visits_ago=1),
            "port_visit_2prev_same_path": trial_context.last_visit_to_port_same_path(visits_ago=2),
            "port_visit_3prev_same_path": trial_context.last_visit_to_port_same_path(visits_ago=3),
            "trial_1prev_rwd": trial_context.was_rewarded_n_trials_ago(trials_ago=1),
            "trial_2prev_rwd": trial_context.was_rewarded_n_trials_ago(trials_ago=2),
            "trial_3prev_rwd": trial_context.was_rewarded_n_trials_ago(trials_ago=3),
            "trial_4prev_rwd": trial_context.was_rewarded_n_trials_ago(trials_ago=4),
            "trial_5prev_rwd": trial_context.was_rewarded_n_trials_ago(trials_ago=5),
        })


@schema
class HexCentroids(dj.Imported):
    """
    Contains a table of hex centroids for each session in the hex maze task
    in video pixel coordinates and cm. The pixels to cm conversion is determined
    from a spatial series for this session in the RawPosition table.
    The session must exist in the HexMazeBlock table (populated via populate_all_hexmaze) 
    and the RawPosition table (populated via sgc.insert_session).
    """

    definition = """
    -> Session  
    ---
    """

    @classmethod
    def get_hex_centroids_dict_cm(cls, session_key):
        """
        Helper to return a dictionary mapping each hex ID to its (x_cm, y_cm) tuple.
        """
        hexes, x_cm, y_cm = (cls.HexCentroidsPart & session_key).fetch('hex', 'x_cm', 'y_cm')
        return {hex_id: (x, y) for hex_id, x, y in zip(hexes, x_cm, y_cm)}

    @classmethod
    def get_hex_centroids_dict_pixels(cls, session_key):
        """
        Helper to return a dictionary mapping each hex ID to its (x_pixels, y_pixels) tuple.
        """
        hexes, x_pixels, y_pixels = (cls.HexCentroidsPart & session_key).fetch('hex', 'x_pixels', 'y_pixels')
        return {hex_id: (x, y) for hex_id, x, y in zip(hexes, x_pixels, y_pixels)}

    @classmethod
    def get_core_hex_centroids_dict_cm(cls, session_key):
        """
        Helper to return a dictionary mapping each hex ID to its (x_cm, y_cm) tuple.
        Includes core hexes only (side hexes by reward ports are removed)
        """
        centroids_dict = cls.get_hex_centroids_dict_cm(session_key)
        # Remove side hex centroids and cast strings to ints
        centroids_dict = {int(k): v for k, v in centroids_dict.items() if "_left" not in k and "_right" not in k}
        return centroids_dict

    @classmethod
    def get_core_hex_centroids_dict_pixels(cls, session_key):
        """
        Helper to return a dictionary mapping each hex ID to its (x_pixels, y_pixels) tuple.
        Includes core hexes only (side hexes by reward ports are removed)
        """
        centroids_dict = cls.get_hex_centroids_dict_pixels(session_key)
        # Remove side hex centroids and cast strings to ints
        centroids_dict = {int(k): v for k, v in centroids_dict.items() if "_left" not in k and "_right" not in k}
        return centroids_dict


    class HexCentroidsPart(dj.Part):
        definition ="""
        -> master
        hex: varchar(10)    # the hex ID in the hex maze (1-49)
        ---
        x_pixels: float     # the x coordinate of the hex centroid, in video pixel coordinates
        y_pixels: float     # the y coordinate of the hex centroid, in video pixel coordinates
        x_cm: float         # the x coordinate of the hex centroid, in cm
        y_cm: float         # the y coordinate of the hex centroid, in cm
        """     

    @staticmethod
    def get_side_hex_centroids(hex_centroids):
        """
        Given a dict of hex centroids, calculate the centroids of the 6 side half-hexes
        near the reward ports (i.e. the sides to the left/right of hexes 4, 49, and 48)
        """
        def find_4th_hex_centroid_parallelogram(top_hex, middle_hex, bottom_hex):
            """ 
            Helper function used for finding centroids of the side half-hexes by reward ports.

            Given 3 (x,y) hex centroids top_hex, middle_hex, and bottom_hex, find the 
            4th hex centroid such that the 4 hexes are arranged in a parallelogram.

            For example, to find the centroid of the side hex to the left of hex 4
            (when facing the reward port), top_hex=1, middle_hex=4, bottom_hex=6.

            Note that 'top' and 'bottom' are relative and interchangeable - generally, I set
            the 'top' hex as one of the reward ports. (it doesn't have to be 'top' and 'bottom' 
            in an x,y coordinate sense, but 'middle' needs to be the hex between them)
            """
            other_middle_hex = np.array(top_hex) + (np.array(bottom_hex) - np.array(middle_hex))
            return tuple(other_middle_hex)

        # Calculate the centroids of the 6 side half-hexes next to the reward ports
        hex4left = find_4th_hex_centroid_parallelogram(hex_centroids[1], hex_centroids[4], hex_centroids[6])
        hex4right = find_4th_hex_centroid_parallelogram(hex_centroids[1], hex_centroids[4], hex_centroids[5])
        hex49left = find_4th_hex_centroid_parallelogram(hex_centroids[2], hex_centroids[49], hex_centroids[47])
        hex49right = find_4th_hex_centroid_parallelogram(hex_centroids[2], hex_centroids[49], hex_centroids[38])
        hex48left = find_4th_hex_centroid_parallelogram(hex_centroids[3], hex_centroids[48], hex_centroids[33])
        hex48right = find_4th_hex_centroid_parallelogram(hex_centroids[3], hex_centroids[48], hex_centroids[43])
        # Return a dict of side hex centroids
        return {"4_left": hex4left, "4_right": hex4right, "49_left": hex49left, "49_right": hex49right, 
                "48_left": hex48left, "48_right": hex48right}

    def make(self, key):
        # Load hex centroids from the NWB file
        nwb_file_path = Nwbfile().get_abs_path(key["nwb_file_name"])
        with NWBHDF5IO(nwb_file_path, mode="r") as io:
            nwbfile = io.read()
            behavior_module = nwbfile.processing["behavior"]
            centroids_df = behavior_module.data_interfaces["hex_centroids"].to_dataframe()
            centroids_dict = centroids_df.set_index('hex')[['x', 'y']].apply(tuple, axis=1).to_dict()

        # Get the number of the first run epoch from the HexMazeBlock table
        first_run_epoch = (HexMazeBlock() & {"nwb_file_name": key["nwb_file_name"]}).fetch('epoch')[0]
        interval_list_name = f"pos {first_run_epoch} valid times"

        # Get the raw position data for the first run epoch
        raw_position = (sgc.RawPosition.PosObject 
                        & {"nwb_file_name": key["nwb_file_name"], 'interval_list_name': interval_list_name})
        spatial_series = raw_position.fetch_nwb()[0]["raw_position"]

        # TODO descriptive error if these don't exist

        # Use the conversion factor from this spatial series (same for all run epochs in a session)
        conversion_factor = spatial_series.conversion  # {unit} per pixel
        conversion_unit = spatial_series.unit.lower()  # assumed to be meters or cm

        if conversion_unit == "meters":
            cm_per_pixel = conversion_factor * 100
        elif conversion_unit == "cm":
            cm_per_pixel = conversion_factor
        else:
            raise ValueError(f"Unexpected spatial series unit '{conversion_unit}'. Expected 'meters' or 'cm'.")

        # Insert the 49 hexes from the centroids table in the nwb
        centroids_to_insert = [
        {
            **key,
            'hex': str(int(row.hex)),
            'x_pixels': row.x,
            'y_pixels': row.y,
            'x_cm': row.x*cm_per_pixel,
            'y_cm': row.y*cm_per_pixel,
            # TODO: add a column for hex size based on distance to nearest neighbor?
        }
        for row in centroids_df.itertuples()
        ]

        # Insert the calculated centroids of the side half-hexes by reward ports
        side_hex_centroids = HexCentroids.get_side_hex_centroids(centroids_dict)
        side_hex_centroids_to_insert = [
        {
            **key,
            'hex': side_hex,
            'x_pixels': side_hex_centroids.get(side_hex)[0],
            'y_pixels': side_hex_centroids.get(side_hex)[1],
            'x_cm': side_hex_centroids.get(side_hex)[0]*cm_per_pixel,
            'y_cm': side_hex_centroids.get(side_hex)[1]*cm_per_pixel,
        }
        for side_hex in side_hex_centroids
        ]

        # Insert nwb_file_name into the HexCentroids table
        self.insert1(key) 
        # Insert the hex centroids into the HexCentroidsPart part table
        self.HexCentroidsPart.insert(centroids_to_insert, skip_duplicates=True)
        self.HexCentroidsPart.insert(side_hex_centroids_to_insert, skip_duplicates=True)


@schema
class HexPositionSelection(SpyglassMixin, dj.Manual):
    """
    Note we inherit from TaskEpoch instead of HexMazeBlock because we want
    nwb_file_name and epoch (but not block) as primary keys.
    The session must exist in the HexMazeBlock table (populated via populate_all_hexmaze).
    """

    definition = """
    -> PositionOutput.proj(pos_merge_id = "merge_id")
    -> TaskEpoch
    -> HexCentroids
    ---
    """

    @classmethod
    def get_all_valid_keys(cls, verbose=True):
        """
        Return a list of valid composite keys (nwb_file_name, epoch, merge_id) 
        for sessions that have HexMazeBlock, PositionOutput, and HexCentroids data.
        These keys can be used to populate the HexPositionSelection table.
        
        Use verbose=False to suppress print output.
        """
        all_valid_keys = []

        # Loop through all unique nwbfiles in the HexMazeBlock table
        for nwb_file_name in set(HexMazeBlock.fetch("nwb_file_name")):
            key = {"nwb_file_name": nwb_file_name}

            # Make sure an entry in HexCentroids exists for this nwbfile
            if not len(HexCentroids & {"nwb_file_name": nwb_file_name}):
                if verbose:
                    print(f"No HexCentroids entry found for nwbfile {nwb_file_name}, skipping.")
                continue

            # Loop through all unique epochs
            for epoch in set((HexMazeBlock & key).fetch("epoch")):
                position_output_key = {
                    "nwb_file_name": key["nwb_file_name"],
                    "interval_list_name": f"pos {epoch} valid times"
                }

                # Fetch the merge_ids for this nwb + epoch combination (if it exists in the PositionOutput table)
                try:
                    merge_ids = (PositionOutput.merge_get_part(position_output_key)).fetch("KEY")
                except ValueError as e:
                    if verbose:
                        print(f"No PositionOutput entry found for {position_output_key}, skipping.")
                    continue

                for merge_id in merge_ids:
                    composite_key = {
                        "nwb_file_name": nwb_file_name,
                        "epoch": epoch,
                        **merge_id
                    }
                    all_valid_keys.append(composite_key)
        return all_valid_keys


@schema
class HexPosition(SpyglassMixin, dj.Computed):
    definition = """
    -> HexPositionSelection
    ---
    -> AnalysisNwbfile
    hex_assignment_object_id: varchar(128)
    """

    def make(self, key):
        # Get a dict of hex: (x, y) centroid in cm for this nwbfile
        hex_centroids = HexCentroids.get_hex_centroids_dict_cm(key)

        # Get the rat's position for this epoch from the PositionOutput table
        pos_key = {"merge_id": key["pos_merge_id"]} # in case the key contains multiple 'merge_id'
        position_df = (PositionOutput & pos_key).fetch1_dataframe()

        # Set up a new df to store assigned hex info for each index in position_df
        # (We use -1 and "None" instead of nan to avoid HDF5 datatype issues)
        hex_df = pd.DataFrame({
            "hex": np.full(len(position_df), -1),
            "hex_including_sides": ["None"] * len(position_df),
            "distance_from_centroid": np.full(len(position_df), -1.0)
        }, index=position_df.index)
    
        # Loop through all blocks within this epoch
        for block in (HexMazeBlock & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]}):

            # Get the block start and end times
            block_start, block_end = (IntervalList & 
                        {'nwb_file_name': key['nwb_file_name'], 
                        'interval_list_name': block['interval_list_name']}
                        ).fetch1('valid_times')[0]

            # Filter position_df to only include times for this block
            block_mask = (position_df.index >= block_start) & (position_df.index <= block_end)
            block_positions = position_df.loc[block_mask]

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
            positions = block_positions[['position_x', 'position_y']].to_numpy()  # shape (n_positions, 2)
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


    def fetch_hex_and_position_dataframe(self, key=None):
        """
        Fetch a combined hex and position dataframe filtered to valid times.

        Works whether called as:
            HexPosition().fetch_hex_and_position_dataframe(key)
        or
            (HexPosition & key).fetch_hex_and_position_dataframe()

        Returns
        -------
        pd.DataFrame
            Combined position + hex dataframe filtered to valid block times.
        """

        # Allow usage with restricted table or explicit key
        entry = self if key is None else (self & key)
        key = entry.fetch1("KEY")

        # Get all blocks for this epoch so we can filter to only valid times
        blocks = (HexMazeBlock & key).fetch()

        if len(blocks) == 0:
            raise ValueError(f"No HexMazeBlock entries found for key: {key}")

        first_block_start, first_block_end = (IntervalList & {
            'nwb_file_name': key["nwb_file_name"],
            'interval_list_name': blocks[0]['interval_list_name']
        }).fetch1('valid_times')[0]

        last_block_start, last_block_end = (IntervalList & {
            'nwb_file_name': key["nwb_file_name"],
            'interval_list_name': blocks[-1]['interval_list_name']
        }).fetch1('valid_times')[0]

        # Get processed xy position from the PositionOutput table
        xy_position_df = (PositionOutput & {"merge_id": key["pos_merge_id"]}).fetch1_dataframe()

        # Get hex position from the HexPosition table
        hex_position_df = entry.fetch1_dataframe()

        # Combine x,y position with assigned hex position
        full_position_df = xy_position_df.join(hex_position_df, on='time')

        # Filter position data to only include times between first block start and last block end
        mask = (full_position_df.index >= first_block_start) & (full_position_df.index <= last_block_end)
        full_position_df  = full_position_df.loc[mask]

        return full_position_df


@schema
class HexPath(SpyglassMixin, dj.Computed):
    """
    Stores each hex transition within a trial, including entry/exit times,
    maze component, and distance to/from ports.
    """

    definition = """
    -> HexPosition
    ---
    -> AnalysisNwbfile
    hex_path_object_id: varchar(128)
    """

    def make(self, key):
        # Get hex position dataframe for this nwb+epoch
        hex_position_df = (HexPosition & key).fetch1_dataframe()

        # Get trials for this nwb+epoch
        trials = (HexMazeBlock().Trial() & {"nwb_file_name": key["nwb_file_name"], "epoch": key["epoch"]})
        
        # Accumulate per-trial dataframes
        all_hex_paths = []

        for trial in trials:
            # Get trial time bounds
            trial_start, trial_end = (IntervalList & {
                'nwb_file_name': trial["nwb_file_name"],
                'interval_list_name': trial['interval_list_name']
            }).fetch1('valid_times')[0]

            # Get maze configuration and attributes
            maze = (HexMazeBlock() & {
                'nwb_file_name': trial['nwb_file_name'], 
                'block': trial['block'],
                'epoch': trial['epoch']
            }).fetch1('config_id')
            critical_choice_points = get_critical_choice_points(maze)
            hex_type_dict = classify_maze_hexes(maze)

            # Filter position data to this trial
            trial_mask = (hex_position_df.index >= trial_start) & (hex_position_df.index <= trial_end)
            trial_position_df = hex_position_df.loc[trial_mask].copy()

            # Identify contiguous hex segments
            trial_position_df["segment"] = (trial_position_df["hex"] != trial_position_df["hex"].shift()).cumsum()

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
            if trial["start_port"] == "None": # first trial does not have a start port, so we just fill with -1
                hex_path["hexes_from_start"] = [-1] * len(hex_path)
            else:
                hex_path["hexes_from_start"] = [
                    get_hexes_from_port(maze, start_hex=h, reward_port=trial["start_port"])
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
                if group_name in {"optimal_hexes", "non_optimal_hexes", "dead_end_hexes"}
                for h in hexes
            }
            hex_path["hex_type"] = hex_path["hex"].map(hex_to_type)
            # Mark choice points
            hex_path["hex_type"] = hex_path.apply(
                lambda row: "choice_point" if row["hex"] in critical_choice_points else row["hex_type"],
                axis=1
            )

            # Map each hex to the section of the maze it's in (1, 2, or 3 for near port A, B, or C)
            thirds = divide_into_thirds(maze)
            hex_to_maze_third = {h: third_num for third_num, hexes in enumerate(thirds, start=1) for h in hexes}
            # Map choice points to section 0
            hex_to_maze_third.update({h: 0 for h in critical_choice_points})

            # Identify the maze sections as 'start', 'chosen', or 'unchosen'
            # Note that for the first trial, start_port is None so start_section and unchosen_section will both be None
            port_map = {"A": 1, "B": 2, "C": 3}
            start_section  = port_map.get(trial["start_port"])
            chosen_section = port_map.get(trial["end_port"])
            unchosen_section = {1, 2, 3} - {chosen_section} - {start_section}
            unchosen_section = unchosen_section.pop() if len(unchosen_section) == 1 else None

            # Map maze section number to its label 
            label = {start_section: "start", chosen_section: "chosen", unchosen_section: "unchosen", 0: "choice_point"}

            # Assign maze section label for each hex (if no label, e.g. first section of first trial, it will be "None")
            hex_path["maze_portion"] = hex_path["hex"].map(lambda h: label.get(hex_to_maze_third.get(h))).astype("str")

            # Add trial number and block columns
            hex_path["block_trial_num"] = trial["block_trial_num"]
            hex_path["block"] = trial["block"]
            # Put the trial and block columns on the left
            hex_path = hex_path[["block", "block_trial_num"] + 
                                [c for c in hex_path.columns if c not in {"block", "block_trial_num"}]]
            
            # Add the hex path for this trial
            all_hex_paths.append(hex_path)

        # Concatenate per-trial dataframes into one big dataframe
        hex_path_all_trials = pd.concat(all_hex_paths, ignore_index=True)

        # Create an empty AnalysisNwbfile with a link to the original nwb
        analysis_file_name = AnalysisNwbfile().create(key["nwb_file_name"])
        # Store the name of this newly created AnalysisNwbfile 
        key["analysis_file_name"] = analysis_file_name
        # Add the hex path dataframe to the AnalysisNwbfile 
        key["hex_path_object_id"] = AnalysisNwbfile().add_nwb_object(
            analysis_file_name, hex_path_all_trials, "hex_path"
        )
        # Create an entry in the AnalysisNwbfile table (like insert1)
        AnalysisNwbfile().add(key["nwb_file_name"], key["analysis_file_name"])
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
            (df["block"] == block) &
            (df["block_trial_num"] == block_trial_num)
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
