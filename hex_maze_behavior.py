import sys
sys.path.append('hex_maze')

from pynwb import NWBHDF5IO
import datajoint as dj
import numpy as np
from spyglass.common import Nwbfile, TaskEpoch, IntervalList, Session
from spyglass.utils.dj_mixin import SpyglassMixin

from hex_maze.hex_maze_utils import get_maze_attributes

schema = dj.schema("hex_maze")

def populate_all_hexmaze(nwb_file_name):
    """Insert all hex maze related tables for a given NWB file"""
    
    # Populate the HexMazeBlock table, Trial part table, and HexMazeConfig table
    HexMazeBlock().load_from_nwb(nwb_file_name)
    # Populate the HexCentroids table
    HexCentroids.populate({'nwb_file_name': nwb_file_name})


@schema
class HexMazeConfig(SpyglassMixin, dj.Manual):
    """
    Contains data for each hex maze configuration, defined as the hexes where
    movable barriers are placed in the hex maze.
    
    TODO: add num_dead_ends: int      # number of dead ends at least 3 hexes deep
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
    """
    
    @staticmethod
    def set_to_string(set):
        """
        Converts a set of ints to a sorted, comma-separated string.
        Used for going from a set of barrier locations to a query-able config_id.
        """
        return ",".join(map(str, sorted(set)))
    
    @staticmethod
    def string_to_set(string):
        """
        Converts a sorted, comma-separated string to a set of ints.
        Used for going from a config_id to a set of barrier locations.
        """
        return set(map(int, string.split(",")))
    
    def insert_config(self, key):
        """
        Calculate secondary keys (maze attributes) based on the primary key (config_id)
        and add them to the HexMazeConfig table.
        """
        # Get config_id as a string
        config_id = key['config_id']
    
        # Calculate maze attributes for this maze
        # TODO: Update hex_maze functions to use our new naming conventions, add num_dead_ends to this function
        maze_attributes = get_maze_attributes(config_id)
        
        # Add maze attributes to key dict
        key.update({
            'len_ab': maze_attributes.get('len12'),
            'len_bc': maze_attributes.get('len23'),
            'len_ac': maze_attributes.get('len13'),
            'path_length_diff': maze_attributes.get('path_length_difference'),
            'num_choice_points': maze_attributes.get('num_choice_points'),
            'num_cycles': maze_attributes.get('num_cycles'),
            'choice_points': list(maze_attributes.get('choice_points'))
        })

        self.insert1(key, skip_duplicates=True)


@schema
class HexMazeBlock(SpyglassMixin, dj.Manual):
    """
    Contains data for each block in the Hex Maze task.
    Calling load_from_nwb to populate this table also
    populates the Trial part table and HexMazeConfig table.

    HexMazeBlock inherits primary keys nwb_file_name and epoch from TaskEpoch, 
    and inherits secondary key config_id from HexMazeConfig
    """

    definition = """
    -> TaskEpoch                    # gives nwb_file_name and epoch
    block: int                      # the block number within the epoch
    ---
    -> HexMazeConfig                # gives config_id
    p_a: float                      # probability of reward at port A
    p_b: float                      # probability of reward at port B
    p_c: float                      # probability of reward at port C
    num_trials: int                 # number of trials in this block
    block_interval: IntervalList    # [start_time, end_time] defining block bounds
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
        epoch_trial_num: int            # trial number within the epoch
        reward: bool                    # if the rat got a reward
        start_port: varchar(5)          # A, B, or C
        end_port: varchar(5)            # A, B, or C
        opto_cond=NULL: varchar(64)     # description of opto condition, if any (delay / no_delay)
        trial_interval: IntervalList    # [start_time, end_time] defining trial bounds
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
                IntervalList.insert1(
                    {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": f"epoch{block.epoch}_block{block.block}",
                        "valid_times": np.array([block.start_time, block.stop_time]),
                        "pipeline": "hex_maze"
                    }, skip_duplicates=True
                )

                # Add the block to the HexMazeBlock table
                block_key = {
                    'nwb_file_name': nwb_file_name,
                    'epoch': block.epoch,
                    'block': block.block,
                    'config_id': block.maze_configuration,
                    'p_a': block.pA,
                    'p_b': block.pB,
                    'p_c': block.pC,
                    'num_trials': block.num_trials,
                    'block_interval': f"epoch{block.epoch}_block{block.block}",
                    'task_type': block.task_type
                }
                self.insert1(block_key, skip_duplicates=True)

            # After populating the HexMazeBlock table, add each trial to the Trial part table
            trials_to_insert = []
            for trial in trial_data.itertuples():

                # Insert the trial interval into the IntervalList table
                IntervalList.insert1(
                    {
                        "nwb_file_name": nwb_file_name,
                        "interval_list_name": f"epoch{trial.epoch}_block{trial.block}_trial{trial.trial_within_block}",
                        "valid_times": np.array([trial.start_time, trial.stop_time]),
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
                    'reward': trial.reward,
                    'start_port': trial.start_port,
                    'end_port': trial.end_port,
                    'opto_cond': trial.opto_condition,
                    'trial_interval': f"epoch{trial.epoch}_block{trial.block}_trial{trial.trial_within_block}",
                    'poke_interval': np.array([trial.poke_in, trial.poke_out]),
                    'duration': trial.duration
                }
                trials_to_insert.append(trial_key)

            HexMazeBlock.Trial.insert(trials_to_insert, skip_duplicates=True)


@schema
class HexCentroids(dj.Imported):
    """
    Contains hex centroids for each session for the hex maze task in video pixel coordinates
    Used for assigning x, y position to a hex
    """

    definition = """
    -> Session           
    hex: int    # the hex ID in the hex maze (1-49)
    ---
    x: float    # the x coordinate of the x centroid, in video pixel coordinates
    y: float    # the y coordinate of the x centroid, in video pixel coordinates
    """

    def make(self, key):
        # Load hex centroids from the NWB file
        nwb_file_path = Nwbfile().get_abs_path(key["nwb_file_name"])
        with NWBHDF5IO(nwb_file_path, mode="r") as io:
            nwbfile = io.read()
            behavior_module = nwbfile.processing["behavior"]
            centroids_data = behavior_module.data_interfaces["hex_centroids"].to_dataframe()
        
        centroids_to_insert = [
        {
            **key,
            'hex': np.int(row.hex),
            'x': row.x,
            'y': row.y
        }
        for row in centroids_data.itertuples()
        ]

        self.insert(centroids_to_insert, skip_duplicates=True)

    def return_closest_hex(self, session_key, x, y):
        hex_ids = self & session_key


# @schema
# class HexMazePosition(SpyglassMixin, dj.Manual):
#     """
#     Contains data for each block in the Hex Maze task.
#     Calling load_from_nwb to populate this table automatically also
#     populates the Trial part table and HexMazeConfig table.

#     HexMazeBlock inherits primary keys nwb_file_name and epoch from TaskEpoch, 
#     and inherits secondary key config_id from HexMazeConfig
#     """

#     definition = """
#     -> TaskEpoch                    # gives nwb_file_name and epoch
#     block: int                      # the block number within the epoch
#     ---
#     -> HexMazeConfig                # gives config_id
#     p_a: float                      # probability of reward at port A
#     p_b: float                      # probability of reward at port B
#     p_c: float                      # probability of reward at port C
#     num_trials: int                 # number of trials in this block
#     block_interval: longblob        # np.array of [start_time, end_time]
#     task_type: varchar(64)          # 'barrier shift' or 'probabilty shift'
#     """
    


 
# Future TODO: add an opto table


# hex ID, dead end?, hexes from port, hexes to port, optimal path?,  choice point?

# entry, exit,