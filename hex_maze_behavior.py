import sys
sys.path.append('hex_maze')

from pynwb import NWBHDF5IO
import datajoint as dj
import numpy as np
from spyglass.common import Nwbfile
from spyglass.utils.dj_mixin import SpyglassMixin

from hex_maze.hex_maze_utils import get_maze_attributes

schema = dj.schema("hex_maze")

@schema
class HexMazeConfig(SpyglassMixin, dj.Computed):
    """
    populate config table each time we get a new maze config
    make secondary keys everything we want to look up by and 
    potentially filter trials by
    
    TODO: add num_dead_ends: int      # number of dead ends at least 3 hexes deep
    """

    definition = """
    config_id: varchar(64)  # maze configuration as a string
    ---
    lenAB: int              # number of hexes on optimal path between ports A and B
    lenBC: int              # number of hexes on optimal path between ports B and C
    lenAC: int              # number of hexes on optimal path between ports A and C
    path_length_diff: int   # max path length difference between lenAB, lenBC, lenAC
    num_choice_points: int  # number of critical choice points for this maze config
    num_cycles: int         # number of graph cycles (closed loops) for this maze config
    choice_points: longblob # set of hexes that are choice points (not query-able)
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
    
    def make(self, key):
        """
        Calculate secondary keys (maze attributes) based on the primary key (config_id)
        and add them to the HexMazeConfig table.
        """
        config_id = key['config_id']
        
        # Convert config_id to barrier set for compatability with hex_maze functions
        barrier_set = HexMazeConfig.string_to_set(config_id)
        
        # Calculate maze attributes for this maze
        # TODO: Update hex_maze functions to use our new naming conventions, add num_dead_ends to this function
        maze_attributes = get_maze_attributes(barrier_set)
        
        # Add maze attributes to key dict
        key.update({
            'lenAB': maze_attributes.get('len12'),
            'lenBC': maze_attributes.get('len23'),
            'lenAC': maze_attributes.get('len13'),
            'path_length_diff': maze_attributes.get('path_length_difference'),
            'num_choice_points': maze_attributes.get('num_choice_points'),
            'num_cycles': maze_attributes.get('num_cycles'),
            'choice_points': maze_attributes.get('choice_points')
        })

        self.insert1(key)


@schema
class HexMazeBlock(SpyglassMixin, dj.Imported):
    """
    Contains data for each block in the Hex Maze task.
    This is an imported table because block data is loaded from the nwbfile.
    
    HexMazeBlock inherits primary keys nwb_file_name and epoch from TaskEpoch, 
    and inherits secondary key config_id from HexMazeConfig
    """

    definition = """
    -> TaskEpoch                    # gives nwb_file_name and epoch
    block: int                      # the block number within the epoch
    ---
    -> HexMazeConfig                # gives config_id
    pA: float                       # probability of reward at port A
    pB: float                       # probability of reward at port B
    pC: float                       # probability of reward at port C
    num_trials: int                 # number of trials in this block
    block_interval: longblob        # np.array of [start_time, end_time]
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
        -> master                   # gives nwb_file_name, epoch, block
        block_trial_num: int        # trial number within the block
        ---
        epoch_trial_num: int        # trial number within the epoch
        reward: bool                # if the rat got a reward
        start_port: char            # A, B, or C
        end_port: char              # A, B, or C
        opto_cond=NULL: varchar(64) # description of opto condition, if any (delay / no_delay)
        trial_interval: longblob    # np.array of [start_time, end_time]
        poke_interval: longblob     # np.array of [poke_in, poke_out]
        duration: float             # trial duration in seconds
        """

    def insert_from_nwb(self, nwb_file_name):

        nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)
        
        with NWBHDF5IO(nwb_file_path, 'r') as io:
            nwbfile = io.read()
        
            # Get trial and block data from the nwb
            block_data = nwbfile.intervals["block"]
            trial_data = nwbfile.intervals["trials"]

            for block in block_data:
                # Add maze for this block to the HexMazeConfig table
                HexMazeConfig.populate({"config_id": block.get("maze_configuration")})
                
                # Add each block to the HexMazeBlock table
                block_key = {
                    'nwb_file_name': nwb_file_name,
                    'epoch': block.get("epoch"),
                    'block': block.get("block"),
                    'config_id': block.get("maze_configuration"),
                    'pA': block.get("pA"),
                    'pB': block.get("pB"),
                    'pC': block.get("pC"),
                    'num_trials': block.get("num_trials"),
                    'block_interval': np.array([block.get("start_time"), block.get("end_time")]),
                    'task_type': block.get("task_type")
                }
                self.insert1(block_key, skip_duplicates=True)
            
            # After populating the HexMazeBlock table, add trials
            for trial in trial_data:

                # Add each trial to the Trial part table
                trial_key = {
                    'nwb_file_name': nwb_file_name,
                    'epoch': trial.get("epoch"),
                    'block': trial.get("block"),
                    'block_trial_num': trial.get("trial_within_block"),
                    'epoch_trial_num': trial.get("trial_within_epoch"),
                    'reward': trial.get("reward"),
                    'start_port': trial.get("start_port"),
                    'end_port': trial.get("end_port"),
                    'opto_cond': trial.get("opto_cond"),
                    'trial_interval': np.array([trial.get("start_time"), trial.get("end_time")]),
                    'poke_interval': np.array([trial.get("poke_in"), trial.get("poke_out")]),
                    'duration':  trial.get("duration")
                }
                self.Trial.insert1(trial_key, skip_duplicates=True)

# Future TODO: add an opto table