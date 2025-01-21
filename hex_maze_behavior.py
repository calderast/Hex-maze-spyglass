import datajoint as dj
import spyglass.common as sgc
import numpy as np
import matplotlib.pyplot as plt
from sgc import TaskEpoch

schema = dj.schema("hex_maze")

@schema
class HexMazeConfig(dj.Computed):
    """
    populate config table each time we get a new maze config
    make secondary keys everything we want to look up by and 
    potentially filter trials by
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
    num_dead_ends: int      # number of dead ends at least 3 hexes deep
    choice_points: longblob # set of hexes that are choice points (not query-able)
    """
    
    def set_to_string(set):
        """
        Converts a set of ints to a sorted, comma-separated string.
        Used for going from a set of barrier locations to a query-able config_id.
        """
        return ", ".join(map(str, sorted(set)))
    
    def string_to_set(string):
        """
        Converts a sorted, comma-separated string to a set of ints.
        Used for going from a config_id to a set of barrier locations.
        """
        return set(map(int, string.split(", ")))
    
    def make(self, key):
        """
        Calculate secondary keys (maze attibutes) based on the primary key (config_id)
        and add them to the HexMazeConfig table.
        """
        # calculate secondary keys based on key
        
        # before calculating, key is a dict with only primary key in it
        config_id = key['config_id']
        
        # add them to "key" dict - keys are secondary key names
        self.insert1(key)
    
    
# to use:
# 1. make primary keys (using set to string) to populate
# 2. key is a dict of {config_id: string value}
# 3. call HexMazeConfig().populate(key) #key can be multiple

@schema
class HexMazeBlock(dj.Imported):
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
    opto_cond=NULL: varchar(64)     # description of opto condition, if any
    num_trials: int                 # number of trials in this block
    block_interval: longblob        # np.array of [start_time, end_time]
    task_type: varchar(64)          # barrier_shift or probabilty_shift
    """
    
    def insert_from_nwb(self,nwb_file_name):
        # load dataframes from NWB
        
        # get all config_ids
        
        # call HexMazeConfig.populate for config_ids (see above)
        
        # for each block in nwb:
        #     make block key (dict with all keys)
        #      self.insert1(block key)
        #       for each trial in block: same
         #           self.trial.insert1(trial key)
         # trial keys need all primary key info of block, 
        pass
        
        
    
    class Trial(dj.Part):
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

# TODO: add an opto table