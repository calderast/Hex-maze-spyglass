import numpy as np
import datajoint as dj
from pynwb import NWBHDF5IO
from scipy.spatial import KDTree
import spyglass.common as sgc
from spyglass.common import Nwbfile, TaskEpoch, IntervalList, Session
from spyglass.utils.dj_mixin import SpyglassMixin

from hexmaze import get_maze_attributes

schema = dj.schema("hex_maze")

def populate_all_hexmaze(nwb_file_name):
    """Insert all hex maze related tables for a given NWB file"""
    
    # Populate the HexMazeBlock table, Trial part table, and HexMazeConfig table
    HexMazeBlock().load_from_nwb(nwb_file_name)
    # Populate the HexCentroids table
    HexCentroids.populate({'nwb_file_name': nwb_file_name})
    
    # Do hex assignment!!


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
    x: float            # the x coordinate of the hex centroid, in video pixel coordinates
    y: float            # the y coordinate of the hex centroid, in video pixel coordinates
    x_meters: float     # the x coordinate of the hex centroid, in meters
    y_meters: float     # the y coordinate of the hex centroid, in meters
    """

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
            in an x,y coordinate sense, just 'middle' needs to be the hex between them)
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
            centroids_data = behavior_module.data_interfaces["hex_centroids"].to_dataframe()

        centroids_to_insert = [
        {
            **key,
            'hex': np.int(row.hex),
            'x': row.x,
            'y': row.y,
            'x_meters': row.x_meters,
            'y_meters': row.y_meters,
        }
        for row in centroids_data.itertuples()
        ]

        self.insert(centroids_to_insert, skip_duplicates=True)

    def return_closest_hex(self, session_key, x, y):
        # Fetch the hex centroids for the given session
        centroids = (self & session_key).fetch(order_by='hex')
        hex_ids = np.array([entry['hex'] for entry in centroids])
        positions = np.array([[entry['x'], entry['y']] for entry in centroids])

        # Use KDTree to find the closest hex
        tree = KDTree(positions)
        _, idx = tree.query([x, y])
        
        # Return the hex ID corresponding to the closest position
        return int(hex_ids[idx])


@schema
class HexMazeTraversal(dj.Computed):
    """
    Stores each hex transition within a trial, including entry/exit times,
    surrounding hexes, and distance to/from ports.
    
    TODO: add fields choice_point, critical_choice_point,
    
    newly available and newly blocked as properties of a maze
    """

    definition = """
    -> HexMazeBlock.Trial
    hex_in_trial: int
    ---
    entry_in_trial: int         # numbered entry into this hex for this trial (1 = first entry)
    hex: int                    # the id of the hex
    entry_time: float           # the time the rat entered this hex
    exit_time: float            # the time the rat exited this hex
    duration: float             # the amount of time spent in this hex during this entry
    from_hex: int               # the id of the previous hex on the rat's path
    to_hex: int                 # the id of the next hex on the rat's path
    hexes_from_port: int        # the distance (in hexes) from the end port of this trial
    hexes_to_port: int          # the distance (in hexes) from the start port of this trial
    hex_type: varchar(20)       # optimal, non-optimal, or dead-end
    # inverse maybe?
    """

    def make(self, key):
        # Fetch trial info
        trial_info = (HexMazeBlock.Trial & key).fetch1()
        config_id = (HexMazeBlock & key).fetch1('config_id')

        start_port = trial_info['start_port']
        end_port = trial_info['end_port']

        # Get the raw position data from sgc RawPosition table
        # TODO change to posv1
        trial_positions = (sgc.RawPosition() 
                           & {"nwb_file_name": key['nwb_file_name'], 
                              "interval_list_name": trial_info['trial_interval']}
                          ).fetch1_dataframe()

        # # Get the trial start and end times
        # trial_interval = (IntervalList & {
        #     'nwb_file_name': key['nwb_file_name'],
        #     'interval_list_name': trial_info['trial_interval']
        # }).fetch1('valid_times')
        # trial_start, trial_end = trial_interval

        # # Filter position data for the trial time interval (maybe not needed bc we did already??)
        # trial_positions = raw_position_df[(raw_position_df.index >= trial_start) 
        #                                   & (raw_position_df.index <= trial_end)]

        # Get centroids
        #centroids = (HexCentroids & {'nwb_file_name': key['nwb_file_name']}).fetch(as_dict=True)
        #centroid_dict = {entry['hex']: np.array([entry['x'], entry['y']]) for entry in centroids}
        
        # Assign each position to a hex using HexCentroids method return_closest_hex
        assigned_hexes = []
        for _, pos in trial_positions[['x', 'y']].iterrows():
            hex_id = HexCentroids().return_closest_hex(
                session_key={'nwb_file_name': key['nwb_file_name'], 'config_id': config_id},
                x=pos['x'], y=pos['y']
            )
            assigned_hexes.append(hex_id)

        # Find transitions
        entries = []
        prev_hex = None
        for i in range(1, len(assigned_hexes)):
            if assigned_hexes[i] != assigned_hexes[i - 1]:
                exit_time = timestamps[i]
                entry_time = timestamps[i - 1]
                from_hex = assigned_hexes[i - 1]
                to_hex = assigned_hexes[i]
                
                # TODO: add a check for if the transition is valid.
                # If not, hopefully the hex position has only jumped over 1 hex
                # Assign a couple indices to the intermediate hex so we dont break things

                hex_id = from_hex
                hexes_from = calculate_hexes_from_port(hex_id, start_port, config_id)
                hexes_to = calculate_hexes_to_port(hex_id, end_port, config_id)

                entries.append({
                    **key,
                    'hex_index': len(entries),
                    'hex': hex_id,
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'duration': exit_time - entry_time,
                    'from_hex': prev_hex if prev_hex is not None else from_hex,
                    'to_hex': to_hex,
                    'hexes_from_port': hexes_from,
                    'hexes_to_port': hexes_to,
                })

                prev_hex = from_hex

        self.insert(entries)



# Future TODO: add an opto table


# hex ID, dead end?, hexes from port, hexes to port, optimal path?,  choice point?

# entry, exit,