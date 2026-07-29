run `Session_Inventory` to see all hex maze sessions and what processing we have done for them so far


run `Berke_Lab_full_process` to insert a Berke Lab session (this is step 0 for Berke Lab users)


run `Hex_Maze_Behavior_Tables` to populate hex tables (if you already ran Berke_Lab_full_process, this is a duplicate of that)


run `Berke_Lab_Sorting_and_Decode_V1` to do spikesorting and decoding for an inserted Berke Lab session


run `Hex_Maze_Decode_Tables` to insert a hex maze session with decoding output into hex maze specific decode tables


run `Berke_Sorting_Decode_Candidates` to see which sessions need spike sorting and decoding


run `Populate_All_Decoding` to just do the spike sorting and decoding for all eligible sessions


run `Hex_Maze_Theta` to populate hex maze theta tables for all eligible sessions


run `Select_Theta_Reference_Electrodes` to pick which electrodes to use as your theta reference for a session
(and save them to `HexMazeThetaReference`, so you can fetch averaged theta phase/power later without redoing
the selection). Run this after `Hex_Maze_Theta`


run `Berke_Decoding_Clusterless` to do clusterless decoding for an inserted Berke Lab session (adapted from the
spyglass tutorial series)


`Berke_Clusterless_Waveform_Features` is a scratch walkthrough of the clusterless pipeline on a single shank,
from sort groups through thresholding and waveform feature extraction. Useful for seeing the steps one at a
time, but `Berke_Decoding_Clusterless` is the one to actually run


`utils` has a helper function for picking sort groups


----

Many of these notebooks are still in development, but I hope they are helpful to you. If you're confused about anything just ask Steph