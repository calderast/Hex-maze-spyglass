# Spyglass Hex Maze

Spyglass extension package for hex maze behavioral and neural analysis. This
package provides DataJoint tables and analysis tools for hex maze experiments
using the Spyglass neurophysiology data analysis framework.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/calderast/Hex-maze-spyglass.git
cd Hex-maze-spyglass
pip install -e .
```

Note that this package is not currently seamlessly compatible with the
Spyglass due to `hex-maze-neuro`'s pin of networkx>=3.3, which is Python 3.10
only. Spyglass depends on a handful of packages limited to Python 3.9.

## Usage

### Tables

This package provides four main modules:

#### Hex Maze Behavior (`hex_maze_behavior`)

- `HexMazeConfig` (Manual) - Hex maze configurations defining barrier
    placements and maze attributes (optimal path lengths between ports, etc)
- `HexMazeBlock` (Manual) - Blocks in the hex maze task, each with maze configuration and reward
    probabilities at ports A, B, C
- `HexMazeBlock.Trial` (Part) - Individual trials within each block,
    including start/end ports and trial outcomes
- `HexMazeChoice` (Computed) - Choice direction, reward probabilities, and path
    length differences for each trial
- `HexMazeTrialHistory` (Computed) - Trial history information for behavioral
    analysis
- `HexCentroids` (Imported) - Hex centroids for each session, used for assigning position to hex
- `HexPositionSelection` (Manual) - Selection table linking position data to hex centroids and
    hex maze epochs
- `HexPosition` (Computed) - Processed position data assigned to hex centroids
- `HexPath` (Computed) - Rat trajectories through the hex maze by trial, and associated hex-level path information

--> Helper class `HexMazeTrialContext` also takes a trial key and provides a number of functions to analyze the trial in context (such as past history of rewards, rewards at the given port, previous visits to the given port on the same vs alternate path, etc)

#### Decoding (`hex_maze_decoding`)

- `HexMazeDecodedPosition` (Computed) - Computes max likelihood x,y decoded position based on DecodingOutput
- `HexMazeDecodedPositionHex` (Computed) - Assigns hex to actual and decoded x,y position from HexMazeDecodedPosition
- `HexMazeDecodedPositionHexV2` (Computed) - Extension of `HexMazeDecodedPositionHex` that adds alternative hex
    assignments, so decodes landing on a barrier hex aren't snapped to the nearest currently-open hex.
    Adds assignments allowing any hex open at some point earlier in the epoch, and allowing all hexes regardless
    of whether they were ever open
- `HexMazeDecodedHexPath` (Computed) - Decoded trajectories through the hex maze. Stores each hex transition
    within a trial, including entry/exit times, maze component, and distance to/from ports

Stephanie's extra tables:
- `HexMazeDecodedPositionHexAnnotated` (Computed) - Same maze annotations as `HexMazeDecodedHexPath`
    (hex_type, maze_portion, hexes_from_start/end/choice, etc) but one row per timepoint instead of per hex
    segment, so you can filter by speed, spatial coverage, etc at each time point
- `HexMazeDecodedHexPathBarrierChange` (Computed) - Extension of `HexMazeDecodedHexPath` for barrier change
    sessions. Adds hex distance from the path divergence point and classifies each hex relative to the barrier
    change (old_path, new_path, before_divergence, after_convergence, or other)
- `HexMazeJunctionDecode` (Computed) - Adds per-timepoint junction context. For every 3-way junction, identifies
    when the rat passes through and classifies the entry direction, which exit is left vs right, which way the
    rat actually went, and which way the decode points. Lets us ask whether the decoder anticipates upcoming
    turns at any junction, not just the critical choice point

#### Theta (`hex_maze_theta`)

- `HexMazeThetaV1` (Computed) - Theta-band analytic signal, phase, and power computed from `LFPBandV1` and saved
    to a single analysis NWB file. Phase is in radians shifted to [0, 2π] (matching spyglass convention) and
    power is amplitude squared. Use `HexMazeThetaV1.setup_theta_pipeline()` to create the upstream
    LFP/LFPBand entries, then populate `LFPBandV1` before populating this table
- `HexMazeThetaReference` (Manual) - A named set of reference electrodes for a `HexMazeThetaV1` entry, so the
    electrode(s) you chose for a layer (e.g. corpus callosum) are saved and don't need to be re-selected.
    Averaging is done on the complex analytic signal, then phase and power
    are derived from that average

--> Add a set with `HexMazeThetaReference.add_reference(key, reference_name, electrode_ids)`, then fetch the
averaged signal later with `fetch1_reference_phase()` / `fetch1_reference_power()`. See the
`Select_Theta_Reference_Electrodes` notebook for how to choose the electrodes in the first place

#### Fiber Photometry (`berke_fiber_photometry`)

- `ExcitationSource` (Manual) - Excitation sources used for fiber photometry
- `Photodetector` (Manual) - Photodetectors used for fiber photometry
- `OpticalFiber` (Manual) - Optical fibers used for fiber photometry
- `Indicator` (Manual) - Fluorescent indicators (e.g. dLight, gACh4h)
- `IndicatorInjection` (Manual) - Maps an indicator to its titer, volume and injection coordinates
- `FiberPhotometrySeries` (Manual) - Stores series data from fiber photometry recordings

### Helpers

#### Spike sorting across labs (`spikesorting_helpers`)

Berke lab (IM-*) sessions use spikesorting **v1** but Frank lab (Lily, Nova, Toby, Luna, etc) sessions
use **v0**. These helpers handle that split so analysis code can be written for both:

- `fetch_good_units(nwb_file_name)` - Well-isolated units from whichever pipeline the session's
    lab uses, always with the same columns (`spike_times`, `unit_label`, `sort_group_id`,
    `peak_channel`). Curation id, sort groups, and sorter params are all looked up rather than
    hardcoded per subject. Note `peak_channel` is always NaN for v0 sessions because that pipeline never populates it
- `electrodes_with_units(units)` - Electrodes carrying a good unit(s), useful for marking on plots and seeing where we are in the brain. Empty for v0 sessions, since those have no peak channel
- `get_electrode_ids(nwb_file_name)` - Electrode IDs from a session's sort groups (so bad
    channels are already excluded). This is what the theta pipeline filters LFP on
- `fetch_electrode_geometry(nwb_file_name)` - Probe geometry plus a normalized `is_bad` column
- `get_raw_electrical_series(nwbf)` - The raw `ElectricalSeries` from an open NWB file, found by type since Berke lab names it "ElectricalSeries" and Frank lab names it "e-series"
- `sorting_version(nwb_file_name)` / `sort_group_table(nwb_file_name)` - Which pipeline a session belongs to (v0 or v1), and its matching `SortGroup` table

--> None of this applies to the theta tables: spyglass LFP is v1-only, so every session goes
through `LFPBandV1` regardless of lab. The v0/v1 split is a spike sorting only.

### Populators

`populate_all_hexmaze(nwb_file_name)`: Populate all basic hex maze tables for a given NWB file. This populates:

- `HexMazeBlock` and `HexMazeBlock.Trial`
- `HexMazeChoice`
- `HexMazeTrialHistory`
- `HexCentroids`
- `HexMazeConfig`

`populate_hex_position(nwb_file_name)`: Populate all position-based hex maze tables for a given NWB file (using all entries associated with the NWB file in `PositionOutput`). This populates:

- `HexPositionSelection`
- `HexPosition`
- `HexPath`

--> Additional method `populate_all_hex_position()` finds all valid `HexPositionSelection` keys (sessions that have HexMazeBlock, PositionOutput, and HexCentroids data) and and uses these to populate the `HexPositionSelection`, `HexPosition`, `HexPath` tables.

`populate_all_fiber_photometry(nwb_file_name)`: Populate all photometry-related tables for a given NWB file. This populates:

- `ExcitationSource`
- `Photodetector`
- `OpticalFiber`
- `Indicator`
- `IndicatorInjection`
- `FiberPhotometrySeries`

---------

### Notes

The `berke_fiber_photometry` schema is in progress and currently relies on an outdated version of `ndx-fiber-photometry==0.1.0` to maintain compatability with spyglass. In the future, each FiberPhotometrySeries will be linked to its associated metadata (ExcitationSource, etc). Photometry series imported from NWB files (currently all added to `FiberPhotometrySeries`) will instead either be added to `RawFiberPhotometrySeries` (raw data, to be processed in spyglass) or `ImportedFiberPhotometrySeries` (already processed). These will be unified in a merge table for downstream processing. This work is planned for fall 2026.
