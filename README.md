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

This package provides three main modules:

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

- `DecodedPosition` (Computed) - Computes max likelihood x,y decoded position based on DecodingOutput
- `DecodedHexPositionSelection` (Manual) - Selection table linking decoded position data to hex centroids and
    hex maze epochs
- `DecodedHexPosition` (Computed) - Decoded position assigned to hex centroids
- `DecodedHexPath` (Computed) - Decoded trajectories through the hex maze

#### Fiber Photometry (`berke_fiber_photometry`)

- `ExcitationSource` (Manual) - Excitation sources used for fiber photometry
- `Photodetector` (Manual) - Photodetectors used for fiber photometry
- `OpticalFiber` (Manual) - Optical fibers used for fiber photometry
- `Indicator` (Manual) - Fluorescent indicators (e.g. dLight, gACh4h)
- `IndicatorInjection` (Manual) - Maps an indicator to its titer, volume and injection coordinates
- `FiberPhotometrySeries` (Manual) - Stores series data from fiber photometry recordings

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
The `berke_fiber_photometry` schema is in progress and currently relies on an outdated version of `ndx-fiber-photometry==0.1.0` to maintain compatability with spyglass. In the future, each FiberPhotometrySeries will be linked to its associated metadata (ExcitationSource, etc). Photometry series imported from NWB files (currently all added to `FiberPhotometrySeries`) will instead either be added to `RawFiberPhotometrySeries` (raw data, to be processed in spyglass) or `ImportedFiberPhotometrySeries` (already processed). These will be unified in a merge table for downstream processing. This work is planned for ~March 2026.