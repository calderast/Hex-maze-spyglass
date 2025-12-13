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

Helper class `HexMazeTrialContext` also takes a trial key and provides a number of helper functions to analyze the trial in context

#### Decoding (`hex_maze_decoding`)

- `DecodedPosition` (Computed) - Decoded position from neural activity using Bayesian decoding
- `DecodedHexPositionSelection` (Manual) - Selection table for hex-aligned decoded positions
- `DecodedHexPosition` (Computed) - Decoded position aligned to hex centroids
- `DecodedHexPath` (Computed) - Decoded trajectories through the hex maze

#### Fiber Photometry (`berke_fiber_photometry`)

- `ExcitationSource` (Manual) - Light sources used for fiber photometry
- `Photodetector` (Manual) - Detectors for fiber photometry signals
- `OpticalFiber` (Manual) - Optical fiber specifications
- `Indicator` (Manual) - Fluorescent indicators (e.g., GCaMP, dLight)
- `IndicatorInjection` (Manual) - Indicator injection details
- `FiberPhotometrySeries` (Manual) - Time series data from fiber photometry
    recordings

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

`populate_all_fiber_photometry(nwb_file_name)`: 
