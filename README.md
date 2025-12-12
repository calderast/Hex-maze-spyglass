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
    placements and optimal path lengths between ports
- `HexMazeBlock` (Manual) - Blocks in the hex maze task, each with reward
    probabilities at ports A, B, C
- `HexMazeBlock.Trial` (Part) - Individual trials within each block,
    including start/end ports and trial outcomes
- `HexMazeChoice` (Computed) - Choice direction, reward probabilities, and path
    length differences for each trial
- `HexMazeTrialHistory` (Computed) - Trial history information for behavioral
    analysis
- `HexPositionSelection` (Manual) - Selection table linking position data to
    hex maze epochs
- `HexPosition` (Computed) - Processed position data aligned to hex centroids
- `HexPath` (Computed) - Trajectory paths through the hex maze

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

### Populator

`populate_all_hexmaze(nwb_file_name)`: Populate all basic hex maze tables for a
given NWB file. This populates:

- `HexMazeBlock` and `HexMazeBlock.Trial`
- `HexMazeChoice`
- `HexMazeTrialHistory`
- `HexCentroids`
- `HexMazeConfig`


