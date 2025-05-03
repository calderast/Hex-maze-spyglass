Note that this repo uses functions from `hex-maze-neuro` (https://pypi.org/project/hex-maze-neuro/). 
This requires python>=3.10.

To use these functions, install via

```bash
pip install hex-maze-neuro
```

Then import functions from the package as follows:

```python
from hexmaze import plot_hex_maze
plot_hex_maze(barriers={37, 7, 39, 41, 14, 46, 20, 23, 30}, show_barriers=False)
```
