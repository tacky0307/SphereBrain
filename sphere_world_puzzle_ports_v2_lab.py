from __future__ import annotations

# Replace the puzzle brain class before loading the existing web UI.
# This keeps the screen and controls identical, so v1/v2 results are comparable.
import sphere_world_puzzle
from sphere_world_puzzle_ports_v2 import ActionPortPuzzleBrainV2

sphere_world_puzzle.PuzzleSphereBrain = ActionPortPuzzleBrainV2

import sphere_world_puzzle_lab as ui  # noqa: E402


if __name__ == "__main__":
    from waitress import serve
    import webbrowser

    webbrowser.open("http://127.0.0.1:5032")
    print("SphereWorld Puzzle — Action Port v2: http://127.0.0.1:5032")
    serve(ui.app, host="127.0.0.1", port=5032, threads=6)
