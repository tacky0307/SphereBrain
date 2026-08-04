from __future__ import annotations

# Reuse the existing Puzzle UI while replacing only its brain implementation.
import sphere_world_puzzle
from sphere_world_puzzle_temporal import TemporalActionPortPuzzleBrain

sphere_world_puzzle.PuzzleSphereBrain = TemporalActionPortPuzzleBrain

import sphere_world_puzzle_lab as ui  # noqa: E402


if __name__ == "__main__":
    from waitress import serve
    import webbrowser

    webbrowser.open("http://127.0.0.1:5033")
    print("SphereWorld Puzzle — Temporal Action Port: http://127.0.0.1:5033")
    serve(ui.app, host="127.0.0.1", port=5033, threads=6)
