from __future__ import annotations

from pathlib import Path
import numpy as np
import plotly.graph_objects as go

from brain import SphereBrain


def _edge_coordinates(brain: SphereBrain, edges) -> tuple[list, list, list]:
    x, y, z = [], [], []
    for a, b in edges:
        x.extend([brain.positions[a, 0], brain.positions[b, 0], None])
        y.extend([brain.positions[a, 1], brain.positions[b, 1], None])
        z.extend([brain.positions[a, 2], brain.positions[b, 2], None])
    return x, y, z


def build_html(
    brain: SphereBrain,
    output_path: str | Path,
    highlighted_edges=None,
    highlighted_nodes=None,
    title="Sphere Brain",
) -> None:
    ordered_highlights = [tuple(sorted((int(e[0]), int(e[1])))) for e in (highlighted_edges or [])]
    # 同じ辺を繰り返し通っても、最初に現れた順序を表示用に保つ。
    ordered_highlights = list(dict.fromkeys(ordered_highlights))
    highlighted_edge_set = set(ordered_highlights)
    highlighted_nodes = set(int(n) for n in (highlighted_nodes or []))

    upper = np.triu_indices(brain.node_count, k=1)
    mask = brain.adjacency[upper]
    edges = [(int(a), int(b)) for a, b in zip(upper[0][mask], upper[1][mask])]
    edges.sort(key=lambda e: brain.weights[e[0], e[1]], reverse=True)
    edges = edges[:700]

    normal_edges = [e for e in edges if tuple(sorted(e)) not in highlighted_edge_set]
    nx, ny, nz = _edge_coordinates(brain, normal_edges)
    hx, hy, hz = _edge_coordinates(brain, ordered_highlights)

    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 36)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=sx,
        y=sy,
        z=sz,
        opacity=0.05,
        showscale=False,
        hoverinfo="skip",
        colorscale=[[0, "#dbeafe"], [1, "#dbeafe"]],
        name="球体境界",
    ))
    fig.add_trace(go.Scatter3d(
        x=nx,
        y=ny,
        z=nz,
        mode="lines",
        line={"width": 1, "color": "rgba(80,100,130,0.16)"},
        hoverinfo="skip",
        name="経路",
    ))
    fig.add_trace(go.Scatter3d(
        x=hx,
        y=hy,
        z=hz,
        mode="lines",
        line={"width": 5, "color": "rgba(238,96,52,0.22)"},
        hoverinfo="skip",
        name="今回の全経路",
    ))

    colors = [1 if i in highlighted_nodes else 0 for i in range(brain.node_count)]
    sizes = [7 if i in highlighted_nodes else 4 for i in range(brain.node_count)]
    fig.add_trace(go.Scatter3d(
        x=brain.positions[:, 0],
        y=brain.positions[:, 1],
        z=brain.positions[:, 2],
        mode="markers",
        marker={
            "size": sizes,
            "color": colors,
            "colorscale": [[0, "#355c7d"], [1, "#ff6b35"]],
            "showscale": False,
            "opacity": 0.92,
        },
        text=[f"ノード {i}<br>使用回数 {brain.node_usage[i]}" for i in range(brain.node_count)],
        hoverinfo="text",
        name="ノード",
    ))

    # アニメーション専用トレース。フレームごとに経路を一本ずつ増やす。
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="lines",
        line={"width": 8, "color": "rgba(245,94,24,0.98)"},
        hoverinfo="skip",
        name="思考経路リプレイ",
    ))
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker={"size": 11, "color": "#facc15", "line": {"width": 2, "color": "#7c2d12"}},
        hoverinfo="text",
        name="現在位置",
    ))

    frames = []
    slider_steps = []
    for index in range(len(ordered_highlights) + 1):
        visible_edges = ordered_highlights[:index]
        rx, ry, rz = _edge_coordinates(brain, visible_edges)
        if index:
            a, b = ordered_highlights[index - 1]
            current_x = [brain.positions[b, 0]]
            current_y = [brain.positions[b, 1]]
            current_z = [brain.positions[b, 2]]
            current_text = [f"ステップ {index}: ノード {a} → {b}"]
        else:
            current_x, current_y, current_z, current_text = [], [], [], []

        frame_name = str(index)
        frames.append(go.Frame(
            name=frame_name,
            data=[
                go.Scatter3d(x=rx, y=ry, z=rz),
                go.Scatter3d(x=current_x, y=current_y, z=current_z, text=current_text),
            ],
            traces=[4, 5],
        ))
        slider_steps.append({
            "label": str(index),
            "method": "animate",
            "args": [[frame_name], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        })

    fig.frames = frames
    controls = []
    sliders = []
    if ordered_highlights:
        controls = [{
            "type": "buttons",
            "direction": "left",
            "x": 0.02,
            "y": 0.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ 再生",
                    "method": "animate",
                    "args": [None, {
                        "fromcurrent": True,
                        "mode": "immediate",
                        "frame": {"duration": 420, "redraw": True},
                        "transition": {"duration": 80},
                    }],
                },
                {
                    "label": "Ⅱ 一時停止",
                    "method": "animate",
                    "args": [[None], {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                    }],
                },
            ],
        }]
        sliders = [{
            "active": 0,
            "x": 0.20,
            "y": 0.025,
            "len": 0.76,
            "currentvalue": {"prefix": "経路ステップ ", "font": {"size": 13}},
            "steps": slider_steps,
        }]

    fig.update_layout(
        title=title,
        scene={
            "xaxis": {"visible": False, "range": [-1.1, 1.1]},
            "yaxis": {"visible": False, "range": [-1.1, 1.1]},
            "zaxis": {"visible": False, "range": [-1.1, 1.1]},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 70 if ordered_highlights else 0},
        updatemenus=controls,
        sliders=sliders,
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    fig.write_html(output_path, include_plotlyjs=True, auto_open=False)
