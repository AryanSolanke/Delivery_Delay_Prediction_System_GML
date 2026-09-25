import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
VIZ_HOME = os.path.join(ROOT, "evaluations", "graph_vizs")

NODE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "node_table.csv")
EDGE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "edge_table.csv")
FULL_BASENAME = "graph_geographic_full"
ZOOMED_BASENAME = "graph_geographic"

MAX_EDGES = 5000
FIG_WIDTH = 14
FIG_HEIGHT = 12
DOT_SIZE = 4
EDGE_ALPHA = 0.15
EDGE_COLOR = "steelblue"
EDGE_WIDTH = 0.6
VIEW_LOW = 0.02
VIEW_HIGH = 0.98
SEED = 42


def next_version(basename: str) -> int:
    prefix = f"{basename}_v"
    versions = []
    for f in os.listdir(VIZ_HOME):
        if f.startswith(prefix) and f.endswith(".png"):
            stem = f[len(prefix) : -4]
            if stem.isdigit():
                versions.append(int(stem))
    return max(versions, default=0) + 1


def output_path(basename: str) -> str:
    return os.path.join(VIZ_HOME, f"{basename}_v{next_version(basename)}.png")


def render(node_table, edge_table, graph, pos, node_colors, state_colors, output_path, title_suffix, zoomed):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrows=False, alpha=EDGE_ALPHA, edge_color=EDGE_COLOR, width=EDGE_WIDTH
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=DOT_SIZE, node_color=node_colors, alpha=0.8
    )

    if zoomed:
        lng_lo, lng_hi = node_table["mean_lng"].quantile([VIEW_LOW, VIEW_HIGH])
        lat_lo, lat_hi = node_table["mean_lat"].quantile([VIEW_LOW, VIEW_HIGH])
        ax.set_xlim(lng_lo, lng_hi)
        ax.set_ylim(lat_lo, lat_hi)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=state)
        for state, color in state_colors.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, title="Brazilian State", ncol=3)

    ax.set_title(
        f"Geographic Logistics Hub Graph {title_suffix}\n{len(node_table)} zip nodes · {len(edge_table)} orders"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    os.makedirs(VIZ_HOME, exist_ok=True)

    node_table = pd.read_csv(NODE_TABLE_PATH).set_index("zip_code_prefix")
    edge_table = pd.read_csv(EDGE_TABLE_PATH)

    graph = nx.DiGraph()
    graph.add_nodes_from(node_table.index)

    edges = list(zip(edge_table["seller_zip_code_prefix"], edge_table["customer_zip_code_prefix"]))
    if len(edges) > MAX_EDGES:
        edges = pd.DataFrame(edges, columns=["src", "dst"]).sample(MAX_EDGES, random_state=SEED)
        edges = list(zip(edges["src"], edges["dst"]))
    graph.add_edges_from(edges)

    pos = {
        node: (row["mean_lng"], row["mean_lat"])
        for node, row in node_table.iterrows()
    }

    states = sorted(node_table["state"].unique())
    state_colors = {
        state: plt.cm.tab20(i % 20) for i, state in enumerate(states)
    }
    node_colors = [state_colors[row["state"]] for _, row in node_table.iterrows()]

    drawn_edges = len(edges)
    edge_note = f"(downsampled to {drawn_edges})" if drawn_edges < len(edge_table) else "(all)"

    render(
        node_table, edge_table, graph, pos, node_colors, state_colors,
        output_path(FULL_BASENAME), f"- Full Extent {edge_note}", zoomed=False,
    )
    render(
        node_table, edge_table, graph, pos, node_colors, state_colors,
        output_path(ZOOMED_BASENAME), f"- Zoomed Core {edge_note}", zoomed=True,
    )


if __name__ == "__main__":
    main()