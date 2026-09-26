import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BACKEND_ROOT, "artifacts")

NODE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "node_table.csv")
EDGE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "edge_table.csv")
GRAPH_PATH = os.path.join(ARTIFACTS_DIR, "graph.pt")

NUM_STATES = 27


def build_graph():
    node_table = pd.read_csv(NODE_TABLE_PATH)
    edge_table = pd.read_csv(EDGE_TABLE_PATH)

    node_table = node_table.sort_values("zip_code_prefix").reset_index(drop=True)
    zip_to_node = {int(z): i for i, z in enumerate(node_table["zip_code_prefix"])}

    node_zip = torch.tensor(node_table["zip_code_prefix"].values, dtype=torch.int64)
    city_idx = torch.tensor(node_table["city_idx"].values, dtype=torch.int64)

    state_ohe = np.zeros((len(node_table), NUM_STATES))
    for i, state in enumerate(node_table["state_idx"].values):
        if state >= 1:
            state_ohe[i, int(state) - 1] = 1.0
    state_ohe = torch.tensor(state_ohe, dtype=torch.float32)

    coord_scaler = StandardScaler()
    coord_feats = torch.tensor(
        coord_scaler.fit_transform(node_table[["mean_lat", "mean_lng"]].values),
        dtype=torch.float32,
    )

    src = edge_table["seller_zip_code_prefix"].map(zip_to_node).values
    dst = edge_table["customer_zip_code_prefix"].map(zip_to_node).values
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.int64)

    distance_raw = torch.tensor(edge_table["haversine_distance_km"].values, dtype=torch.float32)
    count_raw = torch.tensor(edge_table["route_count"].values, dtype=torch.float32)

    distance_scaler = RobustScaler()
    count_scaler = RobustScaler()
    distance_scaled = torch.tensor(
        distance_scaler.fit_transform(distance_raw.numpy().reshape(-1, 1)).reshape(-1),
        dtype=torch.float32,
    )
    count_scaled = torch.tensor(
        count_scaler.fit_transform(count_raw.numpy().reshape(-1, 1)).reshape(-1),
        dtype=torch.float32,
    )

    y = torch.tensor(edge_table["actual_transit_days"].values, dtype=torch.float32)

    graph = {
        "node_zip": node_zip,
        "city_idx": city_idx,
        "state_ohe": state_ohe,
        "coord_feats": coord_feats,
        "edge_index": edge_index,
        "distance_raw": distance_raw,
        "distance_scaled": distance_scaled,
        "count_raw": count_raw,
        "count_scaled": count_scaled,
        "y": y,
        "num_states": NUM_STATES,
    }

    torch.save(graph, GRAPH_PATH)
    joblib.dump(coord_scaler, os.path.join(ARTIFACTS_DIR, "coord_scaler.pkl"))
    joblib.dump(distance_scaler, os.path.join(ARTIFACTS_DIR, "distance_scaler.pkl"))
    joblib.dump(count_scaler, os.path.join(ARTIFACTS_DIR, "count_scaler.pkl"))

    print(f"Nodes: {len(node_table)}")
    print(f"Edges: {edge_index.shape[1]}")
    print(f"State OHE cols: {state_ohe.shape[1]}")
    print(f"Coord feats cols: {coord_feats.shape[1]}")
    print(f"Edge index dtype: {edge_index.dtype}, y mean: {y.mean().item():.4f}")
    print(f"Saved graph to: {GRAPH_PATH}")


if __name__ == "__main__":
    build_graph()