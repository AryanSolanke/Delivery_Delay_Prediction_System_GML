import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET_PATH = os.path.join(ROOT, "Datasets", "raw_dataset", "dataset.csv")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

CITY_VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "city_vocab.json")
STATE_VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "state_vocab.json")
NODE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "node_table.csv")
EDGE_TABLE_PATH = os.path.join(ARTIFACTS_DIR, "edge_table.csv")


def build_vocab(values: pd.Series) -> dict:
    unique = sorted(values.dropna().unique())
    return {name: idx for idx, name in enumerate(unique, start=1)}


def build_node_table(df: pd.DataFrame) -> pd.DataFrame:
    seller = df[["seller_zip_code_prefix", "seller_city", "seller_state", "seller_lat", "seller_lng"]].rename(
        columns={
            "seller_zip_code_prefix": "zip_code_prefix",
            "seller_city": "city",
            "seller_state": "state",
            "seller_lat": "lat",
            "seller_lng": "lng",
        }
    )

    customer = df[["customer_zip_code_prefix", "customer_city", "customer_state", "customer_lat", "customer_lng"]].rename(
        columns={
            "customer_zip_code_prefix": "zip_code_prefix",
            "customer_city": "city",
            "customer_state": "state",
            "customer_lat": "lat",
            "customer_lng": "lng",
        }
    )

    nodes = pd.concat([seller, customer], ignore_index=True)

    grouped = nodes.groupby("zip_code_prefix", sort=True)

    node_table = pd.DataFrame(
        {
            "zip_code_prefix": grouped["zip_code_prefix"].first(),
            "city": grouped["city"].agg(lambda s: s.mode().iloc[0]),
            "state": grouped["state"].agg(lambda s: s.mode().iloc[0]),
            "mean_lat": grouped["lat"].mean(),
            "mean_lng": grouped["lng"].mean(),
        }
    ).reset_index(drop=True)

    return node_table


def report_role_consistency(df: pd.DataFrame, node_table: pd.DataFrame) -> None:
    seller = (
        df.groupby("seller_zip_code_prefix")
        .agg(city=("seller_city", lambda s: s.mode().iloc[0]), state=("seller_state", lambda s: s.mode().iloc[0]))
        .rename_axis("zip_code_prefix")
    )
    customer = (
        df.groupby("customer_zip_code_prefix")
        .agg(city=("customer_city", lambda s: s.mode().iloc[0]), state=("customer_state", lambda s: s.mode().iloc[0]))
        .rename_axis("zip_code_prefix")
    )
    merged = node_table.set_index("zip_code_prefix")[["city", "state"]].join(seller, rsuffix="_seller").join(
        customer, rsuffix="_customer"
    )
    city_conflicts = ((merged["city_seller"].notna()) & (merged["city_customer"].notna()) & (merged["city_seller"] != merged["city_customer"])).sum()
    state_conflicts = ((merged["state_seller"].notna()) & (merged["state_customer"].notna()) & (merged["state_seller"] != merged["state_customer"])).sum()
    print(f"Zips present in BOTH roles with conflicting city (seller vs customer): {city_conflicts}")
    print(f"Zips present in BOTH roles with conflicting state (seller vs customer): {state_conflicts}")


def build_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    edge_table = df[
        ["seller_zip_code_prefix", "customer_zip_code_prefix", "haversine_distance_km", "actual_transit_days"]
    ].copy()

    edge_table["route_count"] = edge_table.groupby(
        ["seller_zip_code_prefix", "customer_zip_code_prefix"]
    )["seller_zip_code_prefix"].transform("count")

    return edge_table


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = pd.read_csv(RAW_DATASET_PATH)

    city_vocab = build_vocab(pd.concat([df["seller_city"], df["customer_city"]]))
    state_vocab = build_vocab(pd.concat([df["seller_state"], df["customer_state"]]))

    node_table = build_node_table(df)
    node_table["city_idx"] = node_table["city"].map(city_vocab)
    node_table["state_idx"] = node_table["state"].map(state_vocab)

    edge_table = build_edge_table(df)

    with open(CITY_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(city_vocab, f, ensure_ascii=False, indent=2)
    with open(STATE_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(state_vocab, f, ensure_ascii=False, indent=2)

    node_table.to_csv(NODE_TABLE_PATH, index=False)
    edge_table.to_csv(EDGE_TABLE_PATH, index=False)

    report_role_consistency(df, node_table)

    print(f"Unique cities (union): {len(city_vocab)}")
    print(f"Unique states (union): {len(state_vocab)}")
    print(f"Node table: {len(node_table)} zips")
    print(f"Edge table: {len(edge_table)} orders")
    print(f"NaN in node city_idx/state_idx: {node_table[['city_idx', 'state_idx']].isna().sum().sum()}")
    print(f"Saved artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()