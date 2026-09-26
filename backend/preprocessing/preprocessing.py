import numpy as np
import pandas as pd

RAW_DATASET_PATH = "C:\\Github_repos\\Delivery_Delay_Prediction_System_GML\\Datasets\\raw_dataset\\datasets.csv"



def compute_transit_days(df: pd.DataFrame) -> pd.Series:
    
    purchase = pd.to_datetime(df["order_purchase_timestamp"])
    delivered = pd.to_datetime(df["order_delivered_customer_date"])
    return (delivered - purchase).dt.total_seconds() / 86400.0


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[["order_purchase_timestamp", "order_delivered_customer_date"]] = df[
        ["order_purchase_timestamp", "order_delivered_customer_date"]
    ].apply(pd.to_datetime)

    df["actual_transit_days"] = compute_transit_days(df)

    df["haversine_distance_km"] = haversine_distance(
        df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"]
    )

    return df


def main():
    df = pd.read_csv(RAW_DATASET_PATH)


if __name__ == "__main__":
    main()