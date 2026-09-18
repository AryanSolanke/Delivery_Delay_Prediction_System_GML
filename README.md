# Delivery Delay Prediction System using Graph Machine Learning

A Graph Neural Network (GNN) based system for accurately predicting package delivery delays by modeling logistics networks as dynamic graphs rather than treating shipments as independent tabular records.

---

## Problem Statement

Accurately predicting package delivery days is a critical challenge in modern supply chains and e-commerce, directly impacting customer satisfaction, warehouse staging, and fleet utilization.

Traditional approaches rely on standard tabular machine learning (such as XGBoost or Random Forests) or static routing heuristics. These models treat shipments as independent events and model routes purely as flat tabular features (origin, destination, distance, transport mode). However, real-world logistics operates over a dynamic, interconnected network where disruptions at a single transit hub or route can cascade across neighboring nodes.

---

## Graph Formulation of the Logistics Network

The logistics system is represented as a graph with the following structure:

| Component     | Description                                                                                              |
|---------------|----------------------------------------------------------------------------------------------------------|
| **Nodes**     | Fulfillment centers, sortation hubs, local delivery stations, and customer drop-off clusters              |
| **Edges**     | Transit lanes, highways, or carrier shipping routes characterized by capacity, historical transit time, distance, and congestion |
| **Dynamic Attributes** | Time-varying factors including weather, seasonal order spikes, vehicle availability, and dwell times at intermediate facilities |

Delivery day prediction is formulated as a **graph-level, edge-level, or subgraph regression/classification task** to estimate either the exact turnaround duration (in days) or the probability that an order arrives by a target delivery date.

---

## Why Graph Machine Learning?

Standard tabular models suffer from critical limitations that Graph Neural Networks address directly:

- **Topology Awareness:** Tabular models ignore how intermediate hubs are wired together. GNNs aggregate topological context via message passing to model network bottlenecks naturally.
- **Cascading Delay Modeling:** Delays in modern fulfillment are rarely isolated. A delay at a primary hub impacts all downstream distribution nodes; graph representations propagate these state changes across neighboring edges.
- **Spatio-Temporal Dependencies:** Delivery timelines depend on both physical distance (spatial) and evolving congestion/processing load over days (temporal). Spatio-temporal GNNs (ST-GNNs) capture both dimensions simultaneously.

---

## Objectives

1. Construct a graph dataset from order tracking and logistics data (e.g., Brazilian E-Commerce/Olist or Supply Chain DataCo).
2. Implement a Graph Neural Network (such as GCN, GAT, or GraphSAGE) to encode warehouse-to-customer path dynamics.
3. Compare GML performance against standard baseline models (Linear Regression, Random Forest, XGBoost) using metrics like **MAE**, **RMSE**, and **On-Time Classification Accuracy**.

---

## Proposed Architecture

```
Order Data + Logistics Network
         |
         v
  Graph Construction
  (Nodes: hubs/stations/clusters)
  (Edges: transit routes)
         |
         v
  Feature Engineering
  (Weather, congestion, dwell times)
         |
         v
  GNN Encoder (GCN / GAT / GraphSAGE)
         |
         v
  Prediction Head
  -> Regression: Estimated Delivery Days
  -> Classification: On-Time Probability
```

---

## Baselines for Comparison

| Model                  | Type           |
|------------------------|----------------|
| Linear Regression      | Tabular        |
| Random Forest          | Tabular (Ensemble) |
| XGBoost                | Tabular (Boosting) |
| GCN                    | Graph          |
| GAT                    | Graph (Attention) |
| GraphSAGE              | Graph (Inductive) |

---

## Evaluation Metrics

- **MAE** (Mean Absolute Error) - Measures average prediction error in delivery days
- **RMSE** (Root Mean Squared Error) - Penalizes larger prediction errors more heavily
- **On-Time Classification Accuracy** - Binary classification accuracy for whether an order arrives by the promised date

---

## Tech Stack

- **Language:** Python
- **Graph Learning:** PyTorch Geometric (PyG) / DGL
- **Baseline ML:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, NetworkX
