# `graph_builder.py` Documentation

The script `graph_builder.py` completes the final transformation from tabular CSV structures into mathematically optimized PyTorch tensors, which are the native format required for training a Graph Neural Network.

## **1. Node Indexing & Alignment**
Neural networks do not understand text or raw zip codes; they require all nodes to be numbered sequentially from $0$ to $N$. The script sorts the `node_table.csv` and assigns a permanent integer index (the row number) to every unique zip code. This `zip_to_node` dictionary ensures that when an edge references a zip code, it accurately points to the correct sequential node ID.
<br><br>

## **2. Node Feature Tensors**
The script mathematically separates the node attributes into three distinct tensors because the GNN must process them differently:

1. **`city_idx`:** The integer index for the city is stored as an `int64` tensor. This is not fed directly as a mathematical feature; rather, the GNN uses it to look up the learnable 32-dimensional embedding vector.
2. **`state_ohe`:** The state mappings are converted into a strict $14,807 \times 27$ matrix of binary values ($0$ or $1$). The code explicitly places the $1$ into the matrix column matching the `state_idx`, ensuring the network perfectly interprets the regional boundaries.
3. **`coord_feats`:** The node latitudes and longitudes are standardized (scaled by mean and standard deviation). This prevents the large raw coordinate values from mathematically dominating the smaller binary state features during gradient calculations.
<br><br>

## **3. Graph Topology Construction**
The `edge_index` is the most critical tensor in the entire script. Using the PyTorch Geometric COO format, the code maps every transaction's seller and customer zip codes to their corresponding node IDs. It then stacks them into a $2 \times 97,331$ matrix. This structure explicitly maps the routing map, telling the GNN exactly which node (seller) sends a message to which node (customer).
<br><br>

## **4. Edge Features & The Target Variable**
The edge metrics—`haversine_distance_km` and `route_count`—are extracted and passed through a `RobustScaler` to mathematically stabilize them against outliers. Crucially, the target variable `y` (`actual_transit_days`) is maintained as raw, unscaled `float32` values. This ensures the model computes its loss against the true number of physical days rather than a distorted metric.

Finally, the entire graph dictionary is saved as a native PyTorch object (`graph.pt`).

<br><br>
 Documentation Ends Here