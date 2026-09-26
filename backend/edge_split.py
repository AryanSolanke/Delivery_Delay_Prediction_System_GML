import numpy as np

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def edge_split(num_edges, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(num_edges)
    train_n = int(num_edges * train_ratio)
    val_n = int(num_edges * val_ratio)
    return perm[:train_n], perm[train_n : train_n + val_n], perm[train_n + val_n :]