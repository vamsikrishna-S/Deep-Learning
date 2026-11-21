"""Build a directed, time-filtered Elliptic transaction graph.

The saved NPZ contains tx_id, x, time, y, and edge_index arrays. The first
feature in x is the time step; labels are 1 (illicit), 0 (licit), and -1
(unknown). Edge indices refer to rows in the node arrays.

Time steps are coarse: edges within the same step have no known ordering.
Dataset-provided aggregated features may also contain information unavailable
at prediction time, so inspect their provenance before temporal evaluation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data" / "elliptic_bitcoin_dataset"


def build_graph(data_dir=DATA_DIR, max_time=None):
    """Return graph arrays for transactions up to and including max_time."""
    data_dir = Path(data_dir)
    features = pd.read_csv(data_dir / "elliptic_txs_features.csv", header=None)
    if features.shape[1] != 167:
        raise ValueError("Expected a transaction ID and 166 feature columns")
    if features[0].duplicated().any():
        raise ValueError("Transaction IDs in the features file must be unique")

    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv", dtype={"class": str})
    if classes["txId"].duplicated().any():
        raise ValueError("Transaction IDs in the classes file must be unique")
    labels = classes.set_index("txId")["class"].reindex(features[0])
    if labels.isna().any() or not labels.isin(["1", "2", "unknown"]).all():
        raise ValueError("Every transaction must have a valid class label")

    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    all_ids = pd.Index(features[0])
    if not edges["txId1"].isin(all_ids).all() or not edges["txId2"].isin(all_ids).all():
        raise ValueError("An edge references a transaction missing from the features file")

    if max_time is not None:
        selected = features[1] <= max_time
        features = features.loc[selected]
        labels = labels.loc[selected.to_numpy()]

    node_ids = pd.Index(features[0])
    edges = edges[edges["txId1"].isin(node_ids) & edges["txId2"].isin(node_ids)]
    source = node_ids.get_indexer(edges["txId1"])
    target = node_ids.get_indexer(edges["txId2"])
    time = features[1].to_numpy(dtype=np.int64)
    time_respecting = time[source] <= time[target]
    dropped_edges = int((~time_respecting).sum())
    edge_index = np.stack((source[time_respecting], target[time_respecting])).astype(np.int64)

    return {
        "tx_id": features[0].to_numpy(dtype=np.int64),
        "x": features.iloc[:, 1:].to_numpy(dtype=np.float32),
        "time": time,
        "y": labels.map({"1": 1, "2": 0, "unknown": -1}).to_numpy(dtype=np.int64),
        "edge_index": edge_index,
    }, dropped_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--max-time", type=int, help="Include only transactions through this time step")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "graph.npz")
    args = parser.parse_args()

    graph, dropped_edges = build_graph(args.data_dir, args.max_time)
    np.savez_compressed(args.output, **graph)
    print(f"Saved {len(graph['tx_id'])} nodes and {graph['edge_index'].shape[1]} directed edges to {args.output}")
    print(f"Dropped {dropped_edges} edges that point backward in time")


if __name__ == "__main__":
    main()