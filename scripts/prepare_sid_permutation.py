"""Prepare original/permuted Two-Tower SIDs for a fixed-tokenizer GenRec control."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from scripts.train_genrec import file_hash, save_history


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def popularity_bins(counts):
    # Exact counts 0..20; then 21..40, 41..80, etc. Unseen stays separate.
    bins = counts.copy()
    head = counts > 20
    bins[head] = 20 + np.ceil(np.log2(counts[head] / 20)).astype(np.int64)
    return bins


def permute_within_bins(counts, seed):
    rng = np.random.default_rng(seed)
    bins = popularity_bins(counts)
    permutation = np.arange(len(counts))
    for label in np.unique(bins):
        rows = np.flatnonzero(bins == label)
        permutation[rows] = rng.permutation(rows)
    return permutation


def prefix_frequency_changes(codes, permutation, weights):
    """Compare train frequency on the SAME prefixes, not sorted histograms."""
    report = []
    for depth in range(1, codes.shape[1] + 1):
        _, buckets = np.unique(codes[:, :depth], axis=0, return_inverse=True)
        before = np.bincount(buckets, weights=weights)
        after = np.bincount(buckets[permutation], weights=weights, minlength=len(before))
        difference = np.abs(after - before)
        report.append({
            "depth": depth,
            "prefixes": len(before),
            "changed_prefixes": int(np.count_nonzero(difference)),
            "max_absolute_count_change": int(difference.max()),
            "total_variation": float(difference.sum() / (2 * weights.sum())) if weights.sum() else 0.0,
        })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Permutation seed, independent of training seed")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing control exports")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("Seed must be nonnegative")

    paths = {role: DATA_DIR / f"item_semantic_ids_twotower_{role}.npy"
             for role in ("original", "permuted")}
    permutation_path = DATA_DIR / "item_semantic_ids_twotower_permutation.npy"
    outputs = [permutation_path, *paths.values(), *(p.with_suffix(".json") for p in paths.values())]
    if not args.overwrite and any(p.exists() for p in outputs):
        parser.error("Control exports already exist; preserve previous runs before using --overwrite")

    source_path = DATA_DIR / "item_semantic_ids_twotower.npy"
    rq_path = CHECKPOINT_DIR / "rqvae_twotower_best.pt"
    codes = np.load(source_path)
    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet", columns=["item_idx"]).sort("item_idx")
    n = len(catalog)
    if catalog["item_idx"].to_list() != list(range(1, n + 1)):
        raise ValueError("Expected contiguous item IDs starting at 1")
    if (codes.shape != (n, 4) or not np.issubdtype(codes.dtype, np.integer)
            or np.any(codes < 0) or len(np.unique(codes, axis=0)) != n):
        raise ValueError("Expected unique, nonnegative, four-token SIDs aligned with catalog")

    interactions = pl.read_parquet(DATA_DIR / "interactions_model.parquet", columns=["item_idx", "split"])
    train_ids = interactions.filter(pl.col("split") == "train")["item_idx"].to_numpy()
    if np.any(train_ids < 1) or np.any(train_ids > n):
        raise ValueError("Invalid training item ID")
    train_counts = np.bincount(train_ids, minlength=n + 1)[1:]

    # Actual next-item targets differ from interaction counts (first items are context only).
    target_counts = np.zeros(n + 1, dtype=np.int64)
    sequence_counts = np.zeros(n + 1, dtype=np.int64)
    users = pl.read_parquet(DATA_DIR / "user_sequences.parquet")
    for row in users.iter_rows(named=True):
        seq = np.asarray(row["item_sequence"], dtype=np.int64)
        if (len(seq) < 4 or np.any(seq < 1) or np.any(seq > n)
                or row["split_sequence"] != ["train"] * (len(seq) - 2) + ["valid", "test"]):
            raise ValueError("Invalid user sequence/split")
        np.add.at(target_counts, seq[1:-2], 1)
        np.add.at(sequence_counts, seq[:-2], 1)
    if not np.array_equal(sequence_counts[1:], train_counts):
        raise ValueError("Training counts disagree between interactions and sequences")

    permutation = permute_within_bins(train_counts, args.seed)
    shuffled = codes[permutation]
    bins = popularity_bins(train_counts)
    assert np.array_equal(np.sort(permutation), np.arange(n))
    assert np.array_equal(bins, bins[permutation])
    assert np.array_equal(np.unique(codes, axis=0), np.unique(shuffled, axis=0))
    metadata = {
        "seed": args.seed,
        "source_sid_file": source_path.name,
        "source_sid_sha256": file_hash(source_path),
        "rq_checkpoint": str(rq_path),
        "rq_checkpoint_sha256": file_hash(rq_path),
        "data_signature": {name: file_hash(DATA_DIR / name) for name in (
            "catalog_model.parquet", "user_sequences.parquet", "interactions_model.parquet"
        )},
        "stratification": "Exact train interaction counts 0..20; 21..40, 41..80, etc.; unseen separate",
        "items": n,
        "moved_items": int(np.count_nonzero(permutation != np.arange(n))),
        "unchanged_prefix_fraction": [float(np.mean(np.all(codes[:, :d] == shuffled[:, :d], axis=1)))
                                      for d in range(1, 5)],
        "bins": [{"bin": int(label), "items": int(np.sum(bins == label)),
                  "moved_items": int(np.sum((bins == label) & (permutation != np.arange(n))))}
                 for label in np.unique(bins)],
        "train_interaction_prefix_changes": prefix_frequency_changes(codes, permutation, train_counts),
        "train_target_prefix_changes": prefix_frequency_changes(codes, permutation, target_counts[1:]),
        "permutation_file": permutation_path.name,
        "permutation_convention": "Zero-based: new_codes[row] = original_codes[permutation[row]]; PAD is not included",
    }
    np.save(permutation_path, permutation)
    metadata["permutation_sha256"] = file_hash(permutation_path)
    for role, path in paths.items():
        np.save(path, codes if role == "original" else shuffled)
        save_history({**metadata, "role": role, "sid_source": f"twotower_{role}",
                      "sid_sha256": file_hash(path)}, path.with_suffix(".json"))
        print("Saved:", path, "and", path.with_suffix(".json"))
    print(f"Moved {metadata['moved_items']:,}/{n:,} items; unchanged prefixes:",
          metadata["unchanged_prefix_fraction"])
    print("Train-target prefix frequency changes:")
    print(json.dumps(metadata["train_target_prefix_changes"], indent=2))


if __name__ == "__main__":
    main()
