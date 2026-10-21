"""Prepare Amazon Reviews 2023 Industrial and Scientific for GenRec."""

import argparse
import json
from pathlib import Path

from common import DATA_DIR
from huggingface_hub import hf_hub_download, snapshot_download
import polars as pl

from src.data.amazon_preprocessing import build_item_text, iterative_k_core


REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = "Industrial_and_Scientific"
RATINGS_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
# Both ratings and metadata Parquet files are available at this revision.
METADATA_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
TEXT_COLUMNS = [
    "parent_asin", "title", "store", "categories", "features", "description", "details",
]


def load_metadata(item_ids, metadata_dir):
    if metadata_dir is None:
        repo_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=METADATA_REVISION,
            allow_patterns=[f"raw_meta_{CATEGORY}/*.parquet"],
        )
        metadata_dir = Path(repo_path) / f"raw_meta_{CATEGORY}"

    # Read only text columns and catalog items; images and review text are unused.
    metadata = (
        pl.scan_parquet(metadata_dir / "*.parquet")
        .select(TEXT_COLUMNS)
        .join(item_ids.lazy(), on="parent_asin", how="semi")
        .collect(engine="streaming")
        .unique(subset=["parent_asin"], keep="first", maintain_order=True)
    )
    return metadata.with_columns(
        pl.struct(TEXT_COLUMNS)
        .map_elements(build_item_text, return_dtype=pl.String)
        .alias("item_text")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--ratings-path", type=Path, help="Existing rating-only CSV")
    parser.add_argument("--metadata-dir", type=Path, help="Existing Industrial & Scientific metadata Parquet directory")
    parser.add_argument("--max-users", type=int, default=100_000, help="0 keeps all eligible users")
    parser.add_argument("--min-user-interactions", type=int, default=5)
    parser.add_argument("--max-user-interactions", type=int, default=200)
    parser.add_argument("--min-item-interactions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.max_users < 0 or args.min_user_interactions < 4
            or args.max_user_interactions < args.min_user_interactions
            or args.min_item_interactions < 1):
        parser.error("Require max-users >= 0, 4 <= min-user <= max-user and min-item >= 1")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = args.ratings_path
    if ratings_path is None:
        ratings_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=RATINGS_REVISION,
            filename=f"benchmark/5core/rating_only/{CATEGORY}.csv",
        ))
    interactions = pl.read_csv(ratings_path).select(
        "user_id", "parent_asin", "rating", "timestamp",
    ).drop_nulls().with_columns(
        pl.col("rating").cast(pl.Float32),
        pl.col("timestamp").cast(pl.Int64),
    )
    raw_rows = len(interactions)

    # Keep the earliest review per user/item so held-out targets cannot be seen items.
    interactions = (
        interactions.sort(["user_id", "timestamp", "parent_asin"])
        .unique(subset=["user_id", "parent_asin"], keep="first", maintain_order=True)
    )
    users = (
        interactions.group_by("user_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n").is_between(args.min_user_interactions, args.max_user_interactions))
        .sort("user_id")
    )
    if args.max_users and len(users) > args.max_users:
        users = users.sample(n=args.max_users, seed=args.seed, shuffle=True)
    interactions = interactions.join(users.select("user_id"), on="user_id", how="semi")
    interactions = iterative_k_core(
        interactions, args.min_user_interactions, args.min_item_interactions,
    )
    if interactions.is_empty():
        raise ValueError("No interactions remain after filtering")

    print("Loading metadata for", interactions["parent_asin"].n_unique(), "items", flush=True)
    metadata = load_metadata(interactions.select("parent_asin").unique(), args.metadata_dir)
    metadata = metadata.filter(pl.col("item_text").str.len_chars() > 0)
    before_metadata = len(interactions)
    interactions = interactions.join(metadata.select("parent_asin"), on="parent_asin", how="semi")
    dropped_rows = before_metadata - len(interactions)
    if dropped_rows:
        interactions = iterative_k_core(
            interactions, args.min_user_interactions, args.min_item_interactions,
        )
    if interactions.is_empty():
        raise ValueError("No interactions remain after metadata filtering")

    # Item 0 is PAD; mappings are stable for a fixed filtered dataset.
    catalog = (
        metadata.join(interactions.select("parent_asin").unique(), on="parent_asin", how="semi")
        .sort("parent_asin")
        .with_row_index("item_idx", offset=1)
        .with_columns(pl.lit(CATEGORY).alias("source_category"))
    )
    user_mapping = interactions.select("user_id").unique().sort("user_id").with_row_index("user_idx")
    interactions = (
        interactions.join(catalog.select("parent_asin", "item_idx"), on="parent_asin")
        .join(user_mapping, on="user_id")
        .sort(["user_idx", "timestamp", "parent_asin"])
        .with_columns(
            pl.int_range(pl.len()).over("user_idx").alias("position"),
            pl.len().over("user_idx").alias("sequence_length"),
        )
        .with_columns(
            pl.when(pl.col("position") == pl.col("sequence_length") - 1).then(pl.lit("test"))
            .when(pl.col("position") == pl.col("sequence_length") - 2).then(pl.lit("valid"))
            .otherwise(pl.lit("train")).alias("split")
        )
    )
    sequences = interactions.group_by("user_idx", maintain_order=True).agg(
        pl.col("user_id").first(),
        pl.col("item_idx").alias("item_sequence"),
        pl.col("split").alias("split_sequence"),
        pl.col("rating").alias("rating_sequence"),
        pl.col("timestamp").alias("timestamp_sequence"),
    )
    catalog.write_parquet(args.data_dir / "catalog.parquet")
    sequences.write_parquet(args.data_dir / "sequences.parquet")
    interactions.write_parquet(args.data_dir / "interactions.parquet")

    summary = {
        "dataset": REPO_ID,
        "category": CATEGORY,
        "ratings_revision": RATINGS_REVISION,
        "metadata_revision": METADATA_REVISION,
        "ratings_path": str(ratings_path),
        "metadata_dir": str(args.metadata_dir) if args.metadata_dir else None,
        "source": "5core_rating_only",
        "seed": args.seed,
        "max_users": args.max_users,
        "min_user_interactions": args.min_user_interactions,
        "max_user_interactions": args.max_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "raw_interactions": raw_rows,
        "dropped_interactions_missing_metadata": dropped_rows,
        "num_users": len(sequences),
        "num_items": len(catalog),
        "num_interactions": len(interactions),
        "num_train_examples": len(interactions) - 3 * len(sequences),
        "num_valid": len(sequences),
        "num_test": len(sequences),
        "split": "per-user last two events: validation, test",
        "order": ["timestamp", "parent_asin"],
    }
    (args.data_dir / "preprocessing.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
