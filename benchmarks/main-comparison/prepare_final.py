"""Build a new Amazon cohort, excluding users from previous experiments."""

import argparse
import json
from pathlib import Path

from common import ROOT, file_hash, save_json
from huggingface_hub import hf_hub_download, snapshot_download
import polars as pl

from src.data.amazon_preprocessing import build_item_text, iterative_k_core

REPO = "McAuley-Lab/Amazon-Reviews-2023"
TEXT_COLUMNS = [
    "parent_asin",
    "title",
    "store",
    "categories",
    "features",
    "description",
    "details",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("final_protocol.json")
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/final-v1")
    parser.add_argument(
        "--ratings-dir",
        type=Path,
        help="Optional directory containing <category>.csv files",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Optional directory containing raw_meta_<category>/*.parquet",
    )
    parser.add_argument(
        "--exclude-users",
        type=Path,
        nargs="*",
        default=[],
        help="Additional previously used Parquet files with user_id",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(
            "Output directory exists; use a new directory to preserve the split"
        )
    protocol = json.loads(args.protocol.read_text())
    excluded_paths = [ROOT / "data/processed/user_mapping.parquet"]
    excluded_paths += sorted(
        (ROOT / "benchmarks").glob("amazon-*/data/processed/sequences.parquet")
    )
    excluded_paths += args.exclude_users
    excluded_paths = sorted(set(p.resolve() for p in excluded_paths))
    excluded = pl.concat(
        [pl.read_parquet(p, columns=["user_id"]) for p in excluded_paths]
    ).unique()
    frames, sources = [], {}
    for category in protocol["categories"]:
        path = (
            args.ratings_dir / f"{category}.csv"
            if args.ratings_dir
            else Path(
                hf_hub_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    revision=protocol["dataset_revision"],
                    filename=f"benchmark/5core/rating_only/{category}.csv",
                )
            )
        )
        frame = (
            pl.scan_csv(path)
            .select("user_id", "parent_asin", "rating", "timestamp")
            .drop_nulls()
            .join(excluded.lazy(), on="user_id", how="anti")
            .with_columns(
                pl.col("rating").cast(pl.Float32), pl.col("timestamp").cast(pl.Int64)
            )
            .collect(engine="streaming")
        )
        frames.append(frame)
        sources[category] = {"path": str(path), "sha256": file_hash(path)}
    interactions = (
        pl.concat(frames)
        .sort(["user_id", "timestamp", "parent_asin"])
        .unique(subset=["user_id", "parent_asin"], keep="first", maintain_order=True)
    )
    users = (
        interactions.group_by("user_id")
        .len()
        .filter(
            pl.col("len").is_between(
                protocol["min_user_interactions"], protocol["max_user_interactions"]
            )
        )
        .sort("user_id")
    )
    if len(users) > protocol["max_users"]:
        users = users.sample(
            n=protocol["max_users"], seed=protocol["split_seed"], shuffle=True
        )
    interactions = interactions.join(users.select("user_id"), on="user_id", how="semi")
    interactions = iterative_k_core(
        interactions,
        protocol["min_user_interactions"],
        protocol["min_item_interactions"],
    )
    if interactions.is_empty():
        raise ValueError("No eligible users after exclusion/filtering")
    items = interactions.select("parent_asin").unique()
    metadata = []
    for category in protocol["categories"]:
        root = args.metadata_root
        if root is None:
            root = Path(
                snapshot_download(
                    repo_id=REPO,
                    repo_type="dataset",
                    revision=protocol["metadata_revisions"][category],
                    allow_patterns=[f"raw_meta_{category}/*.parquet"],
                )
            )
        metadata.append(
            (
                pl.scan_parquet(root / f"raw_meta_{category}/*.parquet")
                .select(TEXT_COLUMNS)
                .join(items.lazy(), on="parent_asin", how="semi")
                .collect(engine="streaming")
            )
        )
    catalog = (
        pl.concat(metadata)
        .unique(subset=["parent_asin"], keep="first", maintain_order=True)
        .with_columns(
            pl.struct(TEXT_COLUMNS)
            .map_elements(build_item_text, return_dtype=pl.String)
            .alias("item_text")
        )
        .filter(pl.col("item_text").str.len_chars() > 0)
    )
    interactions = interactions.join(
        catalog.select("parent_asin"), on="parent_asin", how="semi"
    )
    interactions = iterative_k_core(
        interactions,
        protocol["min_user_interactions"],
        protocol["min_item_interactions"],
    )
    if interactions.is_empty():
        raise ValueError("No interactions after metadata filtering")
    catalog = (
        catalog.join(
            interactions.select("parent_asin").unique(), on="parent_asin", how="semi"
        )
        .sort("parent_asin")
        .with_row_index("item_idx", offset=1)
    )
    user_mapping = (
        interactions.select("user_id")
        .unique()
        .sort("user_id")
        .with_row_index("user_idx")
    )
    if not user_mapping.join(excluded, on="user_id", how="semi").is_empty():
        raise ValueError("Old users leaked into the new cohort")
    interactions = (
        interactions.join(catalog.select("parent_asin", "item_idx"), on="parent_asin")
        .join(user_mapping, on="user_id")
        .sort(["user_idx", "timestamp", "parent_asin"])
        .with_columns(
            pl.int_range(pl.len()).over("user_idx").alias("position"),
            pl.len().over("user_idx").alias("sequence_length"),
        )
        .with_columns(
            pl.when(pl.col("position") == pl.col("sequence_length") - 1)
            .then(pl.lit("test"))
            .when(pl.col("position") == pl.col("sequence_length") - 2)
            .then(pl.lit("valid"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
    )
    sequences = interactions.group_by("user_idx", maintain_order=True).agg(
        pl.col("user_id").first(),
        pl.col("item_idx").alias("item_sequence"),
        pl.col("split").alias("split_sequence"),
        pl.col("timestamp").alias("timestamp_sequence"),
    )
    args.output_dir.mkdir(parents=True)
    for name, frame in [
        ("catalog_model.parquet", catalog),
        ("user_sequences.parquet", sequences),
        ("interactions_model.parquet", interactions),
        ("user_mapping.parquet", user_mapping),
    ]:
        frame.write_parquet(args.output_dir / name)
    signature = {
        name: file_hash(args.output_dir / name)
        for name in (
            "catalog_model.parquet",
            "user_sequences.parquet",
            "interactions_model.parquet",
        )
    }
    save_json(
        args.output_dir / "benchmark_manifest.json",
        {
            "protocol_sha256": file_hash(args.protocol),
            "protocol": protocol,
            "data_signature": signature,
            "excluded_files": {str(p): file_hash(p) for p in excluded_paths},
            "excluded_users": len(excluded),
            "sources": sources,
            "num_users": len(sequences),
            "num_items": len(catalog),
            "num_interactions": len(interactions),
            "old_user_overlap": 0,
            "limitations": "Known catalog, retrospective metadata, per-user chronology (not a global temporal split).",
        },
    )
    print(
        f"Saved fresh cohort: {len(sequences):,} users, {len(catalog):,} items -> {args.output_dir}"
    )
    print("No recommendation models were trained or evaluated.")


if __name__ == "__main__":
    main()
