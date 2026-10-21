"""Download MovieLens 1M and build chronological leave-two-out sequences."""

import argparse
import io
import json
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

from common import BENCHMARK_DIR, DATA_DIR
import polars as pl

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=BENCHMARK_DIR / "data" / "raw")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--min-rating", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if not all((args.raw_dir / name).exists() for name in ("ratings.dat", "movies.dat")):
        print("Downloading:", URL, flush=True)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                archive = response.read()
        except urllib.error.URLError as error:
            parser.exit(
                1,
                f"Download failed: {error}\n"
                "Use --raw-dir with an extracted ml-1m directory.\n",
            )
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for name in ("ratings.dat", "movies.dat", "README"):
                (args.raw_dir / name).write_bytes(source.read(f"ml-1m/{name}"))

    ratings = pl.read_csv(
        io.BytesIO((args.raw_dir / "ratings.dat").read_bytes().replace(b"::", b"\t")),
        separator="\t", has_header=False,
        new_columns=["user_id", "movie_id", "rating", "timestamp"],
    ).filter(pl.col("rating") >= args.min_rating)
    ratings = ratings.unique(subset=["user_id", "movie_id"], keep="first")
    # At least two train events, one validation event, and one test event.
    ratings = ratings.filter(pl.len().over("user_id") >= 4)
    if ratings.is_empty():
        raise ValueError("No users with at least four retained ratings")

    movies = []
    for line in (args.raw_dir / "movies.dat").read_text(encoding="latin-1").splitlines():
        movie_id, title, genres = line.split("::")
        movies.append({
            "movie_id": int(movie_id), "title": title, "genres": genres,
            "item_text": f"Title: {title}\nGenres: {genres.replace('|', ', ')}",
        })
    catalog = (
        pl.DataFrame(movies)
        .join(ratings.select("movie_id").unique(), on="movie_id", how="semi")
        .sort("movie_id")
        .with_row_index("item_idx", offset=1)
    )
    ratings = ratings.join(catalog.select("movie_id", "item_idx"), on="movie_id")
    # Movie ID breaks timestamp ties deterministically.
    users = (
        ratings.sort(["user_id", "timestamp", "movie_id"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_idx").alias("item_sequence"))
        .with_row_index("user_idx", offset=1)
    )
    catalog.write_parquet(args.data_dir / "catalog.parquet")
    users.write_parquet(args.data_dir / "sequences.parquet")
    summary = {
        "source": URL,
        "min_rating": args.min_rating,
        "num_users": len(users), "num_items": len(catalog),
        "num_interactions": len(ratings),
        "num_train_interactions": len(ratings) - 2 * len(users),
        "num_train_examples": len(ratings) - 3 * len(users),
        "num_valid": len(users), "num_test": len(users),
        "order": ["timestamp", "movie_id"],
        "split": "per-user last two events: validation, test",
    }
    (args.data_dir / "preprocessing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
