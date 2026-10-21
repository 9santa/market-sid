from pathlib import Path

import numpy as np
import polars as pl

from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


DATA_DIR = Path("data/processed")

N_COMPONENTS = 64
SEED = 42


def main():
    interactions = pl.read_parquet(DATA_DIR / "interactions_model.parquet")

    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")

    train = interactions.filter(pl.col("split") == "train")

    print(train.head(10))

    num_users = int(interactions["user_idx"].max()) + 1
    num_items = len(catalog)

    # User ids, (len(users))
    rows = train["user_idx"].to_numpy().astype(np.int64)
    # Item ids, (len(items))
    cols = train["item_idx"].to_numpy().astype(np.int64) - 1

    values = np.ones(
        len(train),
        dtype=np.float32,
    )

    # Construct sparse User-Item matrix,
    # with X[i, j] = 1 if user 'i' interacted with item 'j'
    X = coo_matrix(
        (values, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    ).tocsr()

    print("User-item:", X.shape)
    print("NNZ:", X.nnz)  # number of non-zero entries

    # We normalize, so that extremely active user is less informative
    # as evidence that two particular items belong together.
    # Basically so that very active users do not over-dominate.
    X = normalize(
        X,
        norm="l2",
        axis=1,
    )

    # [items, users]
    item_user = X.T.tocsr()

    # SVD approximates the large matrix with a low-rank structure.
    # N_COMPONENTS = 64 means we retain 64 latent behavioral dimensions.
    # The intuition is that SVD discovers latent patterns such as:
    # people who buy phone accessories
    # people who buy PC components
    # people who buy home-office equipment
    # people who buy photography accessories
    # ...
    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        n_iter=10,
        random_state=SEED,
    )

    behavior = svd.fit_transform(item_user).astype(np.float32)

    print("Behavior embeddings:", behavior.shape)

    print("Explained variance:", float(svd.explained_variance_ratio_.sum()))

    # Count number of occurences of each item
    # ==================================
    train_counts = np.zeros(
        num_items,
        dtype=np.int64,
    )

    unique_items, counts = np.unique(
        cols,
        return_counts=True,
    )

    train_counts[unique_items] = counts
    # ==================================

    # Mask items that occur at least once
    warm_mask = train_counts > 0

    print("Warm items:", int(warm_mask.sum()))
    print("Unseen items:", int((~warm_mask).sum()))

    warm_norms = np.linalg.norm(
        behavior[warm_mask],
        axis=1,
    )

    warm_mean = behavior[warm_mask].mean(axis=0, keepdims=True)
    behavior[warm_mask] -= warm_mean

    norms = np.linalg.norm(
        behavior,
        axis=1,
        keepdims=True,
    )

    # Normalize the final behavior vectors
    # because we want direction to be more important than magnitude.
    # This should in theory help with popularity bias.
    behavior = np.divide(
        behavior,
        norms,
        out=np.zeros_like(behavior),
        where=norms > 0,
    )

    print("Warm norm mean:", float(warm_norms.mean()))

    np.save(DATA_DIR / "item_behavior_embeddings.npy", behavior)

    np.save(DATA_DIR / "item_behavior_warm_mask.npy", warm_mask)

    np.save(DATA_DIR / "item_train_counts.npy", train_counts)


if __name__ == "__main__":
    main()
