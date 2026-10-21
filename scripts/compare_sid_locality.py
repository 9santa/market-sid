from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np


DATA_DIR = Path("data/processed")

N_PAIRS = 20_000
SEED = 42


def sample_pairs(
    codes,
    embeddings,
    prefix_len,
    valid_mask=None,
    n_pairs=N_PAIRS,
    seed=SEED,
):
    rng = np.random.default_rng(seed + prefix_len)

    if valid_mask is None:
        valid_mask = np.ones(
            len(codes),
            dtype=bool,
        )

    groups = defaultdict(list)

    for idx, sid in enumerate(codes):
        if not valid_mask[idx]:
            continue

        groups[
            tuple(
                map(
                    int,
                    sid[:prefix_len],
                )
            )
        ].append(idx)

    usable = [indices for indices in groups.values() if len(indices) >= 2]

    if not usable:
        raise ValueError(f"No groups with two items for prefix length {prefix_len}")

    similarities = []

    while len(similarities) < n_pairs:
        group = usable[rng.integers(len(usable))]

        i, j = rng.choice(
            group,
            size=2,
            replace=False,
        )

        similarities.append(float(embeddings[i] @ embeddings[j]))

    return np.asarray(similarities)


def random_pairs(
    embeddings,
    valid_mask=None,
    n_pairs=N_PAIRS,
    seed=SEED,
):
    rng = np.random.default_rng(seed)

    if valid_mask is None:
        indices = np.arange(len(embeddings))
    else:
        indices = np.flatnonzero(valid_mask)

    if len(indices) < 2:
        raise ValueError("At least two eligible items are required")

    a = rng.choice(
        indices,
        size=n_pairs,
        replace=True,
    )

    b = rng.choice(
        indices,
        size=n_pairs,
        replace=True,
    )

    same = a == b

    while same.any():
        b[same] = rng.choice(
            indices,
            size=int(same.sum()),
            replace=True,
        )

        same = a == b

    return np.sum(
        embeddings[a] * embeddings[b],
        axis=1,
    )


def report(
    name,
    codes,
    content,
    behavior,
    warm_mask,
):
    semantic = codes[:, :3]

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("\nCONTENT LOCALITY")

    sims = random_pairs(content)

    print(f"random    {sims.mean():.4f}")

    for prefix_len in (1, 2, 3):
        sims = sample_pairs(
            semantic,
            content,
            prefix_len,
        )

        print(f"prefix={prefix_len} {sims.mean():.4f}")

    print("\nBEHAVIOR LOCALITY (warm only)")

    sims = random_pairs(
        behavior,
        valid_mask=warm_mask,
    )

    print(f"random    {sims.mean():.4f}")

    for prefix_len in (1, 2, 3):
        sims = sample_pairs(
            semantic,
            behavior,
            prefix_len,
            valid_mask=warm_mask,
        )

        print(f"prefix={prefix_len} {sims.mean():.4f}")


def sid_churn(
    base_codes,
    new_codes,
):
    base = base_codes[:, :3]
    new = new_codes[:, :3]

    print(
        "same c1:",
        np.mean(base[:, 0] == new[:, 0]),
    )

    print(
        "same c1,c2:",
        np.mean(
            np.all(
                base[:, :2] == new[:, :2],
                axis=1,
            )
        ),
    )

    print(
        "same full semantic SID:",
        np.mean(
            np.all(
                base == new,
                axis=1,
            )
        ),
    )


def report_svd():
    content = np.load(DATA_DIR / "item_text_embeddings.npy")

    behavior = np.load(DATA_DIR / "item_behavior_embeddings.npy")

    warm_mask = np.load(DATA_DIR / "item_behavior_warm_mask.npy").astype(bool)

    content_sid = np.load(DATA_DIR / "item_semantic_ids_content.npy")

    behavior_001 = np.load(DATA_DIR / "item_semantic_ids_behavior_0p01.npy")

    behavior_005 = np.load(DATA_DIR / "item_semantic_ids_behavior_0p05.npy")

    report(
        "CONTENT SID",
        content_sid,
        content,
        behavior,
        warm_mask,
    )

    report(
        "BEHAVIOR SID lambda=0.01",
        behavior_001,
        content,
        behavior,
        warm_mask,
    )

    report(
        "BEHAVIOR SID lambda=0.05",
        behavior_005,
        content,
        behavior,
        warm_mask,
    )

    print("\nSID CHURN .01")
    sid_churn(
        content_sid,
        behavior_001,
    )

    print("\nSID CHURN .05")
    sid_churn(
        content_sid,
        behavior_005,
    )


def report_locality(name, codes, embedding_name, embeddings):
    assert len(codes) == len(embeddings), "SID and embedding row counts differ"
    print(f"\n{name} -> {embedding_name}", flush=True)
    sims = random_pairs(embeddings)
    print(f"random    {sims.mean():.4f}", flush=True)
    for prefix_len in (1, 2, 3):
        sims = sample_pairs(codes[:, :3], embeddings, prefix_len)
        print(f"prefix={prefix_len} {sims.mean():.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--svd", action="store_true", help="Run the original SVD comparison")
    mode.add_argument("--content-only", action="store_true", help="Baseline before training Two-Tower RQ")
    args = parser.parse_args()
    if args.svd:
        report_svd()
        return

    content = np.load(DATA_DIR / "item_text_embeddings.npy")
    two_tower = np.load(DATA_DIR / "item_two_tower_embeddings.npy")
    content_sid = np.load(DATA_DIR / "item_semantic_ids_content.npy")
    report_locality("CONTENT SID", content_sid, "BGE", content)
    report_locality("CONTENT SID", content_sid, "TWO-TOWER", two_tower)
    if not args.content_only:
        two_tower_sid = np.load(DATA_DIR / "item_semantic_ids_twotower.npy")
        report_locality("TWO-TOWER SID", two_tower_sid, "BGE", content)
        report_locality("TWO-TOWER SID", two_tower_sid, "TWO-TOWER", two_tower)


if __name__ == "__main__":
    main()
