"""Evaluate GenRec retrieval against the full catalog using constrained beams."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.data.sasrec_dataset import _pad_left
from src.models.beam_search import (
    batched_constrained_beam_search,
    filter_seen_items,
)
from src.models.generative_recommender import GenerativeRecommender
from src.models.sid_trie import SIDTrie
from scripts.train_genrec import SID_SOURCES, file_hash


def catalog_prefix_counts(trie):
    codes = trie.item_sids[1:]
    return [len(np.unique(codes[:, :depth], axis=0)) for depth in range(1, codes.shape[1] + 1)]


def make_eval_records(
    sequences,
    train_counts,
    split,
):
    records = []

    for items in sequences:
        items = np.asarray(
            items,
            dtype=np.int64,
        )

        if split == "valid":
            history = items[:-2]
            target = int(items[-2])

        elif split == "test":
            history = items[:-1]
            target = int(items[-1])

        else:
            raise ValueError(split)

        count = int(train_counts[target])

        if count == 0:
            group = "unseen"
        elif count <= 5:
            group = "tail"
        elif count <= 20:
            group = "mid"
        else:
            group = "head"

        records.append(
            {
                "history": history,
                "target": target,
                "group": group,
            }
        )

    return records


@torch.inference_mode()
def evaluate_genrec(
    model,
    trie,
    records,
    max_len,
    device,
    beam_size=200,
    k_values=(10, 20, 50),
    batch_size=64,
    diagnose_prefixes=False,
    prior_alpha=0.0,
    prior_log_probs=None,
    prior_levels=None,
    trie_normalize=False,
):
    model.eval()

    groups = [
        "all",
        "warm",
        "tail",
        "mid",
        "head",
        "unseen",
    ]

    totals = {
        group: {
            "n": 0,
            **{f"hits@{k}": 0 for k in k_values},
            **{f"ndcg@{k}": 0.0 for k in k_values},
        }
        for group in groups
    }

    max_k = max(k_values)
    short_recommendations = 0
    if diagnose_prefixes:
        prefix_totals = {
            group: np.zeros(model.num_levels, dtype=np.int64) for group in groups
        }
        counts = catalog_prefix_counts(trie)
        print("Distinct catalog prefixes by position:", counts)
        print(f"Valid root branches: {counts[0]} | beam size: {beam_size}")
        if counts[0] <= beam_size:
            print("Beam covers every root branch; position 1 requires no pruning.")

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        batch = records[start : start + batch_size]

        histories = np.stack(
            [
                _pad_left(
                    row["history"],
                    max_len,
                )
                for row in batch
            ]
        )

        item_seq = torch.from_numpy(histories).to(device)

        target_sids = None
        if diagnose_prefixes:
            target_ids = torch.as_tensor([row["target"] for row in batch], device=device)
            target_sids = model.item_sids[target_ids]

        results = batched_constrained_beam_search(
            model=model,
            trie=trie,
            item_seq=item_seq,
            beam_size=beam_size,
            target_sids=target_sids,
            prior_alpha=prior_alpha,
            prior_log_probs=prior_log_probs,
            prior_levels=prior_levels,
            trie_normalize=trie_normalize,
        )
        if diagnose_prefixes:
            results, survival = results
            survival = survival.numpy()

        for row_idx, (row, generated) in enumerate(zip(batch, results)):
            recommendations = [
                item_id
                for item_id, _ in filter_seen_items(generated, row["history"], max_k)
            ]
            short_recommendations += len(recommendations) < max_k

            target = row["target"]
            group = row["group"]

            eval_groups = [
                "all",
                group,
            ]

            if group != "unseen":
                eval_groups.append("warm")

            for g in eval_groups:
                totals[g]["n"] += 1
                if diagnose_prefixes:
                    prefix_totals[g] += survival[row_idx]

            if target not in recommendations:
                continue

            rank = recommendations.index(target)

            for k in k_values:
                if rank >= k:
                    continue

                for g in eval_groups:
                    totals[g][f"hits@{k}"] += 1

                    totals[g][f"ndcg@{k}"] += 1.0 / math.log2(rank + 2)

        # print(f"Evaluated {start + len(batch):,}/{len(records):,} users", flush=True)

    metrics = {}

    for group, values in totals.items():
        n = values["n"]

        if n == 0:
            continue

        metrics[group] = {"n": n}

        for k in k_values:
            metrics[group][f"recall@{k}"] = values[f"hits@{k}"] / n

            metrics[group][f"ndcg@{k}"] = values[f"ndcg@{k}"] / n

        if diagnose_prefixes:
            for level, count in enumerate(prefix_totals[group], start=1):
                metrics[group][f"prefix_survival@{level}"] = float(count / n)

    print(f"Users with fewer than {max_k} recommendations: {short_recommendations}")

    if diagnose_prefixes:
        print("\nTarget prefix survival after pruning, before seen-item filtering")
        print("Rates use all targets in each group; full SID is the last position.")
        columns = [f"prefix {level}" for level in range(1, model.num_levels)] + ["full SID"]
        if 10 in k_values:
            columns.append("item top-10")
        print(f"{'group':<8} {'n':>8} " + " ".join(f"{name:>12}" for name in columns))
        for group in ("all", "head", "mid", "tail", "unseen"):
            values = metrics.get(group)
            if values is None:
                print(f"{group:<8} {0:>8} " + " ".join(f"{'N/A':>12}" for _ in columns))
                continue
            rates = [values[f"prefix_survival@{level}"] for level in range(1, model.num_levels + 1)]
            if 10 in k_values:
                rates.append(values["recall@10"])
            print(f"{group:<8} {values['n']:>8} " + " ".join(f"{rate:>12.2%}" for rate in rates))

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--beam-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N users")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trie-normalize", action="store_true",
                        help="Normalize beam probabilities over catalog-valid children")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--diagnose-prefixes", action="store_true")
    diagnostic.add_argument("--diagnose-paths", action="store_true", help="Score true SID paths without beam search")
    parser.add_argument("--prior-smoothing", type=float, default=0.5)
    correction = parser.add_mutually_exclusive_group()
    correction.add_argument("--prior-alpha", type=float, help="Fixed conditional-prior correction at selected SID levels")
    correction.add_argument("--prior-alpha-sweep", type=float, nargs="*", help="Sweep alphas (default: 0 .05 .1 .2 .3 .5)")
    parser.add_argument("--prior-levels", type=int, nargs="+", choices=(1, 2, 3, 4),
                        help="One-based SID positions to correct; default: all levels")
    parser.add_argument("--shuffle-histories", action="store_true", help="Compare target scores under another user's history")
    parser.add_argument(
        "--sid-source",
        choices=SID_SOURCES,
        default="content",
    )
    args = parser.parse_args()
    prior_requested = args.prior_alpha is not None or args.prior_alpha_sweep is not None
    if args.prior_levels is not None:
        if not prior_requested:
            parser.error("--prior-levels requires --prior-alpha or --prior-alpha-sweep")
        args.prior_levels = sorted(set(args.prior_levels))
    if args.prior_alpha_sweep is not None:
        alphas = args.prior_alpha_sweep or [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
        alphas = sorted(set([0.0, *alphas]))
    else:
        alphas = [args.prior_alpha if args.prior_alpha is not None else 0.0]
    if any(not math.isfinite(alpha) or alpha < 0 for alpha in alphas):
        parser.error("Prior alphas must be finite and nonnegative")
    if args.diagnose_paths and prior_requested:
        parser.error("Prior correction applies to beam search, not --diagnose-paths")
    if args.diagnose_paths and args.trie_normalize:
        parser.error("--trie-normalize applies to beam search, not --diagnose-paths")
    if (
        (not args.diagnose_paths and args.beam_size < 50)
        or args.batch_size <= 0
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("Require beam-size >= 50, batch-size > 0 and limit > 0")
    if not math.isfinite(args.prior_smoothing) or args.prior_smoothing <= 0:
        parser.error("Require finite, positive --prior-smoothing")
    if args.shuffle_histories and not args.diagnose_paths:
        parser.error("--shuffle-histories requires --diagnose-paths")
    if (args.diagnose_paths or prior_requested) and args.output and args.output.suffix != ".json":
        parser.error("Diagnostics and prior correction require a .json --output path")

    data_dir = Path("data/processed")
    sid_path = data_dir / f"item_semantic_ids_{args.sid_source}.npy"
    checkpoint_path = Path("checkpoints") / f"genrec_{args.sid_source}_C_best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("SID source:", args.sid_source)
    print("Checkpoint:", checkpoint_path)
    print("Probability normalization:", "trie-valid children" if args.trie_normalize else "full vocabulary")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("sid_source", "content") != args.sid_source:
        parser.error("Checkpoint SID source does not match --sid-source")
    config = checkpoint["config"]
    vocab_sizes = checkpoint["vocab_sizes"]
    num_semantic_levels = checkpoint.get("num_semantic_levels", len(vocab_sizes) - 1)
    data_signature = {
        name: file_hash(data_dir / name)
        for name in ("catalog_model.parquet", "user_sequences.parquet", "interactions_model.parquet")
    }
    if checkpoint.get("data_signature", data_signature) != data_signature:
        raise ValueError("Dataset changed since GenRec training")
    state = checkpoint["model_state_dict"]
    variant_config = checkpoint.get("variant_config", {})
    embedding_mode = variant_config.get("embedding_mode", "scratch")
    rq_codebooks = None
    if embedding_mode == "rq_init":
        rq_codebooks = [
            state[f"sid_embeddings.{level}.weight"]
            for level in range(num_semantic_levels)
        ]
    elif embedding_mode == "rq_anchor":
        rq_codebooks = [
            state[f"rq_base_{level}"] for level in range(num_semantic_levels)
        ]
    # item_sids is a nonpersistent buffer, reconstructed from the exported file.
    # Older checkpoints have no SID checksum; keep those checkpoints loadable.
    sid_sha256 = file_hash(sid_path)
    if checkpoint.get("sid_sha256", sid_sha256) != sid_sha256:
        raise ValueError("SID file differs from the one used for GenRec training")
    semantic_ids = np.load(sid_path)
    catalog = pl.read_parquet(
        data_dir / "catalog_model.parquet", columns=["item_idx"]
    ).sort("item_idx")
    num_items = len(catalog)
    assert catalog["item_idx"].to_list() == list(range(1, num_items + 1))
    assert semantic_ids.shape == (num_items, len(vocab_sizes))
    assert np.all(semantic_ids >= 0) and np.all(semantic_ids < np.asarray(vocab_sizes))
    assert len(np.unique(semantic_ids, axis=0)) == num_items, "SID collisions"

    item_sids = np.zeros((num_items + 1, len(vocab_sizes)), dtype=np.int64)
    item_sids[1:] = semantic_ids
    trie = SIDTrie(item_sids)
    model = GenerativeRecommender(
        item_sids=torch.from_numpy(item_sids),
        vocab_sizes=vocab_sizes,
        max_len=config["max_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        dropout=config["dropout"],
        embedding_mode=embedding_mode,
        tie_semantic_output=variant_config.get("tie_semantic_output", False),
        rq_codebooks=rq_codebooks,
        num_semantic_levels=num_semantic_levels,
    ).to(device)
    model.load_state_dict(state)

    interactions = pl.read_parquet(
        data_dir / "interactions_model.parquet", columns=["item_idx", "split"]
    )
    train_ids = interactions.filter(pl.col("split") == "train")["item_idx"].to_numpy()
    train_counts = np.bincount(train_ids.astype(np.int64), minlength=num_items + 1)
    users = pl.read_parquet(data_dir / "user_sequences.parquet").sort("user_idx")
    if args.diagnose_paths or prior_requested:
        # Build the prior from the full training population even for --limit runs.
        training_sequences = users["item_sequence"].to_list()
    if args.limit is not None:
        users = users.head(args.limit)
    for row in users.iter_rows(named=True):
        assert row["split_sequence"] == ["train"] * (len(row["item_sequence"]) - 2) + [
            "valid",
            "test",
        ], f"Invalid split order for user_idx={row['user_idx']}"
    records = make_eval_records(
        users["item_sequence"].to_list(), train_counts, args.split
    )
    print(
        f"Split: {args.split} | users: {len(records):,} | full catalog: {num_items:,} items"
    )

    if args.diagnose_paths:
        from scripts.diagnose_genrec_paths import diagnose_paths

        suffix = f"_limit{args.limit}" if args.limit is not None else ""
        output = args.output or Path(f"artifacts/main/diagnostics/paths/genrec_{args.sid_source}_C_{args.split}{suffix}_path_diagnostics.json")
        diagnose_paths(
            model, records, training_sequences, users["user_idx"].to_list(),
            config["max_len"], device, output,
            metadata={
                "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
                "sid_source": args.sid_source, "sid_sha256": sid_sha256,
                "split": args.split, "config": config, "data_signature": data_signature,
            },
            batch_size=args.batch_size, smoothing=args.prior_smoothing,
            shuffle_histories=args.shuffle_histories,
        )
        return

    if prior_requested:
        from scripts.sweep_genrec_priors import evaluate_prior_sweep

        suffix = f"_limit{args.limit}" if args.limit is not None else ""
        name = "prior_sweep" if args.prior_alpha_sweep is not None else f"prior_alpha{alphas[0]:g}"
        if args.prior_levels is not None:
            name += "_levels" + "-".join(map(str, args.prior_levels))
        if args.trie_normalize:
            name += "_trie_normalized"
        output = args.output or Path(f"artifacts/main/diagnostics/decoding/genrec_{args.sid_source}_C_{args.split}{suffix}_{name}.json")
        evaluate_prior_sweep(
            model, trie, records, training_sequences, config["max_len"], device,
            alphas, output, beam_size=args.beam_size, batch_size=args.batch_size,
            smoothing=args.prior_smoothing, diagnose_prefixes=args.diagnose_prefixes,
            prior_levels=args.prior_levels,
            trie_normalize=args.trie_normalize,
            metadata={
                "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
                "sid_source": args.sid_source, "sid_sha256": sid_sha256,
                "split": args.split, "config": config, "data_signature": data_signature,
            },
        )
        return

    metrics = evaluate_genrec(
        model,
        trie,
        records,
        config["max_len"],
        device,
        beam_size=args.beam_size,
        batch_size=args.batch_size,
        diagnose_prefixes=args.diagnose_prefixes,
        trie_normalize=args.trie_normalize,
    )
    print(json.dumps(metrics, indent=2))

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    if args.diagnose_prefixes:
        suffix += "_prefix_survival"
    if args.trie_normalize:
        suffix += "_trie_normalized"
    output = args.output or Path(
        f"artifacts/main/evaluation/genrec_{args.sid_source}_C_{args.split}{suffix}_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint["epoch"],
                "sid_source": args.sid_source,
                "genrec_variant": checkpoint.get("variant", "C"),
                "config": config,
                "variant_config": variant_config,
                "sid_control": checkpoint.get("sid_control"),
                "initialization_sha256": checkpoint.get("initialization_sha256"),
                "rq_checkpoint_sha256": checkpoint.get("rq_checkpoint_sha256"),
                "early_stopping_enabled": checkpoint.get("early_stopping_enabled", True),
                "training_history": checkpoint.get("history_file"),
                "num_semantic_levels": num_semantic_levels,
                "data_signature": data_signature,
                "catalog_prefix_counts": catalog_prefix_counts(trie) if args.diagnose_prefixes else None,
                "sid_path": str(sid_path),
                "sid_sha256": sid_sha256,
                "split": args.split,
                "candidate_catalog": "all",
                "num_users": len(records),
                "evaluation_users_sha256": hashlib.sha256(
                    np.asarray(users["user_idx"].to_list(), dtype="<i8").tobytes()
                ).hexdigest(),
                "user_subset": "first_n" if args.limit is not None else "all",
                "beam_size": args.beam_size,
                "batch_size": args.batch_size,
                "diagnose_prefixes": args.diagnose_prefixes,
                "trie_normalize": args.trie_normalize,
                "prior_alpha": 0.0,
                "metrics": metrics,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()
