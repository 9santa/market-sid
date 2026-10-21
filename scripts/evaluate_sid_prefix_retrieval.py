"""
Prefix-level retrieval diagnostics and candidate expansion for frozen GenRec-C.

Can GenRec predict correct semantic area for tail/unseen items, even if
it cannot output its full SID into top-k?

Compare raw and size-normalized prefix scores, quota hybrids, and train-only
popularity controls. Example (run from the repository root):
python -m scripts.evaluate_sid_prefix_retrieval --depths 2 3 --limit 10000 \
    --hybrid-prefix-slots 10 --output artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid10000_v2.json
"""

import argparse
import json
import math
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

from scripts.evaluate_genrec import make_eval_records
from scripts.train_genrec import SID_SOURCES, file_hash
from src.data.sasrec_dataset import _pad_left
from src.data.embedding_cache import embedding_metadata, load_embedding_cache
from src.models.beam_search import (
    batched_constrained_beam_search,
    filter_seen_items,
)
from src.models.generative_recommender import GenerativeRecommender
from src.models.content_two_tower import ContentTwoTower
from src.models.sid_trie import SIDTrie


K_VALUES = (10, 20, 50)
POP_GROUPS = ("all", "head", "mid", "tail", "unseen")


def load_system(sid_source, device):
    """Load the model and the matching SID catalog."""
    data_dir = Path("data/processed")
    sid_path = data_dir / f"item_semantic_ids_{sid_source}.npy"
    checkpoint_path = Path("checkpoints") / f"genrec_{sid_source}_C_best.pt"

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if ckpt.get("sid_source", "content") != sid_source:
        raise ValueError("Checkpoint SID source mismatch")

    signature = {
        name: file_hash(data_dir / name)
        for name in (
            "catalog_model.parquet",
            "user_sequences.parquet",
            "interactions_model.parquet",
        )
    }

    if ckpt.get("data_signature", signature) != signature:
        raise ValueError("Dataset differs from checkpoint")

    sid_hash = file_hash(sid_path)
    if ckpt.get("sid_sha256", sid_hash) != sid_hash:
        raise ValueError("SID file differs from checkpoint")

    sids = np.load(sid_path)
    catalog = pl.read_parquet(
        data_dir / "catalog_model.parquet",
        columns=["item_idx"],
    ).sort("item_idx")

    if catalog["item_idx"].to_list() != list(range(1, len(catalog) + 1)):
        raise ValueError("Unexpected item indexing")

    vocab = ckpt["vocab_sizes"]

    if sids.shape != (len(catalog), len(vocab)):
        raise ValueError("SID shape mismatch")

    if np.any(sids < 0) or np.any(sids >= np.asarray(vocab)):
        raise ValueError("Invalid (out of range) SID token")

    item_sids = np.zeros(
        (len(catalog) + 1, len(vocab)),
        dtype=np.int64,
    )
    item_sids[1:] = sids

    trie = SIDTrie(item_sids)

    config = ckpt["config"]
    variant = ckpt.get("variant_config", {})
    mode = variant.get("embedding_mode", "scratch")
    state = ckpt["model_state_dict"]

    n_semantic = ckpt.get(
        "num_semantic_levels",
        len(vocab) - 1,
    )

    rq_codebooks = None

    if mode == "rq_init":
        rq_codebooks = [state[f"sid_embeddings.{l}.weight"] for l in range(n_semantic)]

    elif mode == "rq_anchor":
        rq_codebooks = [state[f"rq_base_{l}"] for l in range(n_semantic)]

    model = GenerativeRecommender(
        item_sids=torch.from_numpy(item_sids),
        vocab_sizes=vocab,
        max_len=config["max_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        dropout=config["dropout"],
        embedding_mode=mode,
        tie_semantic_output=variant.get(
            "tie_semantic_output",
            False,
        ),
        rq_codebooks=rq_codebooks,
        num_semantic_levels=n_semantic,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    train_items = (
        pl.read_parquet(
            data_dir / "interactions_model.parquet",
            columns=["item_idx", "split"],
        )
        .filter(pl.col("split") == "train")["item_idx"]
        .to_numpy()
    )

    train_counts = np.bincount(
        train_items,
        minlength=len(item_sids),
    )

    provenance = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_sha256": file_hash(checkpoint_path),
        "data_signature": signature,
        "sid_sha256": sid_hash,
        "sid_control": ckpt.get("sid_control"),
        "initialization_sha256": ckpt.get("initialization_sha256"),
        "rq_checkpoint_sha256": ckpt.get("rq_checkpoint_sha256"),
    }
    return model, trie, train_counts, config["max_len"], data_dir, provenance


def build_prefix_index(item_sids, depths):
    """Build prefix -> catalog item IDs."""
    index = {d: defaultdict(list) for d in depths}

    for item_id in range(1, len(item_sids)):
        for d in depths:
            prefix = tuple(map(int, item_sids[item_id, :d]))
            index[d][prefix].append(item_id)

    return {
        d: {p: np.asarray(items, dtype=np.int64) for p, items in groups.items()}
        for d, groups in index.items()
    }


@torch.inference_mode()
def load_two_tower(path, data_dir, device, data_signature):
    """Load both learned towers and build candidate vectors from this checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    catalog = pl.read_parquet(data_dir / "catalog_model.parquet").sort("item_idx")
    mapping_hash = hashlib.sha256(json.dumps(
        catalog.select("item_idx", "parent_asin").rows()
    ).encode("utf-8")).hexdigest()
    if checkpoint.get("mapping_hash") != mapping_hash:
        raise ValueError("Two-Tower checkpoint/catalog mapping mismatch")
    if checkpoint.get("data_signature", data_signature) != data_signature:
        raise ValueError("Two-Tower checkpoint/training dataset mismatch")
    embedding_path = data_dir / "item_text_embeddings.npy"
    embeddings = load_embedding_cache(embedding_path, embedding_metadata(catalog))
    features = np.vstack([np.zeros((1, embeddings.shape[1]), dtype=np.float32), embeddings])
    config = checkpoint["config"]
    model = ContentTwoTower(
        item_features=torch.from_numpy(features),
        **{k: config[k] for k in ("max_len", "d_model", "n_heads", "n_layers", "dropout", "temperature")},
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    vectors = torch.cat([
        model.item_vectors(torch.arange(start, min(start + 8192, len(features)), device=device))
        for start in range(0, len(features), 8192)
    ])
    if not torch.isfinite(vectors).all():
        raise ValueError("Non-finite Two-Tower item vectors")
    return model, vectors, {
        "checkpoint": str(path), "checkpoint_sha256": file_hash(path),
        "checkpoint_epoch": checkpoint["epoch"], "config": config,
        "mapping_hash": mapping_hash, "embeddings_sha256": file_hash(embedding_path),
        # Older checkpoints bind item IDs, but do not fingerprint training data/features.
        "training_data_signature_verified": "data_signature" in checkpoint,
    }


def shuffled_history_indices(n, seed):
    """A seeded random cycle: every selected history is reused once, no self-pairs."""
    if n < 2:
        raise ValueError("Shuffled-history control needs at least two users")
    order = np.random.default_rng(seed).permutation(n)
    donors = np.empty(n, dtype=np.int64)
    donors[order] = np.roll(order, 1)
    return donors


def valid_beam_prefixes(snapshot, user_idx):
    """Ignore artificial beam slots with -inf scores."""
    prefixes = snapshot["prefixes"][user_idx].tolist()
    scores = snapshot["scores"][user_idx].tolist()

    return [
        (tuple(prefix), float(score))
        for prefix, score in zip(prefixes, scores)
        if math.isfinite(score)
    ]


def build_popularity_control(prefix_index, train_counts):
    """Match both train frequency and catalog size within factor-two bins."""
    counts = {p: int(train_counts[ids].sum()) for p, ids in prefix_index.items()}
    strata = defaultdict(list)
    prefix_bin = {}
    for p, ids in prefix_index.items():
        # Zero frequency has its own bin; positive bins are [1,2), [2,4), ...
        key = (counts[p].bit_length(), len(ids).bit_length())
        strata[key].append(p)
        prefix_bin[p] = key
    return {
        "top": sorted(counts, key=lambda p: (-counts[p], p)),
        "bins": {key: sorted(values) for key, values in strata.items()},
        "prefix_bin": prefix_bin,
    }


def matched_prefixes(beams, control, seed, user_id, depth):
    """Sample without targets/history; preserve the predicted bin histogram."""
    quotas = defaultdict(int)
    for p, _ in beams:
        quotas[control["prefix_bin"][p]] += 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, user_id, depth]))
    selected = []
    for key, n in sorted(quotas.items()):
        pool = control["bins"][key]
        selected.extend(pool[i] for i in rng.choice(len(pool), n, replace=False))
    # Original prefixes remain eligible: excluding them would bias the null.
    return [(p, 0.0) for p in selected]


def expand_candidates(beam_prefixes, prefix_index, seen):
    """Expand prefixes; normalization uses bucket sizes BEFORE seen filtering."""
    item_parts = []
    score_parts = []
    log_size_parts = []

    for prefix, score in beam_prefixes:
        ids = prefix_index[prefix]

        item_parts.append(ids)
        score_parts.append(np.full(len(ids), score, dtype=np.float32))
        log_size_parts.append(np.full(len(ids), math.log(len(ids)), dtype=np.float32))

    if not item_parts:
        return (
            np.empty(0, np.int64),
            np.empty(0, np.float32),
            np.empty(0, np.float32),
        )

    ids = np.concatenate(item_parts)
    scores = np.concatenate(score_parts)
    log_sizes = np.concatenate(log_size_parts)

    if len(ids) != len(np.unique(ids)):
        raise AssertionError("Duplicate candidates")

    if seen:
        keep = ~np.isin(
            ids,
            np.asarray(sorted(seen), dtype=np.int64),
        )
        ids = ids[keep]
        scores = scores[keep]
        log_sizes = log_sizes[keep]

    return ids, scores, log_sizes


def pool_size_only(
    beam_prefixes,
    prefix_index,
    item_sids,
    seen,
):
    """Compute pool size without materializing all items."""
    retained = {prefix for prefix, _ in beam_prefixes}

    size = sum(len(prefix_index[prefix]) for prefix in retained)

    depth = len(beam_prefixes[0][0]) if beam_prefixes else 0

    size -= sum(
        tuple(map(int, item_sids[i, :depth])) in retained for i in seen if i > 0
    )

    return size


def load_normalized_content(path, n_items):
    x = np.load(path).astype(np.float32)

    if x.ndim != 2 or len(x) != n_items - 1 or not np.isfinite(x).all():
        raise ValueError("Content/catalog size mismatch")

    x /= np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12,
    )

    padded = np.zeros(shape=(n_items, x.shape[1]), dtype=np.float32)
    padded[1:] = x

    return padded


def user_profile(content, history, window):
    history = np.asarray(
        history[-window:],
        dtype=np.int64,
    )

    profile = content[history].mean(axis=0)

    return profile / max(
        float(np.linalg.norm(profile)),
        1e-12,
    )


def top_k(ids, scores, k=50):
    if not len(ids):
        return []

    # Higher score first, item ID resolves exact ties
    order = np.lexsort((ids, -scores))
    return ids[order[:k]].tolist()


def hybrid_ranking(exact, expanded, slots, k=50):
    """Keep the exact top-(k-slots); add at most slots items outside exact top-k."""
    ranking = exact[:k - slots]
    exact_set = set(exact)
    additions = [i for i in expanded if i not in exact_set][:slots]
    ranking = ranking + additions
    # If expansion has too few novel items, restore the remaining exact results.
    used = set(ranking)
    return (ranking + [i for i in exact if i not in used])[:k]


def metrics_store(methods, depths):
    return {
        "rank": {
            m: {
                g: {
                    "n": 0,
                    "seen_targets": 0,
                    "short_lists": 0,
                    "coverage_n": 0,
                    "candidate_hits": 0,
                    **{f"hit@{k}": 0 for k in K_VALUES},
                    **{f"dcg@{k}": 0.0 for k in K_VALUES},
                }
                for g in POP_GROUPS
            }
            for m in methods
        },
        "prefix": {
            d: {
                g: {
                    "n": 0,
                    "survived": 0,
                    "candidate_hits": 0,
                    "sizes": [],
                    "paired_n": 0,
                    "learned_only": 0,
                    "control_only": 0,
                    "overlap": 0.0,
                }
                for g in POP_GROUPS
            }
            for d in depths
        },
    }


def relevant_groups(group):
    return ("all", group)


def update_ranking(store, method, group, ranking, target, seen_target, covered=None):
    # One-based rank; zero means the target is absent from the top-50.
    rank = ranking.index(target) + 1 if target in ranking else 0
    for g in relevant_groups(group):
        s = store["rank"][method][g]
        s["n"] += 1
        s["seen_targets"] += int(seen_target)
        s["short_lists"] += int(len(ranking) < 50)
        if covered is not None:
            s["coverage_n"] += 1
            s["candidate_hits"] += int(covered)

        for k in K_VALUES:
            if 0 < rank <= k:
                s[f"hit@{k}"] += 1
                s[f"dcg@{k}"] += 1.0 / math.log2(rank + 1)
    return rank


def paired_ranking_comparison(ranks, groups, method, reference):
    """Compare the same users; intervals describe user sampling, not training seeds."""
    result = {"method": method, "reference": reference, "groups": {}}
    for group in POP_GROUPS:
        mask = np.ones(len(groups), dtype=bool) if group == "all" else groups == group
        a, b = ranks[method][mask], ranks[reference][mask]
        n = len(a)
        if not n:
            continue
        values = {"n": n}
        for k in K_VALUES:
            hit_a, hit_b = (a > 0) & (a <= k), (b > 0) & (b <= k)
            values[f"method_only_hits@{k}"] = int(np.sum(hit_a & ~hit_b))
            values[f"reference_only_hits@{k}"] = int(np.sum(hit_b & ~hit_a))
            for metric in ("recall", "ndcg"):
                score_a, score_b = hit_a.astype(float), hit_b.astype(float)
                if metric == "ndcg":
                    score_a /= np.log2(np.maximum(a.astype(float), 1) + 1)
                    score_b /= np.log2(np.maximum(b.astype(float), 1) + 1)
                delta = score_a - score_b
                mean = float(delta.mean())
                se = float(delta.std(ddof=1) / math.sqrt(n)) if n > 1 else None
                values[f"{metric}@{k}"] = {
                    "delta": mean,
                    "standard_error": se,
                    "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
                    "nonzero_differences": int(np.count_nonzero(delta)),
                }
        result["groups"][group] = values
    return result


def update_prefix(store, depth, group, survived, covered, pool_size,
                  learned_covered=None, overlap=0.0):
    for g in relevant_groups(group):
        s = store["prefix"][depth][g]
        s["n"] += 1
        s["survived"] += int(survived)
        s["candidate_hits"] += int(covered)
        s["sizes"].append(pool_size)
        if learned_covered is not None:
            s["paired_n"] += 1
            s["learned_only"] += int(learned_covered and not covered)
            s["control_only"] += int(covered and not learned_covered)
            s["overlap"] += overlap


def summarize(store):
    ranking = {}

    for method, groups in store["rank"].items():
        ranking[method] = {}

        for group, s in groups.items():
            if not s["n"]:
                continue

            ranking[method][group] = {
                "n": s["n"],
                "seen_targets": s["seen_targets"],
                "short_lists": s["short_lists"],
                **{f"hits@{k}": s[f"hit@{k}"] for k in K_VALUES},
                **{f"recall@{k}": s[f"hit@{k}"] / s["n"] for k in K_VALUES},
                **{f"ndcg@{k}": s[f"dcg@{k}"] / s["n"] for k in K_VALUES},
            }
            if s["coverage_n"]:
                ranking[method][group].update({
                    "candidate_hits": s["candidate_hits"],
                    "candidate_coverage": s["candidate_hits"] / s["coverage_n"],
                    **{f"recall@{k}_given_covered":
                       s[f"hit@{k}"] / s["candidate_hits"] if s["candidate_hits"] else None
                       for k in K_VALUES},
                })

    prefix = {}

    for depth, groups in store["prefix"].items():
        prefix[str(depth)] = {}

        for group, s in groups.items():
            if not s["n"]:
                continue

            prefix[str(depth)][group] = {
                "n": s["n"],
                "candidate_hits": s["candidate_hits"],
                "prefix_survival": s["survived"] / s["n"],
                "candidate_coverage": s["candidate_hits"] / s["n"],
                "mean_pool_size": float(np.mean(s["sizes"])),
                "median_pool_size": float(np.median(s["sizes"])),
            }
            if s["paired_n"]:
                prefix[str(depth)][group].update({
                    "learned_only_hits": s["learned_only"],
                    "control_only_hits": s["control_only"],
                    "coverage_delta_learned_minus_control":
                        (s["learned_only"] - s["control_only"]) / s["paired_n"],
                    "mean_prefix_overlap_with_learned": s["overlap"] / s["paired_n"],
                })

    return {
        "ranking": ranking,
        "prefix_diagnostics": prefix,
    }


@torch.inference_mode()
def run(args):
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control_seeds = sorted(set(args.control_seeds or [args.control_seed]))
    slot_budgets = sorted(set(args.hybrid_slot_sweep or [args.hybrid_prefix_slots]))

    def hybrid_name(depth, variant, slots):
        name = f"hybrid{depth}_{variant}"
        return f"{name}_slots{slots}" if len(slot_budgets) > 1 else name

    model, trie, train_counts, max_len, data_dir, provenance = load_system(args.sid_source, device)

    depths = sorted(set(args.depths))

    if not depths or min(depths) < 1 or max(depths) >= model.num_levels:
        raise ValueError("Invalid prefix depths")

    prefix_index = build_prefix_index(trie.item_sids, depths)

    users = pl.read_parquet(data_dir / "user_sequences.parquet").sort("user_idx")

    for row in users.iter_rows(named=True):
        seq = row["item_sequence"]
        expected_splits = ["train"] * (len(seq) - 2) + ["valid", "test"]
        if len(seq) < 4 or row["split_sequence"] != expected_splits:
            raise ValueError("Invalid user split")

    # Count actual next-item training targets across ALL users, before subsampling.
    target_counts = np.zeros(len(trie.item_sids), dtype=np.int64)
    for seq in users["item_sequence"]:
        np.add.at(target_counts, np.asarray(seq[1:-2], dtype=np.int64), 1)
    controls = {d: build_popularity_control(prefix_index[d], target_counts) for d in depths}
    matched_names = {
        seed: f"matched_train_size_seed{seed}" if len(control_seeds) > 1 else "matched_train_size"
        for seed in control_seeds
    }
    control_stores = {name: metrics_store([], depths) for name in ("top_train", *matched_names.values())}
    shuffle_store = metrics_store([], depths) if args.shuffle_histories else None

    if args.limit:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(
            len(users),
            size=min(args.limit, len(users)),
            replace=False,
        )
        selected = np.zeros(len(users), dtype=bool)
        selected[chosen] = True
        users = users.filter(pl.Series(selected))

    records = make_eval_records(
        users["item_sequence"].to_list(),
        train_counts,
        args.split,
    )
    donor_indices = shuffled_history_indices(len(records), args.shuffle_seed) if args.shuffle_histories else None

    content = None
    content_path = args.content_embeddings or data_dir / "item_text_embeddings.npy"
    if not args.coverage_only:
        content = load_normalized_content(
            content_path,
            len(trie.item_sids),
        )

    methods = ["exact"]
    tower, item_vectors, tower_metadata = None, None, None
    if args.two_tower_checkpoint:
        tower, item_vectors, tower_metadata = load_two_tower(
            args.two_tower_checkpoint, data_dir, device, provenance["data_signature"]
        )
        methods.append("two_tower_full")
        methods.extend(f"hybrid_full_two_tower_slots{slots}" for slots in slot_budgets)

    if not args.coverage_only:
        for d in depths:
            methods.extend(
                (
                    f"prefix{d}_only",
                    f"prefix{d}_content",
                    f"prefix{d}_normalized",
                    f"prefix{d}_normalized_content",
                )
            )
            variants = ["normalized", "normalized_content"]
            if tower is not None:
                methods.append(f"prefix{d}_two_tower")
                variants.append("two_tower")
            methods.extend(hybrid_name(d, variant, slots)
                           for variant in variants for slots in slot_budgets)

    store = metrics_store(methods, depths)
    user_ranks = {m: np.zeros(len(records), dtype=np.uint8) for m in methods}
    user_groups = np.asarray([r["group"] for r in records])
    all_item_ids = np.arange(1, len(trie.item_sids), dtype=np.int64)
    timings = {k: 0.0 for k in ("genrec", "shuffled_genrec", "two_tower_full_scoring", "ranking_and_controls")}
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    evaluation_started = time.perf_counter()

    for start in range(0, len(records), args.batch_size):
        batch_started = time.perf_counter()
        batch = records[start : start + args.batch_size]

        histories = np.stack([_pad_left(r["history"], max_len) for r in batch])

        inputs = torch.from_numpy(histories).to(device)

        exact, snapshots = batched_constrained_beam_search(
            model,
            trie,
            inputs,
            beam_size=args.beam_size,
            decode_batch_size=args.decode_batch_size,
            capture_prefixes=True,
        )
        timings["genrec"] += time.perf_counter() - batch_started

        shuffled_snapshots = None
        if donor_indices is not None:
            phase_started = time.perf_counter()
            donor_histories = np.stack([
                _pad_left(records[i]["history"], max_len)
                for i in donor_indices[start:start + len(batch)]
            ])
            _, shuffled_snapshots = batched_constrained_beam_search(
                model, trie, torch.from_numpy(donor_histories).to(device),
                beam_size=args.beam_size, decode_batch_size=args.decode_batch_size,
                capture_prefixes=True,
            )
            timings["shuffled_genrec"] += time.perf_counter() - phase_started

        tower_scores = None
        if tower is not None:
            phase_started = time.perf_counter()
            tower_inputs = torch.from_numpy(np.stack([
                _pad_left(r["history"], tower.max_len) for r in batch
            ])).to(device)
            # Reuse identical scores for full-catalog and prefix-restricted ranking.
            tower_scores = (tower.encode(tower_inputs) @ item_vectors.T).cpu().numpy()
            if not np.isfinite(tower_scores).all():
                raise ValueError("Non-finite Two-Tower scores")
            timings["two_tower_full_scoring"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()

        for j, record in enumerate(batch):
            history = record["history"]
            target = record["target"]
            group = record["group"]

            seen = set(map(int, history))
            seen_target = target in seen

            baseline = [
                i
                for i, _ in filter_seen_items(
                    exact[j],
                    seen,
                    50,
                )
            ]

            def record_ranking(method, ranking, covered=None):
                user_ranks[method][start + j] = update_ranking(
                    store, method, group, ranking, target, seen_target, covered
                )

            record_ranking("exact", baseline)
            if tower_scores is not None:
                eligible = all_item_ids[~np.isin(all_item_ids, list(seen))]
                # Retrieve beyond top-50 so overlap with exact cannot exhaust additions.
                ranking = top_k(eligible, tower_scores[j, eligible], k=100)
                record_ranking("two_tower_full", ranking[:50])
                for slots in slot_budgets:
                    record_ranking(f"hybrid_full_two_tower_slots{slots}",
                                   hybrid_ranking(baseline, ranking, slots))

            if content is not None:
                profile = user_profile(
                    content,
                    history,
                    args.history_window,
                )

            for d in depths:
                beams = valid_beam_prefixes(snapshots[d - 1], j)

                target_prefix = tuple(map(int, trie.item_sids[target, :d]))

                survived = any(p == target_prefix for p, _ in beams)

                if args.coverage_only:
                    size = pool_size_only(
                        beams,
                        prefix_index[d],
                        trie.item_sids,
                        seen,
                    )
                    covered = survived and not seen_target

                else:
                    ids, prefix_scores, log_sizes = expand_candidates(
                        beams,
                        prefix_index[d],
                        seen,
                    )

                    size = len(ids)
                    covered = bool(np.any(ids == target))

                    content_scores = args.content_weight * (content[ids] @ profile)
                    scores = {
                        "only": prefix_scores,
                        "content": prefix_scores + content_scores,
                        "normalized": prefix_scores - log_sizes,
                        "normalized_content": prefix_scores - log_sizes + content_scores,
                    }
                    if tower_scores is not None:
                        # Pure learned ranking isolates candidate retrieval from score mixing.
                        scores["two_tower"] = tower_scores[j, ids]
                    for variant, values in scores.items():
                        # Up to 50 exact items can overlap the expanded ranking.
                        ranking = top_k(ids, values, k=100)
                        record_ranking(f"prefix{d}_{variant}", ranking[:50], covered)
                        if variant.startswith("normalized") or variant == "two_tower":
                            for slots in slot_budgets:
                                hybrid = hybrid_ranking(baseline, ranking, slots)
                                record_ranking(hybrid_name(d, variant, slots), hybrid)

                update_prefix(
                    store,
                    d,
                    group,
                    survived,
                    covered,
                    size,
                )

                control_beams = {
                    "top_train": [(p, 0.0) for p in controls[d]["top"][:len(beams)]],
                    **{matched_names[seed]: matched_prefixes(
                        beams, controls[d], seed,
                        int(users["user_idx"][start + j]), d,
                    ) for seed in control_seeds},
                }
                learned_set = {p for p, _ in beams}
                for name, selected in control_beams.items():
                    selected_set = {p for p, _ in selected}
                    selected_target = target_prefix in selected_set
                    control_size = pool_size_only(selected, prefix_index[d], trie.item_sids, seen)
                    update_prefix(
                        control_stores[name], d, group, selected_target,
                        selected_target and not seen_target, control_size,
                        learned_covered=covered,
                        overlap=len(learned_set & selected_set) / max(1, len(learned_set)),
                    )
                if shuffled_snapshots is not None:
                    selected = valid_beam_prefixes(shuffled_snapshots[d - 1], j)
                    selected_set = {p for p, _ in selected}
                    selected_target = target_prefix in selected_set
                    # Filtering and targets belong to the recipient, never the donor.
                    size = pool_size_only(selected, prefix_index[d], trie.item_sids, seen)
                    update_prefix(shuffle_store, d, group, selected_target,
                                  selected_target and not seen_target, size,
                                  learned_covered=covered,
                                  overlap=len(learned_set & selected_set) / max(1, len(learned_set)))

        timings["ranking_and_controls"] += time.perf_counter() - phase_started
        print(
            f"Evaluated {start + len(batch)}/{len(records)}",
            flush=True,
        )

    control_reports = {}
    for name, control_store in control_stores.items():
        summary = summarize(control_store)["prefix_diagnostics"]
        for groups in summary.values():
            for values in groups.values():
                values["target_prefix_selected"] = values.pop("prefix_survival")
        control_reports[name] = summary

    matched_summary = {}
    for d in depths:
        matched_summary[str(d)] = {}
        for group in control_reports[matched_names[control_seeds[0]]][str(d)]:
            values = [control_reports[matched_names[seed]][str(d)][group] for seed in control_seeds]
            matched_summary[str(d)][group] = {
                "n": values[0]["n"], "seeds": control_seeds,
                "mean_candidate_coverage": float(np.mean([v["candidate_coverage"] for v in values])),
                "std_candidate_coverage_across_seeds": float(np.std([v["candidate_coverage"] for v in values])),
                "mean_pool_size": float(np.mean([v["mean_pool_size"] for v in values])),
            }

    shuffle_report = summarize(shuffle_store)["prefix_diagnostics"] if shuffle_store is not None else None
    if shuffle_report is not None:
        for groups in shuffle_report.values():
            for values in groups.values():
                values["target_prefix_selected"] = values.pop("prefix_survival")

    paired_comparisons = []
    if tower is not None:
        for slots in slot_budgets:
            full = f"hybrid_full_two_tower_slots{slots}"
            paired_comparisons.append(paired_ranking_comparison(user_ranks, user_groups, full, "exact"))
            for d in depths:
                paired_comparisons.append(paired_ranking_comparison(
                    user_ranks, user_groups, hybrid_name(d, "two_tower", slots), full
                ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    per_user_path = args.output.with_name(args.output.stem + "_users.npz")
    np.savez_compressed(
        per_user_path,
        user_idx=users["user_idx"].to_numpy(),
        target=np.asarray([r["target"] for r in records], dtype=np.int64),
        group=user_groups,
        methods=np.asarray(methods),
        target_rank=np.column_stack([user_ranks[m] for m in methods]),
    )

    report = {
        "schema_version": 4,
        **provenance,
        "split": args.split,
        "sid_source": args.sid_source,
        "users": len(records),
        "candidate_catalog": "all",
        "filter_seen": "recipient's complete history",
        "beam_size": args.beam_size,
        "batch_size": args.batch_size,
        "decode_batch_size": args.decode_batch_size,
        "source_sha256": file_hash(Path(__file__)),
        "depths": depths,
        "content_weight": args.content_weight,
        "history_window": args.history_window,
        "coverage_only": args.coverage_only,
        "seed": args.seed,
        "control_seed": control_seeds[0] if len(control_seeds) == 1 else None,
        "control_seeds": control_seeds,
        "limit": args.limit,
        "evaluation_users_sha256": hashlib.sha256(
            np.asarray(users["user_idx"].to_list(), dtype="<i8").tobytes()
        ).hexdigest(),
        "content_embeddings": str(content_path) if content is not None else None,
        "content_sha256": file_hash(content_path) if content is not None else None,
        "hybrid_prefix_slots": slot_budgets[0] if len(slot_budgets) == 1 else None,
        "hybrid_slot_budgets": slot_budgets,
        "two_tower": tower_metadata,
        "per_user_results": {
            "path": str(per_user_path),
            "sha256": file_hash(per_user_path),
            "target_rank": "Rows follow user_idx; columns follow methods; 1..50 = rank, 0 = absent from top-50",
        },
        "paired_comparisons": paired_comparisons,
        "shuffled_history_control": shuffle_report,
        "shuffle_seed": args.shuffle_seed if donor_indices is not None else None,
        "shuffle_donor_users_sha256": hashlib.sha256(
            np.asarray(users["user_idx"].to_numpy()[donor_indices], dtype="<i8").tobytes()
        ).hexdigest() if donor_indices is not None else None,
        "matched_control_summary": matched_summary,
        "timing_seconds": {**timings, "evaluation_total": time.perf_counter() - evaluation_started,
                           "including_setup": time.perf_counter() - started},
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "protocol": {
            "exact": "Full-SID beam search, not exhaustive catalog scoring",
            "normalization": "log P(prefix|history) - log(full catalog bucket size before seen filtering)",
            "content": "Add content_weight * cosine(candidate, normalized mean history embedding); heuristic score",
            "two_tower": "Pure encode(real history) dot item_vectors; full and prefix rankings reuse identical full-catalog scores, no prefix-score mixing",
            "timing": "Diagnostic run costs, not deployment latency: TT prefix ranking reuses full scoring; shuffle adds a second beam pass. CUDA peak includes resident models/features.",
            "shuffle": "Seeded cycle over selected users, no self-pairs; replace GenRec input only, preserve recipient target and full seen history. Pooled across history lengths.",
            "hybrid": "Exact first 50-slots, then up to slots items outside exact top-50; backfill from exact",
            "full_two_tower_hybrid": "Same quotas/novelty rules, but additions come from full-catalog TT without prefix restriction",
            "paired_comparisons": "Method minus reference on identical users, by target group; approximate pointwise 95% normal intervals of paired mean differences. Not adjusted for multiple comparisons or training variation; unreliable for sparse differences (including all-zero samples). Per-user ranks permit resampling without inference.",
            "popularity": "Counts of next-item training targets seq[1:-2] across all users; no held-out targets",
            "top_train": "Top train-frequency prefixes, matching each user's retained prefix count",
            "matched_train_size": "Uniform without replacement within joint factor-two train-frequency/catalog-size bins; zero frequency separate",
            "control_limits": "Matched bins use the learned selection's histogram, not an independent retrieval model; pool sizes are approximate matches. Original prefixes remain eligible; inspect overlap and repeat control seeds.",
        },
        "popularity_controls": control_reports,
        **summarize(store),
    }

    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Saved:", args.output)
    print("Per-user ranks:", per_user_path)

    for d in depths:
        values = report["prefix_diagnostics"][str(d)]
        print(
            f"Depth {d}:",
            {
                g: values.get(g, {}).get("prefix_survival")
                for g in ("head", "mid", "tail", "unseen")
            },
        )
        for name, summary in control_reports.items():
            print(f"  {name} coverage:", {
                g: summary[str(d)].get(g, {}).get("candidate_coverage")
                for g in ("head", "mid", "tail", "unseen")
            })
        if shuffle_report is not None:
            print("  shuffled-history coverage:", {
                g: shuffle_report[str(d)].get(g, {}).get("candidate_coverage")
                for g in ("head", "mid", "tail", "unseen")
            })

    if not args.coverage_only:
        for method in methods:
            recall = report["ranking"][method]["all"]["recall@50"]
            print(f"{method}: Recall@50={recall:.4%}")
    for comparison in paired_comparisons:
        values = comparison["groups"]["all"]["recall@50"]
        print(f"{comparison['method']} minus {comparison['reference']}: "
              f"delta Recall@50={values['delta']:.4%}, "
              f"approximate 95% CI={values['ci95_normal']}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--sid-source",
        choices=SID_SOURCES,
        default="twotower",
    )
    p.add_argument(
        "--split",
        choices=("valid", "test"),
        default="valid",
    )
    p.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3],
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1024,
        help="0 = all users",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--control-seed", type=int, default=42,
                   help="Random matched-control seed, independent of the user sample seed")
    p.add_argument("--control-seeds", type=int, nargs="+",
                   help="Evaluate several matched-control seeds in one beam pass; overrides --control-seed")
    p.add_argument("--shuffle-histories", action="store_true",
                   help="Also measure prefix coverage using a different user's history")
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--beam-size", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--decode-batch-size", type=int, default=512)
    p.add_argument("--history-window", type=int, default=20)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--content-embeddings", type=Path,
                   help="Optional item-aligned embeddings, e.g. data/processed/item_two_tower_embeddings.npy")
    p.add_argument("--hybrid-prefix-slots", type=int, default=10,
                   help="Reserve up to this many of the final 50 slots for novel prefix/full-TT candidates")
    p.add_argument("--hybrid-slot-sweep", type=int, nargs="+",
                   help="Evaluate several quotas in one run, e.g. 5 10; overrides --hybrid-prefix-slots")
    p.add_argument("--two-tower-checkpoint", type=Path,
                   help="Enable learned Two-Tower reranking, full-catalog baseline and quota hybrids")
    p.add_argument("--coverage-only", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/main/diagnostics/prefix_retrieval/prefix_retrieval_valid.json"),
    )

    args = p.parse_args()

    if (args.limit < 0 or min(args.batch_size, args.decode_batch_size) <= 0
            or args.beam_size < 50 or min(args.seed, args.control_seed) < 0):
        p.error("Invalid limit, batch size or beam size")

    if (args.history_window < 1 or not math.isfinite(args.content_weight)
            or args.content_weight < 0 or not 0 <= args.hybrid_prefix_slots <= 50):
        p.error("Invalid history window or content weight")
    if args.shuffle_seed < 0 or any(seed < 0 for seed in (args.control_seeds or [])):
        p.error("Control/shuffle seeds must be nonnegative")
    if any(not 0 <= slots <= 50 for slots in (args.hybrid_slot_sweep or [])):
        p.error("Hybrid quotas must be between 0 and 50")
    if args.coverage_only and args.two_tower_checkpoint:
        p.error("--two-tower-checkpoint requires ranking; omit --coverage-only")
    if args.shuffle_histories and args.limit == 1:
        p.error("Shuffled-history control needs at least two users")

    run(args)


if __name__ == "__main__":
    main()
