"""Teacher-forced SID likelihood and training-frequency diagnostics."""

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr

from src.data.genrec_dataset import _pad_left
from scripts.train_genrec import save_history


GROUPS = ("all", "head", "mid", "tail", "unseen")


def path_frequencies(item_sids, sequences, smoothing=0.5):
    # Exactly the target occurrences in GenRecTrainDataset, without held-out items.
    targets = np.concatenate([np.asarray(seq[1:-2], dtype=np.int64) for seq in sequences])
    if not len(targets) or smoothing <= 0:
        raise ValueError("Require training targets and positive prior smoothing")
    item_counts = np.bincount(targets, minlength=len(item_sids))
    codes = item_sids[1:]
    prefix_counts, token_counts, parent_counts, log_priors = [], [], [], []
    parent_ids = np.zeros(len(codes), dtype=np.int64)
    parent_frequency = np.full(len(codes), len(targets), dtype=np.int64)
    for level in range(codes.shape[1]):
        _, first, prefix_ids = np.unique(
            codes[:, :level + 1], axis=0, return_index=True, return_inverse=True,
        )
        counts = np.bincount(prefix_ids, weights=item_counts[1:]).astype(np.int64)
        frequency = counts[prefix_ids]
        children = np.bincount(parent_ids[first])[parent_ids]
        token_frequency = np.bincount(codes[:, level], weights=item_counts[1:]).astype(np.int64)
        prefix_counts.append(frequency)
        token_counts.append(token_frequency[codes[:, level]])
        parent_counts.append(parent_frequency)
        # Smooth over catalog-valid children, including zero-training-count paths.
        log_priors.append(np.log((frequency + smoothing) / (parent_frequency + smoothing * children)))
        parent_ids, parent_frequency = prefix_ids, frequency
    return {
        "num_training_targets": len(targets),
        "item_counts": item_counts,
        "prefix_counts": np.stack(prefix_counts, axis=1),
        "token_counts": np.stack(token_counts, axis=1),
        "parent_counts": np.stack(parent_counts, axis=1),
        "log_priors": np.stack(log_priors, axis=1),
    }


def correlations(frequency, log_probability):
    if len(frequency) < 2 or np.ptp(frequency) == 0 or np.ptp(log_probability) == 0:
        return {"pearson_log1p_frequency": None, "spearman": None}
    return {
        "pearson_log1p_frequency": float(np.corrcoef(np.log1p(frequency), log_probability)[0, 1]),
        "spearman": float(spearmanr(frequency, log_probability).statistic),
    }


def beam_log_priors(item_sids, sequences, device, smoothing=0.5):
    frequencies = path_frequencies(item_sids, sequences, smoothing)
    priors = []
    for level in range(item_sids.shape[1]):
        # SIDTrie.tensors numbers child nodes by the same sorted unique prefixes.
        _, first = np.unique(item_sids[1:, :level + 1], axis=0, return_index=True)
        priors.append(torch.tensor(frequencies["log_priors"][first, level], dtype=torch.float32, device=device))
    return priors, frequencies["num_training_targets"]


def summarize_paths(rows, levels):
    summary = {}
    print("\nMean target token log probabilities (natural log, teacher-forced)")
    print(f"{'group':<8} {'n':>8} " + " ".join(f"{'c' + str(level):>10}" for level in range(1, levels + 1)))
    for group in GROUPS:
        subset = rows if group == "all" else rows.filter(pl.col("group") == group)
        if not len(subset):
            print(f"{group:<8} {0:>8} N/A")
            continue
        details = []
        for level in range(1, levels + 1):
            token = subset[f"logp_token_{level}"].to_numpy()
            prefix = subset[f"logp_prefix_{level}"].to_numpy()
            frequency = subset[f"prefix_train_count_{level}"].to_numpy()
            # Also weight each distinct prefix once, rather than each test target.
            unique = subset.group_by([f"c{i}" for i in range(1, level + 1)]).agg(
                pl.col(f"prefix_train_count_{level}").first(),
                pl.col(f"logp_prefix_{level}").mean(),
            )
            detail = {
                "mean_token_logp": float(token.mean()),
                "median_token_logp": float(np.median(token)),
                "mean_prefix_logp": float(prefix.mean()),
                "zero_count_targets": int(np.count_nonzero(frequency == 0)),
                "prefix_frequency_vs_prefix_logp": correlations(frequency, prefix),
                "token_frequency_vs_token_logp": correlations(subset[f"token_train_count_{level}"].to_numpy(), token),
                "distinct_prefixes": len(unique),
                "unique_prefix_frequency_vs_mean_logp": correlations(
                    unique[f"prefix_train_count_{level}"].to_numpy(), unique[f"logp_prefix_{level}"].to_numpy(),
                ),
            }
            if f"shuffled_logp_token_{level}" in subset.columns:
                shuffled = subset[f"shuffled_logp_token_{level}"].to_numpy()
                detail["mean_shuffled_token_logp"] = float(shuffled.mean())
                detail["mean_real_minus_shuffled_token_logp"] = float((token - shuffled).mean())
            details.append(detail)
        summary[group] = {"n": len(subset), "levels": details}
        print(f"{group:<8} {len(subset):>8} " + " ".join(f"{d['mean_token_logp']:>10.4f}" for d in details))
    return summary


def plot_paths(rows, levels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, levels, figsize=(4 * levels, 7), squeeze=False)
    for level in range(1, levels + 1):
        for group in GROUPS[1:]:
            subset = rows.filter(pl.col("group") == group)
            if not len(subset):
                continue
            for row, (count_name, score_name) in enumerate((
                (f"prefix_train_count_{level}", f"logp_prefix_{level}"),
                (f"token_train_count_{level}", f"logp_token_{level}"),
            )):
                frequency, score = subset[count_name].to_numpy(), subset[score_name].to_numpy()
                # Log-frequency bins show trends without overplotting repeated targets.
                buckets = np.floor(np.log2(frequency + 1)).astype(np.int64)
                bins = np.unique(buckets)
                xs = [frequency[buckets == b].mean() for b in bins]
                ys = [score[buckets == b].mean() for b in bins]
                axes[row, level - 1].plot(xs, ys, marker="o", markersize=3, label=group)
        for row in range(2):
            ax = axes[row, level - 1]
            ax.set_xscale("symlog", linthresh=1)
            ax.set_title(f"SID position {level}")
            ax.set_xlabel("Prefix training count" if row == 0 else "Token training count (this level)")
            ax.set_ylabel("Mean cumulative log P(prefix | history)" if row == 0 else "Mean conditional log P(token | history, prefix)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Target-weighted means by frequency bin; counts use GenRec training targets")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


@torch.inference_mode()
def diagnose_paths(model, records, sequences, user_ids, max_len, device, output,
                   metadata, batch_size=64, smoothing=0.5, shuffle_histories=False):
    model.eval()
    if not records or len(user_ids) != len(records):
        raise ValueError("Require nonempty records and matching user IDs")
    codes = model.item_sids.cpu().numpy()
    frequencies = path_frequencies(codes, sequences, smoothing)
    histories = np.stack([_pad_left(row["history"], max_len) for row in records])
    target_ids = np.asarray([row["target"] for row in records], dtype=np.int64)
    shuffled_indices = None
    if shuffle_histories:
        if len(records) < 2:
            raise ValueError("Shuffled-history comparison requires at least two users")
        order = np.random.default_rng(42).permutation(len(records))
        shuffled_indices = np.empty(len(records), dtype=np.int64)
        shuffled_indices[order] = np.roll(order, 1)
    scores, shuffled_scores = [], []

    def score_batch(history, targets):
        logits, target_codes = model(torch.from_numpy(history).to(device), targets)
        return torch.stack([
            value.float().log_softmax(dim=-1).gather(1, target_codes[:, level, None]).squeeze(1)
            for level, value in enumerate(logits)
        ], dim=1).cpu().numpy()

    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        targets = torch.from_numpy(target_ids[start:stop]).to(device)
        scores.append(score_batch(histories[start:stop], targets))
        if shuffle_histories:
            shuffled_scores.append(score_batch(histories[shuffled_indices[start:stop]], targets))
        print(f"Scored target paths {stop:,}/{len(records):,}", flush=True)
    logp = np.concatenate(scores)
    if not np.isfinite(logp).all():
        raise ValueError("Nonfinite target log probabilities")
    cumulative = np.cumsum(logp, axis=1)
    data = {
        "user_idx": user_ids, "target_item_idx": target_ids,
        "group": [row["group"] for row in records],
        "target_train_count": frequencies["item_counts"][target_ids],
    }
    shuffled = np.concatenate(shuffled_scores) if shuffle_histories else None
    if shuffled is not None and not np.isfinite(shuffled).all():
        raise ValueError("Nonfinite shuffled-history log probabilities")
    for level in range(model.num_levels):
        label = level + 1
        data[f"c{label}"] = codes[target_ids, level]
        data[f"logp_token_{label}"] = logp[:, level]
        data[f"logp_prefix_{label}"] = cumulative[:, level]
        for name, key in (("prefix_train_count", "prefix_counts"), ("token_train_count", "token_counts"),
                          ("parent_train_count", "parent_counts"), ("log_prior_token", "log_priors")):
            data[f"{name}_{label}"] = frequencies[key][target_ids - 1, level]
        if shuffled is not None:
            data[f"shuffled_logp_token_{label}"] = shuffled[:, level]
            data[f"shuffled_logp_prefix_{label}"] = shuffled[:, :label].sum(axis=1)
    rows = pl.DataFrame(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(output.with_suffix(".parquet"))
    summary = summarize_paths(rows, model.num_levels)
    save_history({
        **metadata, "num_users": len(records), "batch_size": batch_size,
        "num_training_targets": frequencies["num_training_targets"],
        "frequency_definition": "GenRec target occurrences: sequence[1:-2] for all training users",
        "group_definition": "Existing head/mid/tail/unseen groups from all train interactions",
        "probability_definition": "Teacher-forced, full-vocabulary softmax, no trie renormalization or seen filtering",
        "prior_smoothing": smoothing, "shuffle_histories": shuffle_histories, "shuffle_seed": 42,
        "groups": summary,
    }, output)
    plot_paths(rows, model.num_levels, output.with_suffix(".png"))
    print("Saved:", output, output.with_suffix(".parquet"), output.with_suffix(".png"))
