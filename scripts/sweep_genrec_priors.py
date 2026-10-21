"""Evaluate fixed GenRec checkpoints with conditional-prior score correction."""

import polars as pl

from scripts.diagnose_genrec_paths import beam_log_priors
from scripts.train_genrec import save_history


def evaluate_prior_sweep(model, trie, records, sequences, max_len, device, alphas,
                         output, metadata, beam_size=100, batch_size=64,
                         smoothing=0.5, diagnose_prefixes=False, prior_levels=None,
                         trie_normalize=False):
    from scripts.evaluate_genrec import evaluate_genrec

    priors, num_targets = beam_log_priors(trie.item_sids, sequences, device, smoothing)
    levels = list(range(1, model.num_levels + 1)) if prior_levels is None else prior_levels
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        **metadata, "num_users": len(records), "beam_size": beam_size,
        "batch_size": batch_size, "candidate_catalog": "all",
        "prior_smoothing": smoothing, "num_training_targets": num_targets,
        "prior_definition": "GenRec target counts, smoothed over catalog-valid children",
        "score_definition": "sum(log P_model(token|history,prefix) - alpha_level * log P_train(token|prefix))",
        "prior_levels": levels,
        "model_normalization": (
            "catalog-valid children; mask before softmax" if trie_normalize
            else "full vocabulary; mask illegal children without renormalizing"
        ),
        "trie_normalize": trie_normalize,
        "diagnose_prefixes": diagnose_prefixes, "alphas": alphas, "runs": [],
    }
    for alpha in alphas:
        level_alphas = [alpha if level in levels else 0.0 for level in range(1, model.num_levels + 1)]
        print(f"\nConditional-prior alpha={alpha:g} | per-level alphas={level_alphas}", flush=True)
        metrics = evaluate_genrec(
            model, trie, records, max_len, device, beam_size=beam_size,
            batch_size=batch_size, diagnose_prefixes=diagnose_prefixes,
            prior_alpha=alpha, prior_log_probs=priors,
            prior_levels=prior_levels,
            trie_normalize=trie_normalize,
        )
        report["runs"].append({"alpha": alpha, "level_alphas": level_alphas, "metrics": metrics})
        # Preserve completed runs if the sweep is interrupted.
        save_history(report, output)

    rows = []
    print("\nPrior correction comparison (percent)")
    print(f"{'alpha':>6} {'group':<8} {'n':>8} {'R@10':>9} {'NDCG@10':>9} {'R@50':>9} {'NDCG@50':>9}")
    for run in report["runs"]:
        for group in ("all", "head", "mid", "tail", "unseen"):
            values = run["metrics"].get(group)
            if values is None:
                continue
            rows.append({
                "alpha": run["alpha"], "group": group, **values,
                **{f"alpha_c{level}": value for level, value in enumerate(run["level_alphas"], 1)},
            })
            print(f"{run['alpha']:>6g} {group:<8} {values['n']:>8} " + " ".join(
                f"{100 * values[key]:>9.4f}" for key in ("recall@10", "ndcg@10", "recall@50", "ndcg@50")
            ))
    pl.DataFrame(rows).write_csv(output.with_suffix(".csv"))
    print("Saved:", output, output.with_suffix(".csv"))
