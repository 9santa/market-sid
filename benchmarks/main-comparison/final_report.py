"""Summarize the complete frozen benchmark, including paired user uncertainty."""

import argparse
import json
from pathlib import Path

from common import ROOT, GROUPS, KS, file_hash, save_json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmarks/final-main/reports/comparison.md",
    )
    args = parser.parse_args()
    selection = json.loads(args.selection_file.read_text())
    checkpoints = {r["sha256"]: r for r in selection["checkpoints"]}
    runs, seen, reference = {}, set(), None
    for path in args.results:
        result = json.loads(path.read_text())
        digest = result["checkpoint_sha256"]
        if (
            digest not in checkpoints
            or digest in seen
            or result["split"] != "test"
            or result.get("selection_sha256") != file_hash(args.selection_file)
            or result["protocol_sha256"] != selection["protocol_sha256"]
            or result["data_signature"] != selection["data_signature"]
            or result["candidate_catalog"] != "all"
            or result["user_subset"] != "all"
        ):
            parser.error(
                f"Result is not a unique full-test member of the frozen selection: {path}"
            )
        user_path = Path(result["per_user_results"]["path"])
        if file_hash(user_path) != result["per_user_results"]["sha256"]:
            parser.error(f"Per-user results changed: {user_path}")
        with np.load(user_path, allow_pickle=False) as arrays:
            rows = {key: arrays[key] for key in ("user_idx", "target", "group", "rank")}
        if reference is None:
            reference = rows
        if any(
            not np.array_equal(rows[k], reference[k])
            for k in ("user_idx", "target", "group")
        ):
            parser.error("Ordered users/targets/groups differ")
        if np.any((rows["rank"] < 0) | (rows["rank"] > 50)):
            parser.error("Invalid per-user ranks")
        condition, seed = checkpoints[digest]["condition"], checkpoints[digest]["seed"]
        runs.setdefault(condition, {})[seed] = rows["rank"]
        seen.add(digest)
    if seen != set(checkpoints):
        parser.error("Supply exactly one evaluation for every frozen checkpoint")
    lines = [
        "# Final benchmark",
        "",
        "Percentages: mean ± sample SD over training seeds (ItemKNN is deterministic).",
        "",
    ]
    summary = {}
    for group in GROUPS:
        mask = (
            np.ones(len(reference["group"]), dtype=bool)
            if group == "all"
            else (
                reference["group"] != "unseen"
                if group == "warm"
                else reference["group"] == group
            )
        )
        lines += [f"## {group} (n={int(mask.sum()):,})", ""]
        if not mask.any():
            lines += ["No targets; metrics are N/A.", ""]
            continue
        metrics = [f"{metric}@{k}" for k in KS for metric in ("recall", "ndcg")]
        lines += [
            "| Model | " + " | ".join(metrics) + " |",
            "|---|" + "---:|" * len(metrics),
        ]
        summary[group] = {}
        for name, seed_runs in sorted(runs.items()):
            ranks = np.stack([v[mask] for _, v in sorted(seed_runs.items())])
            cells, values = [], {}
            for metric in metrics:
                k = int(metric.split("@")[1])
                hits = (ranks > 0) & (ranks <= k)
                per_seed = (
                    hits
                    if metric.startswith("recall")
                    else hits / np.log2(np.maximum(ranks, 1) + 1)
                ).mean(axis=1)
                mean, sd = (
                    float(per_seed.mean()),
                    float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                )
                values[metric] = {"mean": mean, "sd": sd, "seeds": len(per_seed)}
                cells.append(f"{100 * mean:.3f} ± {100 * sd:.3f}")
            summary[group][name] = values
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Paired Recall@50 differences vs Content Two-Tower",
        "",
        "User bootstrap (5,000 resamples, seed 20261021), averaging the matched training seeds first.",
        "Intervals condition on this split and these fitted seeds; they do not include dataset/split uncertainty.",
        "Subgroup comparisons are exploratory; the registered primary comparison is GenRec C with Two-Tower SIDs vs Two-Tower on all targets. A/B/D are additional ablations, not test-selected replacements for C.",
        "",
        "| Model | Group | Δ percentage points | 95% interval |",
        "|---|---|---:|---:|",
    ]
    paired = []
    base = runs["two_tower"]
    for name, seed_runs in sorted(runs.items()):
        if name == "two_tower":
            continue
        seeds = sorted(set(seed_runs) & set(base))
        difference = np.stack(
            [(seed_runs[s] > 0).astype(float) - (base[s] > 0) for s in seeds]
        ).mean(axis=0)
        for group in ("all", "head", "mid", "tail", "unseen"):
            mask = (
                np.ones(len(difference), dtype=bool)
                if group == "all"
                else reference["group"] == group
            )
            if not mask.any():
                continue
            delta = difference[mask]
            values, counts = np.unique(delta, return_counts=True)
            # Multinomial resampling is exactly the bootstrap of this discrete per-user mean.
            rng = np.random.default_rng(20261021)
            samples = (
                rng.multinomial(len(delta), counts / len(delta), size=5000)
                @ values
                / len(delta)
            )
            low, high = np.quantile(samples, [0.025, 0.975])
            paired.append(
                {
                    "model": name,
                    "group": group,
                    "n": len(delta),
                    "seeds": seeds,
                    "delta": float(delta.mean()),
                    "low": float(low),
                    "high": float(high),
                }
            )
            lines.append(
                f"| {name} | {group} | {100 * delta.mean():.3f} | [{100 * low:.3f}, {100 * high:.3f}] |"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    save_json(
        args.output.with_suffix(".json"),
        {
            "selection_sha256": file_hash(args.selection_file),
            "results": {str(p): file_hash(p) for p in args.results},
            "metrics": summary,
            "paired": paired,
        },
    )
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
