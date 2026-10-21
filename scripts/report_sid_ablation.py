"""Compare main-dataset prefix survival for Current and SID variants A/B/C."""

import argparse
import json
from pathlib import Path

from src.result_paths import resolve_result_path


SOURCES = {"Current": "twotower", "A": "rq4_a", "B": "rq4_b", "C": "rq4_c"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permutation-control", action="store_true",
                        help="Compare original/permuted Two-Tower SIDs instead of RQ-depth variants")
    args = parser.parse_args()
    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    runs = {}
    sources = ({"Original": "twotower_original", "Permuted": "twotower_permuted"}
               if args.permutation_control else SOURCES)
    for name, source in sources.items():
        path = Path(f"artifacts/main/evaluation/genrec_{source}_C_{args.split}{suffix}_prefix_survival_metrics.json")
        runs[name] = json.loads(path.read_text())
        if runs[name]["sid_source"] != source or not runs[name]["diagnose_prefixes"]:
            raise ValueError(f"Wrong SID source or missing diagnostics in {path}")
    baseline = next(iter(runs.values()))
    for name, run in runs.items():
        for key in ("split", "num_users", "beam_size", "batch_size", "candidate_catalog",
                    "data_signature", "config", "variant_config", "genrec_variant"):
            if key not in run or run[key] != baseline[key]:
                raise ValueError(f"{name}: missing or different {key}; regenerate matching evaluations")
        if run["genrec_variant"] != "C":
            raise ValueError("Use GenRec architecture C for every SID variant")
        if run["num_semantic_levels"] != (3 if args.permutation_control or name == "Current" else 4):
            raise ValueError(f"{name}: wrong number of semantic RQ levels")
        if args.permutation_control:
            for key in ("initialization_sha256", "rq_checkpoint_sha256", "evaluation_users_sha256",
                        "catalog_prefix_counts", "trie_normalize", "prior_alpha", "early_stopping_enabled"):
                if run.get(key) is None or run[key] != baseline[key]:
                    raise ValueError(f"{name}: missing or different {key}")
            if run["early_stopping_enabled"]:
                raise ValueError("Use --no-early-stopping for both control runs")
            control = run.get("sid_control")
            if not control or control["role"] != name.lower():
                raise ValueError(f"{name}: missing or incorrect permutation metadata")
            for key in ("seed", "source_sid_sha256", "permutation_sha256", "data_signature", "stratification"):
                if control[key] != baseline["sid_control"][key]:
                    raise ValueError(f"{name}: different control {key}")
            history = json.loads(resolve_result_path(run["training_history"]).read_text())
            if (not history.get("finished_at") or len(history["epochs"]) != run["config"]["epochs"]
                    or history["initialization_sha256"] != run["initialization_sha256"]
                    or history["sid_sha256"] != run["sid_sha256"]):
                raise ValueError(f"{name}: missing or inconsistent completed training budget")
        if {g: v["n"] for g, v in run["metrics"].items()} != {
            g: v["n"] for g, v in baseline["metrics"].items()
        }:
            raise ValueError(f"{name}: target group denominators differ")

    lines = [
        f"GenRec C | split={args.split} | users={baseline['num_users']:,} | beam={baseline['beam_size']}",
        "", "Survival is after pruning, before seen filtering; top-10 is after filtering.",
        "Every percentage uses all targets in that group. Empty groups are N/A.", "",
    ]
    for name, run in runs.items():
        counts = run["catalog_prefix_counts"]
        lines += [
            f"SID {name} ({run['sid_source']})", "",
            f"Distinct catalog prefixes: {counts}. Valid root branches: {counts[0]}.", "",
            "| Target group | n | c1 survives | c1,c2 survive | c1,c2,c3 survive | Full SID survives | Item top-10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for group in ("all", "head", "mid", "tail", "unseen"):
            values = run["metrics"].get(group)
            if values is None:
                lines.append(f"| {group} | 0 | N/A | N/A | N/A | N/A | N/A |")
                continue
            rates = [values[f"prefix_survival@{depth}"] for depth in range(1, 5)]
            rates.append(values["recall@10"])
            if not all(0 <= rate <= 1 for rate in rates) or any(
                later > earlier for earlier, later in zip(rates, rates[1:])
            ):
                raise ValueError(f"{name}/{group}: invalid survival or recall ordering")
            lines.append(f"| {group} | {values['n']:,} | " + " | ".join(f"{rate:.2%}" for rate in rates) + " |")
        lines.append("")
    if args.permutation_control:
        lines += ["Item-level results (percent)", "",
                  "| SID | Group | Recall@10 | Recall@50 | NDCG@50 |",
                  "|---|---|---:|---:|---:|"]
        for name, run in runs.items():
            for group in ("all", "head", "mid", "tail", "unseen"):
                values = run["metrics"].get(group)
                if values:
                    rates = [values[k] for k in ("recall@10", "recall@50", "ndcg@50")]
                    lines.append(f"| {name} | {group} | " + " | ".join(f"{v:.3%}" for v in rates) + " |")
        lines += ["", "Matched initialization, RQ codebooks, training budget, dataset and evaluation users verified.",
                  "One permutation/training seed; not a confidence interval or a claim about all GenRec models.", ""]
    experiment = "sid_permutation" if args.permutation_control else "sid_ablation"
    output = args.output or Path(f"artifacts/main/reports/{experiment}_{args.split}{suffix}.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print("Saved:", output)


if __name__ == "__main__":
    main()
