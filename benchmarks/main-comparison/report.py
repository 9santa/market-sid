"""Compare saved evaluations with matching splits, catalogs, and target populations."""

import argparse
import csv
import json
from pathlib import Path

from common import REPORT_DIR, GROUPS, KS, file_hash
from src.result_paths import resolve_result_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "comparison.md")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    results = []
    reference = None
    for path in args.results:
        path = resolve_result_path(path)
        result = json.loads(path.read_text())
        if not result.get("data_signature") or not result.get("evaluation_users_sha256") or "metrics" not in result:
            parser.error(f"{path}: missing data/user signatures or metrics; rerun its evaluator")
        if result.get("smoke") and not args.allow_smoke:
            parser.error(f"{path}: smoke checkpoint; use --allow-smoke only for a smoke report")
        identity = (result["data_signature"], result["split"], result["candidate_catalog"],
                    result["num_users"], result["evaluation_users_sha256"],
                    {g: m["n"] for g, m in result["metrics"].items()})
        if reference is not None and identity != reference:
            parser.error(f"{path}: data, split, catalog, user count, or group counts differ")
        reference = identity
        if result["candidate_catalog"] != "all":
            parser.error("Only full-catalog evaluations belong in this comparison")
        if result.get("user_subset", "all") != "all" and not args.allow_smoke:
            parser.error(f"{path}: limited evaluation; use full populations for comparison")
        # Existing main GenRec outputs lack a subset field; catch --limit outputs.
        if result.get("model") is None and "_limit" in path.name and not args.allow_smoke:
            parser.error(f"{path}: limited GenRec evaluation")
        name = result.get("model") or f"genrec_{result['sid_source']}_{result.get('genrec_variant', 'C')}"
        results.append((path, name, result))
    beam_sizes = {r["beam_size"] for _, _, r in results if r.get("beam_size") is not None}
    if len(beam_sizes) > 1:
        parser.error("Generative models have different beam sizes; evaluate with the same width")
    lines = ["# Main-dataset model comparison", "",
             f"Split: **{results[0][2]['split']}**. Targets: **{results[0][2]['num_users']:,}**.",
             "Rates and NDCG below are percentages. Exact retrieval and beam retrieval have different computational costs.", ""]
    if args.allow_smoke:
        lines += ["**Smoke/subset comparison — not benchmark-quality results.**", ""]
    metrics = [f"{metric}@{k}" for k in KS for metric in ("recall", "ndcg")]
    csv_rows = []
    for group in GROUPS:
        if group not in results[0][2]["metrics"]:
            continue
        lines += [f"## {group}", "", "| Model | n | " + " | ".join(metrics) + " |",
                  "|---|---:|" + "---:|" * len(metrics)]
        for path, name, result in results:
            row = result["metrics"][group]
            lines.append(f"| {name} | {row['n']} | " + " | ".join(f"{100 * row[k]:4.4f}" for k in metrics) + " |")
            csv_rows.append({"model": name, "group": group, "n": row["n"],
                             **{k: row[k] for k in metrics}, "source": str(path)})
        lines.append("")
    lines += ["## Provenance and selection", "",
              "The comparison checks saved dataset fingerprints and populations. It does not establish equal tuning budgets, parameter counts, or checkpoint selection rules.", ""]
    for path, name, result in results:
        selection = result.get("selection_metric", "Existing GenRec checkpoint; main trainer selects validation token loss")
        lines += [f"- **{name}**: `{path}`; epoch {result['checkpoint_epoch']}; {selection}.",
                  f"  Source SHA-256: `{file_hash(path)}`."]
        if name == "tiger":
            lines.append("  Adapted TIGER with shared-vocabulary normalization; not an exact paper reproduction.")
        if result.get("trie_normalize"):
            lines.append("  GenRec uses trie-valid normalization in this result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {args.output}\nSaved: {csv_path} (metric fractions, not percentages)")


if __name__ == "__main__":
    main()
