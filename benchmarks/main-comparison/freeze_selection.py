"""Freeze all validation-selected models before opening the final test results."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, MainData, file_hash, load_protocol, save_json, checkpoint_matches_protocol, GENERATIVE
import torch


def condition(checkpoint, protocol):
    name = checkpoint["model"]
    if name == "genrec" and "genrec_variants" in protocol:
        return f"genrec_{checkpoint['sid_source']}_{checkpoint['config'].get('variant', 'C')}"
    return name + "_" + checkpoint["sid_source"] if name in GENERATIVE else name


def expected_runs(protocol):
    conditions = [m for m in protocol["models"] if m not in GENERATIVE]
    if "genrec_variants" in protocol:
        conditions += [f"genrec_{source}_{variant}" for source in protocol["sid_sources"]
                       for variant in protocol["genrec_variants"]]
    else:
        conditions += ["genrec_" + s for s in protocol["sid_sources"]]
    conditions += ["tiger_content"]
    return {
        (name, seed)
        for name in conditions
        for seed in ([protocol["seeds"][0]] if name == "itemknn" else protocol["seeds"])
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--allow-missing-seeds", action="store_true",
                        help="Freeze available runs and explicitly record missing condition/seed pairs")
    parser.add_argument("--allow-training-deviations", action="store_true",
                        help="Register additional runs with changed batch size or missing training protocol; label separately")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/final-v1")
    parser.add_argument("--history-dir", type=Path, help="Local history root for checkpoints copied from another machine")
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("final_protocol.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/benchmarks/final-main/selection.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(
            "Selection already frozen; do not overwrite it after reviewing test"
        )
    data = MainData(args.data_dir)
    protocol = load_protocol(args.protocol, data)
    selected, runs = [], set()
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("loss_experiment_sha256") or checkpoint["config"].get("target_gamma", 0) != 0:
            parser.error(f"Objective ablations are separate from the frozen benchmark: {path}")
        if (
            (not checkpoint_matches_protocol(checkpoint, args.protocol, protocol)
             and not (args.allow_training_deviations and checkpoint.get("protocol_sha256") is None))
            or checkpoint["data_signature"] != data.signature
            or checkpoint["smoke"]
            or checkpoint["selection_metric"]
            != "valid.all." + protocol["selection_metric"]
        ):
            parser.error(f"Not a matching protocol checkpoint: {path}")
        fixed = {**protocol["common_training"], **protocol["training_configs"][checkpoint["model"]]}
        if checkpoint["model"] != "itemknn":
            fixed.update({key: protocol[key] for key in ("epochs", "patience", "max_len", "beam_size", "selection_metric")})
        differences = {key: [checkpoint["config"].get(key), value]
                       for key, value in fixed.items() if checkpoint["config"].get(key) != value}
        if differences and not (args.allow_training_deviations and set(differences) <= {"batch_size"}):
            parser.error(f"Training configuration differs from protocol: {path}: {differences}")
        deviations = dict(differences)
        if checkpoint.get("protocol_sha256") is None:
            deviations["training_protocol"] = "not recorded"
        history_path = (args.history_dir / path.parent.name / "history.json"
                        if args.history_dir else Path(checkpoint["history_file"]))
        history = json.loads(history_path.read_text())
        if (
            not history.get("finished_at")
            or history["run_id"] != checkpoint["run_id"]
            or history["data_signature"] != data.signature
        ):
            parser.error(f"Training did not finish normally: {path}")
        if checkpoint["model"] != "itemknn":
            if (
                history["best_epoch"] != checkpoint["epoch"]
                or history["best_metric"] != checkpoint["valid_metric"]
            ):
                parser.error(
                    f"Checkpoint is not the completed run's selected epoch: {path}"
                )
        key = (condition(checkpoint, protocol), checkpoint["config"]["seed"])
        if key in runs:
            parser.error(f"Duplicate condition/seed: {key}")
        runs.add(key)
        selected.append(
            {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "condition": f"{key[0]}_batch{checkpoint['config']['batch_size']}" if deviations else key[0],
                "seed": key[1],
                "epoch": checkpoint["epoch"],
                "valid_metric": checkpoint.get("valid_metric"),
                "training_protocol_sha256": checkpoint["protocol_sha256"],
                "training_deviations": deviations,
                "history_file": str(history_path.resolve()),
                "history_sha256": file_hash(history_path),
            }
        )
    missing = sorted(expected_runs(protocol) - runs)
    unexpected = sorted(runs - expected_runs(protocol))
    if unexpected or (missing and not args.allow_missing_seeds):
        parser.error(
            f"Missing runs: {sorted(expected_runs(protocol) - runs)}; unexpected: {sorted(runs - expected_runs(protocol))}"
        )
    save_json(
        args.output,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": file_hash(args.protocol),
            "data_signature": data.signature,
            "checkpoints": selected,
            "selection_uses_test": False,
            "allow_missing_seeds": args.allow_missing_seeds,
            "additional_comparison": args.allow_training_deviations,
            "missing_runs": [{"condition": name, "seed": seed} for name, seed in missing],
        },
    )
    print("Frozen:", args.output)
    if missing:
        print("Missing condition/seed pairs:", missing)


if __name__ == "__main__":
    main()
