"""Evaluate one comparison checkpoint against the unchanged main catalog."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch

from common import ARTIFACT_DIR, DATA_DIR, MainData, evaluate, file_hash, save_json, users_hash, load_features, load_protocol, checkpoint_matches_protocol, GENERATIVE
from models import ItemKNN, make_model, genrec_variant_config
from common import load_loss_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--loss-experiment", type=Path, help="Registered objective ablation; validation only")
    parser.add_argument("--selection-file", type=Path, help="Frozen checkpoint list required for final test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.decode_batch_size) <= 0 or args.beam_size < 50 or (args.limit is not None and args.limit <= 0):
        parser.error("Require positive batch/limit settings and beam-size >= 50")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data = MainData(args.data_dir)
    if checkpoint["data_signature"] != data.signature:
        raise ValueError("Checkpoint and evaluation data differ")
    name, config = checkpoint["model"], dict(checkpoint["config"])
    config["decode_batch_size"] = args.decode_batch_size
    protocol = load_protocol(args.protocol, data)
    if checkpoint.get("loss_experiment_sha256") and args.loss_experiment is None:
        parser.error("This checkpoint requires --loss-experiment")
    if args.loss_experiment:
        if args.split != "valid":
            parser.error("Review the objective experiment on validation before registering any final test comparison")
        load_loss_experiment(args.loss_experiment, args.protocol, config, checkpoint.get("sid_source"))
        if checkpoint.get("loss_experiment_sha256") not in (None, file_hash(args.loss_experiment)):
            parser.error("Loss experiment changed since training")
    if checkpoint.get("protocol_sha256") and args.protocol is None:
        parser.error("This checkpoint requires --protocol")
    if protocol:
        selected_entry = None
        if args.limit is not None or args.beam_size != protocol["beam_size"]:
            parser.error("Final protocol requires full population and the registered beam size")
        if args.split == "test":
            if args.selection_file is None:
                parser.error("Freeze validation-selected checkpoints first; pass --selection-file")
            selection = json.loads(args.selection_file.read_text())
            if (selection["protocol_sha256"] != file_hash(args.protocol)
                    or selection["data_signature"] != data.signature
                    or file_hash(args.checkpoint) not in [r["sha256"] for r in selection["checkpoints"]]):
                parser.error("Checkpoint/protocol/data not in the frozen selection")
            selected_entry = next(r for r in selection["checkpoints"] if r["sha256"] == file_hash(args.checkpoint))
        if not checkpoint_matches_protocol(checkpoint, args.protocol, protocol):
            if not (selected_entry and selection.get("additional_comparison")
                    and selected_entry.get("training_deviations")
                    and checkpoint.get("protocol_sha256") is None):
                parser.error("Checkpoint was not trained with this protocol or registered as an additional run")
    # Reconstruct GenRec's embedding mode before restoring the complete state.
    inputs, features = load_features(data, "tiger" if name == "genrec" else name,
                                     checkpoint.get("sid_source", "content"))
    for key, value in features.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint feature mismatch: {key}")
    if name == "genrec":
        mode = genrec_variant_config(config)["embedding_mode"]
        if mode == "rq_anchor":
            inputs["rq_codebooks"] = [checkpoint["model_state_dict"][f"rq_base_{i}"] for i in range(3)]
        elif mode == "rq_init":
            inputs["rq_codebooks"] = [checkpoint["model_state_dict"][f"sid_embeddings.{i}.weight"] for i in range(3)]
    if name == "itemknn":
        state = checkpoint["similarities"]
        model = ItemKNN(sp.csr_matrix(tuple(state[k].numpy() for k in ("data", "indices", "indptr")),
                                     shape=state["shape"]))
    else:
        model = make_model(name, config, data, args.device, **inputs)
        model.load_state_dict(checkpoint["model_state_dict"])
    records = data.records(args.split, args.limit)
    started = time.perf_counter()
    per_user = []
    metrics, details = evaluate(model, name, data, records, config, args.device, args.batch_size, args.beam_size, per_user)
    result = {
        "model": name, "implementation": checkpoint["implementation"],
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "config": config, "smoke": checkpoint["smoke"],
        "selection_metric": checkpoint["selection_metric"], "data_signature": data.signature,
        "protocol_sha256": file_hash(args.protocol) if args.protocol else checkpoint.get("protocol_sha256"),
        "training_protocol_sha256": checkpoint.get("protocol_sha256"),
        "training_deviations": selected_entry.get("training_deviations", {}) if protocol and selected_entry else {},
        "loss_experiment_sha256": file_hash(args.loss_experiment) if args.loss_experiment else None,
        "target_weighting": checkpoint.get("target_weighting"),
        "selection_sha256": file_hash(args.selection_file) if args.selection_file else None,
        "split": args.split, "candidate_catalog": "all", "filter_seen": "complete history",
        "num_users": len(records), "num_items": data.num_items, "user_subset": "first_n" if args.limit else "all",
        "dataset_num_users": data.num_users,
        "evaluation_users_sha256": users_hash([r["user_idx"] for r in records]),
        "batch_size": args.batch_size, "beam_size": args.beam_size if name in GENERATIVE else None,
        "retrieval": "catalog-constrained beam" if name in GENERATIVE else "exact full-catalog scores",
        "score_normalization": ("shared full vocabulary" if name == "tiger" else
                                "per-level full vocabulary" if name == "genrec" else "not applicable"),
        "duration_seconds": time.perf_counter() - started, "metrics": metrics, **details,
    }
    if name in GENERATIVE:
        result.update({k: checkpoint[k] for k in ("sid_source", "sid_sha256")})
    if name == "genrec":
        result.update(genrec_variant=config.get("variant", "C"), variant_config=genrec_variant_config(config))
    suffix = f"_limit{args.limit}" if args.limit else ""
    beam = f"_beam{args.beam_size}" if name in GENERATIVE else ""
    output = args.output or ARTIFACT_DIR / f"{args.checkpoint.parent.name}_{args.split}{suffix}{beam}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if protocol and (output.exists() or output.with_suffix(".npz").exists()):
        parser.error("Final result already exists; choose a new --output")
    user_path = output.with_suffix(".npz")
    np.savez_compressed(user_path, **{key: np.asarray([r[key] for r in per_user])
                                    for key in ("user_idx", "target", "group", "rank")})
    result["per_user_results"] = {"path": str(user_path.resolve()), "sha256": file_hash(user_path)}
    save_json(output, result)
    print(f"{'group':8} {'n':>8} {'R@10':>10} {'N@10':>10} {'R@20':>10} {'R@50':>10}")
    for group, values in metrics.items():
        print(f"{group:8} {values['n']:8d} " + " ".join(f"{values[k]:10.6f}" for k in ("recall@10", "ndcg@10", "recall@20", "recall@50")))
    print(f"Short lists: {details['short_recommendations']} | seen targets: {details['seen_targets']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
