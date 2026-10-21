"""Evaluate one comparison checkpoint against the unchanged main catalog."""

import argparse
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
import torch

from common import ARTIFACT_DIR, DATA_DIR, MainData, evaluate, file_hash, save_json, users_hash
from models import ItemKNN, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    item_sids = None
    if name == "tiger":
        path = args.data_dir / f"item_semantic_ids_{checkpoint['sid_source']}.npy"
        if file_hash(path) != checkpoint["sid_sha256"]:
            raise ValueError("SID file changed since TIGER training")
        codes = np.load(path)
        item_sids = np.vstack([np.zeros((1, codes.shape[1]), dtype=np.int64), codes])
    if name == "itemknn":
        state = checkpoint["similarities"]
        model = ItemKNN(sp.csr_matrix(tuple(state[k].numpy() for k in ("data", "indices", "indptr")),
                                     shape=state["shape"]))
    else:
        model = make_model(name, config, data, args.device, item_sids)
        model.load_state_dict(checkpoint["model_state_dict"])
    records = data.records(args.split, args.limit)
    started = time.perf_counter()
    metrics, details = evaluate(model, name, data, records, config, args.device, args.batch_size, args.beam_size)
    result = {
        "model": name, "implementation": checkpoint["implementation"],
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_hash(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "config": config, "smoke": checkpoint["smoke"],
        "selection_metric": checkpoint["selection_metric"], "data_signature": data.signature,
        "split": args.split, "candidate_catalog": "all", "filter_seen": "complete history",
        "num_users": len(records), "num_items": data.num_items, "user_subset": "first_n" if args.limit else "all",
        "dataset_num_users": data.num_users,
        "evaluation_users_sha256": users_hash([r["user_idx"] for r in records]),
        "batch_size": args.batch_size, "beam_size": args.beam_size if name == "tiger" else None,
        "retrieval": "catalog-constrained beam" if name == "tiger" else "exact full-catalog scores",
        "score_normalization": "shared full vocabulary" if name == "tiger" else "not applicable",
        "duration_seconds": time.perf_counter() - started, "metrics": metrics, **details,
    }
    if name == "tiger":
        result.update({k: checkpoint[k] for k in ("sid_source", "sid_sha256")})
    suffix = f"_limit{args.limit}" if args.limit else ""
    beam = f"_beam{args.beam_size}" if name == "tiger" else ""
    output = args.output or ARTIFACT_DIR / f"{args.checkpoint.parent.name}_{args.split}{suffix}{beam}.json"
    save_json(output, result)
    print(f"{'group':8} {'n':>8} {'R@10':>10} {'N@10':>10} {'R@20':>10} {'R@50':>10}")
    for group, values in metrics.items():
        print(f"{group:8} {values['n']:8d} " + " ".join(f"{values[k]:10.6f}" for k in ("recall@10", "ndcg@10", "recall@20", "recall@50")))
    print(f"Short lists: {details['short_recommendations']} | seen targets: {details['seen_targets']}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
