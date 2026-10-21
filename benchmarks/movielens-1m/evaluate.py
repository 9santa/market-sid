"""Evaluate a MovieLens GenRec checkpoint against the full movie catalog."""

import argparse
import json
from pathlib import Path
import time

from common import DATA_DIR, CHECKPOINT_DIR, ARTIFACT_DIR, load_data, load_model
import torch

from scripts.evaluate_genrec import evaluate_genrec, make_eval_records
from scripts.train_genrec import save_history
from src.models.sid_trie import SIDTrie


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "genrec_A_best.pt")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnose-prefixes", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.beam_size < 50 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Require beam-size >= 50, batch-size > 0 and limit > 0")
    device = torch.device(args.device)
    item_sids, sequences, counts, signature = load_data(args.data_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_model(checkpoint, item_sids, signature, device)
    records = make_eval_records(sequences[:args.limit], counts, args.split)
    print(
        f"Device: {device} | split: {args.split} | "
        f"users: {len(records)} | items: {len(item_sids) - 1}"
    )
    started = time.perf_counter()
    metrics = evaluate_genrec(
        model, SIDTrie(item_sids.numpy()), records, checkpoint["model_config"]["max_len"],
        device, beam_size=args.beam_size, batch_size=args.batch_size,
        diagnose_prefixes=args.diagnose_prefixes,
    )
    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    if args.diagnose_prefixes:
        suffix += "_prefix_survival"
    output = args.output or ARTIFACT_DIR / f"{args.checkpoint.stem}_{args.split}{suffix}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_history({
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "variant": checkpoint["variant"], "smoke": checkpoint["smoke"],
        "data_signature": signature, "split": args.split, "candidate_catalog": "all",
        "num_users": len(records), "beam_size": args.beam_size, "batch_size": args.batch_size,
        "diagnose_prefixes": args.diagnose_prefixes,
        "duration_seconds": time.perf_counter() - started, "metrics": metrics,
    }, output)
    print(json.dumps(metrics, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
