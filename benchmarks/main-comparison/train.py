"""Fit ItemKNN, GRU4Rec, LightGCN, or an adapted TIGER on the main split."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import HISTORY_DIR, CHECKPOINT_DIR, DATA_DIR, MainData, TrainExamples, evaluate, file_hash, save_json
from models import ItemKNN, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("itemknn", "gru4rec", "lightgcn", "tiger"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--run-name", help="Defaults to model + timestamp; existing runs are never replaced")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--selection-metric", choices=("ndcg@10", "recall@20"), default="ndcg@10")
    parser.add_argument("--valid-limit", type=int, help="First N validation users; omit for full validation")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, help="Default GRU=1, LightGCN=3, TIGER=4")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--reg-weight", type=float, default=1e-4, help="LightGCN embedding regularization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="CUDA float16 autocast for GRU4Rec/TIGER")
    parser.add_argument("--smoke", action="store_true", help="One epoch, at most 1024 training examples and 32 validation users")
    parser.add_argument("--neighbors", type=int, default=100, help="ItemKNN neighbors per item")
    parser.add_argument("--similarity-block-size", type=int, default=128)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument("--sid-source", choices=("content", "twotower"), default="content")
    parser.add_argument("--n-heads", type=int, default=6, help="TIGER attention heads")
    parser.add_argument("--d-kv", type=int, default=64, help="TIGER per-head attention dimension")
    parser.add_argument("--d-ff", type=int, default=1024, help="TIGER feed-forward dimension")
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--user-buckets", type=int, default=2000, help="TIGER user hash vocabulary; 0 disables")
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    args = parser.parse_args()
    positive = (args.epochs, args.batch_size, args.eval_batch_size, args.patience, args.max_len,
                args.d_model, args.neighbors, args.similarity_block_size, args.n_heads, args.d_kv,
                args.d_ff, args.decoder_layers, args.decode_batch_size)
    if min(positive) <= 0 or args.beam_size < 50 or args.user_buckets < 0 or args.num_workers < 0:
        parser.error("Sizes must be positive, beam-size >= 50, and user-buckets/num-workers >= 0")
    if args.valid_limit is not None and args.valid_limit <= 0:
        parser.error("valid-limit must be positive")
    if args.n_layers is not None and args.n_layers <= 0:
        parser.error("n-layers must be positive")
    if args.lr <= 0 or min(args.weight_decay, args.reg_weight, args.shrink) < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid learning rate, regularization, shrink, or dropout")
    if args.smoke and args.model == "itemknn":
        parser.error("ItemKNN fitting still requires the full matrix; use a small --data-dir for a smoke fit")
    if args.amp and (args.model not in ("gru4rec", "tiger") or args.device != "cuda"):
        parser.error("--amp is only supported for GRU4Rec/TIGER on CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = args.run_name or f"{args.model}_{timestamp}"
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    run_dir = args.checkpoint_dir / (run_name + ("_smoke" if args.smoke else ""))
    if run_dir.exists():
        parser.error(f"Run already exists: {run_dir}; choose a new --run-name")
    data = MainData(args.data_dir)
    config = {k: v for k, v in vars(args).items() if k not in ("data_dir", "checkpoint_dir", "history_dir", "run_name")}
    config["n_layers"] = args.n_layers or {"itemknn": 1, "gru4rec": 1, "lightgcn": 3, "tiger": 4}[args.model]
    config["epochs"] = 1 if args.smoke else args.epochs
    limit = min(args.valid_limit or 32, 32) if args.smoke else args.valid_limit
    config["valid_limit"] = limit
    sid_metadata, item_sids = {}, None
    if args.model == "tiger":
        sid_path = args.data_dir / f"item_semantic_ids_{args.sid_source}.npy"
        codes = np.load(sid_path)
        if codes.shape != (data.num_items, 4) or not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0):
            raise ValueError("Expected four nonnegative integer SID tokens per catalog item")
        if len(np.unique(codes, axis=0)) != data.num_items:
            raise ValueError("SIDs must be unique")
        # Four disjoint 256-token level vocabularies, including unused codes.
        config["vocab_sizes"] = [max(256, int(codes[:, i].max()) + 1) for i in range(4)]
        item_sids = np.vstack([np.zeros((1, 4), dtype=np.int64), codes]).astype(np.int64)
        sid_metadata = {"sid_source": args.sid_source, "sid_path": str(sid_path.resolve()), "sid_sha256": file_hash(sid_path)}
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best.pt"
    metadata = {
        "model": args.model, "implementation": "adapted_tiger_t5" if args.model == "tiger" else args.model,
        "config": config, "data_signature": data.signature, "num_items": data.num_items,
        "num_users": data.num_users, "run_id": timestamp, "smoke": args.smoke,
        "selection_metric": f"valid.all.{args.selection_metric}",
        "versions": {p: version(p) for p in ("torch", "numpy", "scipy", "recbole", "transformers")},
        "source_signature": {name: file_hash(Path(__file__).parent / name)
                             for name in ("common.py", "models.py", "train.py")},
        **sid_metadata,
    }
    print(f"Model: {args.model} | device: {args.device} | users: {data.num_users:,} | items: {data.num_items:,}", flush=True)
    records = data.records("valid", limit)
    if args.model == "itemknn":
        started = time.perf_counter()
        model = ItemKNN.fit(data, args.neighbors, args.similarity_block_size, args.shrink)
        matrix = model.similarities
        torch.save({**metadata, "epoch": 0, "similarities": {
            "data": torch.from_numpy(matrix.data), "indices": torch.from_numpy(matrix.indices),
            "indptr": torch.from_numpy(matrix.indptr), "shape": list(matrix.shape),
        }}, checkpoint_path)
        metrics, details = evaluate(model, args.model, data, records, config, "cpu", args.eval_batch_size)
        save_json(args.history_dir / run_dir.name / "history.json", {**metadata, "fit_seconds": time.perf_counter() - started,
                                             "valid_metrics": metrics, **details})
    else:
        model = make_model(args.model, config, data, args.device, item_sids)
        examples = TrainExamples(data, args.model, args.max_len, args.seed)
        dataset = Subset(examples, range(min(1024, len(examples)))) if args.smoke else examples
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=args.device == "cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        history = {**metadata, "num_train_examples": len(dataset), "epochs": [], "best_epoch": None}
        best, stale = -float("inf"), 0
        print(f"Training examples: {len(dataset):,} | parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        for epoch in range(1, config["epochs"] + 1):
            examples.epoch = epoch
            model.train()
            started = time.perf_counter()
            total, count = 0.0, 0
            for batch in loader:
                batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=args.amp):
                    loss = model.calculate_loss(batch)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                n = len(batch["item_id"])
                total += float(loss.detach()) * n
                count += n
            metrics, details = evaluate(model, args.model, data, records, config, args.device,
                                       args.eval_batch_size, args.beam_size)
            score = metrics["all"][args.selection_metric]
            if score > best:
                best, stale = score, 0
                history["best_epoch"] = epoch
                torch.save({**metadata, "epoch": epoch, "valid_metric": score,
                            "model_state_dict": model.state_dict()}, checkpoint_path)
            else:
                stale += 1
            seconds = time.perf_counter() - started
            history["epochs"].append({"epoch": epoch, "train_loss": total / count,
                                      "valid_metrics": metrics, "seconds": seconds, **details})
            save_json(args.history_dir / run_dir.name / "history.json", history)
            print(f"Epoch {epoch:02d} | loss={total / count:.6f} | valid {args.selection_metric}={score:.6f} | {seconds:.1f}s", flush=True)
            if stale >= args.patience:
                break
    print(f"Saved: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
