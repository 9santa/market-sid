"""Train all comparison models with a shared retrieval-based selection rule."""

import argparse
import json
import math
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import (
    HISTORY_DIR,
    CHECKPOINT_DIR,
    DATA_DIR,
    MainData,
    TrainExamples,
    evaluate,
    file_hash,
    save_json,
    load_features,
    load_protocol,
    load_loss_experiment,
    GENERATIVE,
)
from models import ItemKNN, make_model, genrec_variant_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "itemknn",
            "gru4rec",
            "lightgcn",
            "tiger",
            "sasrec",
            "two_tower",
            "genrec",
        ),
        required=True,
    )
    parser.add_argument(
        "--protocol", type=Path, help="Enforce a frozen benchmark protocol"
    )
    parser.add_argument(
        "--rq-checkpoint",
        type=Path,
        help="Matching RQ checkpoint for GenRec tokenizer/vocabularies",
    )
    parser.add_argument(
        "--variant", choices=("A", "B", "C", "D"), help="GenRec variant; default C"
    )
    parser.add_argument(
        "--anchor-lambda", type=float, help="GenRec D anchor penalty; default 0.01"
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--target-gamma", type=float, default=0.0, help="GenRec target-frequency weighting exponent")
    parser.add_argument("--target-beta", type=float, default=1.0, help="Positive smoothing constant for target counts")
    parser.add_argument("--loss-experiment", type=Path, help="Registered validation-only objective ablation")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument(
        "--run-name",
        help="Defaults to model + timestamp; existing runs are never replaced",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--selection-metric",
        choices=("ndcg@10", "recall@20", "recall@50"),
        default="recall@50",
    )
    parser.add_argument(
        "--valid-limit",
        type=int,
        help="First N validation users; omit for full validation",
    )
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument(
        "--n-layers", type=int, help="Default GRU=1, LightGCN=3, TIGER=4"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1e-4,
        help="LightGCN embedding regularization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--amp", action="store_true", help="CUDA float16 autocast for GRU4Rec/TIGER"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch, at most 1024 training examples and 32 validation users",
    )
    parser.add_argument(
        "--neighbors", type=int, default=100, help="ItemKNN neighbors per item"
    )
    parser.add_argument("--similarity-block-size", type=int, default=128)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument(
        "--sid-source", choices=("content", "twotower"), default="content"
    )
    parser.add_argument(
        "--n-heads", type=int, help="Attention heads: TIGER=6, other Transformers=4"
    )
    parser.add_argument(
        "--d-kv", type=int, default=64, help="TIGER per-head attention dimension"
    )
    parser.add_argument(
        "--d-ff", type=int, default=1024, help="TIGER feed-forward dimension"
    )
    parser.add_argument(
        "--decoder-layers", type=int, help="Decoder layers: TIGER=4, GenRec=2"
    )
    parser.add_argument(
        "--user-buckets",
        type=int,
        default=2000,
        help="TIGER user hash vocabulary; 0 disables",
    )
    parser.add_argument("--beam-size", type=int, default=100)
    parser.add_argument("--decode-batch-size", type=int, default=128)
    preliminary, _ = parser.parse_known_args()
    if preliminary.protocol:
        settings = json.loads(preliminary.protocol.read_text())
        parser.set_defaults(
            **{
                **settings["common_training"],
                **settings["training_configs"][preliminary.model],
            }
        )
    args = parser.parse_args()
    if not math.isfinite(args.target_gamma) or args.target_gamma < 0 or not math.isfinite(args.target_beta) or args.target_beta <= 0:
        parser.error("Require finite target-gamma >= 0 and target-beta > 0")
    if args.model != "genrec" and (args.target_gamma != 0 or args.target_beta != 1 or args.loss_experiment):
        parser.error("Target weighting applies only to GenRec")
    if args.protocol and args.target_gamma > 0 and args.loss_experiment is None:
        parser.error("Weighted loss is an additional experiment; supply --loss-experiment")
    if args.model != "genrec" and (
        args.variant is not None or args.anchor_lambda is not None
    ):
        parser.error("--variant and --anchor-lambda apply only to GenRec")
    if args.model == "genrec":
        args.variant = args.variant or "C"
        if args.anchor_lambda is not None and args.variant != "D":
            parser.error("--anchor-lambda applies only to GenRec D")
        args.anchor_lambda = (
            (args.anchor_lambda if args.anchor_lambda is not None else 0.01)
            if args.variant == "D"
            else 0.0
        )
        if not math.isfinite(args.anchor_lambda) or args.anchor_lambda < 0:
            parser.error("anchor-lambda must be finite and nonnegative")
    args.n_heads = args.n_heads or (6 if args.model == "tiger" else 4)
    args.decoder_layers = args.decoder_layers or (4 if args.model == "tiger" else 2)
    positive = (
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.patience,
        args.max_len,
        args.d_model,
        args.neighbors,
        args.similarity_block_size,
        args.n_heads,
        args.d_kv,
        args.d_ff,
        args.decoder_layers,
        args.decode_batch_size,
    )
    if (
        min(positive) <= 0
        or args.beam_size < 50
        or args.user_buckets < 0
        or args.num_workers < 0
        or args.temperature <= 0
    ):
        parser.error(
            "Sizes must be positive, beam-size >= 50, and user-buckets/num-workers >= 0"
        )
    if args.valid_limit is not None and args.valid_limit <= 0:
        parser.error("valid-limit must be positive")
    if args.n_layers is not None and args.n_layers <= 0:
        parser.error("n-layers must be positive")
    if (
        args.lr <= 0
        or min(args.weight_decay, args.reg_weight, args.shrink) < 0
        or not 0 <= args.dropout < 1
    ):
        parser.error("Invalid learning rate, regularization, shrink, or dropout")
    if args.smoke and args.model == "itemknn":
        parser.error(
            "ItemKNN fitting still requires the full matrix; use a small --data-dir for a smoke fit"
        )
    if args.amp and (
        args.model not in ("gru4rec", "tiger", "sasrec", "two_tower", "genrec")
        or args.device != "cuda"
    ):
        parser.error("--amp is only supported for sequence models on CUDA")
    if args.model in ("sasrec", "two_tower", "genrec") and args.d_model % args.n_heads:
        parser.error("d-model must be divisible by n-heads")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_label = (
        f"genrec_{args.sid_source}_{args.variant}"
        if args.model == "genrec"
        else args.model
    )
    run_name = args.run_name or f"{run_label}_{timestamp}"
    if Path(run_name).name != run_name or run_name in (".", ".."):
        parser.error("run-name must be a single directory name")
    run_dir = args.checkpoint_dir / (run_name + ("_smoke" if args.smoke else ""))
    if run_dir.exists():
        parser.error(f"Run already exists: {run_dir}; choose a new --run-name")
    data = MainData(args.data_dir)
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "data_dir",
            "checkpoint_dir",
            "history_dir",
            "run_name",
            "rq_checkpoint",
            "protocol",
            "loss_experiment",
        )
    }
    config["n_layers"] = (
        args.n_layers
        or {
            "itemknn": 1,
            "gru4rec": 1,
            "lightgcn": 3,
            "tiger": 4,
            "sasrec": 2,
            "two_tower": 2,
            "genrec": 2,
        }[args.model]
    )
    config["epochs"] = 1 if args.smoke else args.epochs
    limit = min(args.valid_limit or 32, 32) if args.smoke else args.valid_limit
    config["valid_limit"] = limit
    protocol = load_protocol(args.protocol, data)
    loss_experiment = (load_loss_experiment(args.loss_experiment, args.protocol, config, args.sid_source)
                       if args.loss_experiment else None)
    if protocol:
        for key in ("epochs", "patience", "max_len", "beam_size", "selection_metric"):
            if getattr(args, key) != protocol[key]:
                parser.error(f"Protocol requires {key}={protocol[key]}")
        fixed = {
            **protocol["common_training"],
            **protocol["training_configs"][args.model],
        }
        if any(config[key] != value for key, value in fixed.items()):
            parser.error("Training configuration differs from the frozen protocol")
        if args.valid_limit is not None or args.smoke:
            parser.error(
                "Final-protocol training requires full validation and no --smoke"
            )
        if args.seed not in protocol["seeds"] or args.model not in protocol["models"]:
            parser.error("Model/seed not registered in protocol")
        if args.model in GENERATIVE and args.sid_source not in protocol["sid_sources"]:
            parser.error("SID source not registered in protocol")
        if args.model == "tiger" and args.sid_source != "content":
            parser.error("The final protocol registers TIGER with content SIDs")
        if args.model == "genrec":
            if args.variant not in protocol.get("genrec_variants", ["C"]):
                parser.error(
                    "Variant not registered; use final_protocol_all_variants.json for A/B/C/D"
                )
            if (
                args.variant == "D"
                and args.anchor_lambda != protocol["genrec_anchor_lambda"]
            ):
                parser.error(
                    f"Protocol requires anchor-lambda={protocol['genrec_anchor_lambda']}"
                )
    inputs, sid_metadata = load_features(
        data, args.model, args.sid_source, args.rq_checkpoint
    )
    if args.model in GENERATIVE:
        codes = inputs["item_sids"][1:]
        config["vocab_sizes"] = [max(256, int(codes[:, i].max()) + 1) for i in range(4)]
        if args.model == "genrec":
            codebooks = inputs["rq_codebooks"]
            config["vocab_sizes"] = [len(x) for x in codebooks] + [
                int(codes[:, -1].max()) + 1
            ]
            if args.variant != "A" and any(
                x.shape[1] != args.d_model for x in codebooks
            ):
                parser.error("GenRec d-model must match RQ latent dimension")
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / "best.pt"
    metadata = {
        "model": args.model,
        "implementation": "adapted_tiger_t5" if args.model == "tiger" else args.model,
        "config": config,
        "data_signature": data.signature,
        "num_items": data.num_items,
        "num_users": data.num_users,
        "run_id": timestamp,
        "smoke": args.smoke,
        "selection_metric": f"valid.all.{args.selection_metric}",
        "protocol_sha256": file_hash(args.protocol) if args.protocol else None,
        "history_file": str(
            (args.history_dir / run_dir.name / "history.json").resolve()
        ),
        "selection_rule": "maximize; strict improvement; earliest tied epoch; full validation"
        if limit is None
        else "subset validation",
        "versions": {
            p: version(p)
            for p in ("torch", "numpy", "scipy", "recbole", "transformers")
        },
        "source_signature": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("common.py", "models.py", "train.py")
        },
        **sid_metadata,
    }
    if args.model == "genrec":
        metadata.update(
            genrec_variant=args.variant, variant_config=genrec_variant_config(config)
        )
    if loss_experiment:
        metadata.update(loss_experiment=loss_experiment,
                        loss_experiment_sha256=file_hash(args.loss_experiment))
    print(
        f"Model: {args.model} | device: {args.device} | users: {data.num_users:,} | items: {data.num_items:,}",
        flush=True,
    )
    if args.model == "genrec":
        print(
            f"Variant: {args.variant} | SID source: {args.sid_source} | {metadata['variant_config']}",
            flush=True,
        )
    records = data.records("valid", limit)
    if args.model == "itemknn":
        started = time.perf_counter()
        model = ItemKNN.fit(
            data, args.neighbors, args.similarity_block_size, args.shrink
        )
        matrix = model.similarities
        torch.save(
            {
                **metadata,
                "epoch": 0,
                "similarities": {
                    "data": torch.from_numpy(matrix.data),
                    "indices": torch.from_numpy(matrix.indices),
                    "indptr": torch.from_numpy(matrix.indptr),
                    "shape": list(matrix.shape),
                },
            },
            checkpoint_path,
        )
        metrics, details = evaluate(
            model, args.model, data, records, config, "cpu", args.eval_batch_size
        )
        save_json(
            args.history_dir / run_dir.name / "history.json",
            {
                **metadata,
                "fit_seconds": time.perf_counter() - started,
                "valid_metrics": metrics,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )
    else:
        model = make_model(args.model, config, data, args.device, **inputs)
        if args.model == "genrec" and args.target_gamma > 0:
            metadata["target_weighting"] = model.target_weighting_stats
            print(f"Target weighting: gamma={args.target_gamma}, beta={args.target_beta} | {model.target_weighting_stats}", flush=True)
        metadata["num_parameters"] = sum(p.numel() for p in model.parameters())
        examples = TrainExamples(data, args.model, args.max_len, args.seed)
        dataset = (
            Subset(examples, range(min(1024, len(examples))))
            if args.smoke
            else examples
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.device == "cuda",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
        history = {
            **metadata,
            "num_train_examples": len(dataset),
            "epochs": [],
            "best_epoch": None,
        }
        best, stale = -float("inf"), 0
        print(
            f"Training examples: {len(dataset):,} | parameters: {sum(p.numel() for p in model.parameters()):,}",
            flush=True,
        )
        for epoch in range(1, config["epochs"] + 1):
            examples.epoch = epoch
            model.train()
            started = time.perf_counter()
            total, count = 0.0, 0
            for batch in loader:
                batch = {
                    k: v.to(args.device, non_blocking=True) for k, v in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=args.device, dtype=torch.float16, enabled=args.amp
                ):
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
            metrics, details = evaluate(
                model,
                args.model,
                data,
                records,
                config,
                args.device,
                args.eval_batch_size,
                args.beam_size,
            )
            score = metrics["all"][args.selection_metric]
            if score > best:
                best, stale = score, 0
                history["best_epoch"] = epoch
                torch.save(
                    {
                        **metadata,
                        "epoch": epoch,
                        "valid_metric": score,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )
            else:
                stale += 1
            seconds = time.perf_counter() - started
            history["epochs"].append(
                {
                    "epoch": epoch,
                    "train_loss": total / count,
                    "valid_metrics": metrics,
                    "seconds": seconds,
                    **details,
                }
            )
            save_json(args.history_dir / run_dir.name / "history.json", history)
            print(
                f"Epoch {epoch:02d} | loss={total / count:.6f} | valid {args.selection_metric}={score:.6f} | {seconds:.1f}s",
                flush=True,
            )
            if stale >= args.patience:
                break
        history["finished_at"] = datetime.now(timezone.utc).isoformat()
        history["best_metric"] = best
        history["stopped_early"] = stale >= args.patience
        save_json(args.history_dir / run_dir.name / "history.json", history)
    print(f"Saved: {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
