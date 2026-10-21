"""Train GenRec on MovieLens prefixes, selecting by validation token loss."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from common import HISTORY_DIR, DATA_DIR, CHECKPOINT_DIR, load_data, make_model
import torch
from torch.utils.data import DataLoader, Subset

from scripts.train_genrec import (
    evaluate_teacher_forced,
    save_history,
    set_seed,
    train_one_epoch,
)
from src.data.genrec_dataset import GenRecTrainDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--rq-checkpoint", type=Path, default=CHECKPOINT_DIR / "rqvae_best.pt")
    parser.add_argument("--variant", choices=("A", "B", "C", "D"), default="B")
    parser.add_argument("--anchor-lambda", type=float, default=0.01)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-encoder-layers", type=int, default=3)
    parser.add_argument("--n-decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch, up to 1024 prefixes and 32 validation users",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if (min(args.max_len, args.d_model, args.n_heads, args.n_encoder_layers,
            args.n_decoder_layers, args.batch_size, args.epochs, args.patience) < 1
            or args.d_model % args.n_heads or args.lr <= 0
            or not 0 <= args.dropout < 1 or min(args.weight_decay, args.anchor_lambda) < 0):
        parser.error("Invalid dimensions, training lengths, or optimizer settings")
    set_seed(args.seed)
    device = torch.device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    item_sids, sequences, _, signature = load_data(args.data_dir)
    rq_checkpoint = torch.load(args.rq_checkpoint, map_location="cpu", weights_only=True)
    if rq_checkpoint["semantic_ids_hash"] != signature["semantic_ids.npy"]:
        raise ValueError("RQ checkpoint does not match these semantic IDs")
    rq_config = rq_checkpoint["config"]
    vocab_sizes = [rq_config["codebook_size"]] * rq_config["num_codebooks"]
    vocab_sizes.append(int(item_sids[:, -1].max()) + 1)
    embedding_modes = {
        "A": "scratch",
        "B": "rq_init",
        "C": "rq_init",
        "D": "rq_anchor",
    }
    variant_config = {
        "embedding_mode": embedding_modes[args.variant],
        "tie_semantic_output": args.variant in ("C", "D"),
        "anchor_lambda": args.anchor_lambda if args.variant == "D" else 0.0,
    }
    codebooks = None
    if args.variant != "A":
        if args.d_model != rq_config["latent_dim"]:
            parser.error("For B/C/D, d-model must match prepare_sids.py --latent-dim")
        codebooks = [
            rq_checkpoint["model_state_dict"][f"quantizer.codebooks.{level}.weight"]
            for level in range(rq_config["num_codebooks"])
        ]
    model_config = {key: getattr(args, key) for key in (
        "max_len", "d_model", "n_heads", "n_encoder_layers", "n_decoder_layers", "dropout",
    )}
    model = make_model(item_sids, model_config, vocab_sizes, variant_config, codebooks).to(device)
    dataset = GenRecTrainDataset(sequences, max_len=args.max_len)
    if args.smoke:
        dataset = Subset(dataset, range(min(1024, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = "_smoke" if args.smoke else ""
    checkpoint_path = args.checkpoint_dir / f"genrec_{args.variant}{suffix}_best.pt"
    history_path = args.history_dir / f"genrec_{args.variant}_{run_id}{suffix}.json"
    history = {
        "run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": model_config, "variant_config": variant_config,
        "data_signature": signature, "num_train_examples": len(dataset),
        "selection_metric": "valid.loss", "best_epoch": None, "best_metric": None,
        "stopped_early": False, "epochs": [],
    }
    save_history(history, history_path)
    print(f"Device: {device} | variant: {args.variant} | users: {len(sequences):,}")
    print(f"Training examples: {len(dataset):,} | batches: {len(loader)}")
    print("Training history:", history_path, flush=True)
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        started = time.perf_counter()
        losses = train_one_epoch(model, loader, optimizer, scaler, device, variant_config)
        valid = evaluate_teacher_forced(
            model, sequences[:32] if args.smoke else sequences, device, args.max_len,
            batch_size=args.batch_size,
        )
        is_best = valid["loss"] < best_loss
        if is_best:
            best_loss = valid["loss"]
            patience_counter = 0
            history["best_epoch"] = epoch
            history["best_metric"] = best_loss
            torch.save({
                "model_state_dict": model.state_dict(), "model_config": model_config,
                "variant": args.variant, "variant_config": variant_config,
                "vocab_sizes": vocab_sizes, "data_signature": signature,
                "epoch": epoch, "val_loss": best_loss, "smoke": args.smoke,
                "history_file": str(history_path),
            }, checkpoint_path)
        else:
            patience_counter += 1
        history["epochs"].append({
            "epoch": epoch, "train_loss": losses["loss"], "train_metrics": losses,
            "valid_metrics": valid, "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.perf_counter() - started, "is_best": is_best,
            "semantic_drift": model.semantic_drift(),
        })
        history["stopped_early"] = patience_counter >= args.patience
        save_history(history, history_path)
        print(
            f"Epoch {epoch:02d} | train={losses['loss']:.4f} | "
            f"val={valid['loss']:.4f} | acc={valid['accuracy']}",
            flush=True,
        )
        if history["stopped_early"]:
            print("Early stopping")
            break
    history["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_history(history, history_path)
    print("Best epoch:", history["best_epoch"], "| checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
