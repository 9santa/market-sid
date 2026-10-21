"""Encode product metadata, train RQ-VAE, and export unique semantic IDs."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from common import HISTORY_DIR, DATA_DIR, CHECKPOINT_DIR
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_genrec import save_history, set_seed
from src.data.embedding_cache import ENCODER_CONFIG
from src.models.rqvae import RQVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    parser.add_argument("--encoder", default=ENCODER_CONFIG["model_name"])
    parser.add_argument("--revision", help="Encoder checkpoint revision")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-codebooks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.encode_batch_size,
            args.max_tokens,
            args.latent_dim,
            args.codebook_size,
            args.num_codebooks,
        )
        < 1
        or args.lr <= 0
    ):
        parser.error(
            "Dimensions, batch sizes, epochs, token limit and lr must be positive"
        )
    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    catalog = pl.read_parquet(args.data_dir / "catalog.parquet").sort("item_idx")
    revision = args.revision
    if revision is None and args.encoder == ENCODER_CONFIG["model_name"]:
        revision = ENCODER_CONFIG["revision"]
    metadata = {
        "catalog_hash": hashlib.sha256(
            (args.data_dir / "catalog.parquet").read_bytes()
        ).hexdigest(),
        "encoder": args.encoder,
        "revision": revision,
        "max_tokens": args.max_tokens,
        "normalize_embeddings": True,
    }
    embedding_path = args.data_dir / "embeddings.npy"
    metadata_path = args.data_dir / "embeddings.json"
    if (
        embedding_path.exists()
        and metadata_path.exists()
        and json.loads(metadata_path.read_text()) == metadata
    ):
        embeddings = np.load(embedding_path)
        print("Loaded cached embeddings:", embeddings.shape)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            args.encoder, revision=revision, device=args.device
        )
        if args.max_tokens > encoder.max_seq_length:
            parser.error(
                f"Requested token limit exceeds encoder limit {encoder.max_seq_length}"
            )
        encoder.max_seq_length = args.max_tokens
        embeddings = encoder.encode(
            catalog["item_text"].to_list(),
            batch_size=args.encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(embedding_path, embeddings)
        save_history(metadata, metadata_path)
        del encoder
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # RQ initialization should be identical whether embeddings were cached or encoded.
    set_seed(args.seed)
    config = {
        "input_dim": embeddings.shape[1],
        "hidden_dim": 256,
        "latent_dim": args.latent_dim,
        "num_codebooks": args.num_codebooks,
        "codebook_size": args.codebook_size,
        "beta": 0.25,
    }
    model = RQVAE(**config).to(args.device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint_path = args.checkpoint_dir / "rqvae_best.pt"
    history = {
        "config": config,
        "encoder": metadata,
        "seed": args.seed,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": [],
    }
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("loss", "reconstruction_loss", "vq_loss")}
        for (x,) in loader:
            x = x.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            output["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += output[key].item() * len(x)
        metrics = {key: value / len(embeddings) for key, value in totals.items()}
        history["epochs"].append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch:02d} | recon={metrics['reconstruction_loss']:.6f}",
            flush=True,
        )
        if metrics["reconstruction_loss"] < best_loss:
            best_loss = metrics["reconstruction_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "encoder": metadata,
                },
                checkpoint_path,
            )
        save_history(history, args.history_dir / "rqvae_history.json")

    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    codes = np.concatenate(
        [
            model.encode_codes(chunk.to(args.device)).cpu().numpy()
            for chunk in torch.from_numpy(embeddings).split(args.batch_size)
        ]
    )
    counts = defaultdict(int)
    collision_tokens = []
    for row in codes:
        key = tuple(row)
        collision_tokens.append(counts[key])
        counts[key] += 1
    codes = np.column_stack([codes, collision_tokens]).astype(np.int64)
    np.save(args.data_dir / "semantic_ids.npy", codes)
    # Use full codebook vocabularies even if some codes are unused in this catalog.
    checkpoint["semantic_ids_hash"] = hashlib.sha256(
        (args.data_dir / "semantic_ids.npy").read_bytes()
    ).hexdigest()
    torch.save(checkpoint, checkpoint_path)
    print(
        "Semantic IDs:", codes.shape, "collision vocabulary:", max(collision_tokens) + 1
    )
    print("Saved:", checkpoint_path)


if __name__ == "__main__":
    main()
