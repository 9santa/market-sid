from pathlib import Path
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.behavior_rqvae import (
    BehaviorAwareRQVAE,
)


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior-weight",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print(
        "Behavior weight:",
        args.behavior_weight,
    )

    content_embeddings = np.load(DATA_DIR / "item_text_embeddings.npy").astype(
        np.float32
    )

    behavior_embeddings = np.load(DATA_DIR / "item_behavior_embeddings.npy").astype(
        np.float32
    )

    warm_mask = np.load(DATA_DIR / "item_behavior_warm_mask.npy").astype(bool)

    assert len(content_embeddings) == len(behavior_embeddings) == len(warm_mask)

    dataset = TensorDataset(
        torch.from_numpy(content_embeddings),
        torch.from_numpy(behavior_embeddings),
        torch.from_numpy(warm_mask),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    content_checkpoint = torch.load(
        CHECKPOINT_DIR / "rqvae_content_best.pt",
        map_location="cpu",
        weights_only=False,
    )

    base_config = content_checkpoint["config"]

    behavior_dim = behavior_embeddings.shape[1]

    model = BehaviorAwareRQVAE(
        **base_config,
        behavior_dim=behavior_dim,
    )

    incompatible = model.load_state_dict(
        content_checkpoint["model_state_dict"],
        strict=False,
    )

    print("Missing keys:", incompatible.missing_keys)

    print("Unexpected keys:", incompatible.unexpected_keys)

    # Should be empty, the only missing parameter should be 'behavior_head'
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith("behavior_head")
    ]

    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys: {bad_missing}")

    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected keys: ", incompatible.unexpected_keys)

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_epoch = None

    tag = str(args.behavior_weight).replace(".", "p")

    checkpoint_path = CHECKPOINT_DIR / f"rqvae_behavior_{tag}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_behavior = 0.0
        total_items = 0

        for x, behavior_target, behavior_mask in loader:
            x = x.to(device, non_blocking=True)

            behavior_target = behavior_target.to(device, non_blocking=True)

            behavior_mask = behavior_mask.to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(
                x=x,
                behavior_target=behavior_target,
                behavior_mask=behavior_mask,
                behavior_weight=args.behavior_weight,
            )

            loss = output["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            n = len(x)

            total_loss += loss.item() * n

            total_recon += output["reconstruction_loss"].item() * n

            total_vq += output["vq_loss"].item() * n

            total_behavior += output["behavior_loss"].item() * n

            total_items += n

        stats = {
            "loss": total_loss / total_items,
            "recon": total_recon / total_items,
            "vq": total_vq / total_items,
            "behavior": total_behavior / total_items,
        }

        print(
            f"Epoch {epoch:02d} | "
            f"loss={stats['loss']:.6f} | "
            f"recon={stats['recon']:.6f} | "
            f"vq={stats['vq']:.6f} | "
            f"behavior={stats['behavior']:.4f}"
        )

        # Пока checkpoint по joint objective (loss)
        if stats["loss"] < best_loss:
            best_loss = stats["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_config": base_config,
                    "behavior_dim": behavior_dim,
                    "behavior_weight": args.behavior_weight,
                    "epoch": epoch,
                    "train_stats": stats,
                },
                checkpoint_path,
            )

    print()
    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )


if __name__ == "__main__":
    main()
