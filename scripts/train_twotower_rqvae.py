from pathlib import Path
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.rqvae import RQVAE


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")

EPOCHS = 30
BATCH_SIZE = 1024
LR = 1e-3
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_path = DATA_DIR / "item_two_tower_embeddings.npy"
    embeddings = np.load(embedding_path).astype(np.float32)
    assert embeddings.ndim == 2 and len(embeddings) > 0
    assert np.isfinite(embeddings).all()
    with embedding_path.open("rb") as stream:
        embedding_hash = hashlib.file_digest(stream, "sha256").hexdigest()

    print("Embeddings:", embeddings.shape)
    print("Device:", device, flush=True)
    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    config = {
        "input_dim": embeddings.shape[1],
        "hidden_dim": 256,
        "latent_dim": 128,
        "num_codebooks": 3,
        "codebook_size": 256,
        "beta": 0.25,
    }
    model = RQVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_reconstruction = float("inf")
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_items = 0

        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = len(x)
            total_loss += output["loss"].item() * n
            total_recon += output["reconstruction_loss"].item() * n
            total_vq += output["vq_loss"].item() * n
            total_items += n

        loss = total_loss / total_items
        recon = total_recon / total_items
        vq = total_vq / total_items
        print(
            f"Epoch {epoch:02d} | loss={loss:.6f} | recon={recon:.6f} | vq={vq:.6f}",
            flush=True,
        )
        if recon < best_reconstruction:
            best_reconstruction = recon
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "reconstruction_loss": recon,
                    "representation": "two_tower",
                    "embedding_sha256": embedding_hash,
                    "seed": SEED,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                },
                CHECKPOINT_DIR / "rqvae_twotower_best.pt",
            )

    print("\nBest epoch:", best_epoch)
    print("Best reconstruction:", best_reconstruction)


if __name__ == "__main__":
    main()
