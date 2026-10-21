from pathlib import Path
from collections import defaultdict
import hashlib

import numpy as np
import torch

from src.models.rqvae import RQVAE


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        CHECKPOINT_DIR / "rqvae_twotower_best.pt",
        map_location=device,
        weights_only=True,
    )
    embedding_path = DATA_DIR / "item_two_tower_embeddings.npy"
    with embedding_path.open("rb") as stream:
        embedding_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checkpoint["embedding_sha256"] == embedding_hash, "Embeddings changed since RQ training"

    model = RQVAE(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    embeddings = np.load(embedding_path).astype(np.float32)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), 4096):
            x = torch.from_numpy(embeddings[start : start + 4096]).to(device)
            batches.append(model.encode_codes(x).cpu().numpy())
    semantic_codes = np.concatenate(batches, axis=0)

    groups = defaultdict(list)
    for idx, sid in enumerate(semantic_codes):
        groups[tuple(map(int, sid))].append(idx)

    # int32 also handles a collapsed codebook with more than 32767 collisions.
    collision = np.zeros(len(semantic_codes), dtype=np.int32)
    for indices in groups.values():
        for suffix, idx in enumerate(indices):
            collision[idx] = suffix
    final_codes = np.column_stack([semantic_codes, collision]).astype(np.int32)
    assert len(np.unique(final_codes, axis=0)) == len(final_codes)
    output_path = DATA_DIR / "item_semantic_ids_twotower.npy"
    np.save(output_path, final_codes)

    print("Shape:", final_codes.shape)
    print("Unique semantic SIDs:", len(groups))
    print("Collision rate:", 1.0 - len(groups) / len(semantic_codes))
    print("Max collision group:", int(collision.max()) + 1)
    codebook_size = checkpoint["config"]["codebook_size"]
    for level in range(semantic_codes.shape[1]):
        counts = np.bincount(semantic_codes[:, level], minlength=codebook_size)
        probs = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probs * np.log(probs))
        print(
            f"Level {level + 1}: used={np.count_nonzero(counts)}/{codebook_size} "
            f"entropy={entropy:.4f}"
        )
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
