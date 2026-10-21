from pathlib import Path
import hashlib
import json

import numpy as np
import polars as pl
import torch

from src.data.embedding_cache import embedding_metadata, load_embedding_cache
from src.models.content_two_tower import ContentTwoTower


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")
    text_embeddings = load_embedding_cache(
        DATA_DIR / "item_text_embeddings.npy",
        embedding_metadata(catalog),
    )
    num_items = len(catalog)
    assert catalog["item_idx"].to_list() == list(range(1, num_items + 1))

    mapping_hash = hashlib.sha256(
        json.dumps(catalog.select("item_idx", "parent_asin").rows()).encode("utf-8")
    ).hexdigest()

    checkpoint_path = CHECKPOINT_DIR / "two_tower_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["mapping_hash"] == mapping_hash
    config = checkpoint["config"]

    features = np.zeros((num_items + 1, text_embeddings.shape[1]), dtype=np.float32)
    features[1:] = text_embeddings
    model = ContentTwoTower(
        item_features=torch.from_numpy(features),
        max_len=config["max_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_vectors = []
    with torch.inference_mode():
        for start in range(1, num_items + 1, 8192):
            end = min(start + 8192, num_items + 1)
            item_ids = torch.arange(start, end, device=device)
            all_vectors.append(model.item_vectors(item_ids).cpu().numpy())

    vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)
    assert vectors.shape == (num_items, config["d_model"])
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    output_path = DATA_DIR / "item_two_tower_embeddings.npy"
    np.save(output_path, vectors)
    with checkpoint_path.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": checkpoint["epoch"],
                "mapping_hash": mapping_hash,
                "shape": list(vectors.shape),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print("Device:", device)
    print("Two-Tower embeddings:", vectors.shape)
    print("Norm mean/std:", norms.mean(), norms.std())
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
