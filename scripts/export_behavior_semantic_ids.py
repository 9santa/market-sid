from pathlib import Path
import argparse
from collections import defaultdict

import numpy as np
import torch

from src.models.behavior_rqvae import (
    BehaviorAwareRQVAE,
)


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior-weight",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = str(args.behavior_weight).replace(".", "p")

    checkpoint = torch.load(
        CHECKPOINT_DIR / f"rqvae_behavior_{tag}_best.pt",
        map_location=device,
        weights_only=False,
    )

    model = BehaviorAwareRQVAE(
        **checkpoint["base_config"],
        behavior_dim=checkpoint["behavior_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    embeddings = np.load(DATA_DIR / "item_text_embeddings.npy").astype(np.float32)

    all_codes = []

    with torch.inference_mode():
        for start in range(0, len(embeddings), 4096):
            x = torch.from_numpy(embeddings[start : start + 4096]).to(device)

            codes = model.encode_codes(x)

            all_codes.append(codes.cpu().numpy())

    semantic_codes = np.concatenate(all_codes, axis=0)

    # Collision-resolution suffix
    groups = defaultdict(list)

    for idx, sid in enumerate(semantic_codes):
        groups[tuple(map(int, sid))].append(idx)

    collision = np.zeros(
        len(semantic_codes),
        dtype=np.int16,
    )

    for indices in groups.values():
        for suffix, idx in enumerate(indices):
            collision[idx] = suffix

    final_codes = np.column_stack(
        [
            semantic_codes,
            collision,
        ]
    ).astype(np.int16)

    assert len(np.unique(final_codes, axis=0)) == len(final_codes)

    output_path = DATA_DIR / f"item_semantic_ids_behavior_{tag}.npy"

    np.save(
        output_path,
        final_codes,
    )

    print(
        "SID shape:",
        final_codes.shape,
    )

    print(
        "Unique semantic SIDs:",
        len(groups),
    )

    print(
        "Semantic collision rate:",
        1.0 - len(groups) / len(semantic_codes),
    )

    print(
        "Max collision suffix:",
        int(collision.max()),
    )

    for level in range(3):
        counts = np.bincount(
            semantic_codes[:, level],
            minlength=256,
        )

        probs = counts[counts > 0] / counts.sum()

        entropy = -np.sum(probs * np.log(probs))

        print(
            f"Level {level + 1}: "
            f"used="
            f"{np.count_nonzero(counts)}/256 "
            f"entropy={entropy:.4f}"
        )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
